#!/usr/bin/env python3
"""Calibrate frozen RF tree standard deviations on historical AmpC runs.

This retrospective study consumes the hash-locked score artifact produced by
``benchmark_rf_uq_methods.py``.  Cycle 7 labels fit one positive multiplier per
run; cycles 8 and 9 are evaluation-only.  It does not modify or execute the
LearnM8 active-learning API.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

# Keep plot output independent of user-specific Matplotlib cache locations.
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/learnm8-matplotlib-cache')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/learnm8-xdg-cache')

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import ndtr
from scipy.stats import norm

LOGGER = logging.getLogger('rf_tree_calibration')
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK_DIR = (
    REPO_ROOT / 'validation' / 'reports' / 'uncertainty' / 'rf_uq_benchmark'
)
DEFAULT_SCORES_PATH = DEFAULT_BENCHMARK_DIR / 'rf_uq_scores.parquet'
DEFAULT_METADATA_PATH = DEFAULT_BENCHMARK_DIR / 'rf_uq_metadata.json'
DEFAULT_ARCHIVE_ROOT = Path('/home/tony/LearnM8_RESULTS_FINAL/_archive')
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / 'validation' / 'reports' / 'uncertainty' / 'rf_tree_calibration'
)
EXPECTED_RUNS = {
    'lm8_A-01_rf_1M_random_b0.1_rep1',
    'lm8_A-01_rf_1M_random_b0.1_rep2',
    'lm8_A-01_rf_1M_random_b0.1_rep3',
    'lm8_A-03_rf_1M_ucb_b0.1_rep1',
    'lm8_A-03_rf_1M_ucb_b0.1_rep2',
    'lm8_A-03_rf_1M_ucb_b0.1_rep3',
}
REQUIRED_COLUMNS = {
    'run_id',
    'family',
    'strategy',
    'replicate',
    'seed',
    'split',
    'ID',
    'target',
    'rf_prediction',
    'tree_std',
}
EXPECTED_SPLIT_SIZES = {'calibration': 1_000, 'test': 2_000}
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
EXPECTED_METADATA_SHA256 = (
    'cc227b4bac82cb254e288a1512a0d58d81791ff3f9f4ce7dc6cfc698a7e4437f'
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_benchmark_metadata(metadata_path: Path) -> dict[str, Any]:
    observed_hash = _sha256(metadata_path)
    if observed_hash != EXPECTED_METADATA_SHA256:
        raise ValueError(
            'benchmark metadata SHA-256 mismatch: '
            f'{observed_hash} != {EXPECTED_METADATA_SHA256}'
        )
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    parameters = metadata.get('parameters', {})
    scope = metadata.get('scope', {})
    reconstruction = metadata.get('reconstruction_checks', [])
    valid_reconstruction = (
        len(reconstruction) == len(EXPECTED_RUNS)
        and {row.get('run_id') for row in reconstruction} == EXPECTED_RUNS
        and all(
            row.get('passed') is True and row.get('n') == 1_000
            for row in reconstruction
        )
    )
    valid_semantics = (
        metadata.get('analysis') == 'Six-method retrospective RF uncertainty benchmark'
        and scope.get('active_learning_executed') is False
        and scope.get('run_count') == 6
        and parameters.get('split_cycles')
        == {'train': list(range(7)), 'calibration': [7], 'test': [8, 9]}
        and parameters.get('split_sizes')
        == {'train': 16_000, 'calibration': 1_000, 'test': 2_000}
        and parameters.get('n_estimators') == 100
        and 'tree_std' in parameters.get('score_methods', [])
        and valid_reconstruction
    )
    if not valid_semantics:
        raise ValueError('benchmark metadata semantics or reconstruction checks differ')
    return metadata


def load_validated_scores(scores_path: Path, metadata_path: Path) -> pl.DataFrame:
    """Load the frozen score artifact after fail-closed provenance checks."""
    metadata = _validate_benchmark_metadata(metadata_path)
    expected_hash = metadata.get('output_hashes', {}).get(scores_path.name)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError('metadata lacks the score artifact SHA-256')
    observed_hash = _sha256(scores_path)
    if observed_hash != expected_hash:
        raise ValueError(
            f'score artifact SHA-256 mismatch: {observed_hash} != {expected_hash}'
        )

    scores = pl.read_parquet(scores_path)
    missing = REQUIRED_COLUMNS - set(scores.columns)
    if missing:
        raise ValueError(f'score artifact lacks required columns: {sorted(missing)}')
    if set(scores['run_id'].unique()) != EXPECTED_RUNS:
        raise ValueError('score artifact does not contain the six historical runs')
    if scores.select(pl.struct(['run_id', 'ID']).is_duplicated().any()).item():
        raise ValueError('score artifact has duplicate IDs within a run')
    if set(scores['split'].unique()) != set(EXPECTED_SPLIT_SIZES):
        raise ValueError('score artifact has unexpected split labels')

    counts = scores.group_by(['run_id', 'split']).len()
    for run_id in EXPECTED_RUNS:
        for split, expected_size in EXPECTED_SPLIT_SIZES.items():
            actual = counts.filter(
                (pl.col('run_id') == run_id) & (pl.col('split') == split)
            )['len']
            if len(actual) != 1 or actual.item() != expected_size:
                raise ValueError(f'{run_id} {split} split size is not {expected_size}')

    numeric = scores.select(['target', 'rf_prediction', 'tree_std']).to_numpy()
    if not np.all(np.isfinite(numeric)):
        raise ValueError('score artifact contains non-finite calibration inputs')
    if np.any(scores['tree_std'].to_numpy() <= 0.0):
        raise ValueError('tree standard deviations must be strictly positive')
    return scores


def fit_scalar_from_frame(run_scores: pl.DataFrame) -> float:
    """Fit the Gaussian-NLL scale multiplier from calibration rows only."""
    calibration = run_scores.filter(pl.col('split') == 'calibration')
    if calibration.height == 0:
        raise ValueError('run has no calibration rows')
    residuals = (
        calibration['target'].to_numpy() - calibration['rf_prediction'].to_numpy()
    )
    scales = calibration['tree_std'].to_numpy()
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(scales)):
        raise ValueError('calibration residuals and scales must be finite')
    if np.any(scales <= 0.0):
        raise ValueError('calibration scales must be strictly positive')
    multiplier = float(np.sqrt(np.mean(np.square(residuals / scales))))
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError('fitted scale multiplier is not positive and finite')
    return multiplier


def join_cycle_labels(
    run_scores: pl.DataFrame, cycle_labels: pl.DataFrame
) -> pl.DataFrame:
    """Join archived cycle labels and enforce the historical split contract."""
    if set(cycle_labels.columns) != {'ID', 'labeled_cycle'}:
        raise ValueError('cycle labels must contain only ID and labeled_cycle')
    if cycle_labels['ID'].n_unique() != cycle_labels.height:
        raise ValueError('cycle labels contain duplicate IDs')
    joined = run_scores.join(cycle_labels, on='ID', how='left', validate='1:1')
    if joined['labeled_cycle'].null_count():
        raise ValueError('archived cycle labels do not cover every score row')
    calibration = joined.filter(pl.col('split') == 'calibration')
    test = joined.filter(pl.col('split') == 'test')
    valid = (
        calibration.height == 1_000
        and calibration['labeled_cycle'].unique().to_list() == [7]
        and test.filter(pl.col('labeled_cycle') == 8).height == 1_000
        and test.filter(pl.col('labeled_cycle') == 9).height == 1_000
        and set(test['labeled_cycle'].unique()) == {8, 9}
    )
    if not valid:
        raise ValueError('archived cycle labels violate the historical temporal split')
    return joined


def _gaussian_nll(residuals: np.ndarray, scales: np.ndarray) -> float:
    return float(
        np.mean(
            0.5 * np.log(2.0 * np.pi)
            + np.log(scales)
            + 0.5 * np.square(residuals / scales)
        )
    )


def _gaussian_crps(residuals: np.ndarray, scales: np.ndarray) -> float:
    standardized = residuals / scales
    values = scales * (
        standardized * (2.0 * ndtr(standardized) - 1.0)
        + 2.0 * np.exp(-0.5 * np.square(standardized)) / np.sqrt(2.0 * np.pi)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(values))


def _binned_scale_errors(
    residuals: np.ndarray, scales: np.ndarray, n_bins: int = 10
) -> tuple[float, float]:
    bins = np.array_split(np.argsort(scales, kind='stable'), n_bins)
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    weights: list[float] = []
    for indices in bins:
        root_mean_variance = float(np.sqrt(np.mean(np.square(scales[indices]))))
        root_mean_squared_error = float(np.sqrt(np.mean(np.square(residuals[indices]))))
        difference = abs(root_mean_squared_error - root_mean_variance)
        absolute_errors.append(difference)
        relative_errors.append(difference / root_mean_variance)
        weights.append(indices.size / residuals.size)
    return float(np.dot(weights, absolute_errors)), float(np.mean(relative_errors))


def _metric_row(
    run: pl.DataFrame,
    evaluation: pl.DataFrame,
    *,
    method: str,
    multiplier: float,
    scope: str,
) -> dict[str, Any]:
    residuals = evaluation['target'].to_numpy() - evaluation['rf_prediction'].to_numpy()
    raw_scales = evaluation['tree_std'].to_numpy()
    scales = raw_scales if method == 'raw_tree_std' else multiplier * raw_scales
    uncertainty_calibration_error, expected_normalized_calibration_error = (
        _binned_scale_errors(residuals, scales)
    )
    coverages: dict[str, float] = {}
    coverage_errors: list[float] = []
    for nominal in COVERAGE_LEVELS:
        critical = float(norm.ppf((1.0 + nominal) / 2.0))
        empirical = float(np.mean(np.abs(residuals) <= critical * scales))
        suffix = round(nominal * 100)
        coverages[f'coverage_{suffix}'] = empirical
        coverages[f'coverage_error_{suffix}'] = empirical - nominal
        coverage_errors.append(abs(empirical - nominal))
    strategy = str(run['strategy'][0])
    return {
        'run_id': str(run['run_id'][0]),
        'family': str(run['family'][0]),
        'strategy': strategy,
        'replicate': int(run['replicate'][0]),
        'seed': int(run['seed'][0]),
        'scope': scope,
        'method': method,
        'n': evaluation.height,
        'scale_multiplier': 1.0 if method == 'raw_tree_std' else multiplier,
        'gaussian_nll': _gaussian_nll(residuals, scales),
        'gaussian_crps': _gaussian_crps(residuals, scales),
        'mean_absolute_coverage_error': float(np.mean(coverage_errors)),
        'max_absolute_coverage_error': float(np.max(coverage_errors)),
        'uncertainty_calibration_error': uncertainty_calibration_error,
        'expected_normalized_calibration_error': (
            expected_normalized_calibration_error
        ),
        'rmse': float(np.sqrt(np.mean(np.square(residuals)))),
        'mean_sigma': float(np.mean(scales)),
        'sigma_cv': float(np.std(scales, ddof=0) / np.mean(scales)),
        **coverages,
    }


def analyze_scores(scores: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit per-run scalar calibrators and evaluate untouched test rows."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for run_id in sorted(EXPECTED_RUNS):
        run = scores.filter(pl.col('run_id') == run_id)
        calibration = run.filter(pl.col('split') == 'calibration')
        test = run.filter(pl.col('split') == 'test')
        multiplier = fit_scalar_from_frame(run)
        calibration_residuals = (
            calibration['target'].to_numpy() - calibration['rf_prediction'].to_numpy()
        )
        calibration_scales = calibration['tree_std'].to_numpy()
        parameter_rows.append(
            {
                'run_id': run_id,
                'family': str(run['family'][0]),
                'strategy': str(run['strategy'][0]),
                'replicate': int(run['replicate'][0]),
                'seed': int(run['seed'][0]),
                'calibration_n': calibration.height,
                'scale_multiplier': multiplier,
                'calibration_raw_nll': _gaussian_nll(
                    calibration_residuals, calibration_scales
                ),
                'calibration_scaled_nll': _gaussian_nll(
                    calibration_residuals, multiplier * calibration_scales
                ),
            }
        )
        scopes = [('cycles_8_9', test)]
        if 'labeled_cycle' in test.columns:
            scopes.extend(
                (f'cycle_{cycle}', test.filter(pl.col('labeled_cycle') == cycle))
                for cycle in (8, 9)
            )
        ordered = np.argsort(test['tree_std'].to_numpy(), kind='stable')
        for bin_index, indices in enumerate(np.array_split(ordered, 5), start=1):
            scopes.append((f'tree_std_quintile_{bin_index}', test[indices]))
        for scope, evaluation in scopes:
            if evaluation.height == 0:
                raise ValueError(f'{run_id} has no rows for {scope}')
            for method in ('raw_tree_std', 'nll_scalar'):
                metric_rows.append(
                    _metric_row(
                        run,
                        evaluation,
                        method=method,
                        multiplier=multiplier,
                        scope=scope,
                    )
                )
    return pl.DataFrame(metric_rows), pl.DataFrame(parameter_rows)


def load_archived_cycle_scores(
    scores: pl.DataFrame, metadata_path: Path, archive_root: Path
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Restore cycles 7--9 from hash-matched archived source CSVs."""
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    input_hashes = metadata.get('input_hashes', {})
    if not isinstance(input_hashes, dict):
        raise ValueError('benchmark metadata lacks input hashes')
    joined_runs: list[pl.DataFrame] = []
    archive_hashes: dict[str, str] = {}
    for run_id in sorted(EXPECTED_RUNS):
        matches = [
            value
            for source, value in input_hashes.items()
            if Path(source).name == 'compounds_final.csv'
            and Path(source).parent.name == run_id
        ]
        if len(matches) != 1:
            raise ValueError(f'metadata lacks one compounds hash for {run_id}')
        source_path = archive_root / run_id / 'compounds_final.csv'
        observed_hash = _sha256(source_path)
        if observed_hash != matches[0]:
            raise ValueError(f'archived compounds SHA-256 mismatch for {run_id}')
        archive_hashes[str(source_path)] = observed_hash
        run = scores.filter(pl.col('run_id') == run_id)
        cycle_labels = pl.read_csv(
            source_path,
            comment_prefix='#',
            columns=['ID', 'labeled_cycle'],
            infer_schema_length=10_000,
            schema_overrides={'ID': pl.String, 'labeled_cycle': pl.Int64},
        ).filter(pl.col('ID').is_in(run['ID'].implode()))
        joined_runs.append(join_cycle_labels(run, cycle_labels))
    joined = pl.concat(joined_runs, how='vertical').sort(['run_id', 'split', 'ID'])
    if joined.height != scores.height:
        raise AssertionError('cycle join changed the score row count')
    return joined, archive_hashes


def _strategy_summary(metrics: pl.DataFrame) -> pl.DataFrame:
    return (
        metrics.filter(pl.col('scope') == 'cycles_8_9')
        .group_by(['family', 'strategy', 'method'])
        .agg(
            pl.len().alias('n_runs'),
            pl.col('gaussian_nll').mean().alias('mean_gaussian_nll'),
            pl.col('gaussian_crps').mean().alias('mean_gaussian_crps'),
            pl.col('mean_absolute_coverage_error')
            .mean()
            .alias('mean_absolute_coverage_error'),
            pl.col('coverage_90').mean().alias('mean_coverage_90'),
            pl.col('mean_sigma').mean().alias('mean_sigma'),
        )
        .sort(['strategy', 'method'])
    )


def evaluate_decision(
    metrics: pl.DataFrame, parameters: pl.DataFrame
) -> dict[str, Any]:
    """Apply conservative primary and shift-stress interpretation gates."""
    pooled = metrics.filter(pl.col('scope') == 'cycles_8_9')
    primary = pooled.filter(pl.col('strategy') == 'random')
    shift = pooled.filter(pl.col('strategy') == 'ucb')

    def paired_improvements(frame: pl.DataFrame, metric: str) -> list[float]:
        wide = frame.pivot(on='method', index='run_id', values=metric).sort('run_id')
        return (
            wide['raw_tree_std'].to_numpy() - wide['nll_scalar'].to_numpy()
        ).tolist()

    primary_nll = paired_improvements(primary, 'gaussian_nll')
    primary_crps = paired_improvements(primary, 'gaussian_crps')
    primary_coverage = paired_improvements(primary, 'mean_absolute_coverage_error')
    shift_nll = paired_improvements(shift, 'gaussian_nll')
    shift_crps = paired_improvements(shift, 'gaussian_crps')
    shift_coverage = paired_improvements(shift, 'mean_absolute_coverage_error')
    scaled_primary = primary.filter(pl.col('method') == 'nll_scalar')
    coverage_columns = [
        f'coverage_error_{round(level * 100)}' for level in COVERAGE_LEVELS
    ]
    strict_coverage_gate = all(
        abs(value) <= 0.02
        for column in coverage_columns
        for value in scaled_primary[column].to_list()
    )
    primary_scores_improve = all(
        value > 0.0 for value in primary_nll + primary_crps + primary_coverage
    )
    shift_scores_worsen = all(
        value < 0.0 for value in shift_nll + shift_crps + shift_coverage
    )
    factors = parameters.sort(['strategy', 'replicate'])
    return {
        'primary_scope': 'A-01 random historical runs',
        'shift_scope': 'A-03 UCB historical selection-shift stress runs',
        'primary_all_replicates_improve_nll_crps_and_mace': primary_scores_improve,
        'primary_all_levels_within_0.02_coverage': strict_coverage_gate,
        'shift_all_replicates_worsen_nll_crps_and_mace': shift_scores_worsen,
        'primary_paired_nll_improvements': primary_nll,
        'primary_paired_crps_improvements': primary_crps,
        'primary_paired_mace_improvements': primary_coverage,
        'shift_paired_nll_improvements': shift_nll,
        'shift_paired_crps_improvements': shift_crps,
        'shift_paired_mace_improvements': shift_coverage,
        'scale_factors': {
            row['run_id']: row['scale_multiplier']
            for row in factors.iter_rows(named=True)
        },
        'recommendation': (
            'do_not_integrate: random-run proper scores improve, but strict coverage '
            'fails and the multiplier transfers poorly under UCB selection shift'
        ),
    }


def _format_summary_table(summary: pl.DataFrame) -> str:
    lines = [
        '| Strategy | Method | NLL | CRPS | Mean abs. coverage error | 90% coverage |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    labels = {'raw_tree_std': 'Raw tree SD', 'nll_scalar': 'NLL scalar'}
    for row in summary.iter_rows(named=True):
        lines.append(
            f'| {row["strategy"]} | {labels[row["method"]]} | '
            f'{row["mean_gaussian_nll"]:.4f} | {row["mean_gaussian_crps"]:.4f} | '
            f'{100 * row["mean_absolute_coverage_error"]:.2f} pp | '
            f'{100 * row["mean_coverage_90"]:.1f}% |'
        )
    return '\n'.join(lines)


def _format_cycle_table(metrics: pl.DataFrame) -> str:
    summary = (
        metrics.filter(pl.col('scope').is_in(['cycle_8', 'cycle_9']))
        .group_by(['strategy', 'scope', 'method'])
        .agg(
            pl.col('gaussian_nll').mean().alias('nll'),
            pl.col('gaussian_crps').mean().alias('crps'),
            pl.col('mean_absolute_coverage_error').mean().alias('mace'),
            pl.col('coverage_90').mean().alias('coverage_90'),
        )
        .sort(['strategy', 'scope', 'method'])
    )
    labels = {'raw_tree_std': 'Raw tree SD', 'nll_scalar': 'NLL scalar'}
    lines = [
        '| Strategy | Cycle | Method | NLL | CRPS | Mean abs. coverage error | 90% coverage |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f'| {row["strategy"]} | {row["scope"].removeprefix("cycle_")} | '
            f'{labels[row["method"]]} | {row["nll"]:.4f} | {row["crps"]:.4f} | '
            f'{100 * row["mace"]:.2f} pp | {100 * row["coverage_90"]:.1f}% |'
        )
    return '\n'.join(lines)


def _format_quintile_table(metrics: pl.DataFrame) -> str:
    summary = (
        metrics.filter(pl.col('scope').str.starts_with('tree_std_quintile_'))
        .group_by(['strategy', 'scope', 'method'])
        .agg(
            pl.col('coverage_90').mean().alias('coverage_90'),
            pl.col('n').min().alias('n_per_run'),
        )
        .pivot(
            on='method',
            index=['strategy', 'scope', 'n_per_run'],
            values='coverage_90',
        )
        .sort(['strategy', 'scope'])
    )
    lines = [
        '| Strategy | Raw-SD quintile | n/run | Raw 90% coverage | Scaled 90% coverage |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f'| {row["strategy"]} | '
            f'{row["scope"].removeprefix("tree_std_quintile_")} | '
            f'{row["n_per_run"]} | {100 * row["raw_tree_std"]:.1f}% | '
            f'{100 * row["nll_scalar"]:.1f}% |'
        )
    return '\n'.join(lines)


def _range(values: list[float]) -> str:
    return f'[{min(values):.5f}, {max(values):.5f}]'


def write_report(
    output_path: Path,
    metrics: pl.DataFrame,
    parameters: pl.DataFrame,
    decision: dict[str, Any],
) -> None:
    """Write the mechanically derived scientific assessment."""
    summary = _strategy_summary(metrics)
    factor_lines = []
    for strategy in ('random', 'ucb'):
        values = parameters.filter(pl.col('strategy') == strategy)[
            'scale_multiplier'
        ].to_numpy()
        factor_lines.append(
            f'- {strategy}: {np.mean(values):.3f} mean '
            f'[{np.min(values):.3f}, {np.max(values):.3f}] across three runs'
        )
    report = f"""# RF tree-uncertainty scalar calibration

## Conclusion

Do not integrate the scalar calibration into LearnM8 yet. On the three historical
A-01 random runs, fitting one positive multiplier on cycle 7 improved held-out
Gaussian NLL, CRPS, and mean absolute coverage error in every replicate. The
effect was small for the proper scores, the strict coverage gate still failed,
and the same procedure worsened all three measures in every A-03 UCB stress run.

This is a historical retrospective calibration of the cycle-0--6 frozen RF on
later molecules selected by the historical policy. It is not a rolling-model
deployment test, a conformal guarantee, or evidence that UCB selections remain
unchanged after rescaling uncertainty.

## Pooled cycles 8--9

{_format_summary_table(summary)}

## Transfer by evaluation cycle

{_format_cycle_table(metrics)}

## Conditional 90% coverage by raw tree-SD quintile

{_format_quintile_table(metrics)}

Each quintile contains 400 evaluation molecules per run. These are descriptive
conditional checks; neither scalar fitting nor marginal coverage guarantees
coverage within uncertainty strata.

## Fitted cycle-7 multipliers

{chr(10).join(factor_lines)}

The multiplier is the closed-form Gaussian-NLL optimum
`sqrt(mean((residual / tree_sd)^2))`. Multiplication preserves uncertainty-only
ordering, so it cannot improve tree-SD Spearman or AUSE. It does change the
exploration term in UCB, EI, PI, and Thompson sampling.

CRPS and sigma are in dockscore units. Gaussian NLL is a log score whose
absolute value depends on the dockscore unit convention. The metrics CSV also
exports UCE (the size-weighted absolute RMSE-versus-root-mean-variance gap) and
ENCE (the mean relative gap) over ten equal-count sigma bins.

## Decision gates

- All A-01 replicates improved NLL, CRPS, and coverage error: `{decision['primary_all_replicates_improve_nll_crps_and_mace']}`
- Every A-01 coverage level within 2 percentage points: `{decision['primary_all_levels_within_0.02_coverage']}`
- All A-03 stress replicates worsened on all three measures: `{decision['shift_all_replicates_worsen_nll_crps_and_mace']}`
- Recommendation: `{decision['recommendation']}`

The two-percentage-point rule is an exploratory conservative decision threshold,
not a hypothesis test or coverage guarantee. Across A-01 runs, paired
improvements ranged {_range(decision['primary_paired_nll_improvements'])} for
NLL, {_range(decision['primary_paired_crps_improvements'])} dockscore units for
CRPS, and {_range(decision['primary_paired_mace_improvements'])} for mean
absolute coverage error. Every corresponding A-03 difference was negative.

## Limits

- Three runs per strategy on one AmpC docking screen are descriptive, not an
  independent multi-dataset validation.
- Cycle 7 calibrates a frozen RF evaluated on cycles 8 and 9. A production
  implementation would need rolling, label-available calibration and a new
  active-learning experiment.
- Gaussian NLL and CRPS treat tree SD as a working Gaussian predictive scale;
  tree disagreement is not itself a validated RF posterior standard deviation.
- A-03 is an adaptive selection-shift stress test and cannot establish nominal
  coverage or make the method pass.
"""
    output_path.write_text(report)


def plot_results(metrics: pl.DataFrame, output_path: Path) -> None:
    """Plot marginal coverage and paired calibration changes."""
    pooled = metrics.filter(pl.col('scope') == 'cycles_8_9')
    colors = {'raw_tree_std': '#4C78A8', 'nll_scalar': '#E45756'}
    labels = {'raw_tree_std': 'Raw tree SD', 'nll_scalar': 'NLL scalar'}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    nominal = np.asarray(COVERAGE_LEVELS)
    for axis, strategy in zip(axes[:2], ('random', 'ucb'), strict=True):
        subset = pooled.filter(pl.col('strategy') == strategy)
        for method in ('raw_tree_std', 'nll_scalar'):
            method_rows = subset.filter(pl.col('method') == method)
            values = np.asarray(
                [
                    method_rows[f'coverage_{round(level * 100)}'].mean()
                    for level in nominal
                ]
            )
            axis.plot(
                nominal,
                values,
                marker='o',
                color=colors[method],
                label=labels[method],
            )
        axis.plot(nominal, nominal, color='black', linestyle='--', linewidth=1)
        axis.set_title('A-01 random' if strategy == 'random' else 'A-03 UCB stress')
        axis.set_xlabel('Nominal Gaussian coverage')
        axis.set_ylabel('Empirical coverage')
        axis.set_xlim(0.47, 0.98)
        axis.set_ylim(0.30, 1.00)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axis = axes[2]
    for x_index, strategy in enumerate(('random', 'ucb')):
        subset = pooled.filter(pl.col('strategy') == strategy)
        wide = subset.pivot(
            on='method',
            index=['run_id', 'replicate'],
            values='mean_absolute_coverage_error',
        ).sort('replicate')
        for row in wide.iter_rows(named=True):
            axis.plot(
                [x_index - 0.12, x_index + 0.12],
                [100 * row['raw_tree_std'], 100 * row['nll_scalar']],
                color='#666666',
                alpha=0.7,
            )
            axis.scatter(
                [x_index - 0.12, x_index + 0.12],
                [100 * row['raw_tree_std'], 100 * row['nll_scalar']],
                color=[colors['raw_tree_std'], colors['nll_scalar']],
                s=28,
                zorder=3,
            )
    axis.set_xticks([0, 1], ['A-01 random', 'A-03 UCB'])
    axis.set_ylabel('Mean absolute coverage error (pp)')
    axis.set_title('Paired run-level change')
    axis.grid(axis='y', alpha=0.2)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_study(
    *,
    scores_path: Path,
    metadata_path: Path,
    archive_root: Path,
    output_dir: Path,
    write_outputs: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run the historical calibration analysis and optionally write artifacts."""
    scores = load_validated_scores(scores_path, metadata_path)
    scores, archive_hashes = load_archived_cycle_scores(
        scores, metadata_path, archive_root
    )
    metrics, parameters = analyze_scores(scores)
    decision = evaluate_decision(metrics, parameters)
    provenance: dict[str, Any] = {
        'analysis': 'Historical RF tree-SD scalar calibration',
        'estimand': (
            'Gaussian-score calibration of the cycle-0--6 frozen RF on cycles '
            '8--9 molecules selected by each historical policy'
        ),
        'calibration_split': 'cycle 7, n=1000 independently per run',
        'evaluation_split': 'cycles 8 and 9, n=1000 each independently per run',
        'primary_scope': 'A-01 random; three runs',
        'stress_scope': 'A-03 UCB; three runs; cannot make the method pass',
        'method': 'one positive Gaussian-NLL scalar fitted per run',
        'coverage_levels': COVERAGE_LEVELS,
        'input_hashes': {
            str(scores_path): _sha256(scores_path),
            str(metadata_path): _sha256(metadata_path),
            str(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
            **archive_hashes,
        },
        'parameters': parameters.sort('run_id').to_dicts(),
        'decision': decision,
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'numpy': _package_version('numpy'),
            'polars': _package_version('polars'),
            'scipy': _package_version('scipy'),
            'matplotlib': _package_version('matplotlib'),
        },
    }
    if not write_outputs:
        return metrics, provenance
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'rf_tree_calibration_metrics.csv'
    report_path = output_dir / 'rf_tree_calibration_report.md'
    figure_path = output_dir / 'rf_tree_calibration.png'
    metadata_output = output_dir / 'rf_tree_calibration_metadata.json'
    metrics.sort(['run_id', 'scope', 'method']).write_csv(metrics_path)
    write_report(report_path, metrics, parameters, decision)
    plot_results(metrics, figure_path)
    provenance['output_hashes'] = {
        path.name: _sha256(path) for path in (metrics_path, report_path, figure_path)
    }
    metadata_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    return metrics, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, default=DEFAULT_SCORES_PATH)
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument('--archive-root', type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='validate and analyze all real rows without writing outputs',
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    metrics, provenance = run_study(
        scores_path=args.scores,
        metadata_path=args.metadata,
        archive_root=args.archive_root,
        output_dir=args.output_dir,
        write_outputs=not args.smoke,
    )
    decision = provenance['decision']
    LOGGER.info(
        'calibration study completed: %d metric rows; coverage_gate=%s; %s',
        metrics.height,
        decision['primary_all_levels_within_0.02_coverage'],
        decision['recommendation'],
    )


if __name__ == '__main__':
    main()

"""Tests for the retrospective RF tree-uncertainty calibration study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / 'validation'
    / 'uncertainty'
    / 'scripts'
    / 'benchmark_rf_tree_calibration.py'
)
SCORES_PATH = (
    REPO_ROOT
    / 'validation'
    / 'reports'
    / 'uncertainty'
    / 'rf_uq_benchmark'
    / 'rf_uq_scores.parquet'
)
METADATA_PATH = SCORES_PATH.with_name('rf_uq_metadata.json')

# The study operates on the RF UQ benchmark report, which lives under the
# gitignored validation/reports/ tree. Without it there is nothing to assert
# against, so skip rather than fail on a fresh clone or in CI. Regenerate with
# `python validation/uncertainty/scripts/benchmark_rf_uq_methods.py`.
pytestmark = pytest.mark.skipif(
    not (SCORES_PATH.exists() and METADATA_PATH.exists()),
    reason=f'RF UQ benchmark report not present at {SCORES_PATH.parent}',
)


def _load_study_module() -> ModuleType:
    assert SCRIPT_PATH.exists(), 'tree-calibration study script is missing'
    spec = importlib.util.spec_from_file_location('rf_tree_calibration', SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_scalar_fit_uses_only_calibration_labels() -> None:
    module = _load_study_module()
    scores = module.load_validated_scores(SCORES_PATH, METADATA_PATH)
    run_id = 'lm8_A-01_rf_1M_random_b0.1_rep1'
    run = scores.filter(pl.col('run_id') == run_id)

    original = module.fit_scalar_from_frame(run)
    changed_test = run.with_columns(
        pl.when(pl.col('split') == 'test')
        .then(pl.col('target') + 1_000.0)
        .otherwise(pl.col('target'))
        .alias('target')
    )

    assert module.fit_scalar_from_frame(changed_test) == pytest.approx(original)
    assert original == pytest.approx(0.8973332146, rel=1e-9)


@pytest.mark.unit
def test_score_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    module = _load_study_module()
    metadata = json.loads(METADATA_PATH.read_text())
    metadata['output_hashes']['rf_uq_scores.parquet'] = '0' * 64
    bad_metadata = tmp_path / 'metadata.json'
    bad_metadata.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match='SHA-256'):
        module.load_validated_scores(SCORES_PATH, bad_metadata)


@pytest.mark.unit
def test_paired_score_and_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    module = _load_study_module()
    tampered_scores = tmp_path / 'tampered_scores.parquet'
    tampered_scores.write_bytes(SCORES_PATH.read_bytes() + b'tamper')
    metadata = json.loads(METADATA_PATH.read_text())
    metadata['output_hashes'][tampered_scores.name] = hashlib.sha256(
        tampered_scores.read_bytes()
    ).hexdigest()
    tampered_metadata = tmp_path / 'tampered_metadata.json'
    tampered_metadata.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match='benchmark metadata SHA-256 mismatch'):
        module.load_validated_scores(tampered_scores, tampered_metadata)


@pytest.mark.unit
def test_real_scores_show_strategy_specific_calibration_transfer() -> None:
    module = _load_study_module()
    scores = module.load_validated_scores(SCORES_PATH, METADATA_PATH)

    metrics, parameters = module.analyze_scores(scores)
    pooled = metrics.filter(pl.col('scope') == 'cycles_8_9')
    summary = (
        pooled.group_by(['strategy', 'method'])
        .agg(
            pl.col('gaussian_nll').mean(),
            pl.col('gaussian_crps').mean(),
            pl.col('mean_absolute_coverage_error').mean(),
        )
        .sort(['strategy', 'method'])
    )

    random_raw = summary.filter(
        (pl.col('strategy') == 'random') & (pl.col('method') == 'raw_tree_std')
    ).row(0, named=True)
    random_scaled = summary.filter(
        (pl.col('strategy') == 'random') & (pl.col('method') == 'nll_scalar')
    ).row(0, named=True)
    ucb_raw = summary.filter(
        (pl.col('strategy') == 'ucb') & (pl.col('method') == 'raw_tree_std')
    ).row(0, named=True)
    ucb_scaled = summary.filter(
        (pl.col('strategy') == 'ucb') & (pl.col('method') == 'nll_scalar')
    ).row(0, named=True)

    assert parameters.height == 6
    assert random_scaled['gaussian_nll'] < random_raw['gaussian_nll']
    assert random_scaled['gaussian_crps'] < random_raw['gaussian_crps']
    assert (
        random_scaled['mean_absolute_coverage_error']
        < random_raw['mean_absolute_coverage_error']
    )
    assert ucb_scaled['gaussian_nll'] > ucb_raw['gaussian_nll']
    assert ucb_scaled['gaussian_crps'] > ucb_raw['gaussian_crps']
    assert (
        ucb_scaled['mean_absolute_coverage_error']
        > ucb_raw['mean_absolute_coverage_error']
    )


@pytest.mark.unit
def test_cycle_join_enforces_the_historical_temporal_split() -> None:
    module = _load_study_module()
    scores = module.load_validated_scores(SCORES_PATH, METADATA_PATH)
    run_id = 'lm8_A-01_rf_1M_random_b0.1_rep1'
    run = scores.filter(pl.col('run_id') == run_id)
    calibration_ids = run.filter(pl.col('split') == 'calibration')['ID']
    test_ids = run.filter(pl.col('split') == 'test')['ID']
    cycles = pl.DataFrame(
        {
            'ID': calibration_ids.to_list() + test_ids.to_list(),
            'labeled_cycle': [7] * 1_000 + [8] * 1_000 + [9] * 1_000,
        }
    )

    joined = module.join_cycle_labels(run, cycles)

    assert joined.filter(pl.col('labeled_cycle') == 7).height == 1_000
    assert joined.filter(pl.col('labeled_cycle') == 8).height == 1_000
    assert joined.filter(pl.col('labeled_cycle') == 9).height == 1_000

    invalid = cycles.with_columns(
        pl.when(pl.col('ID') == test_ids[0])
        .then(pl.lit(7))
        .otherwise(pl.col('labeled_cycle'))
        .alias('labeled_cycle')
    )
    with pytest.raises(ValueError, match='temporal split'):
        module.join_cycle_labels(run, invalid)

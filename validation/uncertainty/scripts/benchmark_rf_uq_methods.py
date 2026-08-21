#!/usr/bin/env python3
"""Benchmark six RF uncertainty scores on existing 1M AmpC runs.

This is a retrospective analysis of three A-01 random and three A-03 UCB
runs. It refits only the 19,000 molecules already labelled in each run. It
does not execute active learning and does not change the LearnM8 API.

Usage::

    conda run -n learnm8 python \
        validation/uncertainty/scripts/benchmark_rf_uq_methods.py --smoke
    conda run -n learnm8 python \
        validation/uncertainty/scripts/benchmark_rf_uq_methods.py --stage all
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

# Make Matplotlib PDF metadata deterministic across plot-only reruns.
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil
from mapie.regression import ConformalizedQuantileRegressor, SplitConformalRegressor
from quantile_forest import RandomForestQuantileRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin

from learnm8.features import extract_features
from learnm8.learners.base import _preprocess_features
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.visualization import style

REPO_ROOT = Path(__file__).resolve().parents[3]


LOGGER = logging.getLogger('rf_uq_benchmark')

EXPECTED_RUNS = (('A-01', 'random'), ('A-03', 'ucb'))
REPLICATE_SEEDS = {1: 42, 2: 142, 3: 242}
TRAIN_CYCLES = tuple(range(7))
CALIBRATION_CYCLE = 7
TEST_CYCLES = (8, 9)
SPLIT_SIZES = {'train': 16_000, 'calibration': 1_000, 'test': 2_000}
N_ESTIMATORS = 100
LOCAL_K_DEFAULT = 25
LOCAL_K_VALUES = (10, 25, 50)
LOCAL_QUERY_CHUNK = 64
IJ_CHUNK_SIZE = 2_048
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
REMOVAL_FRACTIONS = np.linspace(0.0, 0.90, 19)
SCORE_METHODS = (
    'tree_std',
    'ij_se',
    'tree_mad',
    'tree_iqr',
    'qrf_i80_width',
    'local_oob_residual_k25',
)
QRF_SENSITIVITY_WIDTHS = {
    'central_50': (0.25, 0.75),
    'central_80': (0.10, 0.90),
    'central_90': (0.05, 0.95),
}
QRF_QUANTILES = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
PROFILE_REPETITIONS = 3

T = TypeVar('T')


@dataclass(frozen=True)
class RunContext:
    """Validated temporal split for one existing active-learning run."""

    run_id: str
    family: str
    strategy: str
    replicate: int
    seed: int
    path: Path
    train_ids: tuple[str, ...]
    train_smiles: tuple[str, ...]
    train_targets: np.ndarray
    calibration_ids: tuple[str, ...]
    calibration_smiles: tuple[str, ...]
    calibration_targets: np.ndarray
    test_ids: tuple[str, ...]
    test_smiles: tuple[str, ...]
    test_targets: np.ndarray


@dataclass(frozen=True)
class IJResult:
    """Raw and finite-tree-corrected infinitesimal-jackknife values."""

    raw_variance: np.ndarray
    corrected_variance: np.ndarray
    correction: np.ndarray
    standard_error: np.ndarray
    nonpositive_mask: np.ndarray


@dataclass(frozen=True)
class QRFResult:
    """One fitted QRF and its requested conditional quantiles."""

    model: RandomForestQuantileRegressor
    quantiles: np.ndarray


class _PeakRSSMonitor:
    """Sample process-tree RSS while a measured phase executes."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self.start_mb = _process_tree_rss_mb()
        self.peak_mb = self.start_mb
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.peak_mb = max(self.peak_mb, _process_tree_rss_mb())

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, _process_tree_rss_mb())


class _QRFQuantileAdapter(RegressorMixin, BaseEstimator):
    """Expose one quantile of a fitted QRF as a prefit sklearn regressor."""

    def __init__(self, model: RandomForestQuantileRegressor, quantile: float) -> None:
        self.model = model
        self.quantile = quantile
        self.is_fitted_ = True
        self.fitted_ = True
        self.n_features_in_ = model.n_features_in_

    def fit(self, features: np.ndarray, targets: np.ndarray) -> _QRFQuantileAdapter:
        del features, targets
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model.predict(features, quantiles=self.quantile), dtype=np.float64
        )


def _process_tree_rss_mb() -> float:
    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss / (1024.0 * 1024.0)


def _measure(
    phase: str,
    function: Callable[[], T],
    *,
    run_id: str,
    method: str | None,
    measurement: str,
    profile_repetition: int | None,
) -> tuple[T, dict[str, Any]]:
    monitor = _PeakRSSMonitor()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    monitor.start()
    try:
        result = function()
    finally:
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        monitor.stop()
    row = {
        'record_type': 'phase',
        'measurement': measurement,
        'profile_repetition': profile_repetition,
        'run_id': run_id,
        'phase': phase,
        'method': method,
        'wall_seconds': wall_seconds,
        'cpu_seconds': cpu_seconds,
        'start_rss_mb': monitor.start_mb,
        'absolute_peak_rss_mb': monitor.peak_mb,
        'incremental_peak_rss_mb': max(0.0, monitor.peak_mb - monitor.start_mb),
    }
    return result, row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_array(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} contains non-finite values')
    return result


def _manifest_row(manifest: pl.DataFrame, run_id: str) -> dict[str, Any]:
    rows = manifest.filter(pl.col('run_id') == run_id).to_dicts()
    if len(rows) != 1:
        raise ValueError(f'manifest must contain exactly one row for {run_id}')
    return rows[0]


def _read_split(run_dir: Path, run_id: str) -> dict[str, Any]:
    path = run_dir / 'compounds_final.csv'
    if not path.is_file():
        raise FileNotFoundError(f'missing compounds file: {path}')
    compounds = pl.read_csv(
        path,
        comment_prefix='#',
        columns=['ID', 'SMILES', 'labeled_cycle', 'dockscore'],
        infer_schema_length=10_000,
        schema_overrides={
            'ID': pl.String,
            'SMILES': pl.String,
            'labeled_cycle': pl.Int64,
            'dockscore': pl.Float64,
        },
    )
    if compounds.height != 1_000_000:
        raise ValueError(f'{run_id} has {compounds.height} compounds, expected 1M')
    if compounds['ID'].null_count() or compounds['ID'].n_unique() != compounds.height:
        raise ValueError(f'{run_id} has null or duplicate compound IDs')
    if compounds['SMILES'].null_count():
        raise ValueError(f'{run_id} has null SMILES')

    split_frames = {
        'train': compounds.filter(pl.col('labeled_cycle').is_in(TRAIN_CYCLES)),
        'calibration': compounds.filter(pl.col('labeled_cycle') == CALIBRATION_CYCLE),
        'test': compounds.filter(pl.col('labeled_cycle').is_in(TEST_CYCLES)),
    }
    id_sets: dict[str, set[str]] = {}
    for name, frame in split_frames.items():
        if frame.height != SPLIT_SIZES[name]:
            raise ValueError(
                f'{run_id} {name} has {frame.height} rows, expected {SPLIT_SIZES[name]}'
            )
        if frame['ID'].n_unique() != frame.height:
            raise ValueError(f'{run_id} {name} has duplicate IDs')
        _finite_array(frame['dockscore'].to_numpy(), f'{run_id} {name} targets')
        id_sets[name] = set(frame['ID'].to_list())
    if id_sets['train'] & id_sets['calibration']:
        raise ValueError(f'{run_id} train and calibration splits overlap')
    if id_sets['train'] & id_sets['test']:
        raise ValueError(f'{run_id} train and test splits overlap')
    if id_sets['calibration'] & id_sets['test']:
        raise ValueError(f'{run_id} calibration and test splits overlap')

    def values(
        frame: pl.DataFrame,
    ) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
        return (
            tuple(frame['ID'].to_list()),
            tuple(frame['SMILES'].to_list()),
            _finite_array(frame['dockscore'].to_numpy(), 'dockscore'),
        )

    return {name: values(frame) for name, frame in split_frames.items()}


def discover_runs(results_dir: Path) -> list[RunContext]:
    """Discover and validate the exact three A-01 and three A-03 runs."""
    manifest_path = results_dir / 'manifest.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'missing run manifest: {manifest_path}')
    manifest = pl.read_csv(manifest_path, infer_schema_length=None)
    contexts: list[RunContext] = []
    for family, strategy in EXPECTED_RUNS:
        for replicate, seed in REPLICATE_SEEDS.items():
            run_id = f'lm8_{family}_rf_1M_{strategy}_b0.1_rep{replicate}'
            run_dir = results_dir / run_id
            if not run_dir.is_dir():
                raise FileNotFoundError(f'missing required run directory: {run_dir}')
            row = _manifest_row(manifest, run_id)
            expected_values = {
                'family': family,
                'learner': 'rf',
                'pool': '1M',
                'strategy': strategy,
                'rep': replicate,
            }
            for key, expected in expected_values.items():
                if row.get(key) != expected:
                    raise ValueError(
                        f'{run_id} manifest {key}={row.get(key)!r}, expected {expected!r}'
                    )
            if not math.isclose(float(row['batch_fraction_pct']), 0.1):
                raise ValueError(f'{run_id} is not a 0.1% batch run')
            config_path = run_dir / 'config.json'
            if not config_path.is_file():
                raise FileNotFoundError(f'missing config: {config_path}')
            config = json.loads(config_path.read_text())
            if config.get('random_state') != seed:
                raise ValueError(
                    f'{run_id} random_state={config.get("random_state")!r}, '
                    f'expected {seed}'
                )
            if config.get('n_cycles') != 10 or config.get('featurizer') != 'morgan':
                raise ValueError(f'{run_id} is not the expected 10-cycle Morgan run')
            split = _read_split(run_dir, run_id)
            contexts.append(
                RunContext(
                    run_id=run_id,
                    family=family,
                    strategy=strategy,
                    replicate=replicate,
                    seed=seed,
                    path=run_dir,
                    train_ids=split['train'][0],
                    train_smiles=split['train'][1],
                    train_targets=split['train'][2],
                    calibration_ids=split['calibration'][0],
                    calibration_smiles=split['calibration'][1],
                    calibration_targets=split['calibration'][2],
                    test_ids=split['test'][0],
                    test_smiles=split['test'][1],
                    test_targets=split['test'][2],
                )
            )
    return contexts


def extract_split_features(
    context: RunContext, cache_dir: Path, n_jobs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smiles = list(
        context.train_smiles + context.calibration_smiles + context.test_smiles
    )
    features = extract_features(
        smiles,
        'morgan',
        cache_dir=cache_dir,
        n_jobs=n_jobs,
        preferred_dtype='uint8',
    )
    if features.shape[0] != sum(SPLIT_SIZES.values()):
        raise ValueError(f'{context.run_id} feature row count is {features.shape[0]}')
    if not np.isfinite(np.asarray(features, dtype=np.float64)).all():
        raise ValueError(f'{context.run_id} features contain non-finite values')
    train_end = SPLIT_SIZES['train']
    calibration_end = train_end + SPLIT_SIZES['calibration']
    return (
        features[:train_end],
        features[train_end:calibration_end],
        features[calibration_end:],
    )


def _preprocessed_features(
    learner: RandomForestLearner, features: np.ndarray
) -> np.ndarray:
    prepared, _, _ = _preprocess_features(
        features,
        valid_feature_mask=learner._valid_feature_mask,
        remove_zero_variance=learner.remove_zero_variance,
        is_training=False,
        feature_type=learner._feature_type,
        imputer=learner._feature_imputer,
    )
    if learner._feature_scaler is not None:
        prepared = learner._feature_scaler.transform(prepared)
    return prepared


def _tree_predictions(
    learner: RandomForestLearner, prepared_features: np.ndarray
) -> np.ndarray:
    predictions = np.asarray(
        [
            estimator.predict(prepared_features)
            for estimator in learner.model.estimators_
        ],
        dtype=np.float64,
    )
    if predictions.ndim != 2 or predictions.shape[1] != prepared_features.shape[0]:
        raise ValueError(f'unexpected tree prediction shape: {predictions.shape}')
    return predictions


def compute_ij(
    learner: RandomForestLearner,
    tree_predictions: np.ndarray,
    n_train: int,
    *,
    chunk_size: int = IJ_CHUNK_SIZE,
) -> IJResult:
    """Compute chunked IJ variance and the finite-tree Wager correction."""
    samples = getattr(learner.model, 'estimators_samples_', None)
    if samples is None:
        raise RuntimeError('fitted RF does not expose estimators_samples_')
    n_trees, n_eval = tree_predictions.shape
    if len(samples) != n_trees or n_train <= 0 or n_eval <= 0:
        raise ValueError('IJ input dimensions are inconsistent')
    counts = np.zeros((n_trees, n_train), dtype=np.float64)
    for tree_index, bootstrap_indices in enumerate(samples):
        counts[tree_index] = np.bincount(
            np.asarray(bootstrap_indices, dtype=np.int64), minlength=n_train
        )
    counts -= 1.0
    centered = tree_predictions - tree_predictions.mean(axis=0)
    raw_variance = np.zeros(n_eval, dtype=np.float64)
    effective_chunk = max(1, min(chunk_size, n_train))
    for start in range(0, n_train, effective_chunk):
        stop = min(start + effective_chunk, n_train)
        covariance = counts[:, start:stop].T @ centered / float(n_trees)
        raw_variance += np.einsum('ij,ij->j', covariance, covariance)
    correction = n_train * np.var(tree_predictions, axis=0, ddof=0) / n_trees
    corrected_variance = raw_variance - correction
    nonpositive_mask = corrected_variance <= 0.0
    standard_error = np.sqrt(np.maximum(corrected_variance, 0.0))
    for name, values in (
        ('raw IJ variance', raw_variance),
        ('corrected IJ variance', corrected_variance),
        ('IJ correction', correction),
        ('IJ standard error', standard_error),
    ):
        _finite_array(values, name)
    return IJResult(
        raw_variance=raw_variance,
        corrected_variance=corrected_variance,
        correction=correction,
        standard_error=standard_error,
        nonpositive_mask=nonpositive_mask,
    )


def tree_mad(tree_predictions: np.ndarray) -> np.ndarray:
    medians = np.median(tree_predictions, axis=0)
    return np.median(np.abs(tree_predictions - medians), axis=0)


def tree_iqr(tree_predictions: np.ndarray) -> np.ndarray:
    lower, upper = np.quantile(tree_predictions, (0.25, 0.75), axis=0)
    return (upper - lower) / 2.0


def _stable_top_k_indices(similarities: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k > similarities.size:
        raise ValueError(f'k={k} is invalid for {similarities.size} similarities')
    threshold = float(np.partition(similarities, similarities.size - k)[-k])
    above = np.flatnonzero(similarities > threshold)
    tied = np.flatnonzero(similarities == threshold)
    needed = k - above.size
    selected = np.concatenate((above, tied[:needed]))
    return selected[np.lexsort((selected, -similarities[selected]))]


def _morgan_fingerprints(
    smiles: Sequence[str], n_jobs: int
) -> tuple[DataStructs.ExplicitBitVect, ...]:
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    invalid = [index for index, molecule in enumerate(molecules) if molecule is None]
    if invalid:
        raise ValueError(f'invalid SMILES at local-score indices {invalid[:10]}')
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
        useBondTypes=True,
    )
    return tuple(generator.GetFingerprints(molecules, numThreads=n_jobs))


def local_neighbor_scores(
    train_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    query_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    train_oob_absolute_residuals: np.ndarray,
    *,
    k_values: Sequence[int] = LOCAL_K_VALUES,
    query_chunk_size: int = LOCAL_QUERY_CHUNK,
) -> dict[int, np.ndarray]:
    """Average OOB residuals of deterministic top-k Tanimoto neighbors."""
    residuals = _finite_array(train_oob_absolute_residuals, 'OOB residuals')
    if residuals.size != len(train_fingerprints):
        raise ValueError('training fingerprints and OOB residuals differ in length')
    requested = tuple(sorted(set(int(value) for value in k_values)))
    if not requested or requested[0] <= 0 or requested[-1] > residuals.size:
        raise ValueError('local-neighbor k values are invalid')
    if query_chunk_size <= 0:
        raise ValueError('query_chunk_size must be positive')
    outputs = {
        k: np.empty(len(query_fingerprints), dtype=np.float64) for k in requested
    }
    for start in range(0, len(query_fingerprints), query_chunk_size):
        stop = min(start + query_chunk_size, len(query_fingerprints))
        for query_index in range(start, stop):
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(
                    query_fingerprints[query_index], train_fingerprints
                ),
                dtype=np.float64,
            )
            for k in requested:
                neighbors = _stable_top_k_indices(similarities, k)
                outputs[k][query_index] = float(np.mean(residuals[neighbors]))
    return outputs


def expected_retained_mae(
    errors: np.ndarray, scores: np.ndarray, retained: int
) -> float:
    """Expected retained MAE when a score-tie group crosses the boundary."""
    errors = _finite_array(errors, 'absolute errors')
    scores = _finite_array(scores, 'uncertainty scores')
    if errors.shape != scores.shape:
        raise ValueError('errors and scores must have identical shapes')
    if retained <= 0 or retained > errors.size:
        raise ValueError('retained count is outside the sample range')
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    sorted_errors = errors[order]
    boundary = sorted_scores[retained - 1]
    below = sorted_scores < boundary
    tied = sorted_scores == boundary
    below_count = int(np.sum(below))
    tied_needed = retained - below_count
    expected_sum = float(np.sum(sorted_errors[below]))
    expected_sum += (tied_needed / int(np.sum(tied))) * float(
        np.sum(sorted_errors[tied])
    )
    return expected_sum / retained


def tie_aware_sparsification(
    errors: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return tie-aware normalized retained-MAE sparsification values."""
    errors = _finite_array(errors, 'absolute errors')
    baseline = float(np.mean(errors))
    if baseline <= 0.0:
        raise ValueError('cannot normalize sparsification by zero baseline MAE')
    values = []
    for fraction in REMOVAL_FRACTIONS:
        retained = max(1, round((1.0 - fraction) * errors.size))
        values.append(expected_retained_mae(errors, scores, retained) / baseline)
    return REMOVAL_FRACTIONS.copy(), np.asarray(values, dtype=np.float64)


def oracle_sparsification(errors: np.ndarray) -> np.ndarray:
    return tie_aware_sparsification(errors, errors)[1]


def normalized_ause(model_curve: np.ndarray, oracle_curve: np.ndarray) -> float:
    model_curve = _finite_array(model_curve, 'model sparsification curve')
    oracle_curve = _finite_array(oracle_curve, 'oracle sparsification curve')
    if (
        model_curve.shape != REMOVAL_FRACTIONS.shape
        or oracle_curve.shape != model_curve.shape
    ):
        raise ValueError('sparsification curves have unexpected shape')
    return float(np.trapezoid(model_curve - oracle_curve, REMOVAL_FRACTIONS) / 0.90)


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    statistic = spearmanr(first, second).statistic
    return float(statistic) if np.isfinite(statistic) else float('nan')


def _scale_floor(scale: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    scale = _finite_array(scale, 'uncertainty scale')
    positive = scale[scale > 0.0]
    positive_median = float(np.median(positive)) if positive.size else 0.0
    floor = max(1e-8, 1e-6 * positive_median)
    floored = np.maximum(scale, floor)
    return floored, floor, positive_median, float(np.mean(scale < floor))


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    sorted_scores = np.sort(_finite_array(scores, 'conformal scores'))
    if sorted_scores.size == 0 or not 0.0 < coverage < 1.0:
        raise ValueError('finite-sample conformal quantile inputs are invalid')
    rank = min(sorted_scores.size, math.ceil((sorted_scores.size + 1) * coverage))
    return float(sorted_scores[rank - 1])


def interval_score_values(
    targets: np.ndarray, lower: np.ndarray, upper: np.ndarray, coverage: float
) -> np.ndarray:
    targets = _finite_array(targets, 'interval targets')
    lower = _finite_array(lower, 'lower interval bounds')
    upper = _finite_array(upper, 'upper interval bounds')
    alpha = 1.0 - coverage
    return (
        (upper - lower)
        + (2.0 / alpha) * np.maximum(lower - targets, 0.0)
        + (2.0 / alpha) * np.maximum(targets - upper, 0.0)
    )


def _interval_metrics(
    targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: float,
) -> dict[str, float]:
    crossing_fraction = float(np.mean(lower > upper))
    if crossing_fraction:
        raise AssertionError(
            f'interval bounds cross for {crossing_fraction:.3%} of rows'
        )
    widths = upper - lower
    empirical = float(np.mean((targets >= lower) & (targets <= upper)))
    return {
        'empirical_coverage': empirical,
        'signed_coverage_error': empirical - coverage,
        'mean_width': float(np.mean(widths)),
        'median_width': float(np.median(widths)),
        'interval_score': float(
            np.mean(interval_score_values(targets, lower, upper, coverage))
        ),
        'quantile_crossing_fraction': crossing_fraction,
    }


def cqr_bounds(
    calibration_targets: np.ndarray,
    calibration_lower: np.ndarray,
    calibration_upper: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Apply conservative symmetric finite-sample CQR to conditional bounds."""
    conformity = np.maximum(
        calibration_lower - calibration_targets,
        calibration_targets - calibration_upper,
    )
    raw_correction = conformal_quantile(conformity, coverage)
    # MAPIE's symmetric CQR correction can be negative. On heterogeneous test
    # widths that can produce empty/crossed intervals. A zero floor preserves
    # the native QRF interval and is independent of test labels or widths.
    correction = max(0.0, raw_correction)
    lower = test_lower - correction
    upper = test_upper + correction
    return lower, upper, correction, raw_correction


def _quantile_column(quantiles: np.ndarray, value: float) -> np.ndarray:
    index = QRF_QUANTILES.index(value)
    return quantiles[:, index]


def _coverage_quantiles(coverage: float) -> tuple[float, float]:
    alpha_half = (1.0 - coverage) / 2.0
    return round(alpha_half, 3), round(1.0 - alpha_half, 3)


def _score_summary(scale: np.ndarray) -> dict[str, float]:
    scale = _finite_array(scale, 'ranking scale')
    return {
        'zero_fraction': float(np.mean(scale == 0.0)),
        'nonpositive_fraction': float(np.mean(scale <= 0.0)),
        'unique_score_fraction': float(np.unique(scale).size / scale.size),
        'score_mean': float(np.mean(scale)),
        'score_median': float(np.median(scale)),
        'score_min': float(np.min(scale)),
        'score_max': float(np.max(scale)),
    }


def reconstruction_check(
    context: RunContext, calibration_predictions: np.ndarray
) -> dict[str, Any]:
    history_path = context.path / 'selection_history.csv'
    history = pl.read_csv(
        history_path,
        comment_prefix='#',
        columns=['cycle', 'ID', 'prediction_at_selection'],
        infer_schema_length=10_000,
        schema_overrides={
            'cycle': pl.Int64,
            'ID': pl.String,
            'prediction_at_selection': pl.Float64,
        },
    ).filter(pl.col('cycle') == CALIBRATION_CYCLE)
    if history.height != SPLIT_SIZES['calibration']:
        raise ValueError(f'{context.run_id} has an invalid cycle-7 history size')
    refit = pl.DataFrame(
        {'ID': list(context.calibration_ids), 'refit': calibration_predictions}
    )
    joined = history.join(refit, on='ID', how='left', validate='1:1')
    if joined['refit'].null_count():
        raise ValueError(f'{context.run_id} cycle-7 reconstruction IDs differ')
    saved = _finite_array(joined['prediction_at_selection'], 'saved predictions')
    current = _finite_array(joined['refit'], 'refit predictions')
    mae = float(np.mean(np.abs(saved - current)))
    rho = _spearman(saved, current)
    result = {
        'run_id': context.run_id,
        'n': int(history.height),
        'mae': mae,
        'max_absolute_difference': float(np.max(np.abs(saved - current))),
        'spearman': rho,
        'mae_threshold': 0.05,
        'spearman_threshold': 0.999,
        'passed': bool(mae <= 0.05 and rho >= 0.999),
    }
    if not result['passed']:
        raise ValueError(f'{context.run_id} reconstruction check failed: {result}')
    return result


def _cost_context(row: dict[str, Any], context: RunContext) -> dict[str, Any]:
    row.update(
        {
            'family': context.family,
            'strategy': context.strategy,
            'replicate': context.replicate,
            'seed': context.seed,
        }
    )
    return row


def _fit_common_rf(
    features_train: np.ndarray,
    targets_train: np.ndarray,
    seed: int,
    n_jobs: int,
) -> RandomForestLearner:
    learner = RandomForestLearner(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=seed,
        n_jobs=n_jobs,
    )
    learner.train(features_train, targets_train)
    oob = _finite_array(learner.model.oob_prediction_, 'RF OOB predictions')
    if oob.size != targets_train.size:
        raise ValueError('RF OOB prediction count differs from training size')
    return learner


def _fit_qrf(
    prepared_train: np.ndarray,
    targets_train: np.ndarray,
    seed: int,
    n_jobs: int,
) -> RandomForestQuantileRegressor:
    model = RandomForestQuantileRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(prepared_train, targets_train)
    return model


def _ranking_records(
    context: RunContext,
    test_errors: np.ndarray,
    calibration_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    test_ij: IJResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    oracle_curve = oracle_sparsification(test_errors)
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for method in SCORE_METHODS:
        scale = test_scores[method]
        _, retained_mae = tie_aware_sparsification(test_errors, scale)
        row: dict[str, Any] = {
            'record_type': 'ranking',
            'interval_family': None,
            'method': method,
            'test_n': int(test_errors.size),
            'test_mae': float(np.mean(test_errors)),
            'uncertainty_error_spearman': _spearman(scale, test_errors),
            'normalized_ause': normalized_ause(retained_mae, oracle_curve),
            'calibration_nonpositive_fraction': float(
                np.mean(calibration_scores[method] <= 0.0)
            ),
            **_score_summary(scale),
        }
        row['unstable'] = bool(
            max(
                row['calibration_nonpositive_fraction'],
                row['nonpositive_fraction'],
            )
            > 0.05
        )
        if method == 'ij_se':
            row.update(
                {
                    'ij_raw_variance_mean': float(np.mean(test_ij.raw_variance)),
                    'ij_raw_variance_median': float(np.median(test_ij.raw_variance)),
                    'ij_corrected_variance_mean': float(
                        np.mean(test_ij.corrected_variance)
                    ),
                    'ij_corrected_variance_median': float(
                        np.median(test_ij.corrected_variance)
                    ),
                    'ij_correction_mean': float(np.mean(test_ij.correction)),
                    'ij_correction_median': float(np.median(test_ij.correction)),
                    'ij_truncated_scale_mean': float(np.mean(test_ij.standard_error)),
                }
            )
        metrics.append(row)
        for fraction, normalized_mae in zip(
            REMOVAL_FRACTIONS, retained_mae, strict=True
        ):
            curves.append(
                {
                    'curve_type': 'sparsification',
                    'method': method,
                    'removal_fraction': float(fraction),
                    'normalized_retained_mae': float(normalized_mae),
                }
            )
    for method, values in (
        ('oracle', oracle_curve),
        ('random_reference', np.ones_like(oracle_curve)),
    ):
        for fraction, normalized_mae in zip(REMOVAL_FRACTIONS, values, strict=True):
            curves.append(
                {
                    'curve_type': 'sparsification',
                    'method': method,
                    'removal_fraction': float(fraction),
                    'normalized_retained_mae': float(normalized_mae),
                }
            )
    for record in metrics + curves:
        record.update(
            {
                'run_id': context.run_id,
                'family': context.family,
                'strategy': context.strategy,
                'replicate': context.replicate,
                'seed': context.seed,
            }
        )
    return metrics, curves


def _interval_row(
    context: RunContext,
    *,
    interval_family: str,
    method: str,
    nominal_coverage: float,
    targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    prediction_source: str,
    conformal_value: float | None = None,
    scale_floor: float | None = None,
    scale_positive_median: float | None = None,
    calibration_floor_fraction: float | None = None,
    test_floor_fraction: float | None = None,
    calibration_nonpositive_fraction: float | None = None,
    test_nonpositive_fraction: float | None = None,
    raw_conformal_quantile: float | None = None,
    conformal_quantile_floored: bool | None = None,
) -> dict[str, Any]:
    return {
        'record_type': 'interval',
        'interval_family': interval_family,
        'method': method,
        'nominal_coverage': nominal_coverage,
        **_interval_metrics(targets, lower, upper, nominal_coverage),
        'prediction_source': prediction_source,
        'conformal_quantile': conformal_value,
        'raw_conformal_quantile': raw_conformal_quantile,
        'conformal_quantile_floored': conformal_quantile_floored,
        'scale_floor': scale_floor,
        'positive_calibration_scale_median': scale_positive_median,
        'calibration_floor_fraction': calibration_floor_fraction,
        'test_floor_fraction': test_floor_fraction,
        'calibration_nonpositive_fraction': calibration_nonpositive_fraction,
        'test_nonpositive_fraction': test_nonpositive_fraction,
        'calibration_n': SPLIT_SIZES['calibration'],
        'test_n': SPLIT_SIZES['test'],
        'run_id': context.run_id,
        'family': context.family,
        'strategy': context.strategy,
        'replicate': context.replicate,
        'seed': context.seed,
    }


def _interval_records(
    context: RunContext,
    calibration_predictions: np.ndarray,
    test_predictions: np.ndarray,
    calibration_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    qrf_calibration: np.ndarray,
    qrf_test: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    calibration_residuals = np.abs(
        context.calibration_targets - calibration_predictions
    )
    metrics: list[dict[str, Any]] = []
    score_bounds: dict[str, np.ndarray] = {}
    mean_widths: dict[tuple[str, str], list[float]] = {}

    for nominal in COVERAGE_LEVELS:
        marginal_q = conformal_quantile(calibration_residuals, nominal)
        marginal_half_width = np.full(context.test_targets.size, marginal_q)
        marginal = _interval_row(
            context,
            interval_family='marginal_conformal',
            method='marginal',
            nominal_coverage=nominal,
            targets=context.test_targets,
            lower=test_predictions - marginal_half_width,
            upper=test_predictions + marginal_half_width,
            prediction_source='common_rf',
            conformal_value=marginal_q,
        )
        metrics.append(marginal)
        mean_widths.setdefault(('marginal_conformal', 'marginal'), []).append(
            marginal['mean_width']
        )

    for method in SCORE_METHODS:
        raw_calibration = calibration_scores[method]
        raw_test = test_scores[method]
        prepared_calibration, floor, positive_median, cal_floor_fraction = _scale_floor(
            raw_calibration
        )
        prepared_test = np.maximum(raw_test, floor)
        test_floor_fraction = float(np.mean(raw_test < floor))
        normalized_calibration_residuals = calibration_residuals / prepared_calibration
        for nominal in COVERAGE_LEVELS:
            conformal_q = conformal_quantile(normalized_calibration_residuals, nominal)
            half_width = prepared_test * conformal_q
            row = _interval_row(
                context,
                interval_family='scaled_conformal',
                method=method,
                nominal_coverage=nominal,
                targets=context.test_targets,
                lower=test_predictions - half_width,
                upper=test_predictions + half_width,
                prediction_source='common_rf',
                conformal_value=conformal_q,
                scale_floor=floor,
                scale_positive_median=positive_median,
                calibration_floor_fraction=cal_floor_fraction,
                test_floor_fraction=test_floor_fraction,
                calibration_nonpositive_fraction=float(np.mean(raw_calibration <= 0.0)),
                test_nonpositive_fraction=float(np.mean(raw_test <= 0.0)),
            )
            metrics.append(row)
            mean_widths.setdefault(('scaled_conformal', method), []).append(
                row['mean_width']
            )

    for nominal in COVERAGE_LEVELS:
        lower_quantile, upper_quantile = _coverage_quantiles(nominal)
        calibration_lower = _quantile_column(qrf_calibration, lower_quantile)
        calibration_upper = _quantile_column(qrf_calibration, upper_quantile)
        test_lower = _quantile_column(qrf_test, lower_quantile)
        test_upper = _quantile_column(qrf_test, upper_quantile)
        suffix = str(round(nominal * 100))
        score_bounds[f'qrf_native_lower_{suffix}'] = test_lower
        score_bounds[f'qrf_native_upper_{suffix}'] = test_upper
        native = _interval_row(
            context,
            interval_family='qrf_native',
            method='qrf_native',
            nominal_coverage=nominal,
            targets=context.test_targets,
            lower=test_lower,
            upper=test_upper,
            prediction_source='qrf_conditional_quantiles',
        )
        metrics.append(native)
        mean_widths.setdefault(('qrf_native', 'qrf_native'), []).append(
            native['mean_width']
        )
        cqr_lower, cqr_upper, correction, raw_correction = cqr_bounds(
            context.calibration_targets,
            calibration_lower,
            calibration_upper,
            test_lower,
            test_upper,
            nominal,
        )
        score_bounds[f'qrf_cqr_lower_{suffix}'] = cqr_lower
        score_bounds[f'qrf_cqr_upper_{suffix}'] = cqr_upper
        cqr = _interval_row(
            context,
            interval_family='qrf_cqr',
            method='qrf_cqr',
            nominal_coverage=nominal,
            targets=context.test_targets,
            lower=cqr_lower,
            upper=cqr_upper,
            prediction_source='qrf_conditional_quantiles',
            conformal_value=correction,
            raw_conformal_quantile=raw_correction,
            conformal_quantile_floored=raw_correction < 0.0,
        )
        metrics.append(cqr)
        mean_widths.setdefault(('qrf_cqr', 'qrf_cqr'), []).append(cqr['mean_width'])

    for (family, method), widths in mean_widths.items():
        if np.any(np.diff(widths) < -1e-12):
            raise AssertionError(
                f'{context.run_id} {family}/{method} mean widths are not monotonic'
            )
    curves = [
        {
            'curve_type': 'coverage',
            'interval_family': row['interval_family'],
            'method': row['method'],
            'nominal_coverage': row['nominal_coverage'],
            'empirical_coverage': row['empirical_coverage'],
            'signed_coverage_error': row['signed_coverage_error'],
            'mean_width': row['mean_width'],
            'interval_score': row['interval_score'],
            'run_id': row['run_id'],
            'family': row['family'],
            'strategy': row['strategy'],
            'replicate': row['replicate'],
            'seed': row['seed'],
        }
        for row in metrics
    ]
    return metrics, curves, score_bounds


def _sensitivity_records(
    context: RunContext,
    test_errors: np.ndarray,
    qrf_test: np.ndarray,
    local_test: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    if context.family != 'A-01' or context.replicate != 1:
        return []
    oracle = oracle_sparsification(test_errors)
    rows: list[dict[str, Any]] = []

    default_local = local_test[LOCAL_K_DEFAULT]
    default_local_curve = tie_aware_sparsification(test_errors, default_local)[1]
    default_local_rho = _spearman(default_local, test_errors)
    default_local_ause = normalized_ause(default_local_curve, oracle)
    for k in LOCAL_K_VALUES:
        score = local_test[k]
        curve = tie_aware_sparsification(test_errors, score)[1]
        rho = _spearman(score, test_errors)
        ause = normalized_ause(curve, oracle)
        rows.append(
            {
                'run_id': context.run_id,
                'sensitivity_family': 'local_oob_residual_k',
                'option': f'k={k}',
                'parameter_value': float(k),
                'is_default': k == LOCAL_K_DEFAULT,
                'rank_spearman_vs_default': _spearman(score, default_local),
                'uncertainty_error_spearman': rho,
                'normalized_ause': ause,
                'metric_spearman_change': rho - default_local_rho,
                'normalized_ause_change': ause - default_local_ause,
            }
        )

    qrf_scores = {
        label: _quantile_column(qrf_test, upper) - _quantile_column(qrf_test, lower)
        for label, (lower, upper) in QRF_SENSITIVITY_WIDTHS.items()
    }
    default_qrf = qrf_scores['central_80']
    default_qrf_curve = tie_aware_sparsification(test_errors, default_qrf)[1]
    default_qrf_rho = _spearman(default_qrf, test_errors)
    default_qrf_ause = normalized_ause(default_qrf_curve, oracle)
    for label, score in qrf_scores.items():
        central = float(label.split('_')[1])
        curve = tie_aware_sparsification(test_errors, score)[1]
        rho = _spearman(score, test_errors)
        ause = normalized_ause(curve, oracle)
        rows.append(
            {
                'run_id': context.run_id,
                'sensitivity_family': 'qrf_central_width',
                'option': label,
                'parameter_value': central,
                'is_default': label == 'central_80',
                'rank_spearman_vs_default': _spearman(score, default_qrf),
                'uncertainty_error_spearman': rho,
                'normalized_ause': ause,
                'metric_spearman_change': rho - default_qrf_rho,
                'normalized_ause_change': ause - default_qrf_ause,
            }
        )
    return rows


def _build_scores_frame(
    context: RunContext,
    predictions: np.ndarray,
    scores: dict[str, np.ndarray],
    ij: IJResult,
    qrf_quantiles: np.ndarray,
    qrf_test_bounds: dict[str, np.ndarray],
) -> pl.DataFrame:
    n_calibration = SPLIT_SIZES['calibration']
    n_test = SPLIT_SIZES['test']
    columns: dict[str, Any] = {
        'run_id': [context.run_id] * (n_calibration + n_test),
        'family': [context.family] * (n_calibration + n_test),
        'strategy': [context.strategy] * (n_calibration + n_test),
        'replicate': [context.replicate] * (n_calibration + n_test),
        'seed': [context.seed] * (n_calibration + n_test),
        'split': ['calibration'] * n_calibration + ['test'] * n_test,
        'ID': list(context.calibration_ids + context.test_ids),
        'target': np.concatenate((context.calibration_targets, context.test_targets)),
        'rf_prediction': predictions,
        'qrf_median': _quantile_column(qrf_quantiles, 0.50),
        **scores,
        'ij_raw_variance': ij.raw_variance,
        'ij_corrected_variance': ij.corrected_variance,
        'ij_correction': ij.correction,
        'ij_nonpositive': ij.nonpositive_mask,
        'ij_truncated_scale': ij.standard_error,
    }
    for nominal in COVERAGE_LEVELS:
        lower_quantile, upper_quantile = _coverage_quantiles(nominal)
        suffix = str(round(nominal * 100))
        columns[f'qrf_native_lower_{suffix}'] = _quantile_column(
            qrf_quantiles, lower_quantile
        )
        columns[f'qrf_native_upper_{suffix}'] = _quantile_column(
            qrf_quantiles, upper_quantile
        )
        calibration_lower = columns[f'qrf_native_lower_{suffix}'][:n_calibration]
        calibration_upper = columns[f'qrf_native_upper_{suffix}'][:n_calibration]
        test_lower = qrf_test_bounds[f'qrf_cqr_lower_{suffix}']
        test_upper = qrf_test_bounds[f'qrf_cqr_upper_{suffix}']
        correction = float(
            columns[f'qrf_native_lower_{suffix}'][n_calibration] - test_lower[0]
        )
        columns[f'qrf_cqr_lower_{suffix}'] = np.concatenate(
            (calibration_lower - correction, test_lower)
        )
        columns[f'qrf_cqr_upper_{suffix}'] = np.concatenate(
            (calibration_upper + correction, test_upper)
        )
    frame = pl.DataFrame(columns)
    if frame.height != n_calibration + n_test or frame['ID'].n_unique() != frame.height:
        raise AssertionError(f'{context.run_id} score export has invalid IDs')
    return frame


def analyze_features(
    context: RunContext,
    features: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_jobs: int,
    *,
    measurement: str,
    profile_repetition: int | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    pl.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    features_train, features_calibration, features_test = features
    eval_features = np.vstack((features_calibration, features_test))
    costs: list[dict[str, Any]] = []

    def measured(phase: str, function: Callable[[], T], method: str | None = None) -> T:
        result, cost = _measure(
            phase,
            function,
            run_id=context.run_id,
            method=method,
            measurement=measurement,
            profile_repetition=profile_repetition,
        )
        costs.append(_cost_context(cost, context))
        return result

    learner = measured(
        'rf_fit',
        lambda: _fit_common_rf(
            features_train, context.train_targets, context.seed, n_jobs
        ),
        'common_rf',
    )
    predictions = measured(
        'rf_mean_prediction',
        lambda: _finite_array(
            learner.predict(eval_features, compute_uncertainty=False)[0],
            'common RF predictions',
        ),
        'common_rf',
    )
    prepared_train, prepared_eval = measured(
        'shared_feature_preprocessing',
        lambda: (
            _preprocessed_features(learner, features_train),
            _preprocessed_features(learner, eval_features),
        ),
        None,
    )
    tree_predictions = measured(
        'tree_prediction_matrix',
        lambda: _tree_predictions(learner, prepared_eval),
        'tree_scores_common',
    )
    tree_std_score = measured(
        'score_reduction',
        lambda: np.std(tree_predictions, axis=0, ddof=0),
        'tree_std',
    )
    ij = measured(
        'score_reduction',
        lambda: compute_ij(learner, tree_predictions, features_train.shape[0]),
        'ij_se',
    )
    tree_mad_score = measured(
        'score_reduction', lambda: tree_mad(tree_predictions), 'tree_mad'
    )
    tree_iqr_score = measured(
        'score_reduction', lambda: tree_iqr(tree_predictions), 'tree_iqr'
    )

    qrf_model = measured(
        'qrf_fit',
        lambda: _fit_qrf(prepared_train, context.train_targets, context.seed, n_jobs),
        'qrf_i80_width',
    )
    qrf_quantiles = measured(
        'qrf_prediction',
        lambda: _finite_array(
            qrf_model.predict(prepared_eval, quantiles=list(QRF_QUANTILES)),
            'QRF quantiles',
        ),
        'qrf_i80_width',
    )
    if qrf_quantiles.shape != (eval_features.shape[0], len(QRF_QUANTILES)):
        raise ValueError(f'unexpected QRF quantile shape {qrf_quantiles.shape}')
    qrf_crossing = np.diff(qrf_quantiles, axis=1) < -1e-12
    if np.any(qrf_crossing):
        raise AssertionError('QRF conditional quantiles cross')
    qrf_i80 = measured(
        'score_reduction',
        lambda: (
            _quantile_column(qrf_quantiles, 0.90)
            - _quantile_column(qrf_quantiles, 0.10)
        ),
        'qrf_i80_width',
    )

    train_and_query_fingerprints = measured(
        'local_neighbor_indexing',
        lambda: (
            _morgan_fingerprints(context.train_smiles, n_jobs),
            _morgan_fingerprints(
                context.calibration_smiles + context.test_smiles, n_jobs
            ),
        ),
        'local_oob_residual_k25',
    )
    train_fingerprints, query_fingerprints = train_and_query_fingerprints
    oob_residuals = np.abs(
        context.train_targets
        - _finite_array(learner.model.oob_prediction_, 'RF OOB predictions')
    )
    local_scores = measured(
        'local_neighbor_search',
        lambda: local_neighbor_scores(
            train_fingerprints,
            query_fingerprints,
            oob_residuals,
            k_values=LOCAL_K_VALUES,
        ),
        'local_oob_residual_k25',
    )

    all_scores = {
        'tree_std': _finite_array(tree_std_score, 'tree std'),
        'ij_se': _finite_array(ij.standard_error, 'IJ SE'),
        'tree_mad': _finite_array(tree_mad_score, 'tree MAD'),
        'tree_iqr': _finite_array(tree_iqr_score, 'tree IQR'),
        'qrf_i80_width': _finite_array(qrf_i80, 'QRF I80 width'),
        'local_oob_residual_k25': _finite_array(
            local_scores[LOCAL_K_DEFAULT], 'local OOB residual score'
        ),
    }
    n_calibration = SPLIT_SIZES['calibration']
    calibration_scores = {
        method: values[:n_calibration] for method, values in all_scores.items()
    }
    test_scores = {
        method: values[n_calibration:] for method, values in all_scores.items()
    }
    test_errors = np.abs(context.test_targets - predictions[n_calibration:])
    test_ij = IJResult(
        raw_variance=ij.raw_variance[n_calibration:],
        corrected_variance=ij.corrected_variance[n_calibration:],
        correction=ij.correction[n_calibration:],
        standard_error=ij.standard_error[n_calibration:],
        nonpositive_mask=ij.nonpositive_mask[n_calibration:],
    )
    ranking_metrics, ranking_curves = _ranking_records(
        context, test_errors, calibration_scores, test_scores, test_ij
    )
    interval_metrics, interval_curves, qrf_test_bounds = _interval_records(
        context,
        predictions[:n_calibration],
        predictions[n_calibration:],
        calibration_scores,
        test_scores,
        qrf_quantiles[:n_calibration],
        qrf_quantiles[n_calibration:],
    )
    local_test = {k: values[n_calibration:] for k, values in local_scores.items()}
    sensitivity = _sensitivity_records(
        context, test_errors, qrf_quantiles[n_calibration:], local_test
    )
    reconstruction = reconstruction_check(context, predictions[:n_calibration])
    scores_frame = _build_scores_frame(
        context,
        predictions,
        all_scores,
        ij,
        qrf_quantiles,
        qrf_test_bounds,
    )
    return (
        ranking_metrics + interval_metrics,
        ranking_curves + interval_curves,
        scores_frame,
        sensitivity,
        costs,
        reconstruction,
    )


def process_context(
    context: RunContext,
    cache_dir: Path,
    n_jobs: int,
    *,
    measurement: str,
    profile_repetition: int | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    pl.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    features, feature_cost = _measure(
        'feature_loading',
        lambda: extract_split_features(context, cache_dir, n_jobs),
        run_id=context.run_id,
        method=None,
        measurement=measurement,
        profile_repetition=profile_repetition,
    )
    result = analyze_features(
        context,
        features,
        n_jobs,
        measurement=measurement,
        profile_repetition=profile_repetition,
    )
    metrics, curves, scores, sensitivity, costs, reconstruction = result
    costs.insert(0, _cost_context(feature_cost, context))
    return metrics, curves, scores, sensitivity, costs, reconstruction


def _records_frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    if not records:
        raise ValueError('cannot construct a table from no records')
    keys = sorted({key for record in records for key in record})
    normalized = [{key: record.get(key) for key in keys} for record in records]
    return pl.from_dicts(normalized, infer_schema_length=None, strict=False)


def _write_csv(
    records: list[dict[str, Any]], path: Path, sort_columns: Sequence[str]
) -> None:
    frame = _records_frame(records)
    available = [column for column in sort_columns if column in frame.columns]
    if available:
        frame = frame.sort(available, nulls_last=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path, float_scientific=False)


def _summary_stat_rows(
    records: list[dict[str, Any]],
    group_keys: Sequence[str],
    value_keys: Sequence[str],
    *,
    baseline_lookup: dict[tuple[Any, ...], float] | None = None,
    baseline_key_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(tuple(record.get(key) for key in group_keys), []).append(
            record
        )
    rows: list[dict[str, Any]] = []
    for group, members in grouped.items():
        identity = dict(zip(group_keys, group, strict=True))
        for value_key in value_keys:
            values = np.asarray(
                [member[value_key] for member in members], dtype=np.float64
            )
            row = {
                'summary_type': 'replicate',
                **identity,
                'metric': value_key,
                'n': int(values.size),
                'mean': float(np.mean(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
            if baseline_lookup is not None:
                differences = []
                for member in members:
                    lookup_key = tuple(member.get(key) for key in baseline_key_fields)
                    if lookup_key in baseline_lookup:
                        differences.append(
                            member[value_key] - baseline_lookup[lookup_key]
                        )
                if differences:
                    diff = np.asarray(differences, dtype=np.float64)
                    row.update(
                        {
                            'paired_baseline_difference_mean': float(np.mean(diff)),
                            'paired_baseline_difference_min': float(np.min(diff)),
                            'paired_baseline_difference_max': float(np.max(diff)),
                        }
                    )
            rows.append(row)
    return rows


def build_summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking = [row for row in metrics if row['record_type'] == 'ranking']
    ranking_baselines = {
        (row['run_id'], metric): row[metric]
        for row in ranking
        if row['method'] == 'tree_std'
        for metric in ('uncertainty_error_spearman', 'normalized_ause')
    }
    ranking_rows: list[dict[str, Any]] = []
    for metric in ('uncertainty_error_spearman', 'normalized_ause'):
        lookup = {
            (run_id,): value
            for (run_id, metric_name), value in ranking_baselines.items()
            if metric_name == metric
        }
        ranking_rows.extend(
            _summary_stat_rows(
                ranking,
                ('family', 'strategy', 'method'),
                (metric,),
                baseline_lookup=lookup,
                baseline_key_fields=('run_id',),
            )
        )

    intervals = [row for row in metrics if row['record_type'] == 'interval']
    interval_rows = _summary_stat_rows(
        intervals,
        ('family', 'strategy', 'interval_family', 'method', 'nominal_coverage'),
        (
            'empirical_coverage',
            'signed_coverage_error',
            'mean_width',
            'median_width',
            'interval_score',
        ),
    )
    return ranking_rows + interval_rows


def evaluate_decision_gates(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    ranking = {
        (row['run_id'], row['method']): row
        for row in metrics
        if row['record_type'] == 'ranking' and row['strategy'] == 'random'
    }
    intervals = {
        (row['run_id'], row['method'], row['nominal_coverage']): row
        for row in metrics
        if row['record_type'] == 'interval'
        and row['strategy'] == 'random'
        and row['interval_family']
        in {
            'scaled_conformal',
            'marginal_conformal',
        }
    }
    random_run_ids = sorted(
        run_id for run_id, method in ranking if method == 'tree_std'
    )
    methods: dict[str, Any] = {}
    passing: list[str] = []
    for method in SCORE_METHODS:
        ranking_details = []
        relative_improvements = []
        for run_id in random_run_ids:
            baseline = ranking[(run_id, 'tree_std')]
            candidate = ranking[(run_id, method)]
            denominator = baseline['normalized_ause']
            relative = (
                (denominator - candidate['normalized_ause']) / denominator
                if denominator > 0.0
                else float('-inf')
            )
            relative_improvements.append(relative)
            ranking_details.append(
                {
                    'run_id': run_id,
                    'spearman_greater_than_tree_std': bool(
                        candidate['uncertainty_error_spearman']
                        > baseline['uncertainty_error_spearman']
                    ),
                    'ause_lower_than_tree_std': bool(
                        candidate['normalized_ause'] < baseline['normalized_ause']
                    ),
                    'relative_ause_improvement': relative,
                    'unstable': bool(candidate['unstable']),
                }
            )
        ranking_pass = bool(
            method != 'tree_std'
            and ranking_details
            and all(
                detail['spearman_greater_than_tree_std']
                and detail['ause_lower_than_tree_std']
                for detail in ranking_details
            )
            and float(np.mean(relative_improvements)) >= 0.05
        )

        calibration_details = []
        for run_id in random_run_ids:
            candidate_rows = [
                intervals[(run_id, method, coverage)] for coverage in COVERAGE_LEVELS
            ]
            marginal_rows = [
                intervals[(run_id, 'marginal', coverage)]
                for coverage in COVERAGE_LEVELS
            ]
            score_improvements = [
                (marginal['interval_score'] - candidate['interval_score'])
                / marginal['interval_score']
                for candidate, marginal in zip(
                    candidate_rows, marginal_rows, strict=True
                )
            ]
            calibration_details.append(
                {
                    'run_id': run_id,
                    'coverage_within_two_percentage_points_all_levels': all(
                        abs(row['signed_coverage_error']) <= 0.02
                        for row in candidate_rows
                    ),
                    'interval_score_no_worse_all_levels': all(
                        candidate['interval_score']
                        <= marginal['interval_score'] + 1e-12
                        for candidate, marginal in zip(
                            candidate_rows, marginal_rows, strict=True
                        )
                    ),
                    'mean_interval_score_improvement': float(
                        np.mean(score_improvements)
                    ),
                }
            )
        calibration_pass = bool(
            calibration_details
            and all(
                detail['coverage_within_two_percentage_points_all_levels']
                and detail['interval_score_no_worse_all_levels']
                and detail['mean_interval_score_improvement'] >= 0.02
                for detail in calibration_details
            )
        )
        stable = bool(
            ranking_details
            and not any(detail['unstable'] for detail in ranking_details)
        )
        passes_both = bool(ranking_pass and calibration_pass and stable)
        if passes_both:
            passing.append(method)
        methods[method] = {
            'ranking_passed': ranking_pass,
            'ranking_details': ranking_details,
            'mean_relative_ause_improvement': float(np.mean(relative_improvements)),
            'calibration_passed': calibration_pass,
            'calibration_details': calibration_details,
            'stable': stable,
            'passes_both_gates': passes_both,
        }
    return {
        'ranking_gate_scope': 'A-01 random replicates only',
        'calibration_gate_scope': 'A-01 random replicates only',
        'instability_nonpositive_fraction_threshold': 0.05,
        'methods': methods,
        'passing_methods': passing,
        'recommendation': (
            f'consider active-learning rerun for: {", ".join(passing)}'
            if passing
            else 'retain tree_std; alternatives are negative results'
        ),
        'ucb_role': 'selection-shift stress test only; cannot make a method pass',
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _input_hashes(results_dir: Path, contexts: Sequence[RunContext]) -> dict[str, str]:
    paths = [
        results_dir / 'manifest.csv',
        Path(__file__).resolve(),
        REPO_ROOT / 'validation' / 'uncertainty' / 'requirements-rf-uq-benchmark.txt',
    ]
    for context in contexts:
        paths.extend(
            (
                context.path / 'compounds_final.csv',
                context.path / 'selection_history.csv',
                context.path / 'config.json',
            )
        )
    return {str(path.resolve()): _sha256(path) for path in paths}


def _base_metadata(
    results_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    n_jobs: int,
    contexts: Sequence[RunContext],
    reconstruction_checks: list[dict[str, Any]],
    decision_gates: dict[str, Any],
) -> dict[str, Any]:
    return {
        'analysis': 'Six-method retrospective RF uncertainty benchmark',
        'scope': {
            'run_families': ['A-01 random', 'A-03 UCB'],
            'run_count': len(contexts),
            'active_learning_executed': False,
            'claim_boundary': (
                'descriptive AmpC 1M retrospective evidence; no p-values or '
                'general production-performance claim'
            ),
        },
        'paths': {
            'results_dir': str(results_dir.resolve()),
            'output_dir': str(output_dir.resolve()),
            'cache_dir': str(cache_dir.resolve()),
        },
        'parameters': {
            'replicate_seeds': REPLICATE_SEEDS,
            'split_cycles': {
                'train': TRAIN_CYCLES,
                'calibration': (CALIBRATION_CYCLE,),
                'test': TEST_CYCLES,
            },
            'split_sizes': SPLIT_SIZES,
            'n_estimators': N_ESTIMATORS,
            'rf_max_features': 'sqrt',
            'rf_bootstrap': True,
            'local_k_default': LOCAL_K_DEFAULT,
            'local_k_sensitivity': LOCAL_K_VALUES,
            'local_query_chunk_size': LOCAL_QUERY_CHUNK,
            'ij_chunk_size': IJ_CHUNK_SIZE,
            'coverage_levels': COVERAGE_LEVELS,
            'removal_fractions': REMOVAL_FRACTIONS.tolist(),
            'score_methods': SCORE_METHODS,
            'qrf_quantiles': QRF_QUANTILES,
            'scale_floor': 'max(1e-8, 1e-6 * median positive calibration scale)',
            'n_jobs': n_jobs,
            'profile_repetitions': PROFILE_REPETITIONS,
        },
        'semantics': {
            'ranking_prediction': 'common production-equivalent RF mean',
            'tree_std': 'population SD across the common RF trees',
            'ij_se': 'zero-truncated bias-corrected IJ SE; raw terms retained',
            'tree_mad': 'median absolute deviation across common RF trees',
            'tree_iqr': 'half the common RF tree-prediction IQR',
            'qrf_i80_width': 'QRF conditional q90 minus q10 width',
            'local_oob_residual_k25': (
                'mean absolute baseline-RF OOB residual among 25 nearest '
                'training Morgan fingerprints'
            ),
            'qrf_native_and_cqr': 'reported separately from the common-RF comparison',
            'qrf_cqr_correction': (
                'symmetric finite-sample MAPIE correction with a nonnegative '
                'floor; raw correction and floor activation are retained'
            ),
            'ucb': 'selection-shift stress test; no exchangeability claim',
        },
        'software_versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'polars': pl.__version__,
            'scipy': _version('scipy'),
            'scikit-learn': _version('scikit-learn'),
            'rdkit': _version('rdkit'),
            'quantile-forest': _version('quantile-forest'),
            'MAPIE': _version('MAPIE'),
            'matplotlib': _version('matplotlib'),
            'psutil': _version('psutil'),
            'learnm8': _version('learnm8'),
        },
        'input_hashes': _input_hashes(results_dir, contexts),
        'reconstruction_checks': reconstruction_checks,
        'decision_gates': decision_gates,
        'output_hashes': {},
    }


def _output_names() -> tuple[str, ...]:
    return (
        'rf_uq_scores.parquet',
        'rf_uq_metrics.csv',
        'rf_uq_curves.csv',
        'rf_uq_summary.csv',
        'rf_uq_costs.csv',
        'rf_uq_sensitivity.csv',
        'rf_uq_benchmark_report.md',
        'rf_uq_random_ranking.pdf',
        'rf_uq_random_ranking.png',
        'rf_uq_random_calibration.pdf',
        'rf_uq_random_calibration.png',
        'rf_uq_ucb_stress_test.pdf',
        'rf_uq_ucb_stress_test.png',
    )


def _update_output_hashes(metadata: dict[str, Any], output_dir: Path) -> None:
    metadata['output_hashes'] = {
        name: _sha256(output_dir / name)
        for name in _output_names()
        if (output_dir / name).is_file()
    }


def _write_metadata(metadata: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'rf_uq_metadata.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n'
    )


def compute_stage(
    results_dir: Path, output_dir: Path, cache_dir: Path, n_jobs: int
) -> None:
    contexts = discover_runs(results_dir)
    all_metrics: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    all_scores: list[pl.DataFrame] = []
    all_sensitivity: list[dict[str, Any]] = []
    all_costs: list[dict[str, Any]] = []
    reconstruction_checks: list[dict[str, Any]] = []
    for context in contexts:
        LOGGER.info('Computing %s', context.run_id)
        result = process_context(
            context,
            cache_dir,
            n_jobs,
            measurement='actual',
            profile_repetition=None,
        )
        metrics, curves, scores, sensitivity, costs, reconstruction = result
        all_metrics.extend(metrics)
        all_curves.extend(curves)
        all_scores.append(scores)
        all_sensitivity.extend(sensitivity)
        all_costs.extend(costs)
        reconstruction_checks.append(reconstruction)
        gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_frame = pl.concat(all_scores, how='vertical').sort(['run_id', 'split', 'ID'])
    if scores_frame.height != len(contexts) * (
        SPLIT_SIZES['calibration'] + SPLIT_SIZES['test']
    ):
        raise AssertionError('score export row count is incorrect')
    scores_frame.write_parquet(
        output_dir / 'rf_uq_scores.parquet', compression='zstd', statistics=True
    )
    _write_csv(
        all_metrics,
        output_dir / 'rf_uq_metrics.csv',
        (
            'record_type',
            'family',
            'replicate',
            'interval_family',
            'method',
            'nominal_coverage',
        ),
    )
    _write_csv(
        all_curves,
        output_dir / 'rf_uq_curves.csv',
        (
            'curve_type',
            'family',
            'replicate',
            'interval_family',
            'method',
            'nominal_coverage',
            'removal_fraction',
        ),
    )
    _write_csv(
        build_summary(all_metrics),
        output_dir / 'rf_uq_summary.csv',
        (
            'summary_type',
            'family',
            'strategy',
            'interval_family',
            'method',
            'nominal_coverage',
            'metric',
        ),
    )
    _write_csv(
        all_costs,
        output_dir / 'rf_uq_costs.csv',
        ('measurement', 'run_id', 'profile_repetition', 'phase', 'method'),
    )
    _write_csv(
        all_sensitivity,
        output_dir / 'rf_uq_sensitivity.csv',
        ('sensitivity_family', 'parameter_value'),
    )
    metadata = _base_metadata(
        results_dir,
        output_dir,
        cache_dir,
        n_jobs,
        contexts,
        reconstruction_checks,
        evaluate_decision_gates(all_metrics),
    )
    _update_output_hashes(metadata, output_dir)
    _write_metadata(metadata, output_dir)
    LOGGER.info('Wrote compute-stage benchmark tables to %s', output_dir)


def _profile_summary_rows(costs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile = [row for row in costs if row.get('measurement') == 'profile']
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for row in profile:
        groups.setdefault((row['phase'], row.get('method')), []).append(row)
    summaries = []
    for (phase, method), rows in groups.items():
        summary: dict[str, Any] = {
            'record_type': 'profile_summary',
            'measurement': 'profile_summary',
            'run_id': rows[0]['run_id'],
            'family': rows[0]['family'],
            'strategy': rows[0]['strategy'],
            'replicate': rows[0]['replicate'],
            'seed': rows[0]['seed'],
            'phase': phase,
            'method': method,
            'profile_repetitions': len(rows),
        }
        for field in (
            'wall_seconds',
            'cpu_seconds',
            'absolute_peak_rss_mb',
            'incremental_peak_rss_mb',
        ):
            values = np.asarray([row[field] for row in rows], dtype=np.float64)
            summary[f'{field}_median'] = float(np.median(values))
            summary[f'{field}_min'] = float(np.min(values))
            summary[f'{field}_max'] = float(np.max(values))
        summaries.append(summary)
    return summaries


def profile_stage(
    results_dir: Path, output_dir: Path, cache_dir: Path, n_jobs: int
) -> None:
    context = discover_runs(results_dir)[0]
    path = output_dir / 'rf_uq_costs.csv'
    existing = (
        pl.read_csv(path, infer_schema_length=None).to_dicts() if path.is_file() else []
    )
    existing = [
        row
        for row in existing
        if row.get('measurement') not in {'profile', 'profile_summary'}
    ]
    profile_costs: list[dict[str, Any]] = []
    for repetition in range(1, PROFILE_REPETITIONS + 1):
        LOGGER.info(
            'Profiling A-01 replicate 1 (%d/%d)', repetition, PROFILE_REPETITIONS
        )
        result = process_context(
            context,
            cache_dir,
            n_jobs,
            measurement='profile',
            profile_repetition=repetition,
        )
        profile_costs.extend(result[4])
        del result
        gc.collect()
    all_costs = existing + profile_costs + _profile_summary_rows(profile_costs)
    _write_csv(
        all_costs,
        path,
        ('measurement', 'run_id', 'profile_repetition', 'phase', 'method'),
    )
    metadata_path = output_dir / 'rf_uq_metadata.json'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        metadata['profiling'] = {
            'run_id': context.run_id,
            'repetitions': PROFILE_REPETITIONS,
            'reporting': 'median and min-max in rf_uq_costs.csv',
        }
        _update_output_hashes(metadata, output_dir)
        _write_metadata(metadata, output_dir)
    LOGGER.info('Wrote isolated profile repetitions to %s', path)


METHOD_LABELS = {
    'tree_std': 'Tree SD',
    'ij_se': 'IJ SE',
    'tree_mad': 'Tree MAD',
    'tree_iqr': 'Tree IQR/2',
    'qrf_i80_width': 'QRF I80 width',
    'local_oob_residual_k25': 'Local OOB k=25',
    'marginal': 'Marginal',
    'qrf_native': 'QRF native',
    'qrf_cqr': 'QRF CQR',
}
METHOD_COLORS = {
    method: style.CATEGORICAL[index] for index, method in enumerate(SCORE_METHODS)
}
METHOD_COLORS.update(
    {
        'marginal': style.INK,
        'qrf_native': style.ACCENT_ORANGE,
        'qrf_cqr': style.ACCENT_GREEN,
    }
)


def _aggregate(frame: pl.DataFrame, keys: Sequence[str], value: str) -> pl.DataFrame:
    return (
        frame.group_by(list(keys))
        .agg(
            pl.col(value).mean().alias('mean'),
            pl.col(value).min().alias('min'),
            pl.col(value).max().alias('max'),
        )
        .sort(list(keys))
    )


def _save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    creator = 'LearnM8 RF UQ benchmark'
    fig.savefig(
        output_dir / f'{stem}.pdf',
        dpi=style.OUTPUT_DPI,
        bbox_inches='tight',
        pad_inches=0.08,
        facecolor=style.BACKGROUND,
        metadata={'Creator': creator},
    )
    fig.savefig(
        output_dir / f'{stem}.png',
        dpi=style.OUTPUT_DPI,
        bbox_inches='tight',
        pad_inches=0.08,
        facecolor=style.BACKGROUND,
        metadata={'Software': creator},
    )
    plt.close(fig)


def _plot_band(
    ax: Any,
    summary: pl.DataFrame,
    x: str,
    *,
    label: str,
    color: str,
    linestyle: str = '-',
) -> None:
    x_values = summary[x].to_numpy()
    ax.plot(
        x_values,
        summary['mean'].to_numpy(),
        label=label,
        color=color,
        linestyle=linestyle,
    )
    style.band(
        ax,
        x_values,
        summary['min'].to_numpy(),
        summary['max'].to_numpy(),
        color,
        alpha=0.10,
    )


def plot_random_ranking(
    metrics: pl.DataFrame, curves: pl.DataFrame, output_dir: Path
) -> None:
    style.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(style.DOUBLE_COL, 70 * style.MM), constrained_layout=True
    )
    random_curves = curves.filter(
        (pl.col('curve_type') == 'sparsification') & (pl.col('strategy') == 'random')
    )
    for method in SCORE_METHODS:
        summary = _aggregate(
            random_curves.filter(pl.col('method') == method),
            ('removal_fraction',),
            'normalized_retained_mae',
        )
        _plot_band(
            axes[0],
            summary,
            'removal_fraction',
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    oracle = _aggregate(
        random_curves.filter(pl.col('method') == 'oracle'),
        ('removal_fraction',),
        'normalized_retained_mae',
    )
    _plot_band(
        axes[0],
        oracle,
        'removal_fraction',
        label='Oracle',
        color=style.INK,
        linestyle=':',
    )
    axes[0].plot(
        REMOVAL_FRACTIONS,
        np.ones_like(REMOVAL_FRACTIONS),
        color=style.MUTED,
        linestyle='--',
        label='Random removal',
    )
    axes[0].set(
        title='A-01 random: sparsification',
        xlabel='Compounds removed (%)',
        ylabel='Normalized retained MAE',
        xticks=(0.0, 0.3, 0.6, 0.9),
        xticklabels=('0', '30', '60', '90'),
    )
    axes[0].legend(loc='best', fontsize=6.2, ncol=2, handlelength=2.0)

    ranking = metrics.filter(
        (pl.col('record_type') == 'ranking') & (pl.col('strategy') == 'random')
    )
    x = np.arange(len(SCORE_METHODS))
    for replicate in REPLICATE_SEEDS:
        values = [
            float(
                ranking.filter(
                    (pl.col('replicate') == replicate) & (pl.col('method') == method)
                )['normalized_ause'][0]
            )
            for method in SCORE_METHODS
        ]
        axes[1].plot(
            x,
            values,
            color=style.MUTED,
            alpha=0.45,
            linewidth=0.9,
            marker='o',
            label=f'Replicate {replicate}',
        )
    means = [
        float(ranking.filter(pl.col('method') == method)['normalized_ause'].mean())
        for method in SCORE_METHODS
    ]
    axes[1].scatter(x, means, color=style.INK, marker='D', s=20, label='Mean')
    axes[1].set(
        title='Paired run-level normalized AUSE',
        ylabel='Normalized AUSE (lower is better)',
        xticks=x,
        xticklabels=[METHOD_LABELS[method] for method in SCORE_METHODS],
    )
    axes[1].tick_params(axis='x', rotation=35)
    axes[1].legend(loc='best', fontsize=6.5)
    axes[0].text(-0.10, 1.06, 'A', transform=axes[0].transAxes, fontweight='bold')
    axes[1].text(-0.10, 1.06, 'B', transform=axes[1].transAxes, fontweight='bold')
    _save_figure(fig, output_dir, 'rf_uq_random_ranking')


def plot_random_calibration(
    metrics: pl.DataFrame, curves: pl.DataFrame, output_dir: Path
) -> None:
    style.apply()
    fig, axes = plt.subplots(
        1, 3, figsize=(style.DOUBLE_COL, 68 * style.MM), constrained_layout=True
    )
    coverage = curves.filter(
        (pl.col('curve_type') == 'coverage') & (pl.col('strategy') == 'random')
    )
    marginal = _aggregate(
        coverage.filter(pl.col('interval_family') == 'marginal_conformal'),
        ('nominal_coverage',),
        'empirical_coverage',
    ).with_columns(pl.Series('coverage_index', np.arange(len(COVERAGE_LEVELS))))
    _plot_band(
        axes[0],
        marginal,
        'coverage_index',
        label='Marginal',
        color=style.INK,
        linestyle='--',
    )
    for method in SCORE_METHODS:
        summary = _aggregate(
            coverage.filter(
                (pl.col('interval_family') == 'scaled_conformal')
                & (pl.col('method') == method)
            ),
            ('nominal_coverage',),
            'empirical_coverage',
        ).with_columns(pl.Series('coverage_index', np.arange(len(COVERAGE_LEVELS))))
        _plot_band(
            axes[0],
            summary,
            'coverage_index',
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    axes[0].plot(
        np.arange(len(COVERAGE_LEVELS)),
        COVERAGE_LEVELS,
        color=style.MUTED,
        linestyle=':',
        label='Nominal',
    )
    axes[0].set(
        title='Scaled split conformal',
        xlabel='Nominal coverage',
        ylabel='Empirical coverage',
        xticks=np.arange(len(COVERAGE_LEVELS)),
        xticklabels=('50%', '80%', '90%', '95%'),
        ylim=(0.43, 1.0),
    )
    axes[0].legend(loc='best', fontsize=5.5, ncol=2, handlelength=1.8)
    axes[0].title.set_fontsize(9.5)

    for family, method, linestyle in (
        ('qrf_native', 'qrf_native', '-'),
        ('qrf_cqr', 'qrf_cqr', '--'),
    ):
        summary = _aggregate(
            coverage.filter(pl.col('interval_family') == family),
            ('nominal_coverage',),
            'empirical_coverage',
        ).with_columns(pl.Series('coverage_index', np.arange(len(COVERAGE_LEVELS))))
        _plot_band(
            axes[1],
            summary,
            'coverage_index',
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linestyle=linestyle,
        )
    axes[1].plot(
        np.arange(len(COVERAGE_LEVELS)),
        COVERAGE_LEVELS,
        color=style.MUTED,
        linestyle=':',
        label='Nominal',
    )
    axes[1].set(
        title='Native and CQR QRF',
        xlabel='Nominal coverage',
        ylabel='Empirical coverage',
        xticks=np.arange(len(COVERAGE_LEVELS)),
        xticklabels=('50%', '80%', '90%', '95%'),
        ylim=(0.43, 1.0),
    )
    axes[1].legend(loc='best', fontsize=6.5)
    axes[1].title.set_fontsize(9.5)

    interval_90 = metrics.filter(
        (pl.col('record_type') == 'interval')
        & (pl.col('strategy') == 'random')
        & (pl.col('nominal_coverage') == 0.90)
    )
    comparison = [*SCORE_METHODS, 'qrf_native', 'qrf_cqr']
    width_ratios = []
    score_ratios = []
    for method in comparison:
        family = method if method in {'qrf_native', 'qrf_cqr'} else 'scaled_conformal'
        candidate = interval_90.filter(
            (pl.col('interval_family') == family) & (pl.col('method') == method)
        ).sort('replicate')
        marginal_rows = interval_90.filter(
            pl.col('interval_family') == 'marginal_conformal'
        ).sort('replicate')
        width_ratios.append(
            float((candidate['mean_width'] / marginal_rows['mean_width']).mean())
        )
        score_ratios.append(
            float(
                (candidate['interval_score'] / marginal_rows['interval_score']).mean()
            )
        )
    x = np.arange(len(comparison))
    axes[2].axhline(1.0, color=style.MUTED, linestyle=':')
    axes[2].scatter(
        x - 0.12,
        width_ratios,
        color=style.ACCENT_BLUE,
        marker='o',
        label='Mean width',
    )
    axes[2].scatter(
        x + 0.12,
        score_ratios,
        color=style.ACCENT_ORANGE,
        marker='s',
        label='Interval score',
    )
    axes[2].set(
        title='90% ratios vs marginal',
        ylabel='Ratio to marginal (log scale)',
        xticks=x,
        xticklabels=(
            'SD',
            'IJ',
            'MAD',
            'IQR/2',
            'QRF I80',
            'Local',
            'QRF native',
            'QRF CQR',
        ),
    )
    axes[2].set_yscale('log')
    axes[2].tick_params(axis='x', rotation=48)
    axes[2].legend(loc='best', fontsize=6.5)
    axes[2].title.set_fontsize(9.5)
    for label, ax in zip('ABC', axes, strict=True):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontweight='bold')
    _save_figure(fig, output_dir, 'rf_uq_random_calibration')


def plot_ucb_stress_test(curves: pl.DataFrame, output_dir: Path) -> None:
    style.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(style.DOUBLE_COL, 70 * style.MM), constrained_layout=True
    )
    ucb = curves.filter(
        (pl.col('curve_type') == 'sparsification') & (pl.col('strategy') == 'ucb')
    )
    for method in SCORE_METHODS:
        summary = _aggregate(
            ucb.filter(pl.col('method') == method),
            ('removal_fraction',),
            'normalized_retained_mae',
        )
        _plot_band(
            axes[0],
            summary,
            'removal_fraction',
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    axes[0].plot(
        REMOVAL_FRACTIONS,
        np.ones_like(REMOVAL_FRACTIONS),
        color=style.MUTED,
        linestyle='--',
        label='Random removal',
    )
    axes[0].set(
        title='A-03 UCB: sparsification',
        xlabel='Compounds removed (%)',
        ylabel='Normalized retained MAE',
        xticks=(0.0, 0.3, 0.6, 0.9),
        xticklabels=('0', '30', '60', '90'),
    )
    axes[0].legend(loc='best', fontsize=6.0, ncol=2, handlelength=2.0)

    coverage = curves.filter(
        (pl.col('curve_type') == 'coverage')
        & (pl.col('interval_family') == 'scaled_conformal')
    )
    for method in SCORE_METHODS:
        differences = []
        for nominal in COVERAGE_LEVELS:
            paired = []
            for replicate in REPLICATE_SEEDS:
                random_value = float(
                    coverage.filter(
                        (pl.col('strategy') == 'random')
                        & (pl.col('replicate') == replicate)
                        & (pl.col('method') == method)
                        & (pl.col('nominal_coverage') == nominal)
                    )['signed_coverage_error'][0]
                )
                ucb_value = float(
                    coverage.filter(
                        (pl.col('strategy') == 'ucb')
                        & (pl.col('replicate') == replicate)
                        & (pl.col('method') == method)
                        & (pl.col('nominal_coverage') == nominal)
                    )['signed_coverage_error'][0]
                )
                paired.append(abs(ucb_value) - abs(random_value))
            differences.append(100 * float(np.mean(paired)))
        axes[1].plot(
            np.arange(len(COVERAGE_LEVELS)),
            differences,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[1].axhline(0.0, color=style.MUTED, linestyle=':')
    axes[1].set(
        title='Random-to-UCB coverage degradation',
        xlabel='Nominal coverage',
        ylabel='Increase in absolute coverage error (pp)',
        xticks=np.arange(len(COVERAGE_LEVELS)),
        xticklabels=('50%', '80%', '90%', '95%'),
    )
    axes[1].legend(loc='best', fontsize=6.0, ncol=2, handlelength=2.0)
    axes[0].text(-0.10, 1.06, 'A', transform=axes[0].transAxes, fontweight='bold')
    axes[1].text(-0.10, 1.06, 'B', transform=axes[1].transAxes, fontweight='bold')
    _save_figure(fig, output_dir, 'rf_uq_ucb_stress_test')


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n'.join(lines)


def write_report(
    metrics: pl.DataFrame,
    costs: pl.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    gates = metadata['decision_gates']
    ranking = metrics.filter(
        (pl.col('record_type') == 'ranking') & (pl.col('strategy') == 'random')
    )
    ranking_rows = []
    for method in SCORE_METHODS:
        selected = ranking.filter(pl.col('method') == method)
        gate = gates['methods'][method]
        ranking_rows.append(
            (
                METHOD_LABELS[method],
                f'{float(selected["uncertainty_error_spearman"].mean()):.3f}',
                f'{float(selected["normalized_ause"].mean()):.3f}',
                'yes' if gate['ranking_passed'] else 'no',
                'yes' if gate['calibration_passed'] else 'no',
                'yes' if gate['stable'] else 'no',
            )
        )
    interval = metrics.filter(
        (pl.col('record_type') == 'interval')
        & (pl.col('strategy') == 'random')
        & (pl.col('nominal_coverage') == 0.90)
    )
    calibration_rows = []
    for family, method in (
        [('marginal_conformal', 'marginal')]
        + [('scaled_conformal', value) for value in SCORE_METHODS]
        + [('qrf_native', 'qrf_native'), ('qrf_cqr', 'qrf_cqr')]
    ):
        selected = interval.filter(
            (pl.col('interval_family') == family) & (pl.col('method') == method)
        )
        calibration_rows.append(
            (
                METHOD_LABELS[method],
                f'{100 * float(selected["empirical_coverage"].mean()):.1f}%',
                f'{float(selected["mean_width"].mean()):.3f}',
                f'{float(selected["interval_score"].mean()):.3f}',
            )
        )
    cqr_floor_counts = {}
    for strategy in ('random', 'ucb'):
        selected = metrics.filter(
            (pl.col('record_type') == 'interval')
            & (pl.col('strategy') == strategy)
            & (pl.col('interval_family') == 'qrf_cqr')
        )
        cqr_floor_counts[strategy] = int(
            selected['conformal_quantile_floored'].fill_null(False).sum()
        )
    profile = costs.filter(pl.col('measurement') == 'profile_summary')
    cost_rows = []
    if profile.height:
        for row in profile.sort(['phase', 'method']).to_dicts():
            method = row.get('method')
            label = (
                f'{row["phase"]} · {METHOD_LABELS.get(method, method)}'
                if method
                else row['phase']
            )
            cost_rows.append(
                (
                    label,
                    f'{row["wall_seconds_median"]:.3f} '
                    f'[{row["wall_seconds_min"]:.3f}, {row["wall_seconds_max"]:.3f}]',
                    f'{row["incremental_peak_rss_mb_median"]:.1f} '
                    f'[{row["incremental_peak_rss_mb_min"]:.1f}, '
                    f'{row["incremental_peak_rss_mb_max"]:.1f}]',
                )
            )
    else:
        actual = costs.filter(pl.col('measurement') == 'actual')
        for method in SCORE_METHODS:
            selected = actual.filter(pl.col('method') == method)
            if selected.height:
                cost_rows.append(
                    (
                        METHOD_LABELS[method],
                        f'{float(selected["wall_seconds"].median()):.3f} '
                        '(actual-run median)',
                        f'{float(selected["incremental_peak_rss_mb"].median()):.1f}',
                    )
                )
    lines = [
        '# Six-method RF uncertainty benchmark',
        '',
        '## Conclusion',
        '',
        gates['recommendation'][0].upper() + gates['recommendation'][1:] + '.',
        '',
        'A-01 random replicates determine the gates. A-03 UCB is reported only '
        'as a selection-shift stress test. These are descriptive results for '
        'this AmpC 1M retrospective setting.',
        '',
        '## A-01 ranking and gates',
        '',
        _markdown_table(
            (
                'Method',
                'Mean Spearman',
                'Mean nAUSE',
                'Rank gate',
                'Calibration gate',
                'Stable',
            ),
            ranking_rows,
        ),
        '',
        '## A-01 90% interval comparison',
        '',
        _markdown_table(
            ('Method', 'Coverage', 'Mean width', 'Interval score'), calibration_rows
        ),
        '',
        'The raw symmetric CQR correction was negative and conservatively '
        f'floored at zero in {cqr_floor_counts["random"]}/12 A-01 and '
        f'{cqr_floor_counts["ucb"]}/12 A-03 run-by-coverage settings. Raw '
        'corrections and floor flags remain in `rf_uq_metrics.csv`.',
        '',
        '## Computational cost',
        '',
        _markdown_table(
            (
                'Phase/method',
                'Wall seconds median [min, max]',
                'Incremental peak RSS MB',
            ),
            cost_rows,
        ),
        '',
        'Common feature loading, common RF fitting, QRF fitting/prediction, and '
        'local-neighbor indexing/search are retained as separate rows in '
        '`rf_uq_costs.csv`.',
        '',
        '## Figures',
        '',
        '- [Random ranking](rf_uq_random_ranking.pdf)',
        '- [Random calibration](rf_uq_random_calibration.pdf)',
        '- [UCB stress test](rf_uq_ucb_stress_test.pdf)',
        '',
    ]
    (output_dir / 'rf_uq_benchmark_report.md').write_text('\n'.join(lines))


def plot_stage(output_dir: Path) -> None:
    required = {
        'metrics': output_dir / 'rf_uq_metrics.csv',
        'curves': output_dir / 'rf_uq_curves.csv',
        'costs': output_dir / 'rf_uq_costs.csv',
        'metadata': output_dir / 'rf_uq_metadata.json',
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f'plot stage requires {path}')
    metrics = pl.read_csv(required['metrics'], infer_schema_length=None)
    curves = pl.read_csv(required['curves'], infer_schema_length=None)
    costs = pl.read_csv(required['costs'], infer_schema_length=None)
    metadata = json.loads(required['metadata'].read_text())
    plot_random_ranking(metrics, curves, output_dir)
    plot_random_calibration(metrics, curves, output_dir)
    plot_ucb_stress_test(curves, output_dir)
    write_report(metrics, costs, metadata, output_dir)
    for input_path in (
        Path(__file__).resolve(),
        REPO_ROOT / 'validation' / 'uncertainty' / 'requirements-rf-uq-benchmark.txt',
    ):
        metadata['input_hashes'][str(input_path.resolve())] = _sha256(input_path)
    _update_output_hashes(metadata, output_dir)
    _write_metadata(metadata, output_dir)
    LOGGER.info('Wrote plot-only figures and report to %s', output_dir)


def _smoke_context(results_dir: Path) -> RunContext:
    contexts = discover_runs(results_dir)
    return next(
        context
        for context in contexts
        if context.family == 'A-01' and context.replicate == 1
    )


def smoke(results_dir: Path, cache_dir: Path, n_jobs: int) -> None:
    """Run algorithm checks on real A-01 molecules without writing outputs."""
    context = _smoke_context(results_dir)
    train_n, calibration_n, test_n = 64, 16, 16
    smiles = list(
        context.train_smiles[:train_n]
        + context.calibration_smiles[:calibration_n]
        + context.test_smiles[:test_n]
    )
    features = extract_features(
        smiles,
        'morgan',
        cache_dir=cache_dir,
        n_jobs=n_jobs,
        preferred_dtype='uint8',
    )
    train_features = features[:train_n]
    evaluation_features = features[train_n:]
    learner = RandomForestLearner(
        n_estimators=16,
        random_state=context.seed,
        n_jobs=n_jobs,
    )
    learner.train(train_features, context.train_targets[:train_n])
    prepared_train = _preprocessed_features(learner, train_features)
    prepared_evaluation = _preprocessed_features(learner, evaluation_features)
    trees = _tree_predictions(learner, prepared_evaluation)
    ij = compute_ij(learner, trees, train_n, chunk_size=7)
    samples = learner.model.estimators_samples_
    centered = trees - trees.mean(axis=0)
    raw_direct = 0.0
    for train_index in range(train_n):
        counts = np.asarray(
            [np.count_nonzero(sample == train_index) - 1 for sample in samples],
            dtype=np.float64,
        )
        raw_direct += float(np.mean(counts * centered[:, 0]) ** 2)
    if not np.isclose(raw_direct, ij.raw_variance[0], rtol=1e-10, atol=1e-12):
        raise AssertionError('chunked IJ differs from direct scalar IJ')

    direct_mad = np.asarray(
        [
            np.median(np.abs(trees[:, index] - np.median(trees[:, index])))
            for index in range(trees.shape[1])
        ]
    )
    direct_iqr = np.asarray(
        [
            (np.quantile(trees[:, index], 0.75) - np.quantile(trees[:, index], 0.25))
            / 2.0
            for index in range(trees.shape[1])
        ]
    )
    if not np.array_equal(tree_mad(trees), direct_mad):
        raise AssertionError('tree MAD differs from direct NumPy calculation')
    if not np.array_equal(tree_iqr(trees), direct_iqr):
        raise AssertionError('tree IQR differs from direct NumPy calculation')

    errors = np.linspace(0.5, 4.0, 8)
    unique_scores = np.arange(errors.size, dtype=np.float64)[::-1]
    ordinary = []
    order = np.argsort(unique_scores, kind='stable')
    for fraction in REMOVAL_FRACTIONS:
        retained = max(1, round((1.0 - fraction) * errors.size))
        ordinary.append(float(np.mean(errors[order[:retained]]) / np.mean(errors)))
    tie_unique = tie_aware_sparsification(errors, unique_scores)[1]
    if not np.allclose(tie_unique, ordinary, rtol=0.0, atol=1e-15):
        raise AssertionError('tie-aware sparsification differs for unique scores')
    tied_errors = np.asarray([1.0, 3.0, 2.0, 8.0])
    tied_scores = np.asarray([0.0, 1.0, 1.0, 2.0])
    expected = expected_retained_mae(tied_errors, tied_scores, retained=2)
    rng = np.random.default_rng(12345)
    monte_carlo = []
    for _ in range(20_000):
        tied_order = rng.permutation(np.asarray([1, 2]))
        monte_carlo.append(np.mean(tied_errors[[0, tied_order[0]]]))
    if not np.isclose(expected, np.mean(monte_carlo), atol=0.01):
        raise AssertionError(
            'tie-aware boundary differs from Monte Carlo tie averaging'
        )

    train_fingerprints = _morgan_fingerprints(context.train_smiles[:20], n_jobs)
    query_fingerprints = _morgan_fingerprints(context.calibration_smiles[:5], n_jobs)
    residuals = np.abs(
        context.train_targets[:20]
        - _finite_array(learner.model.oob_prediction_[:20], 'smoke OOB')
    )
    chunked_local = local_neighbor_scores(
        train_fingerprints,
        query_fingerprints,
        residuals,
        k_values=(5,),
        query_chunk_size=2,
    )[5]
    direct_local = []
    for query in query_fingerprints:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(query, train_fingerprints)
        )
        direct_local.append(
            float(np.mean(residuals[_stable_top_k_indices(similarities, 5)]))
        )
    if not np.array_equal(chunked_local, np.asarray(direct_local)):
        raise AssertionError('chunked local-neighbor score differs from direct search')

    coverage = 0.80
    marginal_mapie = SplitConformalRegressor(
        learner.model,
        confidence_level=coverage,
        prefit=True,
    ).conformalize(
        prepared_evaluation[:calibration_n],
        context.calibration_targets[:calibration_n],
    )
    marginal_predictions, marginal_intervals = marginal_mapie.predict_interval(
        prepared_evaluation[calibration_n:]
    )
    direct_predictions = learner.model.predict(prepared_evaluation[calibration_n:])
    marginal_q = conformal_quantile(
        np.abs(
            context.calibration_targets[:calibration_n]
            - learner.model.predict(prepared_evaluation[:calibration_n])
        ),
        coverage,
    )
    if not np.allclose(marginal_predictions, direct_predictions):
        raise AssertionError('MAPIE marginal point predictions differ from direct RF')
    if not np.allclose(
        marginal_intervals[:, 0, 0], direct_predictions - marginal_q
    ) or not np.allclose(marginal_intervals[:, 1, 0], direct_predictions + marginal_q):
        raise AssertionError('MAPIE marginal bounds differ from direct calculation')

    qrf = RandomForestQuantileRegressor(
        n_estimators=16,
        max_features='sqrt',
        random_state=context.seed,
        n_jobs=n_jobs,
    ).fit(prepared_train, context.train_targets[:train_n])
    quantiles = _finite_array(
        qrf.predict(prepared_evaluation, quantiles=list(QRF_QUANTILES)),
        'smoke QRF quantiles',
    )
    if np.any(np.diff(quantiles, axis=1) < -1e-12):
        raise AssertionError('smoke QRF bounds cross')
    lower_adapter = _QRFQuantileAdapter(qrf, 0.10)
    upper_adapter = _QRFQuantileAdapter(qrf, 0.90)
    median_adapter = _QRFQuantileAdapter(qrf, 0.50)
    mapie = ConformalizedQuantileRegressor(
        [lower_adapter, upper_adapter, median_adapter],
        confidence_level=coverage,
        prefit=True,
    )
    mapie.conformalize(
        prepared_evaluation[:calibration_n],
        context.calibration_targets[:calibration_n],
    )
    _, mapie_intervals = mapie.predict_interval(
        prepared_evaluation[calibration_n:], symmetric_correction=True
    )
    calibration_lower = quantiles[:calibration_n, QRF_QUANTILES.index(0.10)]
    calibration_upper = quantiles[:calibration_n, QRF_QUANTILES.index(0.90)]
    test_lower = quantiles[calibration_n:, QRF_QUANTILES.index(0.10)]
    test_upper = quantiles[calibration_n:, QRF_QUANTILES.index(0.90)]
    direct_lower, direct_upper, _, raw_correction = cqr_bounds(
        context.calibration_targets[:calibration_n],
        calibration_lower,
        calibration_upper,
        test_lower,
        test_upper,
        coverage,
    )
    if raw_correction < 0.0:
        raise AssertionError('smoke MAPIE comparison requires an unfloored correction')
    if not np.allclose(mapie_intervals[:, 0, 0], direct_lower):
        raise AssertionError('MAPIE lower CQR bound differs from direct calculation')
    if not np.allclose(mapie_intervals[:, 1, 0], direct_upper):
        raise AssertionError('MAPIE upper CQR bound differs from direct calculation')
    if not np.array_equal(
        tie_aware_sparsification(errors, unique_scores)[1], tie_unique
    ):
        raise AssertionError('scientific smoke outputs are not deterministic')
    print(
        'smoke passed: real A-01 molecules; IJ, tie-aware AUSE, MAD, IQR, '
        'local Tanimoto, QRF ordering, and MAPIE/direct marginal and CQR agree'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('/home/tony/LearnM8_RESULTS_FINAL'),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=REPO_ROOT
        / 'validation'
        / 'reports'
        / 'uncertainty'
        / 'rf_uq_benchmark',
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=REPO_ROOT / '.cache' / 'rf_uq_benchmark',
    )
    parser.add_argument('--n-jobs', type=int, default=16)
    parser.add_argument(
        '--stage', choices=('compute', 'profile', 'plot', 'all'), default='all'
    )
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_jobs == 0 or args.n_jobs < -1:
        raise ValueError('--n-jobs must be -1 or a positive integer')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    if args.smoke:
        smoke(args.results_dir, args.cache_dir, args.n_jobs)
        return
    if args.stage in ('compute', 'all'):
        compute_stage(args.results_dir, args.output_dir, args.cache_dir, args.n_jobs)
    if args.stage in ('profile', 'all'):
        profile_stage(args.results_dir, args.output_dir, args.cache_dir, args.n_jobs)
    if args.stage in ('plot', 'all'):
        plot_stage(args.output_dir)


if __name__ == '__main__':
    main()

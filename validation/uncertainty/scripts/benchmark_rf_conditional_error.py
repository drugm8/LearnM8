#!/usr/bin/env python3
"""Benchmark a leakage-safe conditional RF error model on existing AmpC runs.

The analysis is retrospective.  Each of the six existing active-learning runs
is handled independently.  Cycles 0--6 train both the point model and a
scaffold-grouped out-of-fold meta-target; cycle 7 is development-only and
cycles 8--9 are evaluation-only.  No active learning is executed and no
LearnM8 production API is changed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make plot-only PDF output deterministic.
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/learnm8-matplotlib-cache')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/learnm8-xdg-cache')

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from quantile_forest import RandomForestQuantileRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from learnm8.evaluation.metrics.similarity import _scaffold_for_smiles
from learnm8.visualization import style
from validation.uncertainty.scripts import benchmark_rf_uq_methods as uq

REPO_ROOT = Path(__file__).resolve().parents[3]


LOGGER = logging.getLogger('rf_conditional_error')

N_FOLDS = 5
N_ESTIMATORS = 100
LOCAL_K = 25
LOCAL_QUERY_CHUNK = 64
PERMUTATION_REPEATS = 100
SCORE_METHODS = (
    'tree_std',
    'qrf_i80_width',
    'local_oob_residual_k25',
    'fixed_rank_blend',
    'conditional_error_hgb',
)
FEATURE_NAMES = (
    'rf_prediction',
    'rf_distance_from_train_target_median',
    'rf_tree_std',
    'rf_tree_mad',
    'rf_tree_iqr_half',
    'qrf_i80_width',
    'qrf_lower_tail_width',
    'qrf_upper_tail_width',
    'neighbor_nearest_tanimoto',
    'neighbor_25th_tanimoto',
    'neighbor_mean_top25_tanimoto',
    'neighbor_oob_residual_mean',
    'neighbor_oob_residual_median',
    'neighbor_oob_residual_std',
    'neighbor_oob_residual_p90',
    'chem_mol_weight',
    'chem_logp',
    'chem_tpsa',
    'chem_hbd',
    'chem_hba',
    'chem_rotatable_bonds',
    'chem_ring_count',
    'chem_heavy_atoms',
    'chem_fraction_sp3',
    'chem_formal_charge',
)
FEATURE_GROUPS = {
    'rf': tuple(range(0, 5)),
    'qrf': tuple(range(5, 8)),
    'neighborhood': tuple(range(8, 15)),
    'chemistry': tuple(range(15, 25)),
}
METHOD_LABELS = {
    'tree_std': 'Tree SD',
    'qrf_i80_width': 'QRF I80 width',
    'local_oob_residual_k25': 'Local OOB k=25',
    'fixed_rank_blend': 'Fixed rank blend',
    'conditional_error_hgb': 'Conditional error HGB',
}
METHOD_COLORS = {
    method: style.CATEGORICAL[index] for index, method in enumerate(SCORE_METHODS)
}


@dataclass(frozen=True)
class MetaSplit:
    """Meta-features and common RF predictions for one temporal split."""

    features: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True)
class OOFResult:
    """Exactly-once scaffold-grouped out-of-fold training result."""

    meta: MetaSplit
    fold: np.ndarray
    scaffolds: tuple[str, ...]
    costs: list[dict[str, Any]]
    fold_metadata: list[dict[str, Any]]


def _hgb(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss='squared_error',
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def canonical_scaffolds(smiles: Sequence[str]) -> tuple[str, ...]:
    """Return canonical, largest-fragment Bemis--Murcko scaffold SMILES."""
    scaffolds = tuple(_scaffold_for_smiles(value) for value in smiles)
    invalid = [index for index, scaffold in enumerate(scaffolds) if scaffold is None]
    if invalid:
        raise ValueError(f'failed to compute scaffolds at indices {invalid[:10]}')
    return tuple(str(scaffold) for scaffold in scaffolds)


def scaffold_fold_assignment(
    ids: Sequence[str], smiles: Sequence[str], seed: int
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    """Create deterministic shuffled GroupKFold assignments and leakage checks."""
    if len(ids) != len(smiles) or len(set(ids)) != len(ids):
        raise ValueError('fold IDs and SMILES must be aligned with unique IDs')
    scaffolds = canonical_scaffolds(smiles)
    if len(set(scaffolds)) < N_FOLDS:
        raise ValueError('fewer scaffold groups than requested folds')
    splitter = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = np.full(len(ids), -1, dtype=np.int16)
    metadata: list[dict[str, Any]] = []
    indices = np.arange(len(ids))
    for fold_index, (fit_indices, held_out_indices) in enumerate(
        splitter.split(indices, groups=np.asarray(scaffolds, dtype=object))
    ):
        if np.any(fold[held_out_indices] != -1):
            raise AssertionError('an OOF row was assigned more than once')
        fold[held_out_indices] = fold_index
        fit_ids = {ids[index] for index in fit_indices}
        held_out_ids = {ids[index] for index in held_out_indices}
        fit_smiles = {smiles[index] for index in fit_indices}
        held_out_smiles = {smiles[index] for index in held_out_indices}
        fit_scaffolds = {scaffolds[index] for index in fit_indices}
        held_out_scaffolds = {scaffolds[index] for index in held_out_indices}
        if fit_ids & held_out_ids:
            raise AssertionError('ID leakage across a scaffold fold')
        if fit_smiles & held_out_smiles:
            raise AssertionError('SMILES leakage across a scaffold fold')
        if fit_scaffolds & held_out_scaffolds:
            raise AssertionError('scaffold leakage across a scaffold fold')
        digest = hashlib.sha256()
        for compound_id in sorted(held_out_ids):
            digest.update(compound_id.encode())
            digest.update(b'\0')
        metadata.append(
            {
                'fold': fold_index,
                'fit_count': int(fit_indices.size),
                'held_out_count': int(held_out_indices.size),
                'held_out_unique_scaffolds': len(held_out_scaffolds),
                'held_out_ids_sha256': digest.hexdigest(),
            }
        )
    if np.any(fold < 0) or not np.array_equal(np.unique(fold), np.arange(N_FOLDS)):
        raise AssertionError('OOF assignment is not exactly once across five folds')
    return fold, scaffolds, metadata


def chemistry_features(smiles: Sequence[str]) -> np.ndarray:
    """Compute the fixed ten-property RDKit chemistry feature block."""
    output = np.empty((len(smiles), 10), dtype=np.float64)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f'invalid SMILES at chemistry index {index}')
        output[index] = (
            Descriptors.MolWt(molecule),
            Descriptors.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumHAcceptors(molecule),
            Descriptors.NumRotatableBonds(molecule),
            Descriptors.RingCount(molecule),
            molecule.GetNumHeavyAtoms(),
            Descriptors.FractionCSP3(molecule),
            sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
        )
    return uq._finite_array(output, 'chemistry features')


def neighbor_features(
    train_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    query_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    train_oob_absolute_residuals: np.ndarray,
    *,
    k: int = LOCAL_K,
    query_chunk_size: int = LOCAL_QUERY_CHUNK,
) -> np.ndarray:
    """Return similarity and OOB-residual summaries for deterministic top-k."""
    residuals = uq._finite_array(train_oob_absolute_residuals, 'OOB residuals')
    if len(train_fingerprints) != residuals.size:
        raise ValueError('fingerprints and OOB residuals differ in length')
    if k <= 0 or k > residuals.size or query_chunk_size <= 0:
        raise ValueError('invalid local-neighbor configuration')
    output = np.empty((len(query_fingerprints), 7), dtype=np.float64)
    for start in range(0, len(query_fingerprints), query_chunk_size):
        stop = min(start + query_chunk_size, len(query_fingerprints))
        for query_index in range(start, stop):
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(
                    query_fingerprints[query_index], train_fingerprints
                ),
                dtype=np.float64,
            )
            neighbors = uq._stable_top_k_indices(similarities, k)
            selected_similarity = similarities[neighbors]
            selected_residuals = residuals[neighbors]
            output[query_index] = (
                selected_similarity[0],
                selected_similarity[-1],
                np.mean(selected_similarity),
                np.mean(selected_residuals),
                np.median(selected_residuals),
                np.std(selected_residuals, ddof=0),
                np.quantile(selected_residuals, 0.90),
            )
    return uq._finite_array(output, 'local-neighbor features')


def _rf_qrf_features(
    learner: Any,
    qrf: RandomForestQuantileRegressor,
    features: np.ndarray,
    training_target_median: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = uq._finite_array(
        learner.predict(features, compute_uncertainty=False)[0], 'RF predictions'
    )
    prepared = uq._preprocessed_features(learner, features)
    tree_predictions = uq._tree_predictions(learner, prepared)
    quantiles = uq._finite_array(
        qrf.predict(prepared, quantiles=[0.10, 0.50, 0.90]), 'QRF quantiles'
    )
    if quantiles.shape != (features.shape[0], 3):
        raise ValueError(f'unexpected QRF quantile shape: {quantiles.shape}')
    if np.any(np.diff(quantiles, axis=1) < -1e-12):
        raise ValueError('QRF quantiles cross')
    base = np.column_stack(
        (
            predictions,
            np.abs(predictions - training_target_median),
            np.std(tree_predictions, axis=0, ddof=0),
            uq.tree_mad(tree_predictions),
            uq.tree_iqr(tree_predictions),
            quantiles[:, 2] - quantiles[:, 0],
            quantiles[:, 1] - quantiles[:, 0],
            quantiles[:, 2] - quantiles[:, 1],
        )
    )
    return uq._finite_array(base, 'RF/QRF meta-features'), predictions


def _fit_pair(
    features: np.ndarray, targets: np.ndarray, seed: int, n_jobs: int
) -> tuple[Any, RandomForestQuantileRegressor]:
    learner = uq._fit_common_rf(features, targets, seed, n_jobs)
    prepared = uq._preprocessed_features(learner, features)
    qrf = uq._fit_qrf(prepared, targets, seed, n_jobs)
    return learner, qrf


def _annotate_cost(
    row: dict[str, Any], context: uq.RunContext, fold: int | None = None
) -> dict[str, Any]:
    row = uq._cost_context(row, context)
    row['fold'] = fold
    return row


def compute_oof_meta_features(
    context: uq.RunContext,
    train_features: np.ndarray,
    train_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    train_chemistry: np.ndarray,
    n_jobs: int,
) -> OOFResult:
    """Build exactly-once, scaffold-grouped OOF features and predictions."""
    fold, scaffolds, fold_metadata = scaffold_fold_assignment(
        context.train_ids, context.train_smiles, context.seed
    )
    n_train = len(context.train_ids)
    meta_features = np.full((n_train, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    predictions = np.full(n_train, np.nan, dtype=np.float64)
    assigned = np.zeros(n_train, dtype=np.int8)
    costs: list[dict[str, Any]] = []
    all_indices = np.arange(n_train)
    for fold_index in range(N_FOLDS):
        held_out = all_indices[fold == fold_index]
        fit = all_indices[fold != fold_index]
        LOGGER.info(
            '%s OOF fold %d/%d (%d fit, %d held out)',
            context.run_id,
            fold_index + 1,
            N_FOLDS,
            fit.size,
            held_out.size,
        )
        learner, rf_cost = uq._measure(
            'oof_rf_fit',
            lambda fit_indices=fit: uq._fit_common_rf(
                train_features[fit_indices],
                context.train_targets[fit_indices],
                context.seed,
                n_jobs,
            ),
            run_id=context.run_id,
            method='tree_std',
            measurement='actual',
            profile_repetition=None,
        )
        costs.append(_annotate_cost(rf_cost, context, fold_index))
        prepared_fit = uq._preprocessed_features(learner, train_features[fit])
        fit_targets = context.train_targets[fit]
        qrf, qrf_cost = uq._measure(
            'oof_qrf_fit',
            lambda prepared=prepared_fit, targets=fit_targets: uq._fit_qrf(
                prepared, targets, context.seed, n_jobs
            ),
            run_id=context.run_id,
            method='qrf_i80_width',
            measurement='actual',
            profile_repetition=None,
        )
        costs.append(_annotate_cost(qrf_cost, context, fold_index))
        base, held_out_predictions = _rf_qrf_features(
            learner,
            qrf,
            train_features[held_out],
            float(np.median(context.train_targets[fit])),
        )
        oob_residuals = np.abs(
            context.train_targets[fit]
            - uq._finite_array(learner.model.oob_prediction_, 'fold OOB predictions')
        )
        fit_fingerprints = [train_fingerprints[index] for index in fit]
        held_out_fingerprints = [train_fingerprints[index] for index in held_out]
        local, local_cost = uq._measure(
            'oof_neighbor_search',
            lambda train_fps=fit_fingerprints, query_fps=held_out_fingerprints, residuals=oob_residuals: (
                neighbor_features(
                    train_fps,
                    query_fps,
                    residuals,
                )
            ),
            run_id=context.run_id,
            method='local_oob_residual_k25',
            measurement='actual',
            profile_repetition=None,
        )
        costs.append(_annotate_cost(local_cost, context, fold_index))
        meta_features[held_out] = np.column_stack(
            (base, local, train_chemistry[held_out])
        )
        predictions[held_out] = held_out_predictions
        assigned[held_out] += 1
        del learner, qrf, prepared_fit, base, local
        gc.collect()
    if not np.all(assigned == 1):
        raise AssertionError('training rows do not have exactly one OOF prediction')
    uq._finite_array(meta_features, 'OOF meta-features')
    uq._finite_array(predictions, 'OOF predictions')
    return OOFResult(
        meta=MetaSplit(meta_features, predictions),
        fold=fold,
        scaffolds=scaffolds,
        costs=costs,
        fold_metadata=fold_metadata,
    )


def _final_meta_features(
    context: uq.RunContext,
    learner: Any,
    qrf: RandomForestQuantileRegressor,
    train_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    query_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    query_features: np.ndarray,
    query_chemistry: np.ndarray,
    n_jobs: int,
    split_name: str,
) -> tuple[MetaSplit, dict[str, Any]]:
    base, predictions = _rf_qrf_features(
        learner,
        qrf,
        query_features,
        float(np.median(context.train_targets)),
    )
    residuals = np.abs(
        context.train_targets
        - uq._finite_array(learner.model.oob_prediction_, 'final RF OOB predictions')
    )
    local, cost = uq._measure(
        f'{split_name}_neighbor_search',
        lambda: neighbor_features(train_fingerprints, query_fingerprints, residuals),
        run_id=context.run_id,
        method='local_oob_residual_k25',
        measurement='actual',
        profile_repetition=None,
    )
    combined = uq._finite_array(
        np.column_stack((base, local, query_chemistry)),
        f'{split_name} meta-features',
    )
    if combined.shape[1] != len(FEATURE_NAMES):
        raise AssertionError('conditional model feature contract changed')
    return MetaSplit(combined, predictions), _annotate_cost(cost, context)


def _empirical_mid_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(uq._finite_array(reference, 'CDF reference'))
    values = uq._finite_array(values, 'CDF values')
    left = np.searchsorted(reference, values, side='left')
    right = np.searchsorted(reference, values, side='right')
    return (left + right) / (2.0 * reference.size)


def _score_sets(
    oof_meta: np.ndarray,
    development_meta: np.ndarray,
    evaluation_meta: np.ndarray,
    model: HistGradientBoostingRegressor,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    indices = {
        'tree_std': FEATURE_NAMES.index('rf_tree_std'),
        'qrf_i80_width': FEATURE_NAMES.index('qrf_i80_width'),
        'local_oob_residual_k25': FEATURE_NAMES.index('neighbor_oob_residual_mean'),
    }
    outputs: list[dict[str, np.ndarray]] = []
    for matrix in (oof_meta, development_meta, evaluation_meta):
        scores = {
            method: uq._finite_array(matrix[:, column], method)
            for method, column in indices.items()
        }
        percentiles = [
            _empirical_mid_cdf(oof_meta[:, column], matrix[:, column])
            for column in indices.values()
        ]
        scores['fixed_rank_blend'] = np.mean(percentiles, axis=0)
        raw = uq._finite_array(model.predict(matrix), 'raw conditional error')
        scores['conditional_error_raw'] = raw
        scores['conditional_error_hgb'] = np.maximum(raw, 0.0)
        outputs.append(scores)
    return outputs[0], outputs[1], outputs[2]


def _labeled_cycles(context: uq.RunContext) -> np.ndarray:
    compounds = pl.read_csv(
        context.path / 'compounds_final.csv',
        comment_prefix='#',
        columns=['ID', 'labeled_cycle'],
        schema_overrides={'ID': pl.String, 'labeled_cycle': pl.Int64},
    )
    ids = list(context.train_ids + context.calibration_ids + context.test_ids)
    joined = (
        pl.DataFrame({'row': np.arange(len(ids)), 'ID': ids})
        .join(compounds, on='ID', how='left', validate='1:1')
        .sort('row')
    )
    if joined['labeled_cycle'].null_count() or joined.height != len(ids):
        raise ValueError(f'{context.run_id} cycle lookup is incomplete')
    cycles = joined['labeled_cycle'].to_numpy()
    if (
        not np.all(np.isin(cycles[:16_000], uq.TRAIN_CYCLES))
        or not np.all(cycles[16_000:17_000] == uq.CALIBRATION_CYCLE)
        or not np.all(np.isin(cycles[17_000:], uq.TEST_CYCLES))
    ):
        raise AssertionError('temporal split cycles do not match the contract')
    return cycles


def _score_frame(
    context: uq.RunContext,
    cycles: np.ndarray,
    oof: OOFResult,
    development: MetaSplit,
    evaluation: MetaSplit,
    score_sets: tuple[
        dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
    ],
) -> pl.DataFrame:
    split_specs = (
        (
            'oof_train',
            context.train_ids,
            context.train_smiles,
            context.train_targets,
            oof.meta,
            score_sets[0],
            oof.fold,
            oof.scaffolds,
            cycles[:16_000],
        ),
        (
            'development',
            context.calibration_ids,
            context.calibration_smiles,
            context.calibration_targets,
            development,
            score_sets[1],
            np.full(1_000, -1, dtype=np.int16),
            canonical_scaffolds(context.calibration_smiles),
            cycles[16_000:17_000],
        ),
        (
            'evaluation',
            context.test_ids,
            context.test_smiles,
            context.test_targets,
            evaluation,
            score_sets[2],
            np.full(2_000, -1, dtype=np.int16),
            canonical_scaffolds(context.test_smiles),
            cycles[17_000:],
        ),
    )
    frames: list[pl.DataFrame] = []
    for (
        split,
        ids,
        smiles,
        targets,
        meta,
        scores,
        folds,
        scaffolds,
        split_cycles,
    ) in split_specs:
        data: dict[str, Any] = {
            'run_id': [context.run_id] * len(ids),
            'family': [context.family] * len(ids),
            'strategy': [context.strategy] * len(ids),
            'replicate': [context.replicate] * len(ids),
            'seed': [context.seed] * len(ids),
            'split': [split] * len(ids),
            'ID': list(ids),
            'SMILES': list(smiles),
            'labeled_cycle': split_cycles,
            'oof_fold': folds,
            'scaffold': list(scaffolds),
            'target': targets,
            'rf_prediction': meta.predictions,
            'absolute_error': np.abs(targets - meta.predictions),
        }
        for column, name in enumerate(FEATURE_NAMES):
            data[f'meta_{name}'] = meta.features[:, column]
        for method, values in scores.items():
            data[f'score_{method}'] = values
        frames.append(pl.DataFrame(data))
    frame = pl.concat(frames, how='vertical')
    if frame['ID'].n_unique() != frame.height:
        raise AssertionError(f'{context.run_id} exported score IDs are not unique')
    return frame


def _ranking_records(
    context: uq.RunContext,
    split: str,
    targets: np.ndarray,
    predictions: np.ndarray,
    scores: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors = np.abs(
        uq._finite_array(targets, f'{split} targets')
        - uq._finite_array(predictions, f'{split} RF predictions')
    )
    oracle = uq.oracle_sparsification(errors)
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    common = {
        'run_id': context.run_id,
        'family': context.family,
        'strategy': context.strategy,
        'replicate': context.replicate,
        'seed': context.seed,
        'split': split,
        'n': int(errors.size),
        'rf_mae': float(np.mean(errors)),
    }
    for method in SCORE_METHODS:
        score = uq._finite_array(scores[method], f'{split} {method}')
        _, curve = uq.tie_aware_sparsification(errors, score)
        raw_score = (
            scores['conditional_error_raw']
            if method == 'conditional_error_hgb'
            else score
        )
        summary = uq._score_summary(score)
        metrics.append(
            {
                **common,
                'record_type': 'ranking',
                'method': method,
                'uncertainty_error_spearman': uq._spearman(score, errors),
                'normalized_ause': uq.normalized_ause(curve, oracle),
                'raw_nonpositive_fraction': float(np.mean(raw_score <= 0.0)),
                'all_finite': bool(np.isfinite(score).all()),
                **summary,
            }
        )
        for fraction, retained, oracle_value in zip(
            uq.REMOVAL_FRACTIONS, curve, oracle, strict=True
        ):
            curves.append(
                {
                    **common,
                    'curve_type': 'sparsification',
                    'method': method,
                    'removal_fraction': float(fraction),
                    'normalized_retained_mae': float(retained),
                    'oracle_normalized_retained_mae': float(oracle_value),
                }
            )
    for fraction, oracle_value in zip(uq.REMOVAL_FRACTIONS, oracle, strict=True):
        curves.append(
            {
                **common,
                'curve_type': 'sparsification',
                'method': 'oracle',
                'removal_fraction': float(fraction),
                'normalized_retained_mae': float(oracle_value),
                'oracle_normalized_retained_mae': float(oracle_value),
            }
        )
    return metrics, curves


def permutation_feature_groups(
    context: uq.RunContext,
    model: HistGradientBoostingRegressor,
    development_features: np.ndarray,
    development_errors: np.ndarray,
) -> list[dict[str, Any]]:
    """Measure descriptive cycle-7 group importance by row permutation."""
    baseline_score = np.maximum(
        uq._finite_array(model.predict(development_features), 'permutation baseline'),
        0.0,
    )
    oracle = uq.oracle_sparsification(development_errors)
    baseline_curve = uq.tie_aware_sparsification(development_errors, baseline_score)[1]
    baseline_ause = uq.normalized_ause(baseline_curve, oracle)
    baseline_spearman = uq._spearman(baseline_score, development_errors)
    rows: list[dict[str, Any]] = []
    for group_index, (group, columns) in enumerate(FEATURE_GROUPS.items()):
        for repetition in range(PERMUTATION_REPEATS):
            rng = np.random.default_rng(
                context.seed * 100_000 + group_index * 1_000 + repetition
            )
            permuted = development_features.copy()
            order = rng.permutation(permuted.shape[0])
            permuted[:, columns] = permuted[order][:, columns]
            score = np.maximum(
                uq._finite_array(model.predict(permuted), 'permuted HGB score'), 0.0
            )
            curve = uq.tie_aware_sparsification(development_errors, score)[1]
            ause = uq.normalized_ause(curve, oracle)
            spearman = uq._spearman(score, development_errors)
            rows.append(
                {
                    'record_type': 'permutation',
                    'run_id': context.run_id,
                    'family': context.family,
                    'strategy': context.strategy,
                    'replicate': context.replicate,
                    'seed': context.seed,
                    'split': 'development',
                    'feature_group': group,
                    'repetition': repetition + 1,
                    'baseline_normalized_ause': baseline_ause,
                    'permuted_normalized_ause': ause,
                    'delta_normalized_ause': ause - baseline_ause,
                    'baseline_spearman': baseline_spearman,
                    'permuted_spearman': spearman,
                    'delta_spearman': spearman - baseline_spearman,
                }
            )
    return rows


def evaluate_decision_gates(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the prespecified gates; A-03 remains stress-test evidence only."""
    lookup = {
        (row['run_id'], row['split'], row['method']): row
        for row in metrics
        if row['record_type'] == 'ranking'
    }
    a01_ids = sorted(
        {
            row['run_id']
            for row in metrics
            if row['strategy'] == 'random' and row['split'] == 'evaluation'
        }
    )
    development_details: list[dict[str, Any]] = []
    evaluation_details: list[dict[str, Any]] = []
    relative_improvements: list[float] = []
    for run_id in a01_ids:
        dev_baseline = lookup[(run_id, 'development', 'tree_std')]
        dev_candidate = lookup[(run_id, 'development', 'conditional_error_hgb')]
        eval_baseline = lookup[(run_id, 'evaluation', 'tree_std')]
        eval_candidate = lookup[(run_id, 'evaluation', 'conditional_error_hgb')]
        development_details.append(
            {
                'run_id': run_id,
                'baseline_normalized_ause': dev_baseline['normalized_ause'],
                'candidate_normalized_ause': dev_candidate['normalized_ause'],
                'improved': bool(
                    dev_candidate['normalized_ause'] < dev_baseline['normalized_ause']
                ),
            }
        )
        improvement = (
            eval_baseline['normalized_ause'] - eval_candidate['normalized_ause']
        ) / eval_baseline['normalized_ause']
        relative_improvements.append(float(improvement))
        evaluation_details.append(
            {
                'run_id': run_id,
                'spearman_improved': bool(
                    eval_candidate['uncertainty_error_spearman']
                    > eval_baseline['uncertainty_error_spearman']
                ),
                'ause_improved': bool(
                    eval_candidate['normalized_ause'] < eval_baseline['normalized_ause']
                ),
                'relative_ause_improvement': float(improvement),
                'candidate_raw_nonpositive_fraction': eval_candidate[
                    'raw_nonpositive_fraction'
                ],
                'candidate_unique_score_fraction': eval_candidate[
                    'unique_score_fraction'
                ],
                'candidate_all_finite': eval_candidate['all_finite'],
            }
        )
    development_pass = bool(
        development_details
        and np.mean(
            [detail['candidate_normalized_ause'] for detail in development_details]
        )
        < np.mean(
            [detail['baseline_normalized_ause'] for detail in development_details]
        )
        and sum(detail['improved'] for detail in development_details) >= 2
    )
    evaluation_per_run_pass = bool(
        evaluation_details
        and all(
            detail['spearman_improved'] and detail['ause_improved']
            for detail in evaluation_details
        )
    )
    mean_relative_improvement = (
        float(np.mean(relative_improvements)) if relative_improvements else float('nan')
    )
    stability_pass = bool(
        evaluation_details
        and all(
            detail['candidate_all_finite']
            and detail['candidate_raw_nonpositive_fraction'] <= 0.05
            and detail['candidate_unique_score_fraction'] >= 0.10
            for detail in evaluation_details
        )
    )
    passed = bool(
        development_pass
        and evaluation_per_run_pass
        and mean_relative_improvement >= 0.05
        and stability_pass
    )
    return {
        'scope': 'A-01 random replicates only',
        'development': {
            'mean_ause_below_tree_std_and_at_least_two_of_three_improve': development_pass,
            'details': development_details,
        },
        'evaluation': {
            'every_replicate_higher_spearman_and_lower_ause': evaluation_per_run_pass,
            'mean_relative_ause_improvement': mean_relative_improvement,
            'minimum_required_relative_improvement': 0.05,
            'details': evaluation_details,
        },
        'stability': {
            'passed': stability_pass,
            'maximum_nonpositive_fraction': 0.05,
            'minimum_unique_score_fraction': 0.10,
        },
        'passed': passed,
        'recommendation': (
            'validate the conditional-error candidate on a future untouched replicate'
            if passed
            else 'retain tree_std; the conditional-error candidate did not pass all gates'
        ),
        'ucb_role': 'selection-shift stress test only; cannot make the candidate pass',
    }


def process_context(
    context: uq.RunContext, cache_dir: Path, n_jobs: int
) -> tuple[
    pl.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Run one complete, independent retrospective analysis."""
    split_features, feature_cost = uq._measure(
        'feature_loading',
        lambda: uq.extract_split_features(context, cache_dir, n_jobs),
        run_id=context.run_id,
        method=None,
        measurement='actual',
        profile_repetition=None,
    )
    train_features, development_features, evaluation_features = split_features
    all_smiles = context.train_smiles + context.calibration_smiles + context.test_smiles
    fingerprints, fingerprint_cost = uq._measure(
        'fingerprint_indexing',
        lambda: uq._morgan_fingerprints(all_smiles, n_jobs),
        run_id=context.run_id,
        method='local_oob_residual_k25',
        measurement='actual',
        profile_repetition=None,
    )
    chemistry, chemistry_cost = uq._measure(
        'chemistry_features',
        lambda: chemistry_features(all_smiles),
        run_id=context.run_id,
        method='conditional_error_hgb',
        measurement='actual',
        profile_repetition=None,
    )
    train_fingerprints = fingerprints[:16_000]
    development_fingerprints = fingerprints[16_000:17_000]
    evaluation_fingerprints = fingerprints[17_000:]
    train_chemistry = chemistry[:16_000]
    development_chemistry = chemistry[16_000:17_000]
    evaluation_chemistry = chemistry[17_000:]

    oof = compute_oof_meta_features(
        context,
        train_features,
        train_fingerprints,
        train_chemistry,
        n_jobs,
    )
    learner, final_rf_cost = uq._measure(
        'final_rf_fit',
        lambda: uq._fit_common_rf(
            train_features, context.train_targets, context.seed, n_jobs
        ),
        run_id=context.run_id,
        method='tree_std',
        measurement='actual',
        profile_repetition=None,
    )
    prepared_train = uq._preprocessed_features(learner, train_features)
    qrf, final_qrf_cost = uq._measure(
        'final_qrf_fit',
        lambda: uq._fit_qrf(
            prepared_train, context.train_targets, context.seed, n_jobs
        ),
        run_id=context.run_id,
        method='qrf_i80_width',
        measurement='actual',
        profile_repetition=None,
    )
    development, development_cost = _final_meta_features(
        context,
        learner,
        qrf,
        train_fingerprints,
        development_fingerprints,
        development_features,
        development_chemistry,
        n_jobs,
        'development',
    )
    evaluation, evaluation_cost = _final_meta_features(
        context,
        learner,
        qrf,
        train_fingerprints,
        evaluation_fingerprints,
        evaluation_features,
        evaluation_chemistry,
        n_jobs,
        'evaluation',
    )
    reconstruction = uq.reconstruction_check(context, development.predictions)
    meta_targets = np.abs(context.train_targets - oof.meta.predictions)
    model, hgb_cost = uq._measure(
        'conditional_error_fit',
        lambda: _hgb(context.seed).fit(oof.meta.features, meta_targets),
        run_id=context.run_id,
        method='conditional_error_hgb',
        measurement='actual',
        profile_repetition=None,
    )
    score_sets = _score_sets(
        oof.meta.features, development.features, evaluation.features, model
    )
    development_metrics, development_curves = _ranking_records(
        context,
        'development',
        context.calibration_targets,
        development.predictions,
        score_sets[1],
    )
    evaluation_metrics, evaluation_curves = _ranking_records(
        context,
        'evaluation',
        context.test_targets,
        evaluation.predictions,
        score_sets[2],
    )
    feature_group_rows: list[dict[str, Any]] = []
    permutation_cost: dict[str, Any] | None = None
    if context.strategy == 'random':
        development_errors = np.abs(
            context.calibration_targets - development.predictions
        )
        feature_group_rows, permutation_cost = uq._measure(
            'feature_group_permutation',
            lambda: permutation_feature_groups(
                context, model, development.features, development_errors
            ),
            run_id=context.run_id,
            method='conditional_error_hgb',
            measurement='actual',
            profile_repetition=None,
        )
    cycles = _labeled_cycles(context)
    if np.any(cycles[:16_000] >= uq.CALIBRATION_CYCLE):
        raise AssertionError('cycle 7--9 compounds entered meta-model training')
    scores = _score_frame(context, cycles, oof, development, evaluation, score_sets)
    costs = [
        _annotate_cost(feature_cost, context),
        _annotate_cost(fingerprint_cost, context),
        _annotate_cost(chemistry_cost, context),
        *oof.costs,
        _annotate_cost(final_rf_cost, context),
        _annotate_cost(final_qrf_cost, context),
        development_cost,
        evaluation_cost,
        _annotate_cost(hgb_cost, context),
    ]
    if permutation_cost is not None:
        costs.append(_annotate_cost(permutation_cost, context))
    return (
        scores,
        development_metrics + evaluation_metrics,
        development_curves + evaluation_curves,
        feature_group_rows,
        costs,
        reconstruction,
        oof.fold_metadata,
    )


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _input_hashes(
    results_dir: Path, contexts: Sequence[uq.RunContext]
) -> dict[str, str]:
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
    return {str(path.resolve()): uq._sha256(path) for path in paths}


def _output_names() -> tuple[str, ...]:
    return (
        'rf_conditional_error_scores.parquet',
        'rf_conditional_error_metrics.csv',
        'rf_conditional_error_curves.csv',
        'rf_conditional_error_feature_groups.csv',
        'rf_conditional_error_costs.csv',
        'rf_conditional_error_report.md',
        'rf_conditional_error_random_ranking.pdf',
        'rf_conditional_error_random_ranking.png',
        'rf_conditional_error_attribution_stress.pdf',
        'rf_conditional_error_attribution_stress.png',
    )


def _update_output_hashes(metadata: dict[str, Any], output_dir: Path) -> None:
    metadata['output_hashes'] = {
        name: uq._sha256(output_dir / name)
        for name in _output_names()
        if (output_dir / name).is_file()
    }


def _write_metadata(metadata: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'rf_conditional_error_metadata.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n'
    )


def _base_metadata(
    results_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    n_jobs: int,
    contexts: Sequence[uq.RunContext],
    reconstruction: list[dict[str, Any]],
    folds: dict[str, list[dict[str, Any]]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    return {
        'analysis': 'Leakage-safe conditional RF absolute-error model benchmark',
        'scope': {
            'design': 'retrospective',
            'primary': 'A-01 random replicates 1-3',
            'stress_test': 'A-03 UCB replicates 1-3',
            'claim_boundary': (
                'AmpC 1M retrospective evidence only; a future untouched replicate '
                'is required before active-learning adoption'
            ),
        },
        'paths': {
            'results_dir': str(results_dir.resolve()),
            'output_dir': str(output_dir.resolve()),
            'cache_dir': str(cache_dir.resolve()),
        },
        'parameters': {
            'train_cycles': list(uq.TRAIN_CYCLES),
            'development_cycle': uq.CALIBRATION_CYCLE,
            'evaluation_cycles': list(uq.TEST_CYCLES),
            'split_sizes': uq.SPLIT_SIZES,
            'n_scaffold_folds': N_FOLDS,
            'fold_splitter': 'GroupKFold(shuffle=True, random_state=run_seed)',
            'n_estimators': N_ESTIMATORS,
            'morgan_radius': 2,
            'morgan_bits': 2048,
            'morgan_include_chirality': False,
            'local_k': LOCAL_K,
            'feature_names': list(FEATURE_NAMES),
            'feature_groups': {
                name: [FEATURE_NAMES[index] for index in indices]
                for name, indices in FEATURE_GROUPS.items()
            },
            'conditional_model': {
                'class': 'HistGradientBoostingRegressor',
                'loss': 'squared_error',
                'learning_rate': 0.05,
                'max_iter': 200,
                'max_leaf_nodes': 15,
                'min_samples_leaf': 50,
                'l2_regularization': 1.0,
                'early_stopping': False,
                'random_state': 'run_seed',
            },
            'conditional_target': 'absolute scaffold-grouped OOF RF residual',
            'negative_prediction_handling': (
                'raw predictions retained; negative values clamped to zero only '
                'when used as ranking scores'
            ),
            'blend': (
                'mean of tie-aware empirical mid-CDF percentiles for tree_std, '
                'qrf_i80_width, and local_oob_residual_k25; CDFs fitted on OOF '
                'training scores only'
            ),
            'removal_fractions': uq.REMOVAL_FRACTIONS.tolist(),
            'permutation_repeats': PERMUTATION_REPEATS,
            'n_jobs': n_jobs,
        },
        'runs': [
            {
                'run_id': context.run_id,
                'family': context.family,
                'strategy': context.strategy,
                'replicate': context.replicate,
                'seed': context.seed,
            }
            for context in contexts
        ],
        'fold_membership': folds,
        'reconstruction_checks': reconstruction,
        'common_rf_checks': {
            'single_final_rf_per_run': True,
            'all_development_and_evaluation_methods_share_predictions_and_errors': True,
            'cycle_7_reconstruction_required': 'MAE <= 0.05 and Spearman >= 0.999',
        },
        'decision_gates': gates,
        'versions': {
            'python': platform.python_version(),
            'numpy': _version('numpy'),
            'polars': _version('polars'),
            'scikit-learn': _version('scikit-learn'),
            'rdkit': _version('rdkit'),
            'quantile-forest': _version('quantile-forest'),
            'learnm8': _version('learnm8'),
        },
        'input_hashes': _input_hashes(results_dir, contexts),
        'output_hashes': {},
        'determinism': {
            'scientific_tables': 'deterministic for fixed inputs and package versions',
            'timing_values': 'explicitly excluded from determinism comparisons',
            'plot_only': 'performs no model fitting',
        },
    }


def compute_stage(
    results_dir: Path, output_dir: Path, cache_dir: Path, n_jobs: int
) -> None:
    contexts = uq.discover_runs(results_dir)
    scores: list[pl.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    feature_groups: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    reconstruction: list[dict[str, Any]] = []
    fold_metadata: dict[str, list[dict[str, Any]]] = {}
    for context in contexts:
        LOGGER.info('Computing %s', context.run_id)
        result = process_context(context, cache_dir, n_jobs)
        scores.append(result[0])
        metrics.extend(result[1])
        curves.extend(result[2])
        feature_groups.extend(result[3])
        costs.extend(result[4])
        reconstruction.append(result[5])
        fold_metadata[context.run_id] = result[6]
        del result
        gc.collect()
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(scores, how='vertical').sort(
        ['family', 'replicate', 'split', 'labeled_cycle', 'ID']
    ).write_parquet(output_dir / 'rf_conditional_error_scores.parquet')
    uq._write_csv(
        metrics,
        output_dir / 'rf_conditional_error_metrics.csv',
        ('family', 'replicate', 'split', 'method'),
    )
    uq._write_csv(
        curves,
        output_dir / 'rf_conditional_error_curves.csv',
        ('family', 'replicate', 'split', 'method', 'removal_fraction'),
    )
    uq._write_csv(
        feature_groups,
        output_dir / 'rf_conditional_error_feature_groups.csv',
        ('replicate', 'feature_group', 'repetition'),
    )
    uq._write_csv(
        costs,
        output_dir / 'rf_conditional_error_costs.csv',
        ('run_id', 'fold', 'phase', 'method'),
    )
    gates = evaluate_decision_gates(metrics)
    metadata = _base_metadata(
        results_dir,
        output_dir,
        cache_dir,
        n_jobs,
        contexts,
        reconstruction,
        fold_metadata,
        gates,
    )
    _update_output_hashes(metadata, output_dir)
    _write_metadata(metadata, output_dir)
    LOGGER.info('Wrote compute-stage tables to %s', output_dir)


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
    creator = 'LearnM8 conditional-error benchmark'
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


def plot_random_ranking(
    metrics: pl.DataFrame, curves: pl.DataFrame, output_dir: Path
) -> None:
    style.apply()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(
        1, 2, figsize=(style.DOUBLE_COL, 70 * style.MM), constrained_layout=True
    )
    selected_curves = curves.filter(
        (pl.col('strategy') == 'random') & (pl.col('split') == 'evaluation')
    )
    for method in SCORE_METHODS:
        summary = _aggregate(
            selected_curves.filter(pl.col('method') == method),
            ('removal_fraction',),
            'normalized_retained_mae',
        )
        x = summary['removal_fraction'].to_numpy()
        axes[0].plot(
            x,
            summary['mean'].to_numpy(),
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
        axes[0].fill_between(
            x,
            summary['min'].to_numpy(),
            summary['max'].to_numpy(),
            color=METHOD_COLORS[method],
            alpha=0.10,
        )
    oracle = _aggregate(
        selected_curves.filter(pl.col('method') == 'oracle'),
        ('removal_fraction',),
        'normalized_retained_mae',
    )
    axes[0].plot(
        oracle['removal_fraction'].to_numpy(),
        oracle['mean'].to_numpy(),
        color=style.INK,
        linestyle=':',
        label='Oracle',
    )
    axes[0].plot(
        uq.REMOVAL_FRACTIONS,
        np.ones_like(uq.REMOVAL_FRACTIONS),
        color=style.MUTED,
        linestyle='--',
        label='Random removal',
    )
    axes[0].set(
        title='A-01 evaluation sparsification',
        xlabel='Compounds removed (%)',
        ylabel='Normalized retained MAE',
        xticks=(0.0, 0.3, 0.6, 0.9),
        xticklabels=('0', '30', '60', '90'),
    )
    axes[0].legend(loc='best', fontsize=6.0, ncol=2)

    ranking = metrics.filter(
        (pl.col('strategy') == 'random') & (pl.col('split') == 'evaluation')
    )
    x = np.arange(len(SCORE_METHODS))
    for replicate in uq.REPLICATE_SEEDS:
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
            marker='o',
            color=style.MUTED,
            alpha=0.5,
            linewidth=0.9,
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
    axes[1].tick_params(axis='x', rotation=33)
    axes[1].legend(loc='best', fontsize=6.2)
    axes[0].text(-0.10, 1.06, 'A', transform=axes[0].transAxes, fontweight='bold')
    axes[1].text(-0.10, 1.06, 'B', transform=axes[1].transAxes, fontweight='bold')
    _save_figure(fig, output_dir, 'rf_conditional_error_random_ranking')


def plot_attribution_stress(
    metrics: pl.DataFrame, feature_groups: pl.DataFrame, output_dir: Path
) -> None:
    style.apply()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(
        1, 2, figsize=(style.DOUBLE_COL, 68 * style.MM), constrained_layout=True
    )
    groups = tuple(FEATURE_GROUPS)
    x = np.arange(len(groups))
    for replicate in uq.REPLICATE_SEEDS:
        values = [
            float(
                feature_groups.filter(
                    (pl.col('replicate') == replicate)
                    & (pl.col('feature_group') == group)
                )['delta_normalized_ause'].mean()
            )
            for group in groups
        ]
        axes[0].plot(
            x,
            values,
            marker='o',
            color=style.MUTED,
            alpha=0.55,
            linewidth=0.9,
            label=f'Replicate {replicate}',
        )
    means = [
        float(
            feature_groups.filter(pl.col('feature_group') == group)[
                'delta_normalized_ause'
            ].mean()
        )
        for group in groups
    ]
    axes[0].scatter(x, means, color=style.INK, marker='D', s=20, label='Mean')
    axes[0].axhline(0.0, color=style.MUTED, linestyle='--', linewidth=0.8)
    axes[0].set(
        title='Feature-group attribution',
        ylabel='Change in normalized AUSE',
        xticks=x,
        xticklabels=('RF', 'QRF', 'Neighborhood', 'Chemistry'),
    )
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].legend(loc='best', fontsize=6.2)

    ranking = metrics.filter(pl.col('split') == 'evaluation')
    family_x = np.arange(2)
    for replicate in uq.REPLICATE_SEEDS:
        differences = []
        for strategy in ('random', 'ucb'):
            subset = ranking.filter(
                (pl.col('strategy') == strategy) & (pl.col('replicate') == replicate)
            )
            baseline = float(
                subset.filter(pl.col('method') == 'tree_std')['normalized_ause'][0]
            )
            candidate = float(
                subset.filter(pl.col('method') == 'conditional_error_hgb')[
                    'normalized_ause'
                ][0]
            )
            differences.append(candidate - baseline)
        axes[1].plot(
            family_x,
            differences,
            marker='o',
            color=METHOD_COLORS['conditional_error_hgb'],
            alpha=0.65,
            linewidth=1.0,
            label=f'Replicate {replicate}',
        )
    axes[1].axhline(0.0, color=style.MUTED, linestyle='--', linewidth=0.8)
    axes[1].set(
        title='UCB selection-shift stress',
        ylabel='AUSE difference vs tree SD',
        xticks=family_x,
        xticklabels=('A-01 random', 'A-03 UCB'),
    )
    axes[1].legend(loc='best', fontsize=6.2)
    axes[0].text(-0.10, 1.06, 'A', transform=axes[0].transAxes, fontweight='bold')
    axes[1].text(-0.10, 1.06, 'B', transform=axes[1].transAxes, fontweight='bold')
    _save_figure(fig, output_dir, 'rf_conditional_error_attribution_stress')


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n'.join(lines)


def write_report(
    metrics: pl.DataFrame,
    feature_groups: pl.DataFrame,
    costs: pl.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    random_eval = metrics.filter(
        (pl.col('strategy') == 'random') & (pl.col('split') == 'evaluation')
    )
    ranking_rows = []
    for method in SCORE_METHODS:
        selected = random_eval.filter(pl.col('method') == method)
        ranking_rows.append(
            (
                METHOD_LABELS[method],
                f'{float(selected["uncertainty_error_spearman"].mean()):.3f}',
                f'{float(selected["normalized_ause"].mean()):.3f}',
                f'{100 * float(selected["raw_nonpositive_fraction"].max()):.1f}%',
            )
        )
    attribution_rows = []
    for group in FEATURE_GROUPS:
        selected = feature_groups.filter(pl.col('feature_group') == group)
        attribution_rows.append(
            (
                group.upper() if group in {'rf', 'qrf'} else group.title(),
                f'{float(selected["delta_normalized_ause"].mean()):+.4f}',
                f'{float(selected["delta_normalized_ause"].min()):+.4f}',
                f'{float(selected["delta_normalized_ause"].max()):+.4f}',
            )
        )
    cost_rows = []
    for phase in sorted(set(costs['phase'].to_list())):
        selected = costs.filter(pl.col('phase') == phase)
        cost_rows.append(
            (
                phase,
                f'{float(selected["wall_seconds"].sum()):.2f}',
                f'{float(selected["incremental_peak_rss_mb"].max()):.1f}',
            )
        )
    gates = metadata['decision_gates']
    evaluation_details = gates['evaluation']['details']
    replicate_improvements = ', '.join(
        f'{100 * detail["relative_ause_improvement"]:.1f}%'
        for detail in evaluation_details
    )
    ucb_eval = metrics.filter(
        (pl.col('strategy') == 'ucb') & (pl.col('split') == 'evaluation')
    )
    ucb_baseline = ucb_eval.filter(pl.col('method') == 'tree_std')
    ucb_candidate = ucb_eval.filter(pl.col('method') == 'conditional_error_hgb')
    ucb_improved = sum(
        candidate < baseline
        for candidate, baseline in zip(
            ucb_candidate.sort('replicate')['normalized_ause'].to_list(),
            ucb_baseline.sort('replicate')['normalized_ause'].to_list(),
            strict=True,
        )
    )
    lines = [
        '# Leakage-safe conditional RF error model benchmark',
        '',
        '## Conclusion',
        '',
        gates['recommendation'][0].upper() + gates['recommendation'][1:] + '.',
        '',
        'The pass/fail decision uses only A-01 random runs. A-03 UCB results are '
        'a selection-shift stress test and cannot make the candidate pass. This '
        'is retrospective evidence for this AmpC 1M setting, not validated '
        'production performance.',
        '',
        'On A-01 evaluation, normalized AUSE improved in all three replicates '
        f'({replicate_improvements}; mean '
        f'{100 * gates["evaluation"]["mean_relative_ause_improvement"]:.1f}%). '
        'Under A-03 UCB selection shift, the conditional model improved AUSE in '
        f'only {ucb_improved}/3 replicates and its mean AUSE was '
        f'{float(ucb_candidate["normalized_ause"].mean()):.3f} versus '
        f'{float(ucb_baseline["normalized_ause"].mean()):.3f} for tree SD. '
        'The gain is therefore materially dependent on the selection distribution.',
        '',
        '## A-01 evaluation ranking',
        '',
        _markdown_table(
            ('Method', 'Mean Spearman', 'Mean normalized AUSE', 'Max nonpositive'),
            ranking_rows,
        ),
        '',
        '## Development feature-group permutation',
        '',
        'Positive AUSE changes mean the permuted group carried useful ranking '
        'information. These values are descriptive and were not used for tuning.',
        '',
        _markdown_table(
            ('Group', 'Mean delta AUSE', 'Minimum', 'Maximum'), attribution_rows
        ),
        '',
        '## Actual-stage computational cost',
        '',
        _markdown_table(
            ('Phase', 'Total wall seconds', 'Maximum incremental RSS MB'), cost_rows
        ),
        '',
        '## Figures',
        '',
        '- [A-01 ranking](rf_conditional_error_random_ranking.pdf)',
        '- [Feature attribution and A-03 stress test](rf_conditional_error_attribution_stress.pdf)',
        '',
    ]
    (output_dir / 'rf_conditional_error_report.md').write_text('\n'.join(lines))


def plot_stage(output_dir: Path) -> None:
    required = {
        'metrics': output_dir / 'rf_conditional_error_metrics.csv',
        'curves': output_dir / 'rf_conditional_error_curves.csv',
        'feature_groups': output_dir / 'rf_conditional_error_feature_groups.csv',
        'costs': output_dir / 'rf_conditional_error_costs.csv',
        'metadata': output_dir / 'rf_conditional_error_metadata.json',
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f'plot stage requires {path}')
    metrics = pl.read_csv(required['metrics'], infer_schema_length=None)
    curves = pl.read_csv(required['curves'], infer_schema_length=None)
    feature_groups = pl.read_csv(required['feature_groups'], infer_schema_length=None)
    costs = pl.read_csv(required['costs'], infer_schema_length=None)
    metadata = json.loads(required['metadata'].read_text())
    plot_random_ranking(metrics, curves, output_dir)
    plot_attribution_stress(metrics, feature_groups, output_dir)
    write_report(metrics, feature_groups, costs, metadata, output_dir)
    metadata['input_hashes'][str(Path(__file__).resolve())] = uq._sha256(
        Path(__file__).resolve()
    )
    _update_output_hashes(metadata, output_dir)
    _write_metadata(metadata, output_dir)
    LOGGER.info('Wrote plot-only figures and report to %s', output_dir)


def _smoke_context(results_dir: Path) -> uq.RunContext:
    source = next(
        context
        for context in uq.discover_runs(results_dir)
        if context.family == 'A-01' and context.replicate == 1
    )
    train_n, development_n, evaluation_n = 500, 80, 80
    return uq.RunContext(
        run_id=source.run_id,
        family=source.family,
        strategy=source.strategy,
        replicate=source.replicate,
        seed=source.seed,
        path=source.path,
        train_ids=source.train_ids[:train_n],
        train_smiles=source.train_smiles[:train_n],
        train_targets=source.train_targets[:train_n],
        calibration_ids=source.calibration_ids[:development_n],
        calibration_smiles=source.calibration_smiles[:development_n],
        calibration_targets=source.calibration_targets[:development_n],
        test_ids=source.test_ids[:evaluation_n],
        test_smiles=source.test_smiles[:evaluation_n],
        test_targets=source.test_targets[:evaluation_n],
    )


def smoke(results_dir: Path, cache_dir: Path, n_jobs: int) -> None:
    """Run leakage and numerical checks on real A-01 molecules."""
    context = _smoke_context(results_dir)
    all_smiles = context.train_smiles + context.calibration_smiles + context.test_smiles
    features = uq.extract_features(
        list(all_smiles),
        'morgan',
        cache_dir=cache_dir,
        n_jobs=n_jobs,
        preferred_dtype='uint8',
    )
    train_n = len(context.train_ids)
    development_n = len(context.calibration_ids)
    train_features = features[:train_n]
    fingerprints = uq._morgan_fingerprints(all_smiles, n_jobs)
    chemistry = chemistry_features(all_smiles)
    if not np.array_equal(chemistry, chemistry_features(all_smiles)):
        raise AssertionError('chemistry meta-features are not deterministic')

    fold, scaffolds, metadata = scaffold_fold_assignment(
        context.train_ids, context.train_smiles, context.seed
    )
    repeat_fold, repeat_scaffolds, repeat_metadata = scaffold_fold_assignment(
        context.train_ids, context.train_smiles, context.seed
    )
    if (
        not np.array_equal(fold, repeat_fold)
        or scaffolds != repeat_scaffolds
        or metadata != repeat_metadata
    ):
        raise AssertionError('scaffold folds are not deterministic')
    if np.any(np.bincount(fold, minlength=N_FOLDS) == 0):
        raise AssertionError('a scaffold fold is empty')

    oof = compute_oof_meta_features(
        context,
        train_features,
        fingerprints[:train_n],
        chemistry[:train_n],
        n_jobs,
    )
    if not np.array_equal(oof.fold, fold):
        raise AssertionError('OOF features do not use the validated fold assignment')
    if oof.meta.features.shape != (train_n, len(FEATURE_NAMES)):
        raise AssertionError('smoke meta-feature contract changed')
    if not np.isfinite(oof.meta.features).all():
        raise AssertionError('smoke OOF meta-features are non-finite')

    residuals = np.linspace(0.1, 3.0, 30)
    direct_query = fingerprints[train_n : train_n + 4]
    chunked = neighbor_features(
        fingerprints[:30], direct_query, residuals, k=5, query_chunk_size=2
    )
    direct = []
    for query in direct_query:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(query, fingerprints[:30]),
            dtype=np.float64,
        )
        neighbors = uq._stable_top_k_indices(similarities, 5)
        selected_similarity = similarities[neighbors]
        selected_residual = residuals[neighbors]
        direct.append(
            (
                selected_similarity[0],
                selected_similarity[-1],
                np.mean(selected_similarity),
                np.mean(selected_residual),
                np.median(selected_residual),
                np.std(selected_residual, ddof=0),
                np.quantile(selected_residual, 0.90),
            )
        )
    if not np.array_equal(chunked, np.asarray(direct)):
        raise AssertionError(
            'chunked local features differ from direct Tanimoto search'
        )

    meta_targets = np.abs(context.train_targets - oof.meta.predictions)
    first_model = _hgb(context.seed).fit(oof.meta.features, meta_targets)
    second_model = _hgb(context.seed).fit(oof.meta.features, meta_targets)
    development_slice = slice(train_n, train_n + development_n)
    # A finite prediction check on real held-out molecules is sufficient here;
    # full final-model reconstruction is enforced during every scientific run.
    development_proxy = np.column_stack(
        (
            oof.meta.features[:development_n, :15],
            chemistry[development_slice],
        )
    )
    first = uq._finite_array(
        first_model.predict(development_proxy), 'smoke HGB predictions'
    )
    second = uq._finite_array(
        second_model.predict(development_proxy), 'repeat smoke HGB predictions'
    )
    if not np.array_equal(first, second):
        raise AssertionError('conditional-error predictions are not deterministic')
    if set(context.train_ids) & (set(context.calibration_ids) | set(context.test_ids)):
        raise AssertionError('development or evaluation IDs entered smoke training')
    LOGGER.info(
        'Smoke passed on real A-01 molecules: deterministic scaffold folds, '
        'exactly-once OOF features, fold-safe local neighborhoods, direct '
        'Tanimoto agreement, and finite deterministic HGB predictions'
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
        / 'rf_conditional_error',
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=REPO_ROOT / '.cache' / 'rf_conditional_error',
    )
    parser.add_argument('--n-jobs', type=int, default=16)
    parser.add_argument('--stage', choices=('compute', 'plot', 'all'), default='all')
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
    if args.stage in ('plot', 'all'):
        plot_stage(args.output_dir)


if __name__ == '__main__':
    main()

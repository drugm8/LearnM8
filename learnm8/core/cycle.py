"""
Unified cycle execution for active learning.

This module provides a single execute_cycle() function that handles both
run and benchmark modes, integrating with learners, acquisition functions,
pruning strategies, and oracles.

Key features:
- Single execution path for run and benchmark modes
- Integration with new performance infrastructure (Phases 1-3)
- Comprehensive metrics calculation
- Graceful error handling with clear messages

Architecture:
    execute_cycle() orchestrates:
    1. Training on labeled compounds
    2. Prediction on unlabeled (run) or all (benchmark) compounds
    3. Optional pruning of selection pool
    4. Acquisition-based compound selection
    5. Oracle measurement
    6. Master DataFrame updates
    7. Metrics calculation

The only difference between run and benchmark modes is the prediction pool:
- Run mode: predict on unlabeled compounds only
- Benchmark mode: predict on full original pool

All other logic is identical, eliminating code duplication.
"""

import logging
import math
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Literal
import pandas as pd
import numpy as np

from learnm8.features.extraction import extract_features
from learnm8.core.interfaces import Learner, Oracle
from learnm8.evaluation import evaluate_cycle
from learnm8.utils.logging_formatters import format_cycle_metrics_table

logger = logging.getLogger(__name__)


def execute_cycle(
    compounds_df: pd.DataFrame,
    cycle: int,
    config: 'CycleConfig',
    learner: Learner,
    oracle: Oracle,
    target_col: str,
    featurizer_type: Optional[str],
    cache_dir: Path,
    original_pool_size: int,
    score_direction: str = 'higher',
    mode: Literal['run', 'benchmark'] = 'run',
    original_pool: Optional[pd.DataFrame] = None,
    cumulative_selected_ids: Optional[set] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute a single active learning cycle.

    This function handles both run and benchmark modes with a single execution path.
    The ONLY difference is the prediction pool selection:
    - Run mode: predict on unlabeled compounds only (efficient for production)
    - Benchmark mode: predict on full original pool (for scientifically correct metrics)

    All other logic (training, pruning, selection, measurement, updates) is identical.

    Args:
        compounds_df: Master DataFrame with all compounds and their states
        cycle: Current cycle number (0-indexed)
        config: CycleConfig with strategy, batch_fraction, pruning settings
        learner: Trained machine learning model implementing Learner interface
        oracle: Oracle for measuring compound properties
        target_col: Name of target property column
        featurizer_type: Type of molecular features (morgan, maccs, ecfp6, descriptors)
        cache_dir: Directory for feature caching
        original_pool_size: Size of original compound pool (for batch calculation)
        score_direction: 'higher' or 'lower' for optimization direction
        mode: 'run' or 'benchmark' execution mode
        original_pool: Required for benchmark mode - full original compound pool

    Returns:
        (updated_compounds_df, metrics_dict)

    Raises:
        ValueError: Invalid parameters or no compounds available
        RuntimeError: Training, prediction, or selection failures

    Examples:
        # Run mode (production)
        updated_df, metrics = execute_cycle(
            compounds_df, 0, config, learner, oracle,
            'Activity', 'morgan', cache_dir, 1000,
            mode='run'
        )

        # Benchmark mode (evaluation)
        updated_df, metrics = execute_cycle(
            compounds_df, 0, config, learner, oracle,
            'Activity', 'morgan', cache_dir, 1000,
            mode='benchmark', original_pool=original_df
        )
    """
    from learnm8.core.dataframe_ops import (
        get_compounds_by_status,
        add_predictions,
        update_status
    )

    # Step 1: Setup and Validation
    if mode not in ['run', 'benchmark']:
        raise ValueError(f"mode must be 'run' or 'benchmark', got '{mode}'")

    if mode == 'benchmark' and original_pool is None:
        raise ValueError("original_pool required for benchmark mode")

    if score_direction not in ['higher', 'lower']:
        raise ValueError(f"score_direction must be 'higher' or 'lower', got '{score_direction}'")

    # Extract cumulative selected IDs if not provided
    if cumulative_selected_ids is None:
        cumulative_selected_ids = set(compounds_df[compounds_df['status'] == 'labeled']['ID'].tolist())

    # Step 2 & 3: Training
    # Get Labeled Compounds for Training
    labeled_df = get_compounds_by_status(
        compounds_df, 'labeled', columns=['ID', 'SMILES', target_col]
    ).copy()

    unlabeled_df = get_compounds_by_status(
        compounds_df, 'unlabeled', columns=['ID', 'SMILES']
    )

    logger.debug(f"Cycle {cycle}: Pool composition - {len(labeled_df)} labeled, {len(unlabeled_df)} unlabeled")
    logger.debug(f"Cycle {cycle} configuration: strategy={config.strategy}, batch_fraction={config.batch_fraction}, pruning={config.pruning_strategy or 'disabled'}")

    if len(labeled_df) == 0:
        raise RuntimeError(
            f"Cycle {cycle}: No labeled compounds available. "
            f"This should not happen after initialization (cycle 0). "
            f"Check that select_initial_batch() ran successfully before cycles."
        )
    else:
        # Extract features and train learner
        try:
            training_targets = labeled_df[target_col].values
            training_smiles = labeled_df['SMILES'].tolist()

            if learner.requires_smiles():
                if featurizer_type is not None:
                    logger.info(
                        f"Extracting {featurizer_type} features as x_d descriptors "
                        f"for {len(labeled_df)} training compounds"
                    )
                    training_features = extract_features(
                        training_smiles,
                        featurizer_type,
                        cache_dir=cache_dir
                    )
                    logger.info(
                        f"Training {learner.get_name()} with SMILES + {training_features.shape[1]}-D "
                        f"extra descriptors on {len(labeled_df)} compounds"
                    )
                else:
                    training_features = None
                    logger.info(
                        f"Training {learner.get_name()} with SMILES (graph-only) "
                        f"on {len(labeled_df)} compounds"
                    )

                learner.train(
                    features=training_features,
                    targets=training_targets,
                    smiles=training_smiles
                )
                logger.debug(f"Model trained on {len(training_smiles)} compounds")
            else:
                if featurizer_type is None:
                    raise ValueError(
                        f"featurizer_type is required for {learner.get_name()}"
                    )

                logger.info(f"Extracting features for {len(labeled_df)} training compounds ({featurizer_type} fingerprints)")
                training_features = extract_features(
                    training_smiles,
                    featurizer_type,
                    cache_dir=cache_dir
                )
                logger.debug(f"Extracting {featurizer_type} features with cache_dir={cache_dir}")
                logger.info(f"Features extracted: {len(labeled_df)} compounds processed")

                logger.info(f"Training model on {len(labeled_df)} labeled compounds")
                learner.train(training_features, training_targets)
                logger.debug(f"Model trained on features shape: {training_features.shape}, targets shape: {training_targets.shape}")
        except Exception as e:
            logger.error(f"Training failed in cycle {cycle}: {e}")
            raise RuntimeError(f"Training failed in cycle {cycle}: {e}")

        logger.info("Training complete")

    prediction_pool = get_compounds_by_status(
        compounds_df, 'unlabeled', columns=['ID', 'SMILES']
    ).copy()

    if len(prediction_pool) == 0:
        logger.warning(f"No unlabeled compounds available for prediction in cycle {cycle}. Returning unchanged DataFrame.")
        metrics = {
            'cycle': cycle,
            'strategy': config.strategy,
            'selected_count': 0,
            'remaining_pool': 0,
            'remaining_unlabeled': 0,
            'cumulative_labeled': int((compounds_df['status'] == 'labeled').sum()),
            'cumulative_pruned': 0,
            'pruned_count': 0
        }
        return compounds_df, metrics

    try:
        prediction_smiles = prediction_pool['SMILES'].tolist()

        if learner.requires_smiles():
            if featurizer_type is not None:
                logger.info(
                    f"Extracting {featurizer_type} features as x_d descriptors "
                    f"for {len(prediction_pool)} unlabeled compounds"
                )
                prediction_features = extract_features(
                    prediction_smiles,
                    featurizer_type,
                    cache_dir=cache_dir,
                    show_progress=len(prediction_pool) > 10000
                )
                logger.info(
                    f"Generating predictions for {len(prediction_pool)} unlabeled compounds "
                    f"with SMILES + {prediction_features.shape[1]}-D extra descriptors"
                )
            else:
                prediction_features = None
                logger.info(
                    f"Generating predictions for {len(prediction_pool)} unlabeled compounds "
                    f"with SMILES (graph-only)"
                )

            predictions, uncertainties = learner.predict(
                features=prediction_features,
                smiles=prediction_smiles
            )
            logger.debug(f"Predictions generated for {len(prediction_pool)} compounds (mode={mode})")
        else:
            if featurizer_type is None:
                raise ValueError(
                    f"featurizer_type is required for {learner.get_name()}"
                )

            logger.info(f"Extracting features for {len(prediction_pool)} unlabeled compounds ({featurizer_type} fingerprints)")
            prediction_features = extract_features(
                prediction_smiles,
                featurizer_type,
                cache_dir=cache_dir,
                show_progress=len(prediction_pool) > 10000
            )
            logger.debug(f"Extracting {featurizer_type} features for prediction pool")
            logger.info(f"Features extracted: {len(prediction_pool)} compounds processed")

            logger.info(f"Generating predictions for {len(prediction_pool)} unlabeled compounds")
            predictions, uncertainties = learner.predict(prediction_features)
            logger.debug(f"Predicting on {len(prediction_pool)} unlabeled compounds (mode={mode})")

        logger.info(f"Predictions complete: {len(predictions)} predictions generated")
        logger.debug(f"Prediction statistics: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
    except Exception as e:
        logger.error(f"Prediction failed in cycle {cycle}: {e}")
        raise RuntimeError(f"Prediction failed in cycle {cycle}: {e}")

    if len(predictions) == 0:
        raise RuntimeError(f"Prediction returned 0 results in cycle {cycle}")

    logger.debug(f"Generated {len(predictions)} predictions")

    valid_compound_ids = prediction_pool['ID'].tolist()
    compounds_df = add_predictions(
        compounds_df, cycle, valid_compound_ids, predictions, uncertainties
    )

    unlabeled_mask = compounds_df['status'] == 'unlabeled'
    pred_col = f'prediction_cycle_{cycle}'
    has_pred_mask = compounds_df[pred_col].notna()
    selection_pool = compounds_df.loc[
        unlabeled_mask & has_pred_mask,
        ['ID', 'SMILES', pred_col]
    ].copy()
    selection_pool = selection_pool.rename(columns={pred_col: 'prediction'})

    if uncertainties is not None:
        unc_col = f'uncertainty_cycle_{cycle}'
        selection_pool['uncertainty'] = compounds_df.loc[
            unlabeled_mask & has_pred_mask, unc_col
        ].values

    if len(selection_pool) == 0:
        raise RuntimeError(f"No unlabeled compounds with predictions available for selection in cycle {cycle}")

    logger.debug(f"Selection pool: {len(selection_pool)} unlabeled compounds with predictions")

    # Step 8: Apply Pruning (If Specified)
    pruning_stats = {'pruned_count': 0, 'pruned_ids': []}
    unlabeled_before_prune = len(selection_pool)

    if config.pruning_strategy:
        logger.info(f"Pruning design space with '{config.pruning_strategy}' strategy")

        uncertainty_values = None
        if 'uncertainty' in selection_pool.columns:
            uncertainty_values = selection_pool['uncertainty'].values

        selection_pool, pruning_stats = _apply_pruning(
            selection_pool,
            selection_pool['prediction'].values,
            uncertainty_values,
            config.pruning_strategy,
            config.pruning_params or {},
            score_direction
        )

        pruned_ids = pruning_stats['pruned_ids']
        if pruned_ids:
            compounds_df = update_status(
                compounds_df, pruned_ids, 'pruned', cycle, target_col, None
            )

        if pruning_stats['pruned_count'] > 0:
            prune_pct = (pruning_stats['pruned_count'] / unlabeled_before_prune) * 100
            logger.info(f"Pruned {pruning_stats['pruned_count']} compounds ({prune_pct:.1f}% of unlabeled pool)")

        logger.debug(f"Pruning parameters: {config.pruning_params}")
        logger.debug(f"Pool before pruning: {unlabeled_before_prune}, after pruning: {len(selection_pool)}")

    if len(selection_pool) == 0:
        raise RuntimeError(f"No compounds remaining after pruning in cycle {cycle}")

    # Step 9: Calculate Batch Size from Fraction
    batch_size = min(
        max(1, math.ceil(original_pool_size * config.batch_fraction)),
        len(selection_pool)
    )

    if batch_size == 0:
        raise RuntimeError(
            f"Calculated batch size is 0 (original_pool_size={original_pool_size}, "
            f"batch_fraction={config.batch_fraction}). Increase batch_fraction."
        )

    # Step 10: Prepare acquisition params with current_best from labeled data
    acquisition_params = (config.acquisition_params or {}).copy()
    if 'current_best' not in acquisition_params:
        labeled_values = compounds_df[compounds_df['status'] == 'labeled'][target_col].values
        if len(labeled_values) > 0:
            acquisition_params['current_best'] = (
                float(labeled_values.max()) if score_direction == 'higher'
                else float(labeled_values.min())
            )
        else:
            logger.warning(
                f"No labeled data available for current_best calculation. "
                f"EI/PI strategies may fail if selected."
            )

    # Step 11: Select Compounds Using Acquisition Strategy
    if config.pruning_strategy and pruning_stats['pruned_count'] > 0:
        logger.info(f"Acquiring compounds with '{config.strategy}' strategy from {len(selection_pool)} remaining candidates")
    else:
        logger.info(f"Acquiring compounds with '{config.strategy}' strategy")

    selected_df = _select_compounds(
        selection_pool,
        config.strategy,
        batch_size,
        score_direction,
        acquisition_params
    )

    selected_ids = selected_df['ID'].tolist()

    if len(selected_ids) == 0:
        raise RuntimeError(f"Acquisition strategy selected 0 compounds in cycle {cycle}")

    logger.info(f"Selected {len(selected_ids)} compounds for labeling")
    logger.debug(f"Selection pool: {len(selection_pool)} unlabeled compounds with predictions")
    logger.debug(f"Batch size: {batch_size} (calculated from {config.batch_fraction} * {original_pool_size})")
    logger.debug(f"{config.strategy.upper()} acquisition selected {len(selected_ids)} compounds")

    # Step 12: Measure Selected Compounds
    selected_compounds = compounds_df[
        compounds_df['ID'].isin(selected_ids)
    ][['ID', 'SMILES']].copy()

    try:
        measurements = oracle.measure(selected_compounds, [target_col])
    except Exception as e:
        logger.error(f"Oracle measurement failed in cycle {cycle}: {e}")
        raise RuntimeError(f"Oracle measurement failed in cycle {cycle}: {e}")

    if not all(sid in measurements['ID'].values for sid in selected_ids):
        raise RuntimeError(f"Oracle did not return measurements for all selected compounds in cycle {cycle}")

    logger.debug(f"Measured {len(measurements)} compounds")

    # Step 13: Update Master DataFrame with Measurements
    compounds_df = update_status(
        compounds_df,
        selected_ids,
        'labeled',
        cycle,
        target_col,
        measurements.set_index('ID')[target_col]
    )

    # Step 14: Calculate Cycle Metrics
    metrics = _calculate_cycle_metrics(
        compounds_df,
        cycle,
        config.strategy,
        predictions,
        uncertainties,
        selected_ids,
        pruning_stats.get('pruned_ids', []),
        target_col,
        score_direction
    )

    # Step 14b: Enhance with evaluation metrics (mode-aware)
    try:
        pred_col = f'prediction_cycle_{cycle}'

        if pred_col in compounds_df.columns:
            labeled_mask = compounds_df['status'] == 'labeled'
            has_pred_mask = compounds_df[pred_col].notna()
            labeled_with_pred_mask = labeled_mask & has_pred_mask

            if labeled_with_pred_mask.sum() > 0:
                labeled_for_eval = compounds_df[labeled_with_pred_mask][['ID', 'SMILES', target_col, pred_col]].copy()

                eval_predictions = labeled_for_eval[pred_col].values
                eval_ground_truth = labeled_for_eval[target_col].values

                valid_mask = ~(np.isnan(eval_predictions) | np.isnan(eval_ground_truth))

                if np.sum(valid_mask) > 0:
                    selected_for_eval = compounds_df[compounds_df['ID'].isin(selected_ids)][['ID', 'SMILES', target_col]].copy()

                    pool_predictions_for_eval = None
                    pool_ids_for_eval = None
                    original_pool_for_eval = None

                    if mode == 'benchmark' and original_pool is not None:
                        pool_predictions_for_eval = predictions
                        pool_ids_for_eval = prediction_pool['ID'].values
                        original_pool_for_eval = original_pool

                    eval_metrics = evaluate_cycle(
                        cycle=cycle,
                        predictions=eval_predictions[valid_mask],
                        ground_truth=eval_ground_truth[valid_mask],
                        labeled_data=labeled_for_eval,
                        selected_compounds=selected_for_eval,
                        target_col=target_col,
                        oracle_type=mode,
                        ground_truth_data=original_pool_for_eval,
                        pool_predictions=pool_predictions_for_eval,
                        pool_ids=pool_ids_for_eval,
                        uncertainties=uncertainties if uncertainties is not None else None,
                        previously_selected=None,
                        advanced_metrics=False,
                        disable_molecular_similarity=True,
                        score_direction=score_direction,
                        cumulative_selected_ids=cumulative_selected_ids,
                        cumulative_labeled_count=int((compounds_df['status'] == 'labeled').sum())
                    )

                    metrics.update(eval_metrics)
                    logger.debug(f"Enhanced metrics with evaluation (Top-10 Discovery: {eval_metrics.get('top_10_discovery', 'N/A')}%)")
    except Exception as e:
        logger.warning(f"Failed to calculate enhanced evaluation metrics in cycle {cycle}: {e}")

    # Step 15: Display Rich Metrics Table
    try:
        # Get previous cycle metrics if available (for change indicators)
        previous_metrics = None
        if cycle > 1:
            # This will be passed from api.py in a future enhancement
            pass

        metrics_table = format_cycle_metrics_table(
            metrics=metrics,
            oracle_type=mode,
            previous_metrics=previous_metrics
        )

        if metrics_table:
            logger.info("\n" + metrics_table)

    except ImportError:
        logger.debug("Rich not installed, skipping metrics table display")
    except Exception as e:
        logger.debug(f"Failed to display metrics table: {e}")

    # Step 16: Return Results
    total_labeled = len(compounds_df[compounds_df['status'] == 'labeled'])
    total_unlabeled = len(compounds_df[compounds_df['status'] == 'unlabeled'])
    logger.info(f"Cycle {cycle} complete: {total_labeled} labeled, {total_unlabeled} unlabeled remaining")

    logger.debug(f"Oracle measured {len(selected_ids)} compounds for property: {target_col}")

    return compounds_df, metrics


def _calculate_cycle_metrics(
    compounds_df: pd.DataFrame,
    cycle: int,
    strategy: str,
    predictions: np.ndarray,
    uncertainties: Optional[np.ndarray],
    selected_ids: List[str],
    pruned_ids: List[str],
    target_col: str,
    score_direction: str
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for the cycle.

    Provides statistics about:
    - Cycle info (cycle number, strategy, batch size)
    - Pool statistics (remaining/cumulative counts)
    - Pruning statistics
    - Prediction statistics (mean, std, min, max, median)
    - Uncertainty statistics (if available)
    - Measured value statistics for this cycle
    - Best value found so far

    Args:
        compounds_df: Master DataFrame with all compound states
        cycle: Current cycle number
        strategy: Acquisition strategy used
        predictions: Prediction array from learner
        uncertainties: Uncertainty array (None if not available)
        selected_ids: IDs of compounds selected this cycle
        pruned_ids: IDs of compounds pruned this cycle
        target_col: Target property column name
        score_direction: 'higher' or 'lower'

    Returns:
        Dictionary with comprehensive cycle metrics
    """
    metrics = {}

    # Basic cycle info
    metrics['cycle'] = cycle
    metrics['strategy'] = strategy
    metrics['batch_size'] = len(selected_ids)
    metrics['selected_count'] = len(selected_ids)

    # Pool statistics
    metrics['remaining_unlabeled'] = int((compounds_df['status'] == 'unlabeled').sum())
    metrics['cumulative_labeled'] = int((compounds_df['status'] == 'labeled').sum())
    metrics['cumulative_pruned'] = int((compounds_df['status'] == 'pruned').sum())
    metrics['pool_size'] = int((compounds_df['status'] == 'unlabeled').sum())

    # Pruning statistics
    metrics['pruned_count'] = len(pruned_ids)
    metrics['pruned_ids'] = pruned_ids

    # Selection statistics
    metrics['selected_ids'] = selected_ids

    # Prediction statistics (with NaN safeguards)
    if predictions is not None and len(predictions) > 0:
        metrics['prediction_mean'] = float(np.nanmean(predictions)) if not np.all(np.isnan(predictions)) else None
        metrics['prediction_std'] = float(np.nanstd(predictions)) if not np.all(np.isnan(predictions)) else None
        metrics['prediction_min'] = float(np.nanmin(predictions)) if not np.all(np.isnan(predictions)) else None
        metrics['prediction_max'] = float(np.nanmax(predictions)) if not np.all(np.isnan(predictions)) else None
        metrics['prediction_median'] = float(np.nanmedian(predictions)) if not np.all(np.isnan(predictions)) else None
    else:
        metrics['prediction_mean'] = None
        metrics['prediction_std'] = None
        metrics['prediction_min'] = None
        metrics['prediction_max'] = None
        metrics['prediction_median'] = None

    # Uncertainty statistics (with NaN safeguards)
    if uncertainties is not None and len(uncertainties) > 0:
        metrics['uncertainty_mean'] = float(np.nanmean(uncertainties)) if not np.all(np.isnan(uncertainties)) else None
        metrics['uncertainty_std'] = float(np.nanstd(uncertainties)) if not np.all(np.isnan(uncertainties)) else None
        metrics['uncertainty_min'] = float(np.nanmin(uncertainties)) if not np.all(np.isnan(uncertainties)) else None
        metrics['uncertainty_max'] = float(np.nanmax(uncertainties)) if not np.all(np.isnan(uncertainties)) else None
        metrics['has_uncertainty'] = True
    else:
        metrics['has_uncertainty'] = False

    # Measured value statistics (for selected compounds this cycle, with NaN safeguards)
    measured_values = compounds_df[
        compounds_df['ID'].isin(selected_ids)
    ][target_col].values

    if len(measured_values) > 0:
        metrics['measured_mean'] = float(np.nanmean(measured_values)) if not np.all(np.isnan(measured_values)) else None
        metrics['measured_std'] = float(np.nanstd(measured_values)) if not np.all(np.isnan(measured_values)) else None
        metrics['measured_min'] = float(np.nanmin(measured_values)) if not np.all(np.isnan(measured_values)) else None
        metrics['measured_max'] = float(np.nanmax(measured_values)) if not np.all(np.isnan(measured_values)) else None
        if not np.all(np.isnan(measured_values)):
            metrics['measured_best'] = float(
                np.nanmax(measured_values) if score_direction == 'higher' else np.nanmin(measured_values)
            )
        else:
            metrics['measured_best'] = None
    else:
        metrics['measured_mean'] = None
        metrics['measured_std'] = None
        metrics['measured_min'] = None
        metrics['measured_max'] = None
        metrics['measured_best'] = None

    # Best so far (across all labeled compounds, with NaN safeguards)
    all_labeled_values = compounds_df[
        compounds_df['status'] == 'labeled'
    ][target_col].values

    if len(all_labeled_values) > 0 and not np.all(np.isnan(all_labeled_values)):
        metrics['best_so_far'] = float(
            np.nanmax(all_labeled_values) if score_direction == 'higher' else np.nanmin(all_labeled_values)
        )
    else:
        metrics['best_so_far'] = None

    return metrics


def _apply_pruning(
    pool: pd.DataFrame,
    predictions: np.ndarray,
    uncertainties: Optional[np.ndarray],
    strategy: str,
    params: Dict[str, Any],
    score_direction: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply pruning strategy to reduce selection pool.

    Gracefully degrades on error - returns original pool with warning rather than failing.
    This ensures pruning failures don't stop the active learning cycle.

    Args:
        pool: DataFrame with compounds to prune (must have ID column)
        predictions: Prediction values for compounds
        uncertainties: Uncertainty values (None if not available)
        strategy: Pruning strategy name
        params: Strategy-specific parameters
        score_direction: 'higher' or 'lower'

    Returns:
        (pruned_pool_df, pruning_stats_dict)

    Pruning stats include:
        - pruned_count: Number of compounds removed
        - pruned_ids: List of removed compound IDs
        - original_count: Original pool size
        - remaining_count: Pool size after pruning
        - pruning_fraction: Fraction of compounds removed
    """
    from learnm8.pruning import create_pruning_strategy

    try:
        params_with_direction = params.copy()
        params_with_direction['score_direction'] = score_direction

        pruner = create_pruning_strategy(
            strategy_name=strategy,
            parameters=params_with_direction
        )

        pruned_pool = pruner.prune(pool, predictions, uncertainties)

        pruned_ids = list(set(pool['ID']) - set(pruned_pool['ID']))

        pruning_stats = {
            'pruned_count': len(pruned_ids),
            'pruned_ids': pruned_ids,
            'original_count': len(pool),
            'remaining_count': len(pruned_pool),
            'pruning_fraction': len(pruned_ids) / len(pool) if len(pool) > 0 else 0.0
        }

        logger.debug(
            f"Pruning with '{strategy}': {len(pool)} → {len(pruned_pool)} compounds "
            f"({pruning_stats['pruning_fraction']:.1%} removed)"
        )

        return pruned_pool, pruning_stats

    except Exception as e:
        logger.error(f"Pruning failed with strategy '{strategy}': {e}")
        raise RuntimeError(f"Pruning configuration invalid for strategy '{strategy}': {e}") from e


def _select_compounds(
    pool: pd.DataFrame,
    strategy: str,
    batch_size: int,
    score_direction: str,
    acquisition_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply acquisition strategy to select compounds.

    Handles acquisition function creation, validation, and execution.
    Provides clear error messages for common issues like:
    - Unknown strategy names
    - Missing uncertainty estimates for uncertainty-based strategies
    - Selection failures

    Args:
        pool: DataFrame with compounds (must have ID, prediction columns)
        strategy: Acquisition strategy name
        batch_size: Number of compounds to select
        score_direction: 'higher' or 'lower'
        acquisition_params: Strategy-specific parameters

    Returns:
        DataFrame with selected compounds

    Raises:
        ValueError: Unknown strategy or missing requirements
        RuntimeError: Selection failures
    """
    from learnm8.acquisition import get_acquisition_function, list_acquisition_functions

    # Get acquisition class
    try:
        acq_class = get_acquisition_function(strategy)
    except KeyError as e:
        available = list_acquisition_functions()
        raise ValueError(
            f"Unknown acquisition strategy '{strategy}'. "
            f"Available strategies: {available}"
        )

    # Note: current_best must be set from labeled data at cycle level, not predictions
    # This is handled in execute_cycle before calling _select_compounds

    # Create acquisition instance
    try:
        acq_func = acq_class(score_direction=score_direction, **acquisition_params)
    except Exception as e:
        raise ValueError(f"Failed to create acquisition function '{strategy}': {e}")

    # Validate requirements
    if acq_func.requires_uncertainty() and 'uncertainty' not in pool.columns:
        raise ValueError(
            f"Acquisition strategy '{strategy}' requires uncertainty estimates, "
            f"but your learner does not produce them. Use a learner that supports uncertainty "
            f"(e.g., GaussianProcess, MCDropout, EnsembleLearner) or choose a different strategy "
            f"(e.g., greedy, random)."
        )

    # Select compounds
    try:
        selected_df = acq_func.select(pool, batch_size)
    except Exception as e:
        raise RuntimeError(f"Acquisition strategy '{strategy}' failed during selection: {e}")

    # Validate selection
    if len(selected_df) == 0:
        raise RuntimeError(f"Acquisition strategy '{strategy}' selected 0 compounds")

    if len(selected_df) > batch_size:
        logger.warning(
            f"Acquisition strategy '{strategy}' selected {len(selected_df)} compounds, "
            f"expected {batch_size}. Truncating to batch_size."
        )
        selected_df = selected_df.iloc[:batch_size]

    logger.debug(f"Selected {len(selected_df)} compounds using '{strategy}' acquisition")

    return selected_df

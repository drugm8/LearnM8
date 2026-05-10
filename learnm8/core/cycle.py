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
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from learnm8.core.batching import predict_with_batching
from learnm8.core.config import CycleConfig
from learnm8.core.interfaces import Learner, Oracle
from learnm8.evaluation import RunCache, evaluate_cycle
from learnm8.exceptions import (
    AcquisitionError,
    ConfigurationError,
    FeatureExtractionError,
    LearnerError,
    OracleError,
    PruningError,
)
from learnm8.features.extraction import extract_features

logger = logging.getLogger(__name__)


def execute_cycle(
    compounds_df: pl.DataFrame,
    cycle: int,
    config: CycleConfig,
    learner: Learner,
    oracle: Oracle,
    target_col: str,
    featurizer: str | None,
    cache_dir: Path,
    original_pool_size: int,
    score_direction: str = 'higher',
    mode: Literal['run', 'benchmark'] = 'run',
    original_pool: pl.DataFrame | None = None,
    cumulative_selected_ids: set | None = None,
    memory_safety_factor: float = 0.7,
    previous_metrics: dict[str, Any] | None = None,
    n_jobs: int = -1,
    output_dir: Path | None = None,
    run_cache: RunCache | None = None,
    featurizer_obj: Any = None,
    disable_molecular_similarity: bool | Iterable[str] = False,
    random_state: int | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
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
        featurizer: Type of molecular features (morgan, maccs, ecfp6, descriptors)
        cache_dir: Directory for feature caching
        original_pool_size: Size of original compound pool (for batch calculation)
        score_direction: 'higher' or 'lower' for optimization direction
        mode: 'run' or 'benchmark' execution mode
        original_pool: Required for benchmark mode - full original compound pool
        cumulative_selected_ids: Optional set of previously selected compound IDs
        memory_safety_factor: Fraction of available memory to use for prediction
            batching (0.0, 1.0]. Default 0.7. Higher values use more memory.
        previous_metrics: Optional metrics from previous cycle for change indicators

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
        update_status,
    )
    from learnm8.core.persistence import write_cycle_predictions

    # Step 1: Setup and Validation
    if mode not in ['run', 'benchmark']:
        raise ValueError(
            f"mode must be 'run' or 'benchmark', got '{mode}'. "
            f"Mode is auto-detected from oracle type: CSV oracle → 'benchmark', "
            f"Python oracle → 'run'. Override with mode='run' or mode='benchmark'."
        )

    if mode == 'benchmark' and original_pool is None:
        raise ValueError(
            "original_pool is required for benchmark mode because discovery and "
            "ranking metrics need ground truth. Provide original_pool or use mode='run'."
        )

    if score_direction not in ['higher', 'lower']:
        raise ValueError(
            f"score_direction must be 'higher' or 'lower', got '{score_direction}'. "
            f"Use 'higher' when larger target values are better (e.g., activity), "
            f"or 'lower' when smaller values are better (e.g., IC50)."
        )

    # Extract cumulative selected IDs if not provided
    if cumulative_selected_ids is None:
        cumulative_selected_ids = set(compounds_df.filter(pl.col('status') == 'labeled')['ID'].to_list())

    # Start overall cycle timing
    cycle_start_time = time.time()

    # Step 2 & 3: Training
    # Get Labeled Compounds for Training
    labeled_df = get_compounds_by_status(
        compounds_df, 'labeled', columns=['ID', 'SMILES', target_col]
    )

    unlabeled_df = get_compounds_by_status(
        compounds_df, 'unlabeled', columns=['ID', 'SMILES']
    )

    logger.debug(f"Cycle {cycle}: Pool composition - {len(labeled_df)} labeled, {len(unlabeled_df)} unlabeled")
    logger.debug(f"Cycle {cycle} configuration: strategy={config.strategy}, batch_fraction={config.batch_fraction}, pruning={config.pruning_strategy or 'disabled'}")

    if len(labeled_df) == 0:
        raise LearnerError(
            f"No labeled compounds available for training in cycle {cycle}. "
            f"This typically means initialization (cycle 0) failed to label any compounds. "
            f"Check that select_initial_batch() ran successfully, that your oracle returned "
            f"measurements, and that batch_fraction is large enough to select at least 1 compound."
        )
    else:
        # Propagate feature_type from featurizer to learner
        if featurizer is not None:
            from learnm8.features import FEATURIZER_REGISTRY
            if featurizer in FEATURIZER_REGISTRY:
                featurizer_instance = FEATURIZER_REGISTRY[featurizer](n_jobs=1)
                learner._feature_type = featurizer_instance.feature_type
            else:
                learner._feature_type = 'binary'
        else:
            learner._feature_type = 'binary'

        # Extract features and train learner.
        # Feature-extraction time is tracked separately from learner-side
        # training time so that `feature_extraction_time` (HDF5 cache speedup)
        # can be reported independently in cycle metrics.
        training_start_time = time.time()
        train_feature_time = 0.0
        try:
            training_targets = labeled_df[target_col].to_numpy()
            training_smiles = labeled_df['SMILES'].to_list()

            if learner.requires_smiles():
                if featurizer is not None:
                    _t0 = time.time()
                    training_features = extract_features(
                        training_smiles,
                        featurizer,
                        cache_dir=cache_dir,
                        n_jobs=n_jobs,
                        preferred_dtype=learner.preferred_feature_dtype(),
                    )
                    train_feature_time += time.time() - _t0
                    logger.info(
                        f"Training {learner.get_name()} on {len(labeled_df)} compounds "
                        f"(SMILES + {training_features.shape[1]}-D descriptors)"
                    )
                else:
                    training_features = None
                    logger.info(
                        f"Training {learner.get_name()} on {len(labeled_df)} compounds"
                    )

                learner.train(
                    features=training_features,
                    targets=training_targets,
                    smiles=training_smiles
                )
                logger.debug(f"Model trained on {len(training_smiles)} compounds")
            else:
                if featurizer is None:
                    raise ValueError(
                        f"featurizer is required for {learner.get_name()} because it is a "
                        f"feature-based learner. Specify a featurizer (e.g., featurizer='morgan'). "
                        f"SMILES-native learners like 'chemprop' and 'fastprop' can run without "
                        f"a featurizer (featurizer=None)."
                    )

                _t0 = time.time()
                training_features = extract_features(
                    training_smiles,
                    featurizer,
                    cache_dir=cache_dir,
                    n_jobs=n_jobs,
                    preferred_dtype=learner.preferred_feature_dtype(),
                )
                train_feature_time += time.time() - _t0
                logger.debug(f"Extracted {featurizer} features: {len(labeled_df)} training compounds")

                logger.info(f"Training {learner.get_name()} on {len(labeled_df)} compounds ({featurizer})")
                learner.train(training_features, training_targets)
                logger.debug(f"Model trained on features shape: {training_features.shape}, targets shape: {training_targets.shape}")
        except (LearnerError, FeatureExtractionError):
            raise
        except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as e:
            logger.error(f"Training failed in cycle {cycle}: {e}")
            err = LearnerError(
                f"Training failed in cycle {cycle}: {e}. "
                f"Check that the training data is valid, the featurizer is compatible "
                f"with the learner, and there are enough labeled compounds."
            )
            err.add_note(f"Failed during cycle {cycle} with {len(labeled_df)} labeled compounds")
            raise err from e

        # Carve feature-extraction time out of training_time so the two metrics
        # are mutually exclusive (required for stacked-bar compute breakdowns).
        training_time = max(0.0, (time.time() - training_start_time) - train_feature_time)
        logger.info(f"Training complete ({training_time:.2f}s)")

    # Step 4: Prediction on unlabeled compounds (unified always-batch approach)
    prediction_start_time = time.time()
    prediction_pool = get_compounds_by_status(
        compounds_df, 'unlabeled', columns=['ID', 'SMILES']
    )

    if len(prediction_pool) == 0:
        logger.warning(f"No unlabeled compounds available for prediction in cycle {cycle}. Returning unchanged DataFrame.")
        empty_predictions = pl.DataFrame(
            schema={'ID': pl.Utf8, 'prediction': pl.Float64, 'uncertainty': pl.Float64}
        )
        metrics = {
            'cycle': cycle,
            'strategy': config.strategy,
            'selected_count': 0,
            'remaining_pool': 0,
            'remaining_unlabeled': 0,
            'cumulative_labeled': int(compounds_df.filter(pl.col('status') == 'labeled').height),
            'cumulative_pruned': 0,
            'pruned_count': 0,
            'parquet_path': None,
            'selected_predictions': empty_predictions,
        }
        return compounds_df, metrics

    try:
        (
            predictions,
            uncertainties,
            valid_compound_ids,
            predict_feature_time,
        ) = predict_with_batching(
            prediction_pool,
            learner,
            featurizer,
            cache_dir,
            memory_safety_factor=memory_safety_factor,
            n_jobs=n_jobs,
        )
    except (LearnerError, FeatureExtractionError):
        raise
    except (ValueError, RuntimeError, TypeError) as e:
        logger.error(f"Prediction failed in cycle {cycle}: {e}")
        err = LearnerError(
            f"Prediction failed in cycle {cycle}: {e}. "
            f"Check featurizer compatibility with your learner and that training "
            f"completed successfully."
        )
        err.add_note(f"Failed during cycle {cycle}")
        raise err from e

    # Carve feature-extraction time out of prediction_time so the two metrics
    # are mutually exclusive (required for stacked-bar compute breakdowns).
    prediction_time = max(0.0, (time.time() - prediction_start_time) - predict_feature_time)
    logger.info(f"Prediction complete: {len(predictions)} predictions (min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}) in {prediction_time:.2f}s")

    if len(predictions) == 0:
        raise LearnerError(
            f"Prediction returned 0 results in cycle {cycle}. "
            f"This may indicate a featurizer incompatibility, an untrained model, "
            f"or that all unlabeled compounds were filtered out. Check that the "
            f"featurizer produces valid features for your compound pool."
        )

    pred_data: dict[str, Any] = {
        'ID': list(valid_compound_ids),
        'prediction': predictions,
    }
    if uncertainties is not None:
        pred_data['uncertainty'] = uncertainties
    cycle_predictions = pl.DataFrame(pred_data)

    parquet_path = write_cycle_predictions(cycle_predictions, output_dir, cycle)

    selection_pool_cols = ['ID', 'prediction'] + (
        ['uncertainty'] if uncertainties is not None else []
    )
    selection_pool = (
        compounds_df
        .filter(pl.col('status') == 'unlabeled')
        .select(['ID', 'SMILES'])
        .join(cycle_predictions, on='ID', how='inner')
        .select(['ID', 'SMILES'] + [c for c in selection_pool_cols if c != 'ID'])
    )

    if len(selection_pool) == 0:
        raise AcquisitionError(
            f"No unlabeled compounds with predictions available for selection in cycle {cycle}. "
            f"The compound pool may be exhausted. Consider reducing n_cycles or increasing "
            f"the pool size. Current pool: {len(unlabeled_df)} unlabeled, "
            f"{len(labeled_df)} labeled."
        )

    logger.debug(f"Selection pool: {len(selection_pool)} unlabeled compounds with predictions")

    # Step 8: Apply Pruning (If Specified)
    pruning_stats = {'pruned_count': 0, 'pruned_ids': []}
    unlabeled_before_prune = len(selection_pool)

    if config.pruning_strategy:
        logger.debug(f"Pruning design space with '{config.pruning_strategy}' strategy")

        uncertainty_values = None
        if 'uncertainty' in selection_pool.columns:
            uncertainty_values = selection_pool['uncertainty'].to_numpy()

        selection_pool, pruning_stats = _apply_pruning(
            selection_pool,
            selection_pool['prediction'].to_numpy(),
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
            logger.debug(f"Pruned {pruning_stats['pruned_count']} compounds ({prune_pct:.1f}% of unlabeled pool)")

        logger.debug(f"Pruning parameters: {config.pruning_params}")
        logger.debug(f"Pool before pruning: {unlabeled_before_prune}, after pruning: {len(selection_pool)}")

    if len(selection_pool) == 0:
        raise PruningError(
            f"No compounds remaining after pruning in cycle {cycle}. "
            f"Pruning removed all {unlabeled_before_prune} candidates. "
            f"Reduce pruning_fraction (current: {config.pruning_params.get('pruning_fraction', 'N/A') if config.pruning_params else 'N/A'}) "
            f"or disable pruning by setting pruning_strategy=None."
        )

    # Step 9: Calculate Batch Size from Fraction
    batch_size = min(
        max(1, math.ceil(original_pool_size * config.batch_fraction)),
        len(selection_pool)
    )

    if batch_size == 0:
        raise ConfigurationError(
            f"Calculated batch size is 0 (original_pool_size={original_pool_size}, "
            f"batch_fraction={config.batch_fraction}, product={original_pool_size * config.batch_fraction:.4f}). "
            f"Increase batch_fraction so that pool_size * batch_fraction >= 1. "
            f"For a pool of {original_pool_size} compounds, minimum batch_fraction is "
            f"{1.0 / original_pool_size:.6f}."
        )

    # Step 10: Prepare acquisition params with current_best from labeled data
    acquisition_params = (config.acquisition_params or {}).copy()
    if 'current_best' not in acquisition_params:
        labeled_values = compounds_df.filter(pl.col('status') == 'labeled')[target_col].to_numpy()
        if len(labeled_values) > 0:
            acquisition_params['current_best'] = (
                float(labeled_values.max()) if score_direction == 'higher'
                else float(labeled_values.min())
            )
        else:
            logger.warning(
                "No labeled data available for current_best calculation. "
                "EI/PI strategies may fail if selected."
            )

    # Step 11: Select Compounds Using Acquisition Strategy
    acquisition_start_time = time.time()
    logger.debug(f"Acquiring compounds with '{config.strategy}' strategy from {len(selection_pool)} candidates")

    selected_df = _select_compounds(
        selection_pool,
        config.strategy,
        batch_size,
        score_direction,
        acquisition_params
    )

    selected_ids = selected_df['ID'].to_list()
    acquisition_time = time.time() - acquisition_start_time

    if len(selected_ids) == 0:
        raise AcquisitionError(
            f"Acquisition strategy '{config.strategy}' selected 0 compounds in cycle {cycle}. "
            f"Pool had {len(selection_pool)} candidates with batch_size={batch_size}. "
            f"Check acquisition strategy parameters and remaining pool size."
        )

    logger.debug(f"Selected {len(selected_ids)}/{len(selection_pool)} compounds using {config.strategy.upper()} (batch_size={batch_size})")

    # Capture selected predictions at cycle time so save_results() can build
    # selection_history.csv without re-reading the parquet file at save time.
    selected_predictions = cycle_predictions.filter(pl.col('ID').is_in(selected_ids))

    # Step 12: Measure Selected Compounds
    oracle_start_time = time.time()
    # Create temporary mapping to preserve selected_ids order
    id_to_order = {id_val: idx for idx, id_val in enumerate(selected_ids)}

    selected_compounds = compounds_df.filter(
        pl.col('ID').is_in(selected_ids)
    ).select(['ID', 'SMILES']).with_columns(
        pl.col('ID').map_elements(lambda x: id_to_order.get(x, 999), return_dtype=pl.Int64).alias('_order')
    ).sort('_order').drop('_order')

    try:
        measurements = oracle.measure(selected_compounds, [target_col])
    except OracleError:
        raise
    except (ValueError, RuntimeError, TypeError, OSError) as e:
        logger.error(f"Oracle measurement failed in cycle {cycle}: {e}")
        err = OracleError(
            f"Oracle measurement failed in cycle {cycle}: {e}. "
            f"Check that your oracle function handles the requested compound IDs "
            f"and returns valid measurements for target column '{target_col}'."
        )
        err.add_note(f"Failed during cycle {cycle} while measuring {len(selected_ids)} compounds")
        raise err from e

    oracle_time = time.time() - oracle_start_time

    measurement_ids = measurements['ID'].to_list()
    if not all(sid in measurement_ids for sid in selected_ids):
        missing = set(selected_ids) - set(measurement_ids)
        from learnm8.exceptions import _truncate_list
        raise OracleError(
            f"Oracle did not return measurements for {len(missing)} of "
            f"{len(selected_ids)} selected compounds in cycle {cycle}. "
            f"Missing IDs: {_truncate_list(missing)}. "
            f"Check that your oracle contains data for all compound IDs in the pool."
        )

    logger.debug(f"Measured {len(measurements)} compounds")

    # Step 13: Update Master DataFrame with Measurements
    # Create Polars Series with target values indexed by ID
    target_series = measurements.select([target_col]).to_series()

    compounds_df = update_status(
        compounds_df,
        selected_ids,
        'labeled',
        cycle,
        target_col,
        target_series
    )

    # Step 14: Calculate Cycle Metrics
    evaluation_start_time = time.time()
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
    # Predictions are joined from cycle_predictions (temp DF) rather than read
    # from prediction_cycle_N columns, which no longer exist on the master DF.
    try:
        labeled_with_pred = (
            compounds_df.filter(pl.col('status') == 'labeled')
            .select(['ID', 'SMILES', target_col])
            .join(
                cycle_predictions.select(['ID', 'prediction']),
                on='ID',
                how='inner',
            )
        )

        if len(labeled_with_pred) > 0:
            eval_predictions = labeled_with_pred['prediction'].to_numpy()
            eval_ground_truth = labeled_with_pred[target_col].to_numpy()

            valid_mask = ~(np.isnan(eval_predictions) | np.isnan(eval_ground_truth))

            if np.sum(valid_mask) > 0:
                selected_for_eval = compounds_df.filter(
                    pl.col('ID').is_in(selected_ids)
                ).select(['ID', 'SMILES', target_col])

                pool_df_for_eval = None
                original_pool_for_eval = None

                if mode == 'benchmark' and original_pool is not None:
                    pool_df_for_eval = cycle_predictions.select(['ID', 'prediction'])
                    original_pool_for_eval = original_pool

                cumulative_selected_compounds = compounds_df.filter(
                    pl.col('status') == 'labeled'
                ).select(['ID', 'SMILES'])

                eval_metrics = evaluate_cycle(
                    cycle=cycle,
                    predictions=eval_predictions[valid_mask],
                    ground_truth=eval_ground_truth[valid_mask],
                    labeled_data=labeled_with_pred.select(['ID', 'SMILES', target_col]),
                    selected_compounds=selected_for_eval,
                    target_col=target_col,
                    oracle_type=mode,
                    ground_truth_data=original_pool_for_eval,
                    pool_df=pool_df_for_eval,
                    uncertainties=uncertainties if uncertainties is not None else None,
                    cumulative_selected_compounds=cumulative_selected_compounds,
                    advanced_metrics=False,
                    disable_molecular_similarity=disable_molecular_similarity,
                    score_direction=score_direction,
                    cumulative_selected_ids=cumulative_selected_ids,
                    cumulative_labeled_count=int(compounds_df.filter(pl.col('status') == 'labeled').height),
                    run_cache=run_cache,
                    featurizer=featurizer_obj,
                    cache_dir=cache_dir,
                    random_state=random_state,
                )

                metrics.update(eval_metrics)
    except (ValueError, RuntimeError, TypeError, ArithmeticError, KeyError, pl.exceptions.ColumnNotFoundError) as e:
        logger.warning(f"Failed to calculate enhanced evaluation metrics in cycle {cycle}: {e}")

    evaluation_time = time.time() - evaluation_start_time

    # Calculate total cycle time
    total_time = time.time() - cycle_start_time

    # Add timing metrics. `feature_extraction_time` is the wall-clock seconds
    # spent in `extract_features` across both training-side and prediction-side
    # calls; it is carved out of `training_time` and `prediction_time` so the
    # four compute-component metrics can be summed for stacked-bar breakdowns
    # (publication Fig. 5 panel C, HDF5 cache speedup).
    feature_extraction_time = train_feature_time + predict_feature_time
    metrics['training_time'] = training_time
    metrics['prediction_time'] = prediction_time
    metrics['acquisition_time'] = acquisition_time
    metrics['oracle_time'] = oracle_time
    metrics['evaluation_time'] = evaluation_time
    metrics['feature_extraction_time'] = feature_extraction_time
    metrics['total_time'] = total_time

    # Parquet path and cycle-time prediction capture for selected compounds.
    # selected_predictions is a tiny DF (~batch_size rows) used by save_results()
    # to build selection_history.csv without re-reading the parquet file.
    metrics['parquet_path'] = parquet_path
    metrics['selected_predictions'] = selected_predictions

    # Step 15: Log Cycle Metrics
    parts = []
    for key, val in metrics.items():
        if key in ('parquet_path', 'selected_predictions'):
            continue
        if isinstance(val, float):
            parts.append(f"{key}={val:.4f}")
        else:
            parts.append(f"{key}={val}")
    logger.info(f"Cycle {cycle} metrics: " + ", ".join(parts))

    # Step 16: Return Results
    total_labeled = compounds_df.filter(pl.col('status') == 'labeled').height
    total_unlabeled = compounds_df.filter(pl.col('status') == 'unlabeled').height
    logger.info(f"Cycle {cycle} complete: {total_labeled} labeled, {total_unlabeled} unlabeled remaining")

    logger.debug(f"Oracle measured {len(selected_ids)} compounds for property: {target_col}")

    return compounds_df, metrics


def _calculate_cycle_metrics(
    compounds_df: pl.DataFrame,
    cycle: int,
    strategy: str,
    predictions: np.ndarray,
    uncertainties: np.ndarray | None,
    selected_ids: list[str],
    pruned_ids: list[str],
    target_col: str,
    score_direction: str
) -> dict[str, Any]:
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
    metrics['remaining_unlabeled'] = int(compounds_df.filter(pl.col('status') == 'unlabeled').height)
    metrics['cumulative_labeled'] = int(compounds_df.filter(pl.col('status') == 'labeled').height)
    metrics['cumulative_pruned'] = int(compounds_df.filter(pl.col('status') == 'pruned').height)
    metrics['pool_size'] = int(compounds_df.filter(pl.col('status') == 'unlabeled').height)

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
    measured_values = compounds_df.filter(
        pl.col('ID').is_in(selected_ids)
    )[target_col].to_numpy()

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
    all_labeled_values = compounds_df.filter(
        pl.col('status') == 'labeled'
    )[target_col].to_numpy()

    if len(all_labeled_values) > 0 and not np.all(np.isnan(all_labeled_values)):
        metrics['best_so_far'] = float(
            np.nanmax(all_labeled_values) if score_direction == 'higher' else np.nanmin(all_labeled_values)
        )
    else:
        metrics['best_so_far'] = None

    return metrics


def _apply_pruning(
    pool: pl.DataFrame,
    predictions: np.ndarray,
    uncertainties: np.ndarray | None,
    strategy: str,
    params: dict[str, Any],
    score_direction: str
) -> tuple[pl.DataFrame, dict[str, Any]]:
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

    except PruningError:
        raise
    except (ValueError, RuntimeError, TypeError) as e:
        logger.error(f"Pruning failed with strategy '{strategy}': {e}")
        err = PruningError(
            f"Pruning failed with strategy '{strategy}': {e}. "
            f"Check pruning parameters (e.g., pruning_fraction must be 0.0-0.9). "
            f"To disable pruning, set pruning_strategy=None."
        )
        err.add_note(f"Pruning failed on pool of {len(pool)} compounds with params: {params}")
        raise err from e


def _select_compounds(
    pool: pl.DataFrame,
    strategy: str,
    batch_size: int,
    score_direction: str,
    acquisition_params: dict[str, Any]
) -> pl.DataFrame:
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
    except KeyError:
        available = list_acquisition_functions()
        raise ValueError(
            f"Unknown acquisition strategy '{strategy}'. "
            f"Available strategies: {', '.join(available)}. "
            f"Basic strategies (any learner): greedy, random, topk. "
            f"Uncertainty-based (requires supports_uncertainty=True): ucb, ei, pi, thompson, entropy."
        ) from None

    # Note: current_best must be set from labeled data at cycle level, not predictions
    # This is handled in execute_cycle before calling _select_compounds

    # Create acquisition instance
    try:
        acq_func = acq_class(score_direction=score_direction, **acquisition_params)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to create acquisition function '{strategy}': {e}. "
            f"Check that the acquisition parameters are valid for this strategy."
        ) from e
    # Uniqueness of IDs in selection_pool is guaranteed by validate_compound_pool() upstream
    acq_func._skip_unique_id_check = True

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
    except AcquisitionError:
        raise
    except (ValueError, RuntimeError, TypeError) as e:
        raise AcquisitionError(
            f"Acquisition strategy '{strategy}' failed during selection: {e}. "
            f"Pool had {len(pool)} candidates. Check strategy parameters and "
            f"ensure predictions/uncertainties are valid."
        ) from e

    if len(selected_df) == 0:
        raise AcquisitionError(
            f"Acquisition strategy '{strategy}' selected 0 compounds from "
            f"{len(pool)} candidates (batch_size={batch_size}). "
            f"Check remaining pool size and strategy parameters."
        )

    if len(selected_df) > batch_size:
        logger.warning(
            f"Acquisition strategy '{strategy}' selected {len(selected_df)} compounds, "
            f"expected {batch_size}. Truncating to batch_size."
        )
        selected_df = selected_df.head(batch_size)

    logger.debug(f"Selected {len(selected_df)} compounds using '{strategy}' acquisition")

    return selected_df

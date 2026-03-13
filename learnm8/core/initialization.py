"""Initialization functions for active learning experiments.

Provides clean initialization with all compounds starting unlabeled,
followed by a separate initialization phase (cycle 0) that selects the
initial batch before active learning cycles begin.
"""

import polars as pl
import pandas as pd
import numpy as np
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Set
from .data_structures import STATUS_UNLABELED, VALID_STATUSES
from .interfaces import Oracle
from learnm8.exceptions import AcquisitionError, OracleError

logger = logging.getLogger(__name__)


def initialize_master_dataframe_empty(
    valid_compounds: 'pl.DataFrame | pd.DataFrame',
    target_col: str
) -> pl.DataFrame:
    """Create master DataFrame with all compounds unlabeled.

    All compounds start unlabeled. The first cycle (cycle 0) will select
    and measure the initial batch as part of normal cycle execution.

    Args:
        valid_compounds: Polars or Pandas DataFrame with 'ID' and 'SMILES' columns (already validated)
        target_col: Name of target column for measurements

    Returns:
        Master DataFrame with columns: ID, SMILES, status, labeled_cycle,
        selected_cycle, pruned_cycle, target_col

        All compounds have status='unlabeled', all tracking columns are empty.

    Example:
        >>> master_df = initialize_master_dataframe_empty(
        ...     valid_compounds=validated_df,
        ...     target_col='Activity'
        ... )
        >>> # All compounds unlabeled, cycle 0 will select initial batch
    """
    # Convert pandas DataFrame to Polars for backward compatibility
    if isinstance(valid_compounds, pd.DataFrame):
        valid_compounds = pl.from_pandas(valid_compounds)

    if 'ID' not in valid_compounds.columns or 'SMILES' not in valid_compounds.columns:
        missing = [c for c in ['ID', 'SMILES'] if c not in valid_compounds.columns]
        raise ValueError(
            f"valid_compounds is missing required columns: {missing}. "
            f"Available columns: {list(valid_compounds.columns)}. "
            f"Ensure your compound pool CSV has 'ID' and 'SMILES' columns."
        )

    master_df = valid_compounds.select(['ID', 'SMILES']).clone()

    # All compounds start unlabeled
    master_df = master_df.with_columns([
        pl.lit(STATUS_UNLABELED).cast(pl.Enum(VALID_STATUSES)).alias('status')
    ])

    # Initialize empty tracking columns
    master_df = master_df.with_columns([
        pl.lit(None).cast(pl.Int64).alias('labeled_cycle'),
        pl.lit(None).cast(pl.Int64).alias('selected_cycle'),
        pl.lit(None).cast(pl.Int64).alias('pruned_cycle'),
        pl.lit(None).cast(pl.Float64).alias(target_col)
    ])

    logger.info(
        f"Initialized master DataFrame: {len(master_df)} compounds (all unlabeled)"
    )

    return master_df


def calculate_initialization_metrics(
    compounds_df: pl.DataFrame,
    selected_ids: List[str],
    target_col: str,
    strategy: str,
    score_direction: str,
    mode: str,
    original_pool: Optional[pl.DataFrame] = None,
    cumulative_selected_ids: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """Calculate metrics for initialization phase (cycle 0).

    Generates metrics similar to execute_cycle() but simplified for initialization:
    - No predictions/uncertainties (no model trained yet)
    - No pruning (always 0)
    - Focus on pool state and measured values
    - Includes discovery metrics if in benchmark mode

    Args:
        compounds_df: Master DataFrame after initialization
        selected_ids: IDs of compounds selected in initialization
        target_col: Target property column name
        strategy: Acquisition strategy used (typically 'random')
        score_direction: 'higher' or 'lower' for optimization
        mode: 'run' or 'benchmark' execution mode
        original_pool: Full original pool (required for benchmark mode)
        cumulative_selected_ids: Set of all selected IDs (for discovery metrics)

    Returns:
        Dictionary with cycle 0 metrics matching structure of execute_cycle() metrics
    """
    metrics = {}

    # Basic cycle info
    metrics['cycle'] = 0
    metrics['strategy'] = strategy
    metrics['batch_size'] = len(selected_ids)
    metrics['selected_count'] = len(selected_ids)

    # Pool statistics
    metrics['remaining_unlabeled'] = int((compounds_df['status'] == 'unlabeled').sum())
    metrics['cumulative_labeled'] = int((compounds_df['status'] == 'labeled').sum())
    metrics['cumulative_pruned'] = 0
    metrics['pool_size'] = metrics['remaining_unlabeled']

    # Pruning statistics (none in initialization)
    metrics['pruned_count'] = 0
    metrics['pruned_ids'] = []

    # Selection statistics
    metrics['selected_ids'] = selected_ids

    # Prediction statistics (None - no model yet)
    metrics['prediction_mean'] = None
    metrics['prediction_std'] = None
    metrics['prediction_min'] = None
    metrics['prediction_max'] = None
    metrics['prediction_median'] = None

    # Uncertainty statistics (None - no model yet)
    metrics['has_uncertainty'] = False

    # Measured value statistics for selected compounds
    measured_values = compounds_df.filter(
        pl.col('ID').is_in(selected_ids)
    )[target_col].to_numpy()

    if len(measured_values) > 0 and not np.all(np.isnan(measured_values)):
        metrics['measured_mean'] = float(np.nanmean(measured_values))
        metrics['measured_std'] = float(np.nanstd(measured_values))
        metrics['measured_min'] = float(np.nanmin(measured_values))
        metrics['measured_max'] = float(np.nanmax(measured_values))
        metrics['measured_best'] = float(
            np.nanmax(measured_values) if score_direction == 'higher' else np.nanmin(measured_values)
        )
        metrics['best_so_far'] = metrics['measured_best']
        metrics['avg_score_selected'] = metrics['measured_mean']
    else:
        metrics['measured_mean'] = None
        metrics['measured_std'] = None
        metrics['measured_min'] = None
        metrics['measured_max'] = None
        metrics['measured_best'] = None
        metrics['best_so_far'] = None
        metrics['avg_score_selected'] = None

    # Enhanced metrics from evaluation framework (if benchmark mode)
    if mode == 'benchmark' and original_pool is not None:
        try:
            from learnm8.evaluation import evaluate_cycle

            # For cycle 0, we have measurements but no predictions
            # Focus on discovery metrics (what was found) not ranking metrics
            labeled_for_eval = compounds_df.filter(
                pl.col('status') == 'labeled'
            ).select(['ID', 'SMILES', target_col])

            selected_for_eval = compounds_df.filter(
                pl.col('ID').is_in(selected_ids)
            ).select(['ID', 'SMILES', target_col])

            if cumulative_selected_ids is None:
                cumulative_selected_ids = set(selected_ids)

            eval_metrics = evaluate_cycle(
                cycle=0,
                predictions=None,
                ground_truth=None,
                labeled_data=labeled_for_eval,
                selected_compounds=selected_for_eval,
                target_col=target_col,
                oracle_type='benchmark',
                ground_truth_data=original_pool,
                pool_predictions=None,
                pool_ids=None,
                uncertainties=None,
                previously_selected=None,
                advanced_metrics=False,
                disable_molecular_similarity=True,
                score_direction=score_direction,
                cumulative_selected_ids=cumulative_selected_ids,
                cumulative_labeled_count=len(selected_ids)
            )

            metrics.update(eval_metrics)
            logger.debug(f"Cycle 0 enhanced metrics: Top-10 Discovery: {eval_metrics.get('top_10_discovery', 'N/A')}%")

        except (ValueError, RuntimeError, TypeError, ArithmeticError) as e:
            logger.warning(f"Failed to calculate enhanced evaluation metrics for cycle 0: {e}")

    return metrics


def select_initial_batch(
    compounds_df: pl.DataFrame,
    oracle: Oracle,
    target_col: str,
    strategy: str,
    batch_fraction: float,
    featurizer: str,
    cache_dir: Path,
    original_pool_size: int,
    random_state: int = 42,
    acquisition_params: Optional[Dict] = None,
    score_direction: str = 'higher',
    mode: str = 'run',
    original_pool: Optional[pl.DataFrame] = None
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Select and measure initial batch before active learning cycles.

    This is the initialization phase (cycle 0) - no model training/prediction needed.
    Uses acquisition strategies that don't require predictions (random, bitbirch).

    Args:
        compounds_df: Master DataFrame (all unlabeled)
        oracle: Oracle for measurements
        target_col: Target property column name
        strategy: Acquisition strategy ('random' or 'bitbirch')
        batch_fraction: Fraction of pool to select
        featurizer: Molecular featurizer ('morgan', 'ecfp6', etc.)
        cache_dir: Feature cache directory
        original_pool_size: Original pool size for batch calculation
        random_state: Random seed for reproducibility
        acquisition_params: Optional parameters for acquisition function
        score_direction: 'higher' or 'lower' for optimization (default: 'higher')
        mode: 'run' or 'benchmark' execution mode (default: 'run')
        original_pool: Full original pool (required for benchmark mode metrics)

    Returns:
        Tuple of (updated DataFrame, cycle 0 metrics dict)
        - DataFrame: With initial batch labeled (labeled_cycle=0)
        - metrics: Dictionary with cycle 0 metrics

    Raises:
        ValueError: If calculated batch size is 0
        RuntimeError: If oracle measurement fails or no unlabeled compounds

    Example:
        >>> compounds_df, metrics = select_initial_batch(
        ...     compounds_df=master_df,
        ...     oracle=csv_oracle,
        ...     target_col='Activity',
        ...     strategy='random',
        ...     batch_fraction=0.01,
        ...     featurizer='morgan',
        ...     cache_dir=Path('.cache'),
        ...     original_pool_size=10000,
        ...     score_direction='higher',
        ...     mode='benchmark',
        ...     original_pool=original_df
        ... )
    """
    from .dataframe_ops import update_status
    from learnm8.acquisition import get_acquisition_function

    batch_size = min(
        max(1, math.ceil(original_pool_size * batch_fraction)),
        len(compounds_df)
    )

    if batch_size == 0:
        raise ValueError(
            f"Calculated batch size is 0 (pool_size={original_pool_size}, "
            f"batch_fraction={batch_fraction}). "
            f"The pool is too small for the given batch_fraction. "
            f"Increase batch_fraction or use a larger compound pool."
        )

    logger.debug(f"Starting initialization phase (cycle 0)")
    logger.debug(f"Strategy: {strategy}, batch_fraction: {batch_fraction}")
    logger.debug(f"Original pool size: {original_pool_size}, calculated batch: {batch_size}")

    logger.info(f"Initialization: selecting {batch_size} compounds using '{strategy}' strategy")

    unlabeled = compounds_df.filter(
        pl.col('status') == 'unlabeled'
    ).select(['ID', 'SMILES'])

    if len(unlabeled) == 0:
        raise AcquisitionError(
            "No unlabeled compounds available for initialization (cycle 0). "
            "All compounds may have been pre-labeled or filtered out during validation. "
            "Check that your compound pool contains valid, unlabeled compounds."
        )

    if strategy != 'random':
        logger.warning(
            f"Strategy '{strategy}' not yet supported for initialization. "
            f"Using 'random' strategy instead. "
            f"(BitBIRCH and other strategies deferred to future release)"
        )
        strategy = 'random'

    acq_class = get_acquisition_function('random')
    acq_func = acq_class(random_state=random_state)

    logger.debug(f"Acquiring from {len(unlabeled)} unlabeled compounds")
    logger.debug(f"Acquisition function: {acq_class.__name__}")

    selected_df = acq_func.select(unlabeled, batch_size)

    selected_ids = selected_df['ID'].to_list()

    if len(selected_ids) == 0:
        raise AcquisitionError(
            f"Acquisition strategy '{strategy}' selected 0 compounds during initialization. "
            f"The pool had {len(unlabeled)} unlabeled compounds with batch_size={batch_size}. "
            f"This is unexpected — check the acquisition strategy implementation."
        )

    logger.info(f"Measuring {len(selected_ids)} selected compounds...")

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
        logger.error(f"Oracle measurement failed during initialization: {e}")
        raise OracleError(
            f"Oracle measurement failed during initialization (cycle 0) "
            f"for {len(selected_ids)} compounds: {e}. "
            f"Check that your oracle function/CSV contains data for the selected compound IDs."
        ) from e

    measurement_ids = measurements['ID'].to_list()
    if not all(sid in measurement_ids for sid in selected_ids):
        missing = set(selected_ids) - set(measurement_ids)
        from learnm8.exceptions import _truncate_list
        raise OracleError(
            f"Oracle did not return measurements for {len(missing)} of "
            f"{len(selected_ids)} compounds during initialization. "
            f"Missing IDs: {_truncate_list(missing)}. "
            f"Check that your oracle contains data for all compound IDs in the pool."
        )

    logger.debug(f"Oracle measured {len(selected_ids)} compounds")
    logger.debug(f"Updating DataFrame: status=labeled, labeled_cycle=0")

    # Extract target values from measurements
    target_values = measurements.get_column(target_col)

    compounds_df = update_status(
        compounds_df,
        selected_ids,
        'labeled',
        cycle=0,
        target_col=target_col,
        target_values=target_values
    )

    logger.info(
        f"Initialization (cycle 0) complete: {len(selected_ids)} compounds labeled "
        f"(strategy='{strategy}')"
    )

    # Calculate cycle 0 metrics
    cumulative_selected_ids = set(selected_ids)
    metrics = calculate_initialization_metrics(
        compounds_df=compounds_df,
        selected_ids=selected_ids,
        target_col=target_col,
        strategy=strategy,
        score_direction=score_direction,
        mode=mode,
        original_pool=original_pool,
        cumulative_selected_ids=cumulative_selected_ids
    )

    return compounds_df, metrics

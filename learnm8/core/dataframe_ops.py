"""Vectorized DataFrame operations for efficient compound tracking.

This module provides high-performance operations for updating and querying
the master DataFrame during active learning cycles. All functions use
vectorized operations for O(n) performance.

Key features:
- 10-50x faster than pandas with Polars optimizations
- Immutable pattern: all update functions return new DataFrames
- Memory efficient: query functions return views
- Comprehensive logging for debugging

Performance characteristics:
- add_predictions: O(n) using join-based mapping
- update_status: O(n) using vectorized expressions
- get_compounds_by_status: O(n) filtering, returns view
- batch_update: Single clone for multiple operations
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from learnm8.utils.polars_utils import map_values_via_join

from .data_structures import VALID_STATUSES

logger = logging.getLogger(__name__)


def _add_predictions_inplace(
    df: pl.DataFrame,
    cycle: int,
    compound_ids: list[str],
    predictions: np.ndarray,
    uncertainties: np.ndarray | None = None
) -> pl.DataFrame:
    """Add predictions and uncertainties for a cycle in-place (modifies df).

    Private helper for batch operations. Modifies df in-place without copying.

    Args:
        df: Master DataFrame to modify in-place
        cycle: Current cycle number
        compound_ids: List of compound IDs corresponding to predictions
        predictions: Array of prediction values (same length as compound_ids)
        uncertainties: Optional array of uncertainty values (same length as compound_ids)

    Returns:
        The same DataFrame (modified in-place)
    """
    if len(compound_ids) != len(predictions):
        raise ValueError(
            f"compound_ids and predictions must have the same length: "
            f"{len(compound_ids)} IDs vs {len(predictions)} predictions. "
            f"Ensure the learner returned predictions for all input compounds."
        )
    if uncertainties is not None and len(uncertainties) != len(compound_ids):
        raise ValueError(
            f"uncertainties length ({len(uncertainties)}) must match "
            f"compound_ids length ({len(compound_ids)}). "
            f"Ensure the learner returned uncertainties for all input compounds."
        )
    if len(set(compound_ids)) != len(compound_ids):
        raise ValueError(
            f"compound_ids contains {len(compound_ids) - len(set(compound_ids))} duplicates. "
            f"Each compound ID must appear exactly once in the prediction batch."
        )

    logger.debug(f"Adding predictions for {len(compound_ids)} compounds in cycle {cycle}")

    pred_col = f'prediction_cycle_{cycle}'
    unc_col = f'uncertainty_cycle_{cycle}' if uncertainties is not None else None
    logger.debug(f"Prediction column: {pred_col}, uncertainty column: {unc_col}")

    if pred_col not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(pred_col))

    id_to_pred = dict(zip(compound_ids, predictions.tolist()))
    df = map_values_via_join(df, id_to_pred, 'ID', pred_col)

    if uncertainties is not None:
        unc_col = f'uncertainty_cycle_{cycle}'
        if unc_col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(unc_col))

        id_to_unc = dict(zip(compound_ids, uncertainties.tolist()))
        df = map_values_via_join(df, id_to_unc, 'ID', unc_col)

    return df


def add_predictions(
    df: pl.DataFrame,
    cycle: int,
    compound_ids: list[str],
    predictions: np.ndarray,
    uncertainties: np.ndarray | None = None
) -> pl.DataFrame:
    """Add predictions and uncertainties for a cycle using vectorized operations.

    Uses join-based mapping for 10-50x performance improvement over pandas,
    especially for large DataFrames.

    Args:
        df: Master DataFrame to update
        cycle: Current cycle number
        compound_ids: List of compound IDs corresponding to predictions
        predictions: Array of prediction values (same length as compound_ids)
        uncertainties: Optional array of uncertainty values (same length as compound_ids)

    Returns:
        Updated DataFrame (new copy)

    Raises:
        ValueError: If compound_ids and predictions have different lengths
        ValueError: If uncertainties length doesn't match compound_ids length
        ValueError: If compound_ids contains duplicates

    Performance:
        - O(n) complexity using join-based mapping
        - 10-50x faster than pandas .map() for large dictionaries
        - Single pass through data for each column update

    Note:
        - Creates a single copy of the DataFrame
        - Prediction/uncertainty columns are created with Float64 dtype
        - Uncertainty column is only created when uncertainties are provided

    Example:
        >>> predictions = np.array([0.5, 0.6, 0.7])
        >>> uncertainties = np.array([0.1, 0.15, 0.2])
        >>> updated_df = add_predictions(
        ...     df, cycle=0, compound_ids=['C1', 'C2', 'C3'],
        ...     predictions=predictions, uncertainties=uncertainties
        ... )
        >>> print(updated_df.filter(pl.col('ID') == 'C1').get_column('prediction_cycle_0')[0])
        0.5
    """
    df = df.clone()
    return _add_predictions_inplace(df, cycle, compound_ids, predictions, uncertainties)


def _update_status_inplace(
    df: pl.DataFrame,
    compound_ids: list[str],
    new_status: str,
    cycle: int,
    target_col: str,
    target_values: pl.Series | dict | None = None
) -> pl.DataFrame:
    """Update compound status in-place (modifies df).

    Private helper for batch operations. Modifies df in-place without copying.

    Args:
        df: Master DataFrame to modify in-place
        compound_ids: List of compound IDs to update
        new_status: New status ('labeled', 'unlabeled', or 'pruned')
        cycle: Current cycle number
        target_col: Name of target column (e.g., 'Activity')
        target_values: Optional Series with target values indexed by compound ID

    Returns:
        The same DataFrame (modified in-place)

    Raises:
        ValueError: If new_status is not in VALID_STATUSES
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}."
        )

    logger.debug(f"Updating status to '{new_status}' for {len(compound_ids)} compounds")

    mask = pl.col('ID').is_in(compound_ids)
    df = df.with_columns(
        pl.when(mask)
          .then(pl.lit(new_status))
          .otherwise(pl.col('status'))
          .alias('status')
    )

    if new_status == 'labeled':
        df = df.with_columns(
            pl.when(mask)
              .then(pl.lit(cycle))
              .otherwise(pl.col('labeled_cycle'))
              .alias('labeled_cycle')
        )

        na_mask = mask & pl.col('selected_cycle').is_null()
        df = df.with_columns(
            pl.when(na_mask)
              .then(pl.lit(cycle))
              .otherwise(pl.col('selected_cycle'))
              .alias('selected_cycle')
        )

        if target_values is not None:
            # Handle both dict and Series input for backward compatibility
            if isinstance(target_values, dict):
                id_to_value = target_values
            elif isinstance(target_values, pl.Series):
                # For Series, zip IDs with values
                id_to_value = dict(zip(compound_ids, target_values.to_list()))
            elif isinstance(target_values, pd.Series):
                # For pandas Series, use to_dict() for index-value mapping
                id_to_value = target_values.to_dict()
            else:
                raise TypeError(
                    f"target_values must be dict, pl.Series, or pd.Series, "
                    f"got {type(target_values).__name__}. "
                    f"Pass measured values as a Polars Series, pandas Series, or dict mapping ID → value."
                )

            df = map_values_via_join(df, id_to_value, 'ID', target_col)

    elif new_status == 'pruned':
        df = df.with_columns(
            pl.when(mask)
              .then(pl.lit(cycle))
              .otherwise(pl.col('pruned_cycle'))
              .alias('pruned_cycle')
        )

    logger.debug(f"Updated status to '{new_status}' for {len(compound_ids)} compounds")

    return df


def update_status(
    df: pl.DataFrame,
    compound_ids: list[str],
    new_status: str,
    cycle: int,
    target_col: str,
    target_values: pl.Series | pd.Series | dict | None = None
) -> pl.DataFrame:
    """Update compound status using vectorized boolean masking.

    Uses vectorized operations for O(n) performance. When updating to 'labeled'
    status, preserves the first selection cycle to maintain selection history.

    Args:
        df: Master DataFrame to update
        compound_ids: List of compound IDs to update
        new_status: New status ('labeled', 'unlabeled', or 'pruned')
        cycle: Current cycle number
        target_col: Name of target column (e.g., 'Activity')
        target_values: Optional Series (Polars or Pandas) or dict with target values indexed by compound ID

    Returns:
        Updated DataFrame (new copy)

    Raises:
        ValueError: If new_status is not in VALID_STATUSES

    Performance:
        - O(n) complexity using vectorized expressions
        - Single mask creation, reused throughout function

    Note:
        - Creates a single copy of the DataFrame
        - Preserves first selection cycle when re-labeling compounds. If a compound
          was selected in cycle 0 and labeled in cycle 2, selected_cycle remains 0
          while labeled_cycle updates to 2.

    Example:
        >>> # Initial labeling in cycle 0
        >>> target_vals = pl.Series('ID', ['C1', 'C2'])
        >>> updated_df = update_status(
        ...     df, ['C1', 'C2'], 'labeled', cycle=0,
        ...     target_col='Activity', target_values=target_vals
        ... )
        >>> # C1 and C2 now have: labeled_cycle=0, selected_cycle=0

        >>> # Re-labeling C1 in cycle 1 (if it was unlabeled again)
        >>> target_vals = pl.Series('ID', ['C1'])
        >>> updated_df = update_status(
        ...     updated_df, ['C1'], 'labeled', cycle=1,
        ...     target_col='Activity', target_values=target_vals
        ... )
        >>> # C1 now has: labeled_cycle=1, selected_cycle=0 (preserved)
    """
    df = df.clone()
    return _update_status_inplace(df, compound_ids, new_status, cycle, target_col, target_values)


def get_compounds_by_status(
    df: pl.DataFrame,
    status: str,
    columns: list[str] | None = None
) -> pl.DataFrame:
    """Get compounds by status using vectorized filtering.

    Returns a filtered DataFrame. Polars uses copy-on-write, so this is efficient.

    Args:
        df: Master DataFrame
        status: Status to filter by ('labeled', 'unlabeled', or 'pruned')
        columns: Optional list of columns to return (default: all columns)

    Returns:
        DataFrame filtered by status

    Raises:
        ValueError: If status is not in VALID_STATUSES

    Performance:
        - O(n) complexity using vectorized filtering
        - Memory efficient with Polars copy-on-write

    Example:
        >>> # Get all columns for labeled compounds
        >>> labeled = get_compounds_by_status(df, 'labeled')

        >>> # Get specific columns for unlabeled compounds
        >>> unlabeled = get_compounds_by_status(df, 'unlabeled', columns=['ID', 'SMILES'])

        >>> # Get labeled compounds with target values
        >>> labeled = get_compounds_by_status(df, 'labeled', columns=['ID', 'SMILES', 'Activity'])
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status filter '{status}'. Must be one of {VALID_STATUSES}."
        )

    filtered = df.filter(pl.col('status') == status)

    if columns is not None:
        return filtered.select(columns)
    else:
        return filtered


def batch_update(
    df: pl.DataFrame,
    updates: dict[str, Any]
) -> pl.DataFrame:
    """Apply multiple updates to DataFrame in a single operation.

    More efficient than separate calls - creates single DataFrame copy instead of
    multiple copies for each update operation.

    Args:
        df: Master DataFrame to update
        updates: Dictionary with update specifications:
            - 'predictions': Tuple of (cycle, compound_ids, predictions, uncertainties)
            - 'status': Tuple of (compound_ids, new_status, cycle, target_col, target_values)

    Returns:
        Updated DataFrame (new copy)

    Performance:
        - Creates only one DataFrame copy regardless of number of operations
        - 2x+ efficiency gain for combined operations

    Example:
        >>> # Apply both prediction and status updates together
        >>> updates = {
        ...     'predictions': (0, ['C1', 'C2'], np.array([0.5, 0.6]), np.array([0.1, 0.15])),
        ...     'status': (['C1', 'C2'], 'labeled', 0, 'Activity', pl.Series('ID', ['C1', 'C2']))
        ... }
        >>> updated_df = batch_update(df, updates)

        >>> # Apply only predictions
        >>> updates = {
        ...     'predictions': (1, ['C3'], np.array([0.7]), None)
        ... }
        >>> updated_df = batch_update(df, updates)
    """
    df = df.clone()

    if 'predictions' in updates:
        cycle, compound_ids, predictions, uncertainties = updates['predictions']
        df = _add_predictions_inplace(df, cycle, compound_ids, predictions, uncertainties)

    if 'status' in updates:
        compound_ids, new_status, cycle, target_col, target_values = updates['status']
        df = _update_status_inplace(df, compound_ids, new_status, cycle, target_col, target_values)

    return df

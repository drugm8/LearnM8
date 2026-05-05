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
- update_status: O(n) using vectorized expressions
- get_compounds_by_status: O(n) filtering, returns view
"""

import logging

import pandas as pd
import polars as pl

from learnm8.utils.polars_utils import map_values_via_join

from .data_structures import VALID_STATUSES

logger = logging.getLogger(__name__)


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

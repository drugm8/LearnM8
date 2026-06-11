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

import polars as pl

from .initialization import VALID_STATUSES

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
            if isinstance(target_values, dict):
                id_to_value = target_values
            elif isinstance(target_values, pl.Series):
                id_to_value = dict(zip(compound_ids, target_values.to_list(), strict=True))
            else:
                raise TypeError(
                    f"target_values must be dict or pl.Series, "
                    f"got {type(target_values).__name__}. "
                    f"Pass measured values as a Polars Series or dict mapping ID → value."
                )

            lookup_df = pl.DataFrame({
                'ID': list(id_to_value.keys()),
                f'{target_col}_new': list(id_to_value.values())
            })
            df = df.join(lookup_df, on='ID', how='left')
            if target_col in df.columns:
                df = df.with_columns(
                    pl.coalesce(
                        pl.col(f'{target_col}_new'),
                        pl.col(target_col)
                    ).alias(target_col)
                ).drop(f'{target_col}_new')
            else:
                df = df.rename({f'{target_col}_new': target_col})

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
    target_values: pl.Series | dict | None = None
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
        target_values: Optional Series (Polars) or dict with target values indexed by compound ID

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


def iter_status_chunks(
    df: pl.DataFrame,
    status: str,
    chunk_size: int,
    columns: tuple[str, ...] = ('ID', 'SMILES'),
):
    """Yield ``(global_offset, chunk_df)`` over rows with the given status.

    Streaming pass 1 uses this instead of materializing a full
    ``df.filter(status==...).select(ID, SMILES)`` copy (the ~8 GB
    ``prediction_pool`` at 100M). Row indices of the matching rows are computed
    once (UInt32, ~0.4 GB at 100M) and each chunk is gathered positionally.

    ``global_offset`` is the chunk's start position within the filtered
    sequence (0-based), so it aligns with the parquet write order and the
    ``StreamingTopK`` global index space.

    Args:
        df: Master DataFrame.
        status: Status to iterate ('unlabeled', 'labeled', 'pruned').
        chunk_size: Rows per chunk (must be positive).
        columns: Columns to project into each chunk.

    Yields:
        ``(global_offset, chunk_df)`` tuples; no chunk is empty.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status filter '{status}'. Must be one of {VALID_STATUSES}."
        )
    if chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size}')

    idx = df.select(
        pl.arg_where(pl.col('status') == status).alias('i')
    ).to_series()
    total = idx.len()
    cols = list(columns)
    for start in range(0, total, chunk_size):
        chunk_idx = idx.slice(start, chunk_size)
        yield start, df[chunk_idx].select(cols)


def mark_pruned_by_ids(
    df: pl.DataFrame,
    pruned_ids: pl.Series,
    cycle: int,
) -> pl.DataFrame:
    """Set ``status='pruned'`` + ``pruned_cycle=cycle`` for ``pruned_ids`` via join.

    Streaming pruning can mark tens of millions of compounds; a join against a
    Polars/Arrow ID Series avoids the per-element Python-object overhead of
    ``is_in(python_list)`` in :func:`update_status`. Only the status and
    ``pruned_cycle`` columns change (no target values), matching the
    ``new_status == 'pruned'`` branch of :func:`_update_status_inplace`.

    Args:
        df: Master DataFrame.
        pruned_ids: Polars Series of compound IDs to mark pruned.
        cycle: Current cycle number.

    Returns:
        New DataFrame with the pruned rows updated.
    """
    if pruned_ids.len() == 0:
        return df
    marker = pl.DataFrame({'ID': pruned_ids}).with_columns(
        pl.lit(True).alias('_pruned_marker')
    )
    df = df.join(marker, on='ID', how='left')
    mask = pl.col('_pruned_marker').fill_null(False)
    df = df.with_columns(
        pl.when(mask)
        .then(pl.lit('pruned'))
        .otherwise(pl.col('status'))
        .alias('status'),
        pl.when(mask)
        .then(pl.lit(cycle))
        .otherwise(pl.col('pruned_cycle'))
        .alias('pruned_cycle'),
    ).drop('_pruned_marker')
    return df


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

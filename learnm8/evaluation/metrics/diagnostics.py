"""Diagnostic cycle metrics — feature 014.

Two pure functions and one trivial helper used by ``evaluate_cycle``:

- ``compute_selected_percentile``: mean rank-percentile of the cycle's selected
  compounds against the full pool's true score distribution. Benchmark mode
  only — returns ``None`` when ground truth is unavailable. Honours
  ``score_direction`` so that ``100`` always means "best".
- ``compute_prediction_entropy``: Shannon entropy (base 2) of the model's
  predictions over the **unlabeled subset** of the pool, computed via a
  50-bin density histogram. Returns ``None`` when no unlabeled compound is
  available.

See ``specs/014-diagnostic-metrics/spec.md`` for the full requirements.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

__all__ = [
    'compute_prediction_entropy',
    'compute_selected_percentile',
]


def compute_selected_percentile(
    selected_compounds: pl.DataFrame,
    ground_truth_data: pl.DataFrame | None,
    target_col: str,
    score_direction: str = 'higher',
) -> float | None:
    """Return mean rank-percentile of selected compounds vs. full pool true scores.

    Higher-better: ``percentile = (s >= pool_scores).mean() * 100``
    Lower-better:  ``percentile = (s <= pool_scores).mean() * 100``

    Both conventions yield ``100`` when ``s`` is the best in the pool.

    Args:
        selected_compounds: This cycle's selections (must contain ``target_col``).
        ground_truth_data: Full pool ground truth (must contain ``target_col``);
            ``None`` in run mode.
        target_col: Name of the score column.
        score_direction: ``'higher'`` (default) or ``'lower'``.

    Returns:
        Mean percentile in [0, 100], or ``None`` if any required input is
        missing / empty.
    """
    if ground_truth_data is None:
        return None
    if target_col not in ground_truth_data.columns:
        return None
    if selected_compounds is None or len(selected_compounds) == 0:
        return None
    if target_col not in selected_compounds.columns:
        return None

    pool_scores = ground_truth_data.get_column(target_col).to_numpy()
    selected_scores = selected_compounds.get_column(target_col).to_numpy()

    pool_scores = pool_scores[~np.isnan(pool_scores)]
    selected_scores = selected_scores[~np.isnan(selected_scores)]
    if len(pool_scores) == 0 or len(selected_scores) == 0:
        return None

    sorted_pool = np.sort(pool_scores)
    n = len(sorted_pool)

    if score_direction == 'lower':
        # percentile = (count of pool >= s) / n
        # = (n - count strictly less than s) / n
        counts_below = np.searchsorted(sorted_pool, selected_scores, side='left')
        percentiles = (n - counts_below) / n * 100.0
    else:
        # percentile = (count of pool <= s) / n
        counts_at_or_below = np.searchsorted(sorted_pool, selected_scores, side='right')
        percentiles = counts_at_or_below / n * 100.0

    return float(np.mean(percentiles))


def compute_prediction_entropy(
    pool_df: pl.DataFrame | None,
    cumulative_selected_ids: set | None = None,
    n_bins: int = 50,
) -> float | None:
    """Shannon entropy (base 2, bits) of the model's predictions over unlabeled pool.

    Histogram is density-normalised over ``n_bins`` equal-width bins. A small
    additive smoothing (``+ 1e-10``) keeps the entropy finite when bins are
    empty (e.g. all predictions identical).

    Args:
        pool_df: Pool predictions DataFrame with at least ``ID`` and
            ``prediction`` columns. ``None`` to skip.
        cumulative_selected_ids: IDs of compounds already labeled (excluded
            from the histogram). ``None`` keeps the full pool.
        n_bins: Number of histogram bins (default ``50``).

    Returns:
        Entropy in bits (``>= 0``), or ``None`` if the pool / unlabeled subset
        is unavailable.
    """
    if pool_df is None or len(pool_df) == 0:
        return None
    if 'prediction' not in pool_df.columns:
        return None

    if cumulative_selected_ids and 'ID' in pool_df.columns:
        unlabeled = pool_df.filter(~pl.col('ID').is_in(cumulative_selected_ids))
    else:
        unlabeled = pool_df

    if len(unlabeled) == 0:
        return None

    preds = unlabeled.get_column('prediction').to_numpy()
    preds = preds[~np.isnan(preds)]
    if len(preds) == 0:
        return None

    hist, _ = np.histogram(preds, bins=n_bins, density=True)
    return float(stats.entropy(hist + 1e-10, base=2))

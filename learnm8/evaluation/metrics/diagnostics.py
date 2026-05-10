"""Diagnostic cycle metrics — feature 014, refined in feature 019.

Two pure functions and one trivial helper used by ``evaluate_cycle``:

- ``compute_selected_percentile``: mean rank-percentile of the cycle's selected
  compounds against the full pool's true score distribution. Benchmark mode
  only — returns ``None`` when ground truth is unavailable. Honours
  ``score_direction`` so that ``100`` always means "best".
- ``compute_prediction_entropy``: Shannon entropy (units: nats) of the model's
  predictions over the **unlabeled subset** of the pool, computed via a fixed
  50-bin count histogram with post-normalisation smoothing. Scale-invariant by
  construction; units changed from bits → nats in feature 019 for cross-contract
  consistency with ``EntropyAcquisition``.

See ``specs/014-diagnostic-metrics/spec.md`` and
``specs/019-math-correctness/contracts/prediction_entropy.md`` for the full
requirements.
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
    """Shannon entropy (units: nats) of the model's predictions over unlabeled pool.

    Bin count is fixed at 50 to preserve scale-invariance: data-dependent rules
    (Sturges, Freedman-Diaconis) would defeat the property ``H(p) == H(c·p)``
    by yielding different bin counts at different scales. Histogram is built
    with ``density=False`` (counts), normalised to probabilities, then smoothed
    with ``+1e-10/n_bins`` AFTER normalisation and renormalised so
    ``Σ p log p`` is unbiased. Units are nats (base = e), matching
    ``EntropyAcquisition`` and ``scipy.stats.norm.entropy``.

    Predictions are guaranteed to be NaN-free upstream (cycle.py FR-005 guard);
    the previous internal silent-NaN filter has been removed.

    Args:
        pool_df: Pool predictions DataFrame with at least ``ID`` and
            ``prediction`` columns. ``None`` to skip.
        cumulative_selected_ids: IDs of compounds already labeled (excluded
            from the histogram). ``None`` keeps the full pool.
        n_bins: Number of histogram bins (default ``50``; fixed for cross-run
            comparability — see contract).

    Returns:
        Entropy in nats (``>= 0``, ``≤ log(n_bins)``), or ``None`` if the
        pool / unlabeled subset is unavailable.
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
    if len(preds) == 0:
        return None

    # Spec US3 #2: constant predictions → exactly 0.0 (no smoothing bias).
    if preds.size > 0 and float(preds.min()) == float(preds.max()):
        return 0.0

    counts, _edges = np.histogram(preds, bins=n_bins, density=False)
    total = int(counts.sum())
    if total == 0:
        return None
    probs = counts.astype(np.float64)
    probs /= total
    probs += 1e-10 / n_bins
    probs /= probs.sum()
    return float(stats.entropy(probs))

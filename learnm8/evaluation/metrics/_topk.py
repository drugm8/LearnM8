"""Private top-K index helper (Spec 022 FR-005).

Shared by ``top_k_overlap.py`` and ``unlabeled_enrichment_factor.py``: the
helper centralises the ``np.argpartition`` recipe so the two callers cannot
diverge in tie-handling or fallback semantics.

The key win over ``pl.DataFrame.sort(...).head(k)`` is replacing 4-5 full
O(N log N) sorts per cycle with **one** O(N) partition plus a sort over only
the K-element top window. The ``kth=-max_k`` form avoids materialising a
``-scores`` copy (which would cost ~400 MB at 100M float32).

Internal — not exported in the public API.
"""

from __future__ import annotations

import numpy as np


def top_k_indices(scores: np.ndarray, max_k: int) -> np.ndarray:
    """Return indices of the top-``max_k`` scores in descending order.

    Uses ``np.argpartition`` with ``kth=-max_k`` (ascending partition; the
    right side is the top-K unsorted window). After the partition we
    ``argsort`` only the K-element top slice, not the full array.

    Falls back to a full ``argsort`` when ``max_k >= n`` because
    ``np.argpartition`` raises on out-of-range ``kth``.

    The fallback uses ``-scores`` since the array is at most ``max_k`` long
    in the hot path — at the K=N edge a one-off negated copy is acceptable.

    Args:
        scores: 1-D array of scores (any numeric dtype).
        max_k: number of top entries to return (1 <= max_k).

    Returns:
        Index array of length ``min(max_k, len(scores))`` in descending-score
        order. Dtype is platform-default integer (typically int64); callers
        that need uint32 should cast explicitly.

    Notes:
        Ties at the K-th score are non-deterministic under ``argpartition``
        (introselect is unstable). Downstream callers that compare top-K
        sets across runs should either (a) use distinct scores or
        (b) tie-break by index lexicographically before set comparison.
    """
    n = scores.size
    if max_k >= n:
        # Full-sort fallback; ``kind='stable'`` makes ties deterministic.
        return np.argsort(-scores, kind="stable")
    # Partition once at the kth-from-end; right partition is the top-max_k unsorted.
    top_unsorted = np.argpartition(scores, kth=-max_k)[-max_k:]
    # Sort only the K-element window in descending order.
    return top_unsorted[np.argsort(-scores[top_unsorted], kind="stable")]


__all__ = ["top_k_indices"]

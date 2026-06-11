"""Bounded streaming top-K reducer for streaming acquisition.

Pass 2 of the streaming cycle scores the candidate pool one chunk at a time and
must end with the globally best ``k`` candidates without ever holding all ``N``
scores at once. :class:`StreamingTopK` keeps an **index-only** running buffer of
the best ``k`` ``(score, global_index)`` pairs; the payload (ID, prediction,
uncertainty) is gathered by index afterwards by the caller. At 100M scale with
the ``top_k`` strategy (``k = k_fraction · N`` ≈ 10M) the buffer is ~160 MB and
carries no Python strings.

Canonical order (the NEW normative selection order — see the streaming feature
plan): best first, ties broken by **smaller global index**. Equivalently, items
are ranked by a key where *smaller = better*:

- ``descending=True``  (maximise): key = -score  -> highest score wins.
- ``descending=False`` (minimise): key =  score  → lowest score wins.

This order is deterministic and invariant to how the pool was chunked, including
when scores tie at the k-th-place boundary — unlike the legacy
``np.argpartition`` membership in ``_safe_select_top_k`` (pivot-arbitrary at
ties). On tie-free score boundaries it reproduces the legacy order exactly.
"""

from __future__ import annotations

import numpy as np

from learnm8.exceptions import AcquisitionError


class StreamingTopK:
    """Running top-``k`` over chunked scores, retaining only ``(score, index)``.

    Args:
        k: Number of best candidates to retain. Must be positive.
        descending: If True keep the highest scores; if False the lowest.
    """

    def __init__(self, k: int, descending: bool = True):
        if k <= 0:
            raise AcquisitionError(
                f'StreamingTopK requires k > 0, got {k}. This usually means the '
                f'batch/shortlist size was computed as zero — check batch_fraction '
                f'and pool size.'
            )
        self.k = int(k)
        self.descending = bool(descending)
        self._scores = np.empty(0, dtype=np.float64)
        self._idx = np.empty(0, dtype=np.uint64)

    def update(self, scores: np.ndarray, global_offset: int) -> None:
        """Fold one chunk of scores (positions ``[offset, offset + len)``) in.

        Args:
            scores: 1-D array of acquisition scores for this chunk, in pool order.
            global_offset: Global index of the chunk's first row.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        n = scores.shape[0]
        if n == 0:
            return
        if global_offset < 0:
            raise AcquisitionError(
                f'global_offset must be non-negative, got {global_offset}.'
            )
        idx = np.arange(global_offset, global_offset + n, dtype=np.uint64)

        merged_scores: np.ndarray = (
            scores if self._scores.size == 0 else np.concatenate([self._scores, scores])
        )
        merged_idx: np.ndarray = (
            idx if self._idx.size == 0 else np.concatenate([self._idx, idx])
        )

        if merged_scores.shape[0] <= self.k:
            self._scores, self._idx = merged_scores, merged_idx
            return

        keep = self._select_best_indices(merged_scores, merged_idx, self.k)
        self._scores, self._idx = merged_scores[keep], merged_idx[keep]

    def _select_best_indices(
        self, scores: np.ndarray, idx: np.ndarray, k: int
    ) -> np.ndarray:
        """Positions of the ``k`` best items; ties resolved by smaller index.

        ``key`` is defined so smaller is always better; selection is the ``k``
        smallest keys, breaking key-ties by ascending ``idx``. O(n) partition
        plus an O(t log t) sort only over the boundary tie-group ``t``.
        """
        key = -scores if self.descending else scores
        # k-th smallest key value is the inclusion threshold.
        threshold = np.partition(key, k - 1)[k - 1]
        strictly_better = key < threshold
        m = int(np.count_nonzero(strictly_better))
        if m == k:
            return np.flatnonzero(strictly_better)

        tie = np.flatnonzero(key == threshold)
        # Need (k - m) tied items with the smallest global index → deterministic.
        need = k - m
        tie_sorted = tie[np.argsort(idx[tie], kind='stable')][:need]
        return np.concatenate([np.flatnonzero(strictly_better), tie_sorted])

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(global_indices, scores)`` in canonical selection order.

        Best first; ties broken by smaller global index. Length is
        ``min(k, total seen)``.
        """
        key = -self._scores if self.descending else self._scores
        order = np.lexsort((self._idx, key))  # primary key asc, then idx asc
        return self._idx[order], self._scores[order]

    def __len__(self) -> int:
        return int(self._scores.shape[0])

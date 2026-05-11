"""Spec 022 US3 / FR-005: ``top_k_indices`` private helper coverage."""

from __future__ import annotations

import numpy as np
import pytest

from learnm8.evaluation.metrics._topk import top_k_indices

pytestmark = pytest.mark.unit


def test_top_k_basic_descending() -> None:
    """Top-3 of a 10-element scores array; indices in descending-score order."""
    scores = np.array([0.1, 0.5, 0.9, 0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.0])
    idx = top_k_indices(scores, max_k=3)
    assert len(idx) == 3
    # Scores at returned indices should be 0.9, 0.8, 0.7 in order.
    assert scores[idx[0]] == 0.9
    assert scores[idx[1]] == 0.8
    assert scores[idx[2]] == 0.7


def test_top_k_full_sort_when_k_equals_n() -> None:
    """When max_k >= n, fallback returns all indices in descending order."""
    scores = np.array([0.3, 0.1, 0.2])
    idx = top_k_indices(scores, max_k=3)
    assert len(idx) == 3
    assert list(scores[idx]) == [0.3, 0.2, 0.1]


def test_top_k_k_greater_than_n() -> None:
    """max_k > n falls back to a full sort and returns all n indices."""
    scores = np.array([0.3, 0.1, 0.2])
    idx = top_k_indices(scores, max_k=100)
    assert len(idx) == 3
    assert list(scores[idx]) == [0.3, 0.2, 0.1]


def test_top_k_single_element() -> None:
    """max_k=1 returns just the argmax."""
    scores = np.array([0.5, 0.7, 0.2, 0.9, 0.1])
    idx = top_k_indices(scores, max_k=1)
    assert len(idx) == 1
    assert idx[0] == 3
    assert scores[idx[0]] == 0.9


def test_top_k_preserves_int_dtype() -> None:
    """Returned dtype is a valid integer (int64, uint32, etc.)."""
    scores = np.random.RandomState(42).rand(100).astype(np.float32)
    idx = top_k_indices(scores, max_k=10)
    assert np.issubdtype(idx.dtype, np.integer)


def test_top_k_works_on_float32() -> None:
    """Top-K on float32 input works (no implicit float64 promotion needed)."""
    rng = np.random.RandomState(7)
    scores = rng.rand(10_000).astype(np.float32)
    idx = top_k_indices(scores, max_k=100)
    assert len(idx) == 100
    # Top-100 of float32 must satisfy descending-order invariant.
    top_vals = scores[idx]
    assert np.all(top_vals[:-1] >= top_vals[1:])


def test_top_k_tied_scores_keeps_descending_property() -> None:
    """Ties at the K-th boundary: returned set still has K largest scores.

    Tie membership is non-deterministic under argpartition (introselect),
    but the K returned elements MUST all have score >= the next-best
    non-returned score. This is the property argpartition guarantees.
    """
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6])
    idx = top_k_indices(scores, max_k=5)
    assert len(idx) == 5
    top_set = scores[idx]
    # Top-5 scores: must include 0.6 and four of the five 0.5's.
    assert 0.6 in top_set
    assert (top_set == 0.5).sum() == 4


def test_top_k_descending_compared_to_argsort() -> None:
    """argpartition+slice produces same set as full argsort for ties-free input.

    Validates bit-identical AGREEMENT on the top-K SET (not order — argsort
    of unique scores is canonical descending, but argpartition is unstable).
    """
    rng = np.random.RandomState(123)
    scores = rng.rand(1000)  # high-precision floats; no expected ties
    K = 50
    via_partition = top_k_indices(scores, max_k=K)
    via_argsort = np.argsort(-scores, kind='stable')[:K]
    assert set(via_partition.tolist()) == set(via_argsort.tolist())

"""Tests for StreamingTopK (streaming acquisition reducer).

Pins the load-bearing properties:
- chunk-size invariance: the result is identical no matter how scores are split;
- reference equivalence: matches a single-shot canonical (key, index) sort;
- tie determinism: a defined, chunk-invariant order even at the k-th boundary;
- legacy equivalence on tie-free data: matches _safe_select_top_k's selection.
"""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learnm8.acquisition.base import AcquisitionFunction
from learnm8.acquisition.streaming import StreamingTopK
from learnm8.exceptions import AcquisitionError

pytestmark = pytest.mark.unit


def reference_topk(scores: np.ndarray, k: int, descending: bool) -> np.ndarray:
    """Canonical single-shot top-k: best first, ties by ascending index."""
    key = -scores if descending else scores
    idx = np.arange(scores.shape[0], dtype=np.uint64)
    order = np.lexsort((idx, key))
    return idx[order][: min(k, scores.shape[0])]


def run_streaming(scores: np.ndarray, k: int, descending: bool, chunk: int):
    topk = StreamingTopK(k, descending=descending)
    for start in range(0, scores.shape[0], chunk):
        topk.update(scores[start : start + chunk], start)
    idx, _ = topk.result()
    return idx


class TestChunkInvariance:
    @pytest.mark.parametrize('descending', [True, False])
    @pytest.mark.parametrize('chunk', [1, 7, 64, 1000])
    def test_matches_reference_across_chunks(self, descending, chunk):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=500)
        k = 25
        expected = reference_topk(scores, k, descending)
        got = run_streaming(scores, k, descending, chunk)
        np.testing.assert_array_equal(got, expected)

    def test_chunk_one_equals_single_shot(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(size=300)
        a = run_streaming(scores, 10, True, 1)
        b = run_streaming(scores, 10, True, 300)
        np.testing.assert_array_equal(a, b)


class TestTieDeterminism:
    def test_all_equal_scores_pick_smallest_indices(self):
        scores = np.zeros(100)
        # descending: all tied → canonical keeps smallest indices, in order.
        idx = run_streaming(scores, 10, True, 7)
        np.testing.assert_array_equal(idx, np.arange(10, dtype=np.uint64))

    def test_boundary_ties_are_chunk_invariant(self):
        # Scores with a heavy tie exactly at the k-boundary.
        scores = np.array([3.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0])
        k = 3  # one clear winner (3.0) + two of the four 2.0s
        for chunk in [1, 2, 3, 7]:
            idx = run_streaming(scores, k, True, chunk)
            # winner index 0, then the two smallest-index 2.0s: 1 and 2.
            np.testing.assert_array_equal(idx, np.array([0, 1, 2], dtype=np.uint64))


class TestLegacyEquivalenceTieFree:
    """On tie-free scores, StreamingTopK selection == _safe_select_top_k selection."""

    class _Acq(AcquisitionFunction):
        def select(self, compounds, n_select):  # pragma: no cover - unused
            raise NotImplementedError

    @pytest.mark.parametrize('descending', [True, False])
    def test_same_selected_set_and_order(self, descending):
        rng = np.random.default_rng(2)
        scores = rng.permutation(np.linspace(-5, 5, 200))  # distinct → tie-free
        k = 30
        ids = [f'c{i}' for i in range(200)]
        df = pl.DataFrame({'ID': ids, 'SMILES': ids, 'prediction': scores})

        acq = self._Acq(score_direction='higher' if descending else 'lower')
        legacy = acq._safe_select_top_k(df, scores, k, ascending=not descending)
        legacy_ids = legacy['ID'].to_list()

        topk = StreamingTopK(k, descending=descending)
        topk.update(scores, 0)
        sidx, _ = topk.result()
        streaming_ids = [ids[i] for i in sidx]

        assert streaming_ids == legacy_ids


class TestEdgeCases:
    def test_k_larger_than_pool_keeps_all(self):
        scores = np.array([1.0, 3.0, 2.0])
        topk = StreamingTopK(10, descending=True)
        topk.update(scores, 0)
        idx, sc = topk.result()
        np.testing.assert_array_equal(idx, np.array([1, 2, 0], dtype=np.uint64))
        np.testing.assert_array_equal(sc, np.array([3.0, 2.0, 1.0]))

    def test_empty_updates(self):
        topk = StreamingTopK(5, descending=True)
        topk.update(np.array([]), 0)
        idx, sc = topk.result()
        assert idx.shape == (0,)
        assert sc.shape == (0,)

    def test_k_zero_raises(self):
        with pytest.raises(AcquisitionError):
            StreamingTopK(0)

    def test_negative_offset_raises(self):
        with pytest.raises(AcquisitionError):
            StreamingTopK(3).update(np.array([1.0]), -1)

    def test_global_offset_tracked_across_updates(self):
        # Offsets must produce the true global indices, not chunk-local ones.
        topk = StreamingTopK(2, descending=True)
        topk.update(np.array([1.0, 2.0]), 0)
        topk.update(np.array([9.0, 0.5]), 2)  # index 2 has score 9.0
        idx, _ = topk.result()
        np.testing.assert_array_equal(idx, np.array([2, 1], dtype=np.uint64))


class TestPropertyBased:
    @settings(max_examples=200, deadline=None)
    @given(
        data=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False),
            min_size=1,
            max_size=300,
        ),
        k=st.integers(min_value=1, max_value=50),
        descending=st.booleans(),
        chunk=st.integers(min_value=1, max_value=64),
    )
    def test_streaming_equals_reference_under_random_partitions(
        self, data, k, descending, chunk
    ):
        scores = np.array(data, dtype=np.float64)
        expected = reference_topk(scores, k, descending)
        got = run_streaming(scores, k, descending, chunk)
        np.testing.assert_array_equal(got, expected)

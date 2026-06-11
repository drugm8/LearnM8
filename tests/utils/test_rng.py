"""Tests for counter-based RNG (streaming acquisition).

Pins the load-bearing property: stream-position invariance under any chunking,
so stochastic acquisition draws do not depend on how the pool was split.
"""

import numpy as np
import pytest

from learnm8.utils.rng import (
    DEFAULT_RNG_SEED,
    chunk_normal,
    chunk_uniform,
)

pytestmark = pytest.mark.unit

SEED = (42, 3)


class TestChunkUniformPositionInvariance:
    def test_slice_equals_full_stream(self):
        full = chunk_uniform(SEED, 0, 1000)
        # arbitrary, NON-multiple-of-4 offsets (Philox block is 4 uint64s, so a
        # naive block-advance would only be correct at multiples of 4).
        for start, n in [(1, 10), (7, 13), (37, 101), (250, 500), (999, 1)]:
            chunk = chunk_uniform(SEED, start, n)
            np.testing.assert_array_equal(chunk, full[start : start + n])

    def test_concatenated_chunks_equal_single_draw(self):
        full = chunk_uniform(SEED, 0, 100)
        pieces = [
            chunk_uniform(SEED, 0, 7),
            chunk_uniform(SEED, 7, 1),
            chunk_uniform(SEED, 8, 50),
            chunk_uniform(SEED, 58, 42),
        ]
        np.testing.assert_array_equal(np.concatenate(pieces), full)

    def test_chunk_size_one_reproduces_stream(self):
        full = chunk_uniform(SEED, 0, 64)
        per_element = np.array([chunk_uniform(SEED, i, 1)[0] for i in range(64)])
        np.testing.assert_array_equal(per_element, full)

    def test_values_in_unit_interval(self):
        u = chunk_uniform(SEED, 0, 10_000)
        assert u.min() >= 0.0
        assert u.max() < 1.0

    def test_empty_chunk(self):
        out = chunk_uniform(SEED, 100, 0)
        assert out.shape == (0,)
        assert out.dtype == np.float64

    def test_negative_args_raise(self):
        with pytest.raises(ValueError):
            chunk_uniform(SEED, 0, -1)
        with pytest.raises(ValueError):
            chunk_uniform(SEED, -1, 5)


class TestSeedComponents:
    def test_different_components_independent(self):
        a = chunk_uniform((42, 0), 0, 100)
        b = chunk_uniform((42, 1), 0, 100)
        assert not np.array_equal(a, b)

    def test_none_falls_back_to_default(self):
        # (None, 3) must resolve identically to (DEFAULT_RNG_SEED, 3).
        a = chunk_uniform((None, 3), 0, 50)
        b = chunk_uniform((DEFAULT_RNG_SEED, 3), 0, 50)
        np.testing.assert_array_equal(a, b)

    def test_same_components_reproduce_draws(self):
        # Calling twice with identical components resets the counter (documented
        # divergence from a stateful Generator).
        a = chunk_uniform(SEED, 0, 20)
        b = chunk_uniform(SEED, 0, 20)
        np.testing.assert_array_equal(a, b)


class TestChunkNormal:
    def test_position_invariance(self):
        full = chunk_normal(SEED, 0, 500)
        for start, n in [(3, 17), (60, 200), (499, 1)]:
            np.testing.assert_array_equal(
                chunk_normal(SEED, start, n), full[start : start + n]
            )

    def test_no_infinities(self):
        # ndtri(0.0) would be -inf; the floor guard must prevent it.
        z = chunk_normal((1, 1), 0, 100_000)
        assert np.isfinite(z).all()

    def test_approximately_standard_normal(self):
        z = chunk_normal((7, 9), 0, 200_000)
        assert abs(z.mean()) < 0.02
        assert abs(z.std() - 1.0) < 0.02

    def test_empty(self):
        out = chunk_normal(SEED, 10, 0)
        assert out.shape == (0,)

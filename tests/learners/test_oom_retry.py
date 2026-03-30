from __future__ import annotations

import numpy as np
import pytest

from learnm8.exceptions import LearnerError
from learnm8.learners.base import _predict_with_oom_retry

pytestmark = pytest.mark.unit


def test_oom_retry_succeeds_on_first_attempt():
    features = np.random.rand(100, 10).astype(np.float32)
    expected = np.random.rand(100, 1).astype(np.float32)

    def predict_fn(x):
        return expected[: len(x)]

    result = _predict_with_oom_retry(features, predict_fn, min_batch=256, n_output_cols=1)
    assert result.shape == (100, 1)


def test_oom_retry_halves_batch_on_oom():
    features = np.arange(100 * 5, dtype=np.float32).reshape(100, 5)
    call_sizes = []

    def predict_fn(x):
        call_sizes.append(len(x))
        if len(x) >= 100:
            raise RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB')
        return np.ones((len(x), 1), dtype=np.float32)

    result = _predict_with_oom_retry(features, predict_fn, min_batch=1, n_output_cols=1)
    assert result.shape[0] == 100
    assert any(s < 100 for s in call_sizes)


def test_oom_retry_preserves_output_ordering():
    n = 80
    features = np.arange(n * 4, dtype=np.float32).reshape(n, 4)

    def predict_fn(x):
        if len(x) >= n:
            raise RuntimeError('out of memory')
        return x[:, :1].copy()

    result = _predict_with_oom_retry(features, predict_fn, min_batch=1, n_output_cols=1)
    assert result.shape == (n, 1)
    np.testing.assert_array_equal(result[:, 0], features[:, 0])


def test_oom_retry_raises_after_min_batch_exceeded():
    features = np.random.rand(64, 8).astype(np.float32)

    def predict_fn(x):
        raise RuntimeError('out of memory')

    with pytest.raises(LearnerError):
        _predict_with_oom_retry(features, predict_fn, min_batch=256, n_output_cols=1)


def test_oom_retry_output_shape_matches_input():
    features = np.random.rand(73, 12).astype(np.float32)

    def predict_fn(x):
        if len(x) > 40:
            raise RuntimeError('CUDA out of memory')
        return np.zeros((len(x), 2), dtype=np.float32)

    result = _predict_with_oom_retry(features, predict_fn, min_batch=1, n_output_cols=2)
    assert result.shape[0] == features.shape[0]

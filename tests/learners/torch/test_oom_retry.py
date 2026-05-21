"""Regression tests for MCDropoutLearner GPU-OOM chunk-split retry path."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from learnm8.exceptions import LearnerError
from learnm8.learners.torch.mc_dropout import MCDropoutLearner


def _make_data(n_samples, n_features=10, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = X[:, 0] * 2.0 + X[:, 1] * (-1.5) + rng.randn(n_samples) * 0.1
    return X, y


def _make_trained_learner(X, y, predict_batch_size, seed=42):
    learner = MCDropoutLearner(
        hidden_sizes=(32, 16),
        n_dropout_samples=20,
        max_epochs=5,
        batch_size=16,
        predict_batch_size=predict_batch_size,
        random_state=seed,
    )
    learner.train(X, y)
    return learner


class TestMCDropoutOOMRetry:
    @pytest.mark.unit
    def test_oom_split_odd_chunk_keeps_every_row(self):
        n_samples = 1025
        X, y = _make_data(n_samples)

        learner = _make_trained_learner(X, y, predict_batch_size=n_samples)

        original = learner._predict_chunk
        call_count = [0]

        def side_effect(X_chunk):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError('CUDA out of memory')
            return original(X_chunk)

        with patch.object(learner, '_predict_chunk', side_effect=side_effect):
            preds, unc = learner.predict(X)

        assert call_count[0] > 1
        assert preds.shape == (n_samples,)
        assert unc.shape == (n_samples,)
        assert np.all(np.isfinite(preds))
        assert np.all(np.isfinite(unc))

    @pytest.mark.unit
    def test_cuda_oom_error_raises_learner_error_not_unbound(self):
        n_samples = 2048
        X, y = _make_data(n_samples)

        learner = _make_trained_learner(X, y, predict_batch_size=n_samples)

        def side_effect(_X_chunk):
            raise torch.cuda.OutOfMemoryError('CUDA out of memory')

        with (
            patch.object(learner, '_predict_chunk', side_effect=side_effect),
            pytest.raises(LearnerError) as exc_info,
        ):
            learner.predict(X)

        assert not isinstance(exc_info.value.__cause__, UnboundLocalError)
        assert isinstance(exc_info.value.__cause__, torch.cuda.OutOfMemoryError)
        assert '3 retries' in str(exc_info.value)

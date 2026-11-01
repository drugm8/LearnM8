"""Tests for MCDropoutLearner chunked prediction."""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch.nn as nn

from learnm8.exceptions import LearnerError
from learnm8.learners.torch.mc_dropout import MCDropoutLearner


def _make_data(n_samples=100, n_features=10, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = X[:, 0] * 2.0 + X[:, 1] * (-1.5) + rng.randn(n_samples) * 0.1
    return X, y


def _make_trained_learner(
    X,
    y,
    hidden_sizes=(32, 16),
    n_dropout_samples=20,
    batch_norm=False,
    predict_batch_size=None,
    seed=42,
):
    learner = MCDropoutLearner(
        hidden_sizes=hidden_sizes,
        n_dropout_samples=n_dropout_samples,
        max_epochs=5,
        batch_size=16,
        batch_norm=batch_norm,
        predict_batch_size=predict_batch_size,
        random_state=seed,
    )
    learner.train(X, y)
    return learner


class TestMCDropoutChunked:
    @pytest.mark.unit
    def test_chunked_predict_matches_full_predict(self):
        X, y = _make_data(100, 10)

        learner = _make_trained_learner(X, y)

        learner.predict_batch_size = 7
        preds_chunked, unc_chunked = learner.predict(X)

        learner.predict_batch_size = 200
        preds_full, unc_full = learner.predict(X)

        assert preds_chunked.shape == preds_full.shape == (100,)
        assert unc_chunked.shape == unc_full.shape == (100,)
        assert np.all(np.isfinite(preds_chunked))
        assert np.all(np.isfinite(preds_full))
        assert np.all(unc_chunked >= 0)
        assert np.all(unc_full >= 0)
        assert np.any(unc_chunked > 0)
        assert np.any(unc_full > 0)

    @pytest.mark.unit
    def test_predict_is_reproducible_across_calls(self):
        X, y = _make_data(100, 10)
        learner = _make_trained_learner(X, y)
        learner.predict_batch_size = 13

        preds_a, unc_a = learner.predict(X)
        preds_b, unc_b = learner.predict(X)

        # Repeated predict() on the same input gives identical mean and sigma.
        np.testing.assert_array_equal(preds_a, preds_b)
        np.testing.assert_array_equal(unc_a, unc_b)

    @pytest.mark.unit
    @pytest.mark.parametrize('chunk', [1, 7, 13, 64])
    def test_predictions_do_not_depend_on_chunk_size(self, chunk):
        X, y = _make_data(100, 10)
        learner = _make_trained_learner(X, y)

        learner.predict_batch_size = 200
        preds_full, unc_full = learner.predict(X)
        learner.predict_batch_size = chunk
        preds, unc = learner.predict(X)

        # Equal up to float32 matmul rounding, not Monte Carlo noise.
        np.testing.assert_allclose(preds, preds_full, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(unc, unc_full, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_predictions_do_not_depend_on_row_order_or_pool(self):
        X, y = _make_data(100, 10)
        learner = _make_trained_learner(X, y)
        preds_full, unc_full = learner.predict(X)

        perm = np.random.RandomState(0).permutation(100)
        preds_perm, unc_perm = learner.predict(X[perm])
        np.testing.assert_allclose(preds_perm, preds_full[perm], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(unc_perm, unc_full[perm], rtol=1e-4, atol=1e-6)

        # A compound predicted inside a smaller pool gets the same value.
        preds_sub, unc_sub = learner.predict(X[40:45])
        np.testing.assert_allclose(preds_sub, preds_full[40:45], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(unc_sub, unc_full[40:45], rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_different_seeds_give_different_masks(self):
        X, y = _make_data(100, 10)
        learner = _make_trained_learner(X, y)
        _, unc_a = learner.predict(X)
        learner.random_state = learner.random_state + 1
        _, unc_b = learner.predict(X)
        assert not np.allclose(unc_a, unc_b)

    @pytest.mark.unit
    def test_predict_does_not_perturb_global_rng(self):
        import torch

        X, y = _make_data(50, 10)
        learner = _make_trained_learner(X, y)

        torch.manual_seed(12345)
        before = torch.get_rng_state()
        learner.predict(X)
        after = torch.get_rng_state()

        assert torch.equal(before, after)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        'n_samples,predict_batch_size',
        [
            (50, 16),
            (50, 7),
            (100, 10),
            (100, 33),
            (23, 5),
        ],
    )
    def test_chunks_processed_count(self, n_samples, predict_batch_size):
        X, y = _make_data(n_samples, 10)
        learner = _make_trained_learner(X, y)

        expected_chunks = math.ceil(n_samples / predict_batch_size)
        learner.predict_batch_size = predict_batch_size

        with patch.object(
            learner,
            '_predict_chunk_with_oom_retry',
            wraps=learner._predict_chunk_with_oom_retry,
        ) as mock_predict:
            learner.predict(X)

        assert mock_predict.call_count == expected_chunks

    @pytest.mark.unit
    def test_predict_batch_size_auto_from_estimate(self):
        X, y = _make_data(100, 10)

        learner = _make_trained_learner(X, y, predict_batch_size=None)
        batch_size = learner._compute_predict_batch_size(100, 10)
        assert isinstance(batch_size, int)
        assert 0 < batch_size <= 100

        with patch('learnm8.core.batching.estimate_batch_size', return_value=50):
            batch_size = learner._compute_predict_batch_size(100, 10)
            assert batch_size == 50

    @pytest.mark.unit
    def test_predict_batch_size_real_estimate_is_used(self):
        """The unmocked estimate must run (it once always raised on a torch.device)."""
        X, y = _make_data(5000, 10)
        learner = _make_trained_learner(X, y, predict_batch_size=None)
        with patch(
            'learnm8.core.batching.get_available_memory', return_value=10**12
        ) as mem:
            batch_size = learner._compute_predict_batch_size(5000, 10)
        mem.assert_called_once_with(str(learner.device))
        assert batch_size == 5000 != 4 * learner.batch_size

    @pytest.mark.unit
    def test_predict_batch_size_auto_fallback_when_estimate_fails(self):
        X, y = _make_data(100, 10)

        learner = _make_trained_learner(X, y, predict_batch_size=None)

        with patch(
            'learnm8.core.batching.estimate_batch_size',
            side_effect=ImportError('no batching module'),
        ):
            batch_size = learner._compute_predict_batch_size(100, 10)

        assert isinstance(batch_size, int)
        assert 0 < batch_size <= 100
        assert batch_size == max(1, min(4 * learner.batch_size, 100))

    @pytest.mark.unit
    def test_predict_batch_size_user_supplied_overrides_estimate(self):
        X, y = _make_data(100, 10)

        learner = _make_trained_learner(X, y, predict_batch_size=23)
        batch_size = learner._compute_predict_batch_size(100, 10)
        assert batch_size == 23

    @pytest.mark.unit
    def test_batchnorm_in_eval_mode_during_predict(self):
        X, y = _make_data(100, 10)

        learner = _make_trained_learner(X, y, batch_norm=True)
        learner.predict_batch_size = 10

        bn_modes = []
        original = learner._predict_chunk_with_oom_retry

        def record_after(chunk, *args, **kwargs):
            result = original(chunk, *args, **kwargs)
            for m in learner.model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    bn_modes.append(m.training)
            return result

        with patch.object(
            learner, '_predict_chunk_with_oom_retry', side_effect=record_after
        ):
            learner.predict(X)

        assert len(bn_modes) > 0
        assert all(not training for training in bn_modes)

    @pytest.mark.unit
    def test_batchnorm_in_eval_mode_after_predict(self):
        X, y = _make_data(100, 10)

        learner = _make_trained_learner(X, y, batch_norm=True)
        learner.predict_batch_size = 10
        learner.predict(X)

        for m in learner.model.modules():
            if isinstance(m, nn.BatchNorm1d):
                assert not m.training

    @pytest.mark.unit
    def test_oom_retry_halves_and_succeeds(self):
        n_samples = 600
        X, y = _make_data(n_samples, 10)

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
        assert np.all(unc >= 0)

    @pytest.mark.unit
    def test_oom_retry_stops_at_max_retries(self):
        n_samples = 2048
        X, y = _make_data(n_samples, 10)

        learner = _make_trained_learner(X, y, predict_batch_size=n_samples)

        def side_effect(_X_chunk):
            raise RuntimeError('CUDA out of memory')

        with (
            patch.object(learner, '_predict_chunk', side_effect=side_effect),
            pytest.raises(LearnerError) as exc_info,
        ):
            learner.predict(X)

        assert '3 retries' in str(exc_info.value)

    @pytest.mark.unit
    def test_oom_retry_stops_at_effective_min(self):
        n_samples = 520
        X, y = _make_data(n_samples, 10)

        learner = _make_trained_learner(X, y, predict_batch_size=n_samples)

        def side_effect(_X_chunk):
            raise RuntimeError('CUDA out of memory')

        with (
            patch.object(learner, '_predict_chunk', side_effect=side_effect),
            pytest.raises(LearnerError) as exc_info,
        ):
            learner.predict(X)

        assert 'minimum size' in str(exc_info.value)

    @pytest.mark.unit
    def test_oom_retry_checks_count_before_min(self):
        n_samples = 2048
        X, y = _make_data(n_samples, 10)

        learner = _make_trained_learner(X, y, predict_batch_size=n_samples)

        def side_effect(_X_chunk):
            raise RuntimeError('CUDA out of memory')

        with (
            patch.object(learner, '_predict_chunk', side_effect=side_effect),
            pytest.raises(LearnerError) as exc_info,
        ):
            learner.predict(X)

        msg = str(exc_info.value)
        assert '3 retries' in msg
        assert 'minimum size' not in msg


class TestMaskGenerator:
    @pytest.mark.unit
    def test_mix32_matches_uint64_reference(self):
        import torch

        from learnm8.learners.torch.mc_dropout import _mix32

        def ref(x):
            m = np.uint64(0xFFFFFFFF)
            x = x ^ (x >> np.uint64(16))
            x = (x * np.uint64(0x7FEB352D)) & m
            x = x ^ (x >> np.uint64(15))
            x = (x * np.uint64(0x846CA68B)) & m
            return x ^ (x >> np.uint64(16))

        vals = np.array(
            [0, 1, 2, 12345, 0x7FFFFFFF, 0x80000000, 0xDEADBEEF, 0xFFFFFFFF],
            dtype=np.uint64,
        )
        as_i32 = torch.from_numpy(vals.astype(np.uint32).view(np.int32))
        got = _mix32(as_i32).numpy().view(np.uint32).astype(np.uint64)
        np.testing.assert_array_equal(got, ref(vals))

    @pytest.mark.unit
    @pytest.mark.parametrize('p', [0.1, 0.2, 0.5])
    def test_keep_rate_matches_dropout_rate(self, p):
        import torch

        from learnm8.learners.torch.mc_dropout import _dropout_mask, _row_keys

        X = np.random.RandomState(1).randn(2000, 10)
        lo, hi = (torch.from_numpy(k) for k in _row_keys(X))
        mask = _dropout_mask(lo, hi, seed=42, layer=0, sample=3, width=256, p=p)
        keep = (mask > 0).float().mean().item()
        assert abs(keep - (1 - p)) < 0.01
        # Kept units are rescaled by 1 / (1 - p), as in nn.Dropout.
        np.testing.assert_allclose(mask[mask > 0].numpy(), 1 / (1 - p), rtol=1e-6)

    @pytest.mark.unit
    def test_masks_identical_on_cpu_and_cuda(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')
        from learnm8.learners.torch.mc_dropout import _dropout_mask, _row_keys

        X = np.random.RandomState(2).randn(500, 10)
        lo, hi = (torch.from_numpy(k) for k in _row_keys(X))
        kw = dict(seed=7, layer=1, sample=42, width=128, p=0.2)
        cpu = _dropout_mask(lo, hi, **kw)
        gpu = _dropout_mask(lo.cuda(), hi.cuda(), **kw).cpu()
        assert torch.equal(cpu, gpu)

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from learnm8.core.batching import (
    FALLBACK_MEMORY_BYTES,
    MIN_BATCH_CPU,
    MIN_BATCH_GPU,
    _configure_cuda_allocator,
    _get_learner_device,
    estimate_batch_size,
    get_available_memory,
    predict_with_batching,
)
from learnm8.exceptions import LearnerError

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_allocator_flag():
    import learnm8.core.batching as batching_module
    original = batching_module._allocator_configured
    yield
    batching_module._allocator_configured = original


@pytest.fixture
def mock_learner():
    learner = MagicMock()
    learner.memory_profile.return_value = {
        "bytes_per_sample": 2048 * 8,
        "working_multiplier": 1.3,
        "fixed_overhead": 0,
    }
    learner.predict.return_value = (np.ones(100), None)
    learner.get_name.return_value = "MockLearner"
    learner.requires_smiles.return_value = False
    learner.supports_uncertainty.return_value = False
    del learner.device
    del learner._device
    del learner.accelerator
    return learner


@pytest.fixture
def pool_df():
    n = 200
    return pl.DataFrame(
        {
            "ID": [f"mol_{i}" for i in range(n)],
            "SMILES": ["CCO"] * n,
        }
    )


class TestGetAvailableMemory:
    def test_cpu_memory_via_psutil(self):
        mock_vm = MagicMock()
        mock_vm.available = 8 * 1024**3
        with patch("psutil.virtual_memory", return_value=mock_vm):
            result = get_available_memory("cpu")
        assert result == 8 * 1024**3

    def test_gpu_memory_via_torch(self):
        free_bytes = 6 * 1024**3
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(free_bytes, 8 * 1024**3)),
            patch("torch.cuda.empty_cache") as mock_empty,
        ):
            result = get_available_memory("cuda")
        mock_empty.assert_called_once()
        assert result == free_bytes

    def test_fallback_when_both_unavailable(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("psutil.virtual_memory", side_effect=Exception("no psutil")),
        ):
            result = get_available_memory("cpu")
        assert result == FALLBACK_MEMORY_BYTES

    def test_gpu_fallback_to_cpu(self):
        mock_vm = MagicMock()
        mock_vm.available = 4 * 1024**3
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", side_effect=RuntimeError("GPU error")),
            patch("psutil.virtual_memory", return_value=mock_vm),
        ):
            result = get_available_memory("cuda")
        assert result == 4 * 1024**3


class TestConfigureCudaAllocator:
    def test_sets_env_var_when_not_set(self):
        import learnm8.core.batching as batching_module

        batching_module._allocator_configured = False
        env_backup = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        try:
            _configure_cuda_allocator()
            assert "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        finally:
            if env_backup is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env_backup

    def test_idempotent_second_call(self):
        import learnm8.core.batching as batching_module

        batching_module._allocator_configured = False
        env_backup = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        try:
            _configure_cuda_allocator()
            first_val = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            _configure_cuda_allocator()
            second_val = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            assert first_val == second_val
        finally:
            if env_backup is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env_backup


class TestGetLearnerDevice:
    def test_device_attribute(self):
        learner = MagicMock()
        learner.device = "cuda"
        del learner._device
        del learner.accelerator
        assert _get_learner_device(learner) == "cuda"

    def test_underscore_device(self):
        import torch

        learner = MagicMock()
        del learner.device
        learner._device = torch.device("cuda")
        del learner.accelerator
        assert _get_learner_device(learner) == "cuda"

    def test_accelerator_gpu(self):
        learner = MagicMock()
        del learner.device
        del learner._device
        learner.accelerator = "gpu"
        assert _get_learner_device(learner) == "cuda"

    def test_default_cpu(self):
        learner = MagicMock()
        del learner.device
        del learner._device
        del learner.accelerator
        assert _get_learner_device(learner) == "cpu"


class TestEstimateBatchSize:
    def _make_learner(self, bytes_per_sample, working_multiplier, fixed_overhead=0, name="MockLearner"):
        learner = MagicMock()
        learner.memory_profile.return_value = {
            "bytes_per_sample": bytes_per_sample,
            "working_multiplier": working_multiplier,
            "fixed_overhead": fixed_overhead,
        }
        learner.get_name.return_value = name
        return learner

    def test_rf_learner_cpu_16gb(self):
        n_features = 2048
        learner = self._make_learner(n_features * 8, 1.3)
        available_memory = 16 * 1024**3
        with patch("learnm8.core.batching.get_available_memory", return_value=available_memory):
            result = estimate_batch_size(learner, n_samples=10**6, n_features=n_features, device="cpu", memory_safety_factor=0.7, n_jobs=1)
        cost = n_features * 8 * 1.3
        expected = int(available_memory * 0.7 / cost)
        assert result == expected

    def test_mlp_gpu_6gb(self):
        n_features = 2048
        learner = self._make_learner(n_features * 4, 3.0)
        free_bytes = 6 * 1024**3
        with patch("learnm8.core.batching.get_available_memory", return_value=free_bytes):
            result = estimate_batch_size(learner, n_samples=10**6, n_features=n_features, device="cuda", memory_safety_factor=0.7, n_jobs=1)
        from learnm8.core.batching import GPU_ALIGNMENT
        assert result % GPU_ALIGNMENT == 0

    def test_gp_n_train_5000(self):
        n_features = 2048
        n_train = 5000
        bytes_per_sample = n_features * 8 + n_train * 8 * 2
        learner = self._make_learner(bytes_per_sample, 1.0)
        available = 8 * 1024**3
        with patch("learnm8.core.batching.get_available_memory", return_value=available):
            result = estimate_batch_size(learner, n_samples=10**6, n_features=n_features, device="cpu", memory_safety_factor=0.7, n_jobs=1)
        expected = int(available * 0.7 / bytes_per_sample)
        assert result == expected

    def test_cpu_n_jobs_does_not_shrink_prediction_batch_budget(self):
        n_features = 512
        learner = self._make_learner(n_features * 8, 1.3)
        available = 8 * 1024**3
        with patch("learnm8.core.batching.get_available_memory", return_value=available):
            batch_1 = estimate_batch_size(learner, n_samples=10**6, n_features=n_features, device="cpu", memory_safety_factor=0.7, n_jobs=1)
            batch_8 = estimate_batch_size(learner, n_samples=10**6, n_features=n_features, device="cpu", memory_safety_factor=0.7, n_jobs=8)
        assert batch_1 == batch_8

    def test_n_jobs_minus_one_does_not_shrink_prediction_batch_budget(self):
        n_features = 512
        learner = self._make_learner(n_features * 8, 1.3)
        available = 8 * 1024**3
        cpu_count = 4
        with (
            patch("learnm8.core.batching.get_available_memory", return_value=available),
            patch("os.cpu_count", return_value=cpu_count),
        ):
            batch_minus1 = estimate_batch_size(learner, n_samples=10**6, n_features=n_features, device="cpu", memory_safety_factor=0.7, n_jobs=-1)
            batch_explicit = estimate_batch_size(learner, n_samples=10**6, n_features=n_features, device="cpu", memory_safety_factor=0.7, n_jobs=cpu_count)
        assert batch_minus1 == batch_explicit

    def test_80gb_32_threads_gp_profile_is_not_over_partitioned(self):
        n_features = 2048
        n_train = 5000
        bytes_per_sample = n_features * 8 + n_train * 8 * 2
        learner = self._make_learner(bytes_per_sample, 1.0)
        available = 80 * 1024**3

        with patch("learnm8.core.batching.get_available_memory", return_value=available):
            batch_size = estimate_batch_size(
                learner,
                n_samples=10**6,
                n_features=n_features,
                device="cpu",
                memory_safety_factor=0.7,
                n_jobs=32,
            )

        expected = int(available * 0.7 / bytes_per_sample)
        assert batch_size == expected
        assert batch_size > 500_000

    def test_gpu_alignment_to_32(self):
        n_features = 2048
        learner = self._make_learner(n_features * 4, 3.0)
        available = 6 * 1024**3
        with patch("learnm8.core.batching.get_available_memory", return_value=available):
            result = estimate_batch_size(learner, n_samples=10**6, n_features=n_features, device="cuda", memory_safety_factor=0.7, n_jobs=1)
        from learnm8.core.batching import GPU_ALIGNMENT
        assert result % GPU_ALIGNMENT == 0

    def test_min_batch_gpu(self):
        learner = self._make_learner(2048 * 4, 3.0)
        tiny_available = 2048 * 4 * 3.0 * 1.5
        with patch("learnm8.core.batching.get_available_memory", return_value=int(tiny_available)):
            result = estimate_batch_size(learner, n_samples=10**6, n_features=2048, device="cuda", memory_safety_factor=0.7, n_jobs=1)
        assert result >= MIN_BATCH_GPU

    def test_min_batch_cpu(self):
        learner = self._make_learner(2048 * 8, 1.3)
        tiny_available = 2048 * 8 * 1.3 * 1.5
        with patch("learnm8.core.batching.get_available_memory", return_value=int(tiny_available)):
            result = estimate_batch_size(learner, n_samples=10**6, n_features=2048, device="cpu", memory_safety_factor=0.7, n_jobs=1)
        assert result >= MIN_BATCH_CPU

    def test_caps_at_n_samples(self):
        n_features = 512
        learner = self._make_learner(n_features * 8, 1.3)
        available = 100 * 1024**3
        n_samples = 50
        with patch("learnm8.core.batching.get_available_memory", return_value=available):
            result = estimate_batch_size(learner, n_samples=n_samples, n_features=n_features, device="cpu", memory_safety_factor=0.7, n_jobs=1)
        assert result <= n_samples

    def test_insufficient_memory_raises(self):
        n_features = 2048
        learner = self._make_learner(n_features * 8, 1.3)
        with (
            patch("learnm8.core.batching.get_available_memory", return_value=1),
            pytest.raises(LearnerError, match="Insufficient memory"),
        ):
            estimate_batch_size(learner, n_samples=100, n_features=n_features, device="cpu", memory_safety_factor=0.7, n_jobs=1)

    def test_malformed_profile_raises(self):
        learner = MagicMock()
        learner.memory_profile.return_value = {
            "bytes_per_sample": 0,
            "working_multiplier": 1.3,
            "fixed_overhead": 0,
        }
        learner.get_name.return_value = "BadLearner"
        with pytest.raises(LearnerError, match="Invalid memory profile"):
            estimate_batch_size(learner, n_samples=100, n_features=2048, device="cpu")


class TestPredictWithBatching:
    def _make_learner(self, supports_uncertainty=False):
        learner = MagicMock()
        learner.memory_profile.return_value = {
            "bytes_per_sample": 2048 * 8,
            "working_multiplier": 1.3,
            "fixed_overhead": 0,
        }
        learner.get_name.return_value = "MockLearner"
        learner.requires_smiles.return_value = False
        learner.supports_uncertainty.return_value = supports_uncertainty
        del learner.device
        del learner._device
        del learner.accelerator
        return learner

    def _make_pool(self, n):
        return pl.DataFrame(
            {
                "ID": [f"mol_{i}" for i in range(n)],
                "SMILES": ["CCO"] * n,
            }
        )

    def test_single_batch_for_small_pool(self, tmp_path):
        n = 5
        pool = self._make_pool(n)
        learner = self._make_learner()
        learner.predict.return_value = (np.ones(n), None)
        fake_features = np.ones((n, 2048))
        large_memory = 100 * 1024**3

        with (
            patch("learnm8.core.batching.get_available_memory", return_value=large_memory),
            patch("learnm8.core.batching.extract_features", return_value=fake_features),
        ):
            preds, uncerts, ids, feat_time = predict_with_batching(pool, learner, "morgan", tmp_path)

        assert len(preds) == n
        assert uncerts is None
        assert len(ids) == n
        assert feat_time >= 0.0

    def test_multi_batch_concatenation(self, tmp_path):
        n = 50
        pool = self._make_pool(n)
        learner = self._make_learner()

        call_count = []

        def side_effect(features):
            call_count.append(1)
            batch_n = len(features)
            return np.arange(batch_n, dtype=float), None

        learner.predict.side_effect = side_effect

        small_memory = int(2048 * 8 * 1.3 * 12)

        with (
            patch("learnm8.core.batching.get_available_memory", return_value=small_memory),
            patch("learnm8.core.batching.extract_features", side_effect=lambda smiles, *a, **kw: np.ones((len(smiles), 2048))),
        ):
            preds, _uncerts, ids, _feat_time = predict_with_batching(pool, learner, "morgan", tmp_path)

        assert len(preds) == n
        assert len(ids) == n
        assert len(call_count) > 1

    def test_uncertainty_propagation(self, tmp_path):
        n = 10
        pool = self._make_pool(n)
        learner = self._make_learner(supports_uncertainty=True)
        learner.predict.return_value = (np.ones(n), np.zeros(n))
        large_memory = 100 * 1024**3

        with (
            patch("learnm8.core.batching.get_available_memory", return_value=large_memory),
            patch("learnm8.core.batching.extract_features", return_value=np.ones((n, 2048))),
        ):
            _preds, uncerts, _ids, _feat_time = predict_with_batching(pool, learner, "morgan", tmp_path)

        assert uncerts is not None
        assert len(uncerts) == n

    def test_oom_recovery_halves_batch(self, tmp_path):
        n = 4000
        pool = self._make_pool(n)
        learner = self._make_learner()

        call_count = [0]

        def predict_side_effect(features):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")
            batch_n = len(features)
            return np.ones(batch_n), None

        learner.predict.side_effect = predict_side_effect

        large_memory = 100 * 1024**3

        with (
            patch("learnm8.core.batching.get_available_memory", return_value=large_memory),
            patch("learnm8.core.batching.extract_features", side_effect=lambda smiles, *a, **kw: np.ones((len(smiles), 2048))),
        ):
            preds, _uncerts, _ids, _feat_time = predict_with_batching(pool, learner, "morgan", tmp_path)

        assert len(preds) == n
        assert call_count[0] > 1

    def test_oom_recovery_keeps_halving_subchunks_until_success(self, tmp_path):
        n = 4000
        pool = self._make_pool(n)
        learner = self._make_learner()
        call_sizes = []

        def predict_side_effect(features):
            batch_n = len(features)
            call_sizes.append(batch_n)
            if batch_n > MIN_BATCH_CPU:
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")
            return np.ones(batch_n), None

        learner.predict.side_effect = predict_side_effect

        large_memory = 100 * 1024**3

        with (
            patch("learnm8.core.batching.get_available_memory", return_value=large_memory),
            patch("learnm8.core.batching.extract_features", side_effect=lambda smiles, *a, **kw: np.ones((len(smiles), 2048))),
        ):
            preds, _uncerts, ids, _feat_time = predict_with_batching(pool, learner, "morgan", tmp_path)

        assert len(preds) == n
        assert len(ids) == n
        assert max(call_sizes) == n
        assert call_sizes.count(MIN_BATCH_CPU) == 4

    def test_oom_at_min_batch_raises(self, tmp_path):
        n = 5
        pool = self._make_pool(n)
        learner = self._make_learner()
        learner.predict.side_effect = MemoryError("out of memory")
        large_memory = 100 * 1024**3

        with (
            patch("learnm8.core.batching.get_available_memory", return_value=large_memory),
            patch("learnm8.core.batching.extract_features", return_value=np.ones((n, 2048))),
            pytest.raises(LearnerError, match="OOM at minimum batch size"),
        ):
            predict_with_batching(pool, learner, "morgan", tmp_path)

    def test_info_logging_emitted(self, tmp_path):
        n = 10
        pool = self._make_pool(n)
        learner = self._make_learner()
        learner.predict.return_value = (np.ones(n), None)
        large_memory = 100 * 1024**3

        with (
            patch("learnm8.core.batching.get_available_memory", return_value=large_memory),
            patch("learnm8.core.batching.extract_features", return_value=np.ones((n, 2048))),
            patch("learnm8.core.batching.logger") as mock_logger,
        ):
            predict_with_batching(pool, learner, "morgan", tmp_path)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Batched prediction" in call_args

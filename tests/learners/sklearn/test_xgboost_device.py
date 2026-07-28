"""Tests for XGBoostLearner device parameter and aggressive GC behavior."""

import gc
from unittest.mock import patch

import numpy as np
import pytest

from learnm8.exceptions import LearnerError
from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner

pytestmark = pytest.mark.unit


@pytest.fixture
def features():
    rng = np.random.default_rng(0)
    X = rng.random((30, 10)).astype(np.float64)
    X[:, 2] = 1.0  # one zero-variance col to exercise mask logic
    return X


@pytest.fixture
def targets():
    rng = np.random.default_rng(1)
    return rng.random(30).astype(np.float64)


class TestXGBoostDeviceCpu:
    def test_device_cpu_preserves_existing_behavior(self):
        learner_default = XGBoostLearner()
        learner_explicit = XGBoostLearner(device='cpu')
        assert learner_default.device == 'cpu'
        assert learner_explicit.device == 'cpu'
        assert learner_default.model.get_params()['device'] == 'cpu'
        assert learner_explicit.model.get_params()['device'] == 'cpu'

    def test_device_cpu_tree_method_hist(self):
        learner = XGBoostLearner(device='cpu')
        assert learner.model.get_params()['tree_method'] == 'hist'


class TestXGBoostDeviceCuda:
    def test_device_cuda_passes_to_xgbregressor(self):
        learner = XGBoostLearner(device='cuda')
        assert learner.model.get_params()['device'] == 'cuda'

    def test_device_cuda_sets_tree_method_hist(self):
        learner = XGBoostLearner(device='cuda')
        assert learner.model.get_params()['tree_method'] == 'hist'

    def test_no_construction_time_cuda_probe(self):
        # Must not raise even without CUDA available
        try:
            learner = XGBoostLearner(device='cuda')
        except Exception as exc:
            pytest.fail(f'XGBoostLearner(device="cuda") raised at construction: {exc}')
        assert learner.device == 'cuda'

    def test_cuda_failure_at_train_raises_learner_error(self, features, targets):
        learner = XGBoostLearner(device='cuda', n_estimators=5)
        cuda_err = RuntimeError('CUDA error: no kernel image is available for execution on the device')
        with patch.object(learner.model, 'fit', side_effect=cuda_err), \
                pytest.raises(LearnerError, match='XGBoost CUDA training failed'):
            learner.train(features, targets)

    def test_no_silent_cpu_substitution_on_cuda_failure(self, features, targets):
        learner = XGBoostLearner(device='cuda', n_estimators=5)
        cuda_err = RuntimeError('CUDA: device not available')
        with patch.object(learner.model, 'fit', side_effect=cuda_err):
            with pytest.raises(LearnerError):
                learner.train(features, targets)
            # Must not have silently completed training
            assert not learner.is_trained


class TestXGBoostDeviceAuto:
    """`--device auto` (the CLI default) used to reach XGBoost verbatim.

    XGBoost accepts only 'cpu' / 'cuda' / 'cuda:N', so every CPU run launched
    with default flags died at train time. Resolution happens in __init__,
    matching how fastprop/svgp/gpu_gp resolve 'auto' themselves.
    """

    def test_auto_resolves_to_concrete_device(self):
        learner = XGBoostLearner(device='auto')
        assert learner.device in ('cpu', 'cuda')
        assert learner.model.get_params()['device'] == learner.device

    def test_auto_resolves_to_cpu_without_cuda(self):
        with patch('torch.cuda.is_available', return_value=False):
            learner = XGBoostLearner(device='auto')
        assert learner.device == 'cpu'
        assert learner.model.get_params()['device'] == 'cpu'

    def test_auto_resolves_to_cuda_when_available(self):
        with patch('torch.cuda.is_available', return_value=True):
            learner = XGBoostLearner(device='auto')
        assert learner.device == 'cuda'
        assert learner.model.get_params()['device'] == 'cuda'

    def test_auto_trains_without_device_error(self, features, targets):
        with patch('torch.cuda.is_available', return_value=False):
            learner = XGBoostLearner(device='auto', n_estimators=5)
        learner.train(features, targets)
        assert learner.is_trained


class TestXGBoostAggressiveGc:
    def test_enable_aggressive_gc_default_true(self):
        learner = XGBoostLearner()
        assert learner.enable_aggressive_gc is True

    @pytest.mark.slow
    def test_enable_aggressive_gc_deletes_old_model_before_refit(self, features, targets):
        learner = XGBoostLearner(n_estimators=5, enable_aggressive_gc=True)
        learner.train(features, targets)
        assert learner.is_trained

        gc_calls = []
        original_gc = gc.collect

        def tracking_gc():
            gc_calls.append(1)
            return original_gc()

        with patch('gc.collect', side_effect=tracking_gc):
            learner.train(features, targets)

        assert len(gc_calls) > 0, 'gc.collect was not called during refit'

    @pytest.mark.slow
    def test_enable_aggressive_gc_false_skips_gc(self, features, targets):
        learner = XGBoostLearner(n_estimators=5, enable_aggressive_gc=False)
        learner.train(features, targets)

        with patch('gc.collect') as mock_gc:
            learner.train(features, targets)
            mock_gc.assert_not_called()

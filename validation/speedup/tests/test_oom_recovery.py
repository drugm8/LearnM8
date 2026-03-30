from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    X_train = rng.random((500, 30)).astype(np.float32)
    y_train = rng.random(500).astype(np.float32)
    X_test = rng.random((100, 30)).astype(np.float32)
    return X_train, y_train, X_test


class TestRfFilOomRecovery:
    def test_oom_triggers_chunked_fallback(self, synthetic_data):
        from learnm8.learners.base import _predict_with_oom_retry
        X_train, y_train, X_test = synthetic_data

        call_count = [0]

        def mock_predict_fn(chunk):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MemoryError('CUDA out of memory')
            return np.ones((len(chunk), 10))

        result = _predict_with_oom_retry(X_test, mock_predict_fn, n_output_cols=10, min_batch=1)
        assert result.shape[0] == len(X_test)
        assert result.shape[1] == 10
        assert call_count[0] > 1, 'should have retried after OOM'

    def test_oom_result_finite(self, synthetic_data):
        from learnm8.learners.base import _predict_with_oom_retry
        _, _, X_test = synthetic_data

        call_count = [0]

        def mock_predict_fn(chunk):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MemoryError('std::bad_alloc CUDA out of memory')
            rng = np.random.default_rng(call_count[0])
            return rng.random((len(chunk), 5))

        result = _predict_with_oom_retry(X_test, mock_predict_fn, n_output_cols=5, min_batch=1)
        assert np.all(np.isfinite(result))

    def test_non_oom_error_propagates(self, synthetic_data):
        from learnm8.learners.base import _predict_with_oom_retry
        from learnm8.exceptions import LearnerError
        _, _, X_test = synthetic_data

        def mock_predict_fn(chunk):
            raise RuntimeError('something else went wrong')

        with pytest.raises((RuntimeError, LearnerError)):
            _predict_with_oom_retry(X_test, mock_predict_fn, n_output_cols=5, min_batch=1)

    def test_success_on_first_call_no_retry(self, synthetic_data):
        from learnm8.learners.base import _predict_with_oom_retry
        _, _, X_test = synthetic_data

        call_count = [0]

        def mock_predict_fn(chunk):
            call_count[0] += 1
            return np.ones((len(chunk), 8))

        result = _predict_with_oom_retry(X_test, mock_predict_fn, n_output_cols=8, min_batch=1)
        assert result.shape == (len(X_test), 8)
        assert call_count[0] == 1, 'should not retry if first call succeeds'


class TestRidgeCumlOomFallback:
    def test_leverage_fallback_on_oom(self):
        cuml = pytest.importorskip('cuml', reason='RAPIDS cuML not available')

        from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
        rng = np.random.default_rng(42)
        X_train = rng.random((300, 20))
        y_train = X_train[:, 0] * 2.0 + rng.normal(0, 0.1, 300)
        X_test = rng.random((50, 20))

        learner = RidgeCumlLearner(alpha=1.0, random_state=42)
        learner.train(X_train, y_train)

        with patch('learnm8.learners.gpu.ridge_cuml.cho_solve') as mock_cho_solve:
            call_count = [0]
            original_result = None

            import scipy.linalg
            real_cho_solve = scipy.linalg.cho_solve

            def fake_cho_solve(c_and_lower, b, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception('out of memory (simulated GPU OOM)')
                return real_cho_solve(c_and_lower, b, **kwargs)

            mock_cho_solve.side_effect = fake_cho_solve

            preds, unc = learner.predict(X_test)
            assert preds.shape == (50,)
            assert np.all(np.isfinite(preds))
            assert unc is not None

    def test_non_oom_leverage_error_raises(self):
        cuml = pytest.importorskip('cuml', reason='RAPIDS cuML not available')

        from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
        from learnm8.exceptions import LearnerError
        rng = np.random.default_rng(42)
        X_train = rng.random((200, 15))
        y_train = rng.random(200)
        X_test = rng.random((30, 15))

        learner = RidgeCumlLearner(alpha=1.0, random_state=42)
        learner.train(X_train, y_train)

        with patch('learnm8.learners.gpu.ridge_cuml.cho_solve') as mock_cho_solve:
            mock_cho_solve.side_effect = ValueError('singular matrix')
            with pytest.raises(LearnerError):
                learner.predict(X_test)


class TestXGBCudaOomRecovery:
    def test_cuda_oom_wrapped_as_learner_error(self):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip('CUDA GPU not available')
        except ImportError:
            pytest.skip('torch not available')

        from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
        from learnm8.exceptions import LearnerError

        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)
        y = rng.random(100)

        learner = XGBoostLearner(n_estimators=5, device='cuda')

        with patch.object(learner, 'model') as mock_model:
            mock_model.fit.side_effect = MemoryError('CUDA out of memory')
            with pytest.raises((LearnerError, MemoryError)):
                learner.train(X, y)

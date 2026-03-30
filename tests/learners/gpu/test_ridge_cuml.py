"""Unit tests for RidgeCumlLearner — CPU-mockable (no GPU required)."""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.linalg import cho_factor, solve_triangular
from sklearn.linear_model import Ridge as SkRidge

from learnm8.core.interfaces import Learner
from learnm8.exceptions import LearnerError
from learnm8.learners.base import SklearnLearner, _preprocess_features
from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(42)


def _make_data(n: int = 50, p: int = 10) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.standard_normal((n, p)).astype(np.float32)
    y = (X @ RNG.standard_normal(p) + 0.1 * RNG.standard_normal(n)).astype(np.float32)
    return X, y


def _make_cuml_mock(n_train: int):
    """Return a (mock_cuml_module, mock_ridge_instance) pair."""
    mock_instance = MagicMock()
    mock_instance.fit.return_value = mock_instance
    mock_instance.predict.return_value = np.zeros(n_train)
    mock_cuml = MagicMock()
    mock_cuml.linear_model.Ridge = MagicMock(return_value=mock_instance)
    return mock_cuml, mock_instance


def _build_trained(X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> RidgeCumlLearner:
    """Build RidgeCumlLearner with internal state set directly (no cuml needed)."""
    learner = RidgeCumlLearner(alpha=alpha)
    features_pp, mask = _preprocess_features(X, remove_zero_variance=True, is_training=True)
    n, p = features_pp.shape
    Xd = features_pp.astype(np.float64)
    gram = Xd.T @ Xd + alpha * np.eye(p, dtype=np.float64)

    sk_model = SkRidge(alpha=alpha, fit_intercept=True).fit(features_pp, y)
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda f: sk_model.predict(f)

    learner._valid_feature_mask = mask
    learner._model = mock_model
    learner._gram_chol_cpu = cho_factor(gram)
    preds = sk_model.predict(features_pp).astype(np.float64)
    residuals = y.astype(np.float64) - preds
    learner._sigma_hat_sq = float(np.sum(residuals ** 2)) / max(n - p, 1)
    learner.is_trained = True
    return learner


class TestConstruction:
    def test_construction_with_valid_parameters(self):
        learner = RidgeCumlLearner(alpha=0.5, fit_intercept=False, solver='svd', random_state=0)
        assert learner.alpha == 0.5
        assert learner.fit_intercept is False
        assert learner.solver == 'svd'
        assert learner.random_state == 0
        assert learner.is_trained is False
        assert learner._model is None
        assert learner._gram_chol_cpu is None
        assert learner._sigma_hat_sq is None
        assert learner._valid_feature_mask is None

    def test_inherits_learner_abc_not_sklearn_learner(self):
        learner = RidgeCumlLearner()
        assert isinstance(learner, Learner)
        assert not isinstance(learner, SklearnLearner)

    def test_alpha_none_raises_error(self):
        with pytest.raises(LearnerError, match='alpha'):
            RidgeCumlLearner(alpha=None)

    def test_get_name_returns_ridge_cuml(self):
        assert RidgeCumlLearner().get_name() == 'ridge_cuml'

    def test_supports_uncertainty_returns_true(self):
        assert RidgeCumlLearner().supports_uncertainty() is True


class TestMissingCuml:
    def test_missing_cuml_raises_learner_error_with_message(self):
        X, y = _make_data()
        learner = RidgeCumlLearner(alpha=0.1)
        with patch.dict('sys.modules', {'cuml': None, 'cuml.linear_model': None}), \
             pytest.raises(LearnerError, match='cuml'):
            learner.train(X, y)


class TestTrain:
    def test_train_fits_cuml_ridge(self):
        X, y = _make_data()
        mock_cuml, mock_instance = _make_cuml_mock(len(y))
        with patch.dict('sys.modules', {'cuml': mock_cuml, 'cuml.linear_model': mock_cuml.linear_model}):
            learner = RidgeCumlLearner(alpha=0.1)
            learner.train(X, y)
        mock_instance.fit.assert_called_once()
        assert learner.is_trained is True

    def test_train_computes_gram_cholesky_in_float64(self):
        X, y = _make_data()
        mock_cuml, _mock_instance = _make_cuml_mock(len(y))
        with patch.dict('sys.modules', {'cuml': mock_cuml, 'cuml.linear_model': mock_cuml.linear_model}):
            learner = RidgeCumlLearner(alpha=0.1)
            learner.train(X, y)
        assert learner._gram_chol_cpu is not None
        L, _lower = learner._gram_chol_cpu
        assert L.dtype == np.float64

    def test_empty_input_raises_error(self):
        learner = RidgeCumlLearner(alpha=0.1)
        with pytest.raises(LearnerError):
            learner.train(np.empty((0, 5)), np.empty(0))

    def test_gram_matrix_high_condition_number_warns(self):
        rng = np.random.default_rng(0)
        n = 50
        base = rng.standard_normal(n)
        X = np.column_stack([base + 1e-8 * rng.standard_normal(n) for _ in range(8)])
        y = rng.standard_normal(n)
        mock_cuml, _ = _make_cuml_mock(n)

        with patch.dict('sys.modules', {'cuml': mock_cuml, 'cuml.linear_model': mock_cuml.linear_model}):
            learner = RidgeCumlLearner(alpha=1e-12)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                learner.train(X, y)

        cond_warns = [w for w in caught if 'condition' in str(w.message).lower() or 'Gram' in str(w.message)]
        assert len(cond_warns) > 0


class TestPredict:
    def test_predict_before_train_raises_learner_error(self):
        learner = RidgeCumlLearner(alpha=0.1)
        X = RNG.standard_normal((5, 3)).astype(np.float32)
        with pytest.raises(LearnerError, match='trained'):
            learner.predict(X)

    def test_predict_returns_predictions_and_uncertainty(self):
        X, y = _make_data()
        learner = _build_trained(X, y)
        preds, unc = learner.predict(X)
        assert preds is not None
        assert unc is not None

    def test_predict_output_shapes_match_input_length(self):
        X, y = _make_data(n=50, p=8)
        learner = _build_trained(X, y)
        preds, unc = learner.predict(X)
        assert preds.shape == (50,)
        assert unc.shape == (50,)

    def test_uncertainty_is_finite_and_non_negative(self):
        X, y = _make_data()
        learner = _build_trained(X, y)
        _, unc = learner.predict(X)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)

    def test_uncertainty_is_deterministic_for_fixed_input(self):
        X, y = _make_data()
        learner = _build_trained(X, y)
        _, unc1 = learner.predict(X)
        _, unc2 = learner.predict(X)
        np.testing.assert_array_equal(unc1, unc2)

    def test_uncertainty_formula_matches_cpu_lr(self):
        """RidgeCumlLearner uncertainty must agree with LinearRegressionLearner."""
        X, y = _make_data(n=80, p=12)
        alpha = 0.1

        cpu_lr = LinearRegressionLearner(alpha=alpha, random_state=42)
        cpu_lr.train(X.astype(np.float64), y.astype(np.float64))
        _, cpu_unc = cpu_lr.predict(X.astype(np.float64))

        gpu_learner = _build_trained(X, y, alpha=alpha)
        _, gpu_unc = gpu_learner.predict(X)

        np.testing.assert_allclose(gpu_unc, cpu_unc, rtol=1e-4, atol=1e-6)

    def test_cpu_leverage_oom_raises_learner_error(self):
        """CPU-only leverage OOM raises LearnerError (no redundant retry)."""
        X, y = _make_data(n=30, p=6)
        learner = _build_trained(X, y)

        with patch('learnm8.learners.gpu.ridge_cuml.cho_solve', side_effect=MemoryError('out of memory')), \
             pytest.raises(LearnerError):
            learner.predict(X)


class TestCuPyGPULeverage:
    """Tests for CuPy GPU leverage computation (T045)."""

    @staticmethod
    def _gpu_chol_from_cpu(learner):
        """Derive lower Cholesky factor for GPU from CPU cho_factor state."""
        c, lower = learner._gram_chol_cpu
        return np.tril(c) if lower else np.triu(c).T

    @staticmethod
    def _mock_cupy_modules(solve_tri_effect=None):
        """Build mock cupy/cupyx sys.modules for GPU leverage tests."""
        if solve_tri_effect is None:
            def solve_tri_effect(L, b, lower=True):
                return solve_triangular(np.asarray(L), np.asarray(b), lower=lower)

        mock_cp = MagicMock()
        mock_cp.asarray.side_effect = lambda x, **kw: np.asarray(x)
        mock_cp.asnumpy.side_effect = lambda x: np.asarray(x)
        mock_cp.einsum.side_effect = lambda *a, **kw: np.einsum(*a, **kw)

        mock_solve_tri = MagicMock(side_effect=solve_tri_effect)

        mock_cupyx_linalg = MagicMock()
        mock_cupyx_linalg.solve_triangular = mock_solve_tri

        modules = {
            'cupy': mock_cp,
            'cupy.linalg': mock_cp.linalg,
            'cupyx': MagicMock(),
            'cupyx.scipy': MagicMock(),
            'cupyx.scipy.linalg': mock_cupyx_linalg,
        }
        return mock_solve_tri, modules

    def test_leverage_uses_cupy_gpu_when_available(self):
        """predict() uses GPU solve_triangular when _gram_chol_gpu is set."""
        X, y = _make_data(n=30, p=6)
        learner = _build_trained(X, y)
        learner._gram_chol_gpu = self._gpu_chol_from_cpu(learner)
        learner._gram_chol_cpu = None

        mock_solve_tri, modules = self._mock_cupy_modules()

        with patch.dict('sys.modules', modules):
            _preds, unc = learner.predict(X)

        mock_solve_tri.assert_called()
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0)

    def test_leverage_falls_back_to_cpu_on_oom(self):
        """predict() falls back to CPU when GPU leverage raises OOM."""
        X, y = _make_data(n=30, p=6)
        learner = _build_trained(X, y)
        learner._gram_chol_gpu = self._gpu_chol_from_cpu(learner)

        mock_solve_tri, modules = self._mock_cupy_modules(
            solve_tri_effect=MemoryError('GPU out of memory'),
        )

        with patch.dict('sys.modules', modules):
            _preds, unc = learner.predict(X)

        mock_solve_tri.assert_called()
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0)

    def test_gpu_and_cpu_leverage_produce_same_result(self):
        """GPU and CPU leverage produce numerically identical uncertainty."""
        X, y = _make_data(n=50, p=8)

        learner_cpu = _build_trained(X, y)
        _, unc_cpu = learner_cpu.predict(X)

        learner_gpu = _build_trained(X, y)
        learner_gpu._gram_chol_gpu = self._gpu_chol_from_cpu(learner_gpu)
        learner_gpu._gram_chol_cpu = None

        _, modules = self._mock_cupy_modules()

        with patch.dict('sys.modules', modules):
            _, unc_gpu = learner_gpu.predict(X)

        np.testing.assert_allclose(unc_gpu, unc_cpu, atol=1e-10)

    def test_chunked_cpu_dtrsm_fallback_for_large_n(self):
        """CPU fallback chunks leverage for n > chunk threshold."""
        X, y = _make_data(n=50, p=6)
        learner = _build_trained(X, y)

        _, unc_ref = learner.predict(X)

        with patch('learnm8.learners.gpu.ridge_cuml._LEVERAGE_CHUNK_SIZE', 10):
            _, unc_chunked = learner.predict(X)

        np.testing.assert_allclose(unc_chunked, unc_ref, atol=1e-10)

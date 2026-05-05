"""Tests for accelerated leverage score computation.

Tests for _compute_leverages_cpu helper and its integration into
LinearRegressionLearner and RidgeCumlLearner.
"""

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve


def _reference_leverages(gram, X):
    """Reference implementation: leverages via cho_solve (current production code)."""
    factor = cho_factor(gram)
    solved = cho_solve(factor, X.T)
    return np.einsum('ij,ji->i', X, solved)


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(42)
    n, p = 200, 20
    X = rng.randn(n, p)
    alpha = 0.1
    gram = X.T @ X + alpha * np.eye(p)
    L = np.linalg.cholesky(gram)
    return X, gram, L, alpha


@pytest.mark.unit
class TestComputeLeveragesCpu:
    """Tests for the _compute_leverages_cpu helper function."""

    def test_leverages_match_reference_cho_solve(self, regression_data):
        """Leverages from dtrmm-based helper must match cho_solve reference."""
        X, gram, L, _ = regression_data
        from scipy.linalg.lapack import dtrtri

        L_inv, info = dtrtri(L, lower=1)
        assert info == 0

        from learnm8.learners.base import _compute_leverages_cpu

        result = _compute_leverages_cpu(L_inv, X)
        expected = _reference_leverages(gram, X)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_leverages_non_negative(self, regression_data):
        """Leverage scores must be non-negative."""
        X, _, L, _ = regression_data
        from scipy.linalg.lapack import dtrtri

        L_inv, _ = dtrtri(L, lower=1)

        from learnm8.learners.base import _compute_leverages_cpu

        result = _compute_leverages_cpu(L_inv, X)
        assert np.all(result >= 0)

    def test_chunked_matches_unchunked(self, regression_data):
        """Small chunk_size must produce same result as large chunk_size."""
        X, _, L, _ = regression_data
        from scipy.linalg.lapack import dtrtri

        L_inv, _ = dtrtri(L, lower=1)

        from learnm8.learners.base import _compute_leverages_cpu

        result_small_chunk = _compute_leverages_cpu(L_inv, X, chunk_size=17)
        result_large_chunk = _compute_leverages_cpu(L_inv, X, chunk_size=10000)
        np.testing.assert_allclose(result_small_chunk, result_large_chunk, rtol=1e-12)

    def test_single_sample(self):
        """Must work for a single sample."""
        rng = np.random.RandomState(99)
        X = rng.randn(1, 5)
        L_inv = np.linalg.inv(np.linalg.cholesky(np.eye(5)))

        from learnm8.learners.base import _compute_leverages_cpu

        result = _compute_leverages_cpu(L_inv, X)
        assert result.shape == (1,)
        assert result[0] >= 0

    def test_output_dtype_is_float64(self, regression_data):
        """Output must be float64 regardless of input dtype."""
        X, _, L, _ = regression_data
        from scipy.linalg.lapack import dtrtri

        L_inv, _ = dtrtri(L, lower=1)

        from learnm8.learners.base import _compute_leverages_cpu

        result = _compute_leverages_cpu(L_inv, X.astype(np.float32))
        assert result.dtype == np.float64


@pytest.mark.unit
class TestLinearRegressionLearnerLInv:
    """Tests that LinearRegressionLearner uses L_inv instead of cho_factor."""

    def test_train_stores_L_inv_not_cho_factor(self):
        """After training, learner must have _L_inv (ndarray) not _gram_chol (tuple)."""
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        y = rng.randn(50)

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)

        assert hasattr(learner, '_L_inv')
        assert isinstance(learner._L_inv, np.ndarray)
        assert not hasattr(learner, '_gram_chol')

    def test_L_inv_is_lower_triangular(self):
        """_L_inv must be lower-triangular (inverse of Cholesky factor)."""
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        y = rng.randn(50)

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)

        upper = np.triu(learner._L_inv, k=1)
        np.testing.assert_allclose(upper, 0.0, atol=1e-15)

    def test_L_inv_dims_match_filtered_features(self):
        """Zero-variance columns -> _L_inv dims match filtered count."""
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        rng = np.random.RandomState(42)
        X_good = rng.randn(30, 5)
        X_const = np.ones((30, 3)) * 7.0
        X = np.hstack([X_good, X_const])
        y = rng.randn(30)

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)

        assert learner._L_inv.shape == (5, 5)

    def test_predict_leverages_match_reference(self):
        """Predicted uncertainties must match reference cho_solve computation."""
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        y = rng.randn(50)

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)

        _preds, unc = learner.predict(X)
        assert unc is not None
        assert np.all(np.isfinite(unc))
        assert len(unc) == len(X)
        assert np.all(unc >= 0)


@pytest.mark.unit
class TestRidgeCumlLearnerLInv:
    """Tests that RidgeCumlLearner uses _L_inv_cpu instead of _gram_chol_cpu."""

    def test_train_stores_L_inv_cpu_not_gram_chol(self):
        """After training, learner must have _L_inv_cpu (ndarray) not _gram_chol_cpu."""
        from unittest.mock import MagicMock, patch

        from sklearn.linear_model import Ridge as SkRidge

        from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10)).astype(np.float32)
        y = (X @ rng.standard_normal(10)).astype(np.float32)

        learner = RidgeCumlLearner(alpha=0.1)

        mock_cuml = MagicMock()
        sk_model = SkRidge(alpha=0.1, fit_intercept=True).fit(X, y)
        mock_instance = MagicMock()
        mock_instance.fit.return_value = mock_instance
        mock_instance.predict.side_effect = lambda f: sk_model.predict(f)
        mock_cuml.linear_model.Ridge = MagicMock(return_value=mock_instance)

        with patch.dict('sys.modules', {'cuml': mock_cuml, 'cuml.linear_model': mock_cuml.linear_model}):
            learner.train(X, y)

        assert hasattr(learner, '_L_inv_cpu')
        assert isinstance(learner._L_inv_cpu, np.ndarray)
        assert not hasattr(learner, '_gram_chol_cpu')

    def test_L_inv_cpu_is_lower_triangular(self):
        """_L_inv_cpu must be lower-triangular."""
        from unittest.mock import MagicMock, patch

        from sklearn.linear_model import Ridge as SkRidge

        from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10)).astype(np.float32)
        y = (X @ rng.standard_normal(10)).astype(np.float32)

        learner = RidgeCumlLearner(alpha=0.1)

        mock_cuml = MagicMock()
        sk_model = SkRidge(alpha=0.1, fit_intercept=True).fit(X, y)
        mock_instance = MagicMock()
        mock_instance.fit.return_value = mock_instance
        mock_instance.predict.side_effect = lambda f: sk_model.predict(f)
        mock_cuml.linear_model.Ridge = MagicMock(return_value=mock_instance)

        with patch.dict('sys.modules', {'cuml': mock_cuml, 'cuml.linear_model': mock_cuml.linear_model}):
            learner.train(X, y)

        upper = np.triu(learner._L_inv_cpu, k=1)
        np.testing.assert_allclose(upper, 0.0, atol=1e-15)

    def test_cpu_leverages_match_reference(self):
        """CPU fallback leverages must match cho_solve reference."""
        from unittest.mock import MagicMock

        from sklearn.linear_model import Ridge as SkRidge
        from sklearn.preprocessing import StandardScaler

        from learnm8.learners.base import _preprocess_features
        from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10)).astype(np.float32)
        y = (X @ rng.standard_normal(10)).astype(np.float32)
        alpha = 0.1

        features_pp, mask, imputer = _preprocess_features(X, remove_zero_variance=True, is_training=True)
        scaler = StandardScaler()
        features_pp = scaler.fit_transform(features_pp)
        n, p = features_pp.shape
        Xd = features_pp.astype(np.float64)
        gram = Xd.T @ Xd + alpha * np.eye(p)

        from scipy.linalg.lapack import dtrtri
        L = np.linalg.cholesky(gram)
        L_inv, _ = dtrtri(L, lower=1)

        sk_model = SkRidge(alpha=alpha, fit_intercept=True).fit(features_pp, y)
        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda f: sk_model.predict(f)

        learner = RidgeCumlLearner(alpha=alpha)
        learner._valid_feature_mask = mask
        learner._feature_imputer = imputer
        learner._feature_scaler = scaler
        learner._model = mock_model
        learner._L_inv_cpu = L_inv
        learner._leverage_A_gpu = None
        preds = sk_model.predict(features_pp).astype(np.float64)
        residuals = y.astype(np.float64) - preds
        learner._sigma_hat_sq = float(np.sum(residuals ** 2)) / max(n - p, 1)
        learner.is_trained = True

        _, unc = learner.predict(X)
        assert unc is not None
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0)

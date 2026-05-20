"""Tests for LinearRegressionLearner implementation."""

import numpy as np
import polars as pl
import pytest

from learnm8.exceptions import LearnerError
from learnm8.features.extraction import extract_features
from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.molecular
class TestLinearRegressionLearner:
    """Test LinearRegressionLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create LinearRegressionLearner instance for testing."""
        return LinearRegressionLearner(alpha=None, random_state=42)

    @pytest.fixture
    def ridge_learner(self):
        """Create Ridge regression learner instance for testing."""
        return LinearRegressionLearner(alpha=1.0, random_state=42)

    @pytest.fixture
    def compounds_20(self, small_real_compounds):
        return small_real_compounds.head(20)

    @pytest.fixture
    def features_20(self, small_real_morgan_features):
        return small_real_morgan_features[:20]

    def test_initialization_sets_linear_mode_defaults_and_uncertainty_support(self, learner):
        """Test learner initialization with default Linear Regression mode."""
        assert learner.alpha is None
        assert learner.fit_intercept is True
        assert learner.n_jobs == -1
        assert learner.random_state == 42
        assert not learner.is_trained
        assert learner.supports_uncertainty() is True
        assert learner.is_ridge is False

    def test_initialization_ridge_mode(self, ridge_learner):
        """Test learner initialization with Ridge mode."""
        assert ridge_learner.alpha == 1.0
        assert ridge_learner.fit_intercept is True
        assert ridge_learner.n_jobs is None
        assert ridge_learner.random_state == 42
        assert not ridge_learner.is_trained
        assert ridge_learner.supports_uncertainty() is True
        assert ridge_learner.is_ridge is True

    def test_predict_returns_finite_values_and_uncertainty_after_training(self, learner, compounds_20, features_20):
        """Test training and prediction with real molecular data."""
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = features_20
        targets = compounds['Activity'].to_numpy()

        learner.train(features, targets)
        assert learner.is_trained

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape == predictions.shape
        assert np.all(uncertainty >= 0)
        assert np.all(np.isfinite(predictions))

    def test_ridge_vs_linear_mode(self, compounds_20, features_20):
        """Test both Ridge and Linear modes produce valid predictions."""
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = features_20
        targets = compounds['Activity'].to_numpy()

        linear_learner = LinearRegressionLearner(alpha=None, random_state=42)
        linear_learner.train(features, targets)
        linear_predictions, _ = linear_learner.predict(features)

        ridge_learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        ridge_learner.train(features, targets)
        ridge_predictions, _ = ridge_learner.predict(features)

        assert linear_predictions.shape == ridge_predictions.shape
        assert np.all(np.isfinite(linear_predictions))
        assert np.all(np.isfinite(ridge_predictions))
        assert not np.allclose(linear_predictions, ridge_predictions)

    def test_train_with_empty_arrays(self, learner):
        """Test error handling with empty arrays."""
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises((ValueError, LearnerError), match="empty dataset"):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        """Test error handling with mismatched feature/target shapes."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises((ValueError, LearnerError), match="mismatched lengths"):
            learner.train(features, targets)

    def test_predict_without_training(self, learner):
        """Test error when predicting without training."""
        features = np.random.randn(5, 10)
        with pytest.raises((RuntimeError, LearnerError), match="must be trained"):
            learner.predict(features)

    def test_get_name_distinguishes_linear_and_ridge_modes(self, learner, ridge_learner):
        """Test name generation for both Ridge and Linear modes."""
        linear_name = learner.get_name()
        assert "LinearRegression" in linear_name
        assert "with_intercept" in linear_name

        ridge_name = ridge_learner.get_name()
        assert "Ridge" in ridge_name
        assert "alpha=1.000" in ridge_name
        assert "with_intercept" in ridge_name

    def test_get_name_no_intercept(self):
        """Test name generation when fit_intercept=False."""
        learner = LinearRegressionLearner(alpha=None, fit_intercept=False, random_state=42)
        name = learner.get_name()
        assert "LinearRegression" in name
        assert "no_intercept" in name

    def test_supports_uncertainty_returns_true_for_linear_and_ridge_modes(self, learner, ridge_learner):
        """Test that uncertainty is supported in both modes."""
        assert learner.supports_uncertainty() is True
        assert ridge_learner.supports_uncertainty() is True

    def test_ridge_alpha_values_change_predictions_while_remaining_finite(self, compounds_20, features_20):
        """Test Ridge regression with different alpha values."""
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = features_20
        targets = compounds['Activity'].to_numpy()

        alphas = [0.1, 1.0, 10.0]
        predictions_dict = {}

        for alpha in alphas:
            learner = LinearRegressionLearner(alpha=alpha, random_state=42)
            learner.train(features, targets)
            predictions, _ = learner.predict(features)
            predictions_dict[alpha] = predictions

        for alpha in alphas:
            assert np.all(np.isfinite(predictions_dict[alpha]))

        assert not np.allclose(predictions_dict[0.1], predictions_dict[10.0])

    def test_small_diverse_dataset_trains_and_predicts_finite_values(self, learner, tmp_path):
        """Test with small diverse dataset."""
        small_compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(5)],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'C1CCNCC1'],
            'Activity': [0.5, 0.3, 0.8, 0.2, 0.6]
        })

        features = extract_features(small_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        learner.train(features, small_compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert len(predictions) == 5
        assert np.all(np.isfinite(predictions))

    def test_n_jobs_parameter(self, compounds_20, features_20):
        """Test that n_jobs parameter is respected for LinearRegression mode."""
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = features_20
        targets = compounds['Activity'].to_numpy()

        learner_parallel = LinearRegressionLearner(alpha=None, n_jobs=-1, random_state=42)
        assert learner_parallel.n_jobs == -1
        learner_parallel.train(features, targets)
        predictions, _ = learner_parallel.predict(features)
        assert np.all(np.isfinite(predictions))

        learner_single = LinearRegressionLearner(alpha=None, n_jobs=1, random_state=42)
        assert learner_single.n_jobs == 1
        learner_single.train(features, targets)
        predictions_single, _ = learner_single.predict(features)
        assert np.allclose(predictions, predictions_single)

    def test_ridge_mode_no_n_jobs(self, ridge_learner):
        """Test that Ridge mode does not use n_jobs parameter."""
        assert ridge_learner.n_jobs is None

    def test_uncertainty_shape_matches_predictions(self, learner, compounds_20, features_20):
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = features_20
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = learner.predict(features)
        assert uncertainty is not None
        assert uncertainty.shape == predictions.shape

    def test_uncertainty_non_negative(self, learner, compounds_20, features_20):
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = features_20
        learner.train(features, compounds['Activity'].to_numpy())
        _, uncertainty = learner.predict(features)
        assert uncertainty is not None
        assert np.all(uncertainty >= 0)
        assert np.all(np.isfinite(uncertainty))

    def test_uncertainty_consistency(self, learner, compounds_20, features_20):
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = features_20
        learner.train(features, compounds['Activity'].to_numpy())
        _, unc1 = learner.predict(features)
        _, unc2 = learner.predict(features)
        np.testing.assert_array_equal(unc1, unc2)

    def test_leverage_property(self):
        """OOD points should have higher uncertainty than in-distribution."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 5)
        y_train = X_train @ rng.randn(5) + rng.randn(50) * 0.1

        X_in = rng.randn(10, 5)
        X_ood = rng.randn(10, 5) * 10 + 20

        learner = LinearRegressionLearner(random_state=42)
        learner.train(X_train, y_train)

        _, unc_in = learner.predict(X_in)
        _, unc_ood = learner.predict(X_ood)

        assert np.mean(unc_ood) > np.mean(unc_in)

    def test_ridge_vs_ols_uncertainty(self):
        """Both alpha=None (OLS) and alpha>0 (Ridge) produce uncertainty."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        y = X @ rng.randn(5) + rng.randn(30) * 0.1

        lr_ols = LinearRegressionLearner(alpha=None, random_state=42)
        lr_ols.train(X, y)
        _, unc_ols = lr_ols.predict(X)

        lr_ridge = LinearRegressionLearner(alpha=1.0, random_state=42)
        lr_ridge.train(X, y)
        _, unc_ridge = lr_ridge.predict(X)

        assert unc_ols is not None
        assert unc_ridge is not None
        assert np.all(unc_ols >= 0)
        assert np.all(unc_ridge >= 0)

    def test_uncertainty_auto_epsilon(self):
        """alpha=0 with p > n should not crash (auto-epsilon)."""
        rng = np.random.RandomState(42)
        X = rng.randn(5, 20)
        y = rng.randn(5)

        learner = LinearRegressionLearner(alpha=None, random_state=42)
        learner.train(X, y)
        _preds, unc = learner.predict(X)

        assert unc is not None
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0)

    def test_sigma_hat_sq_zero_targets(self):
        """Identical targets -> zero uncertainty."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        y = np.ones(20) * 3.0

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)
        _, unc = learner.predict(X)

        assert unc is not None
        np.testing.assert_allclose(unc, 0.0, atol=1e-10)

    def test_leverage_formula_manual_validation(self):
        """Compare leverage uncertainty vs scipy.linalg.inv reference."""
        from scipy.linalg import inv
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        y = X @ rng.randn(5) + rng.randn(30) * 0.1

        alpha = 1.0
        learner = LinearRegressionLearner(alpha=alpha, random_state=42)
        learner.train(X, y)

        preprocessed = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        variance = np.var(preprocessed, axis=0)
        mask = variance > 1e-10
        preprocessed = preprocessed[:, mask]

        scaler = StandardScaler()
        preprocessed = scaler.fit_transform(preprocessed)

        n, p = preprocessed.shape
        gram = preprocessed.T @ preprocessed + alpha * np.eye(p)
        gram_inv = inv(gram)

        residuals = y - learner.model.predict(preprocessed)
        sigma_hat_sq = np.sum(residuals**2) / max(n - p, 1)

        leverages = np.einsum("ij,jk,ik->i", preprocessed, gram_inv, preprocessed)
        expected_unc = np.sqrt(sigma_hat_sq * np.maximum(1.0 + leverages, 0.0))

        _, actual_unc = learner.predict(X)
        np.testing.assert_allclose(actual_unc, expected_unc, rtol=1e-6)

    def test_gram_computed_on_preprocessed_features(self):
        """Zero-variance columns -> _L_inv dims match filtered count."""
        rng = np.random.RandomState(42)
        X_good = rng.randn(30, 5)
        X_const = np.ones((30, 3)) * 7.0
        X = np.hstack([X_good, X_const])
        y = rng.randn(30)

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)

        assert learner._L_inv.shape == (5, 5)

        _, unc = learner.predict(X)
        assert unc is not None
        assert np.all(np.isfinite(unc))

    def test_uncertainty_with_near_singular_features(self):
        """Features with extreme variance ratios produce no NaN/Inf."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        X[:, 0] = X[:, 1] + rng.randn(30) * 1e-8
        y = rng.randn(30)

        learner = LinearRegressionLearner(alpha=None, random_state=42)
        learner.train(X, y)
        _, unc = learner.predict(X)

        assert unc is not None
        assert np.all(np.isfinite(unc))

    def test_uncertainty_with_p_equals_n(self):
        """n == p: sigma_hat_sq uses max(n-p, 1) = 1 denominator."""
        rng = np.random.RandomState(42)
        n = 10
        X = rng.randn(n, n)
        y = rng.randn(n)

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)
        _, unc = learner.predict(X)

        assert unc is not None
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0)

    def test_uncertainty_with_p_greater_than_n(self):
        """More features than samples -- no crash."""
        rng = np.random.RandomState(42)
        X = rng.randn(5, 20)
        y = rng.randn(5)

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)
        _, unc = learner.predict(X)

        assert unc is not None
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0)

    @pytest.mark.parametrize("alpha", [None, 0.1, 1.0, 10.0])
    def test_uncertainty_across_alpha_values(self, alpha):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        y = rng.randn(30)

        learner = LinearRegressionLearner(alpha=alpha, random_state=42)
        learner.train(X, y)
        preds, unc = learner.predict(X)

        assert unc is not None
        assert unc.shape == preds.shape
        assert np.all(unc >= 0)
        assert np.all(np.isfinite(unc))

    def test_uncertainty_after_zero_variance_removal(self):
        rng = np.random.RandomState(42)
        X_good = rng.randn(30, 5)
        X_const = np.full((30, 3), 5.0)
        X = np.hstack([X_good, X_const])
        y = rng.randn(30)

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)
        _, unc = learner.predict(X)

        assert unc is not None
        assert unc.shape == (30,)
        assert np.all(np.isfinite(unc))

    def test_uncertainty_with_tiny_dataset(self):
        rng = np.random.RandomState(42)
        X = rng.randn(3, 2)
        y = rng.randn(3)

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(X, y)
        preds, unc = learner.predict(X)

        assert preds.shape == (3,)
        assert unc is not None
        assert unc.shape == (3,)
        assert np.all(np.isfinite(unc))


@pytest.mark.unit
class TestRidgeFeatureScaling:
    def test_scale_features_default_true(self):
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        assert learner.scale_features is True

    def test_feature_scaler_fitted_after_training(self):
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        rng = np.random.RandomState(42)
        features = rng.randn(30, 5) * 100
        targets = rng.randn(30)
        learner = LinearRegressionLearner(alpha=1.0, random_state=42)
        learner.train(features, targets)
        assert learner._feature_scaler is not None

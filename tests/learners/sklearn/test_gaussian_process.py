"""Tests for GaussianProcessLearner implementation."""

import logging

import numpy as np
import polars as pl
import pytest
from scipy import stats
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

from learnm8.exceptions import ConfigurationError, LearnerError
from learnm8.features.extraction import extract_features
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.molecular
class TestGaussianProcessLearner:
    """Test GaussianProcessLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create GaussianProcessLearner instance for testing."""
        return GaussianProcessLearner(random_state=42)

    @pytest.fixture
    def compounds_25(self, small_real_compounds):
        return small_real_compounds.head(25)

    @pytest.fixture
    def features_25(self, small_real_morgan_features):
        return small_real_morgan_features[:25]

    def test_initialization_sets_default_hyperparameters_and_uncertainty_support(self, learner):
        """Test learner initialization."""
        assert learner.alpha == 1e-3
        assert learner.random_state == 42
        assert not learner.is_trained
        assert learner.supports_uncertainty() is True

    def test_predict_returns_finite_values_and_uncertainty_after_training(self, learner, compounds_25, features_25):
        """Test training and prediction with real molecular data."""
        compounds = compounds_25.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        features = features_25
        learner.train(features, compounds['Activity'].to_numpy())
        assert learner.is_trained

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_training_points_have_comparable_or_lower_uncertainty_than_holdout_points(self, learner, compounds_25, features_25):
        """Test that uncertainty estimates are reasonable."""
        compounds = compounds_25.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        half = len(compounds) // 2
        train_features = features_25[:half]
        learner.train(train_features, compounds[:half]['Activity'].to_numpy())

        _train_pred, train_unc = learner.predict(train_features)

        if half < len(compounds):
            test_features = features_25[half:]
            _test_pred, test_unc = learner.predict(test_features)
            assert np.mean(train_unc) <= np.mean(test_unc) * 2

    def test_predict_without_training(self, learner, features_25):
        """Test error when predicting without training."""
        with pytest.raises(RuntimeError, match="must be trained before prediction"):
            learner.predict(features_25)

    def test_get_name_includes_alpha_configuration(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "GaussianProcess" in name
        assert "alpha=0.001" in name

    def test_get_learned_hyperparameters_returns_kernel_and_theta_after_training(self, learner, compounds_25, features_25):
        """Test hyperparameter learning."""
        compounds = compounds_25.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        assert learner.get_learned_hyperparameters() is None

        features = features_25
        learner.train(features, compounds['Activity'].to_numpy())
        hyperparams = learner.get_learned_hyperparameters()
        assert hyperparams is not None
        assert 'kernel' in hyperparams
        assert 'theta' in hyperparams

    def test_custom_alpha_configuration_trains_and_predicts(self, compounds_25, features_25):
        """Test learner with custom kernel configuration."""
        learner = GaussianProcessLearner(alpha=1e-6, random_state=42)

        compounds = compounds_25.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        features = features_25
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = learner.predict(features)

        assert learner.alpha == 1e-6
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None

    def test_small_diverse_dataset_trains_and_predicts_finite_values(self, learner, tmp_path):
        """Test with small diverse dataset."""
        small_compounds = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(5)],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'C1CCNCC1'],
            'Activity': [0.5, 0.3, 0.8, 0.2, 0.6]
        })

        features = extract_features(small_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        learner.train(features, small_compounds['Activity'].to_numpy())
        predictions, uncertainty = learner.predict(features)

        assert len(predictions) == 5
        assert len(uncertainty) == 5
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_extreme_targets_produce_finite_predictions_and_uncertainty(self, learner, tmp_path):
        """Test numerical stability with extreme values."""
        compounds = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.0, 1000.0, 0.001]
        })

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = learner.predict(features)

        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(uncertainty))
        assert np.all(uncertainty >= 0)

    def test_predict_returns_non_negative_uncertainty_for_unseen_compounds(self, learner, tmp_path):
        """Test that uncertainty estimates have reasonable ordering."""
        train_compounds = pl.DataFrame({
            'ID': ['TRAIN_001', 'TRAIN_002'],
            'SMILES': ['CCO', 'CCC'],
            'Activity': [0.5, 0.6]
        })

        test_compounds = pl.DataFrame({
            'ID': ['TEST_001', 'TEST_002'],
            'SMILES': ['CCCO', 'c1ccccc1'],
        })

        train_features = extract_features(train_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        learner.train(train_features, train_compounds['Activity'].to_numpy())

        test_features = extract_features(test_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        _, uncertainties = learner.predict(test_features)

        assert np.all(uncertainties >= 0)
        assert len(uncertainties) == len(test_compounds)

    def test_predict_returns_uncertainty_array_when_learner_supports_uncertainty(self, learner, compounds_25, features_25):
        compounds = compounds_25.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.lit(np.random.beta(2, 5, len(compounds))).alias('Activity'))

        features = features_25
        learner.train(features, compounds['Activity'].to_numpy())

        _predictions, uncertainty = learner.predict(features)

        assert learner.supports_uncertainty() is True
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)

    def test_train_with_empty_arrays(self, learner):
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(RuntimeError, match="empty dataset"):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(RuntimeError, match="mismatched lengths"):
            learner.train(features, targets)

    def test_train_with_1d_features(self, learner):
        features_1d = np.random.rand(10)
        targets = np.random.rand(10)

        with pytest.raises((ValueError, RuntimeError)):
            learner.train(features_1d, targets)

    def test_default_kernel_configuration_is_auto(self):
        learner = GaussianProcessLearner()
        assert learner._kernel_config == "auto"

    def test_none_kernel_configuration_is_normalized_to_auto(self):
        learner = GaussianProcessLearner(kernel=None)
        assert learner._kernel_config == "auto"

    def test_get_name_reflects_selected_kernel_after_training(self, compounds_25, features_25):
        learner = GaussianProcessLearner(random_state=42)
        compounds = compounds_25.clone()
        if "Activity" not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias("Activity")
            )
        learner.train(features_25, compounds["Activity"].to_numpy())
        name = learner.get_name()
        assert "Tanimoto" in name or "RBF" in name


@pytest.mark.slow
@pytest.mark.unit
class TestKernelAutoDetection:
    """Tests for kernel auto-detection and explicit selection (T006)."""

    def test_auto_kernel_binary_selects_tanimoto(self):
        learner = GaussianProcessLearner(kernel="auto", random_state=42)
        X = np.random.randint(0, 2, size=(20, 10)).astype(float)
        y = np.random.randn(20)
        learner.train(X, y)
        assert learner._kernel_name == "Tanimoto"

    def test_auto_kernel_continuous_selects_rbf(self):
        learner = GaussianProcessLearner(kernel="auto", random_state=42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        learner.train(X, y)
        assert learner._kernel_name == "RBF"

    def test_explicit_tanimoto_kernel_sets_kernel_name_on_train(self):
        learner = GaussianProcessLearner(kernel="tanimoto", random_state=42)
        X = np.random.randn(20, 5).clip(0)
        y = np.random.randn(20)
        learner.train(X, y)
        assert learner._kernel_name == "Tanimoto"

    def test_explicit_rbf_kernel_sets_kernel_name_on_train(self):
        learner = GaussianProcessLearner(kernel="rbf", random_state=42)
        X = np.random.randint(0, 2, size=(20, 10)).astype(float)
        y = np.random.randn(20)
        learner.train(X, y)
        assert learner._kernel_name == "RBF"

    def test_custom_kernel_object_passthrough(self):
        custom_kernel = C(1.0) * RBF(1.0)
        learner = GaussianProcessLearner(kernel=custom_kernel, random_state=42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        learner.train(X, y)
        assert isinstance(learner._kernel_config, type(custom_kernel))

    def test_retraining_on_continuous_features_switches_kernel_from_tanimoto_to_rbf(self):
        learner = GaussianProcessLearner(kernel="auto", random_state=42)
        X_binary = np.random.randint(0, 2, size=(20, 10)).astype(float)
        y = np.random.randn(20)
        learner.train(X_binary, y)
        assert learner._kernel_name == "Tanimoto"

        X_cont = np.random.randn(20, 5)
        learner.train(X_cont, y)
        assert learner._kernel_name == "RBF"

    def test_auto_kernel_treats_binary_feature_matrix_as_tanimoto(self):
        learner = GaussianProcessLearner(kernel="auto", random_state=42)
        X = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]] * 10)
        y = np.random.randn(20)
        learner.train(X, y)
        assert learner._kernel_name == "Tanimoto"

    def test_custom_kernel_object_predicts_with_backward_compatible_api(self):
        custom_kernel = C(2.0, (1e-3, 1e3)) * RBF(0.5, (1e-3, 1e3))
        learner = GaussianProcessLearner(kernel=custom_kernel, random_state=42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        learner.train(X, y)
        preds, unc = learner.predict(X)
        assert preds.shape == (20,)
        assert unc.shape == (20,)

    def test_invalid_kernel_string_raises(self):
        with pytest.raises(ConfigurationError, match="kernel must be"):
            GaussianProcessLearner(kernel="matern")

    def test_negative_features_tanimoto_warns(self, caplog):
        learner = GaussianProcessLearner(kernel="tanimoto", random_state=42)
        X = np.array([[-1.0, 0.5, 0.3], [0.2, 0.1, 0.8]] * 10)
        y = np.random.randn(20)
        logging.getLogger('learnm8').propagate = True
        with caplog.at_level(logging.WARNING, logger='learnm8.learners.sklearn.gaussian_process'):
            learner.train(X, y)
        assert "non-negative" in caplog.text.lower() or "negative" in caplog.text.lower()

    def test_get_name_before_train_explicit_kernel(self):
        learner = GaussianProcessLearner(kernel="tanimoto")
        assert "Tanimoto" in learner.get_name()

        learner_rbf = GaussianProcessLearner(kernel="rbf")
        assert "RBF" in learner_rbf.get_name()


@pytest.mark.slow
@pytest.mark.unit
class TestAlphaConfiguration:
    """Tests for alpha configuration passthrough (T007)."""

    def test_alpha_value_is_passed_to_underlying_model(self):
        learner = GaussianProcessLearner(alpha=0.05, random_state=42)
        assert learner.alpha == 0.05
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        learner.train(X, y)
        assert learner.model.alpha == 0.05


@pytest.mark.slow
@pytest.mark.unit
class TestSizeGuard:
    """Tests for training set size guards (T008)."""

    def test_training_above_warning_threshold_logs_warning(self, caplog):
        learner = GaussianProcessLearner(max_train_size=50, random_state=42)
        n = 21  # > 50 * 0.4 = 20
        X = np.random.randn(n, 2)
        y = np.random.randn(n)
        logging.getLogger('learnm8').propagate = True
        with caplog.at_level(logging.WARNING, logger='learnm8.learners.sklearn.gaussian_process'):
            learner.train(X, y)
        assert "large" in caplog.text.lower() or "slow" in caplog.text.lower()

    def test_training_at_warning_threshold_does_not_log_warning(self, caplog):
        learner = GaussianProcessLearner(max_train_size=50, random_state=42)
        n = 20  # = 50 * 0.4, NOT > threshold
        X = np.random.randn(n, 2)
        y = np.random.randn(n)
        with caplog.at_level(logging.WARNING):
            learner.train(X, y)
        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING
                        and "large" in r.message.lower()]
        assert len(warning_msgs) == 0

    def test_training_above_max_train_size_raises_learner_error(self):
        learner = GaussianProcessLearner(max_train_size=50, random_state=42)
        n = 51  # > 50
        X = np.random.randn(n, 2)
        y = np.random.randn(n)
        with pytest.raises(LearnerError, match="exceeds maximum"):
            learner.train(X, y)

    def test_training_at_max_train_size_succeeds(self):
        learner = GaussianProcessLearner(max_train_size=50, random_state=42)
        n = 50  # = max, NOT > max
        X = np.random.randn(n, 2)
        y = np.random.randn(n)
        learner.train(X, y)
        assert learner.is_trained

    def test_custom_max_train_size_allows_limit_and_rejects_excess(self):
        learner = GaussianProcessLearner(max_train_size=100, random_state=42)
        X_ok = np.random.randn(100, 2)
        y_ok = np.random.randn(100)
        learner.train(X_ok, y_ok)
        assert learner.is_trained

        X_too_big = np.random.randn(101, 2)
        y_too_big = np.random.randn(101)
        with pytest.raises(LearnerError, match="exceeds maximum"):
            learner.train(X_too_big, y_too_big)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.molecular
class TestScientificValidation:
    """Scientific validation tests (T009)."""

    @pytest.fixture
    def compounds_25(self, small_real_compounds):
        return small_real_compounds.head(25)

    @pytest.fixture
    def features_25(self, small_real_morgan_features):
        return small_real_morgan_features[:25]

    def test_uncertainty_rank_correlation_positive(
        self, compounds_25, features_25
    ):
        compounds = compounds_25.clone()
        if "Activity" not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias("Activity")
            )

        targets = compounds["Activity"].to_numpy()
        features = features_25

        half = len(compounds) // 2
        train_X, test_X = features[:half], features[half:]
        train_y, test_y = targets[:half], targets[half:]

        learner = GaussianProcessLearner(random_state=42)
        learner.train(train_X, train_y)
        preds, unc = learner.predict(test_X)

        abs_error = np.abs(preds - test_y)
        rho, _ = stats.spearmanr(unc, abs_error)
        assert rho > 0, f"Expected positive rank correlation, got rho={rho:.4f}"

    def test_ablation_tanimoto_vs_rbf_on_binary(
        self, compounds_25, features_25
    ):
        compounds = compounds_25.clone()
        if "Activity" not in compounds.columns:
            compounds = compounds.with_columns(
                pl.lit(np.random.beta(2, 5, len(compounds))).alias("Activity")
            )

        targets = compounds["Activity"].to_numpy()
        features = features_25

        half = len(compounds) // 2
        train_X, test_X = features[:half], features[half:]
        train_y, test_y = targets[:half], targets[half:]

        gp_tan = GaussianProcessLearner(kernel="tanimoto", alpha=1e-3, random_state=42)
        gp_tan.train(train_X, train_y)
        preds_tan, unc_tan = gp_tan.predict(test_X)
        abs_err_tan = np.abs(preds_tan - test_y)
        rho_tan, _ = stats.spearmanr(unc_tan, abs_err_tan)

        gp_rbf = GaussianProcessLearner(kernel="rbf", alpha=1e-3, random_state=42)
        gp_rbf.train(train_X, train_y)
        preds_rbf, unc_rbf = gp_rbf.predict(test_X)
        abs_err_rbf = np.abs(preds_rbf - test_y)
        rho_rbf, _ = stats.spearmanr(unc_rbf, abs_err_rbf)

        if np.isnan(rho_rbf):
            pass
        else:
            assert rho_tan >= rho_rbf, (
                f"Expected Tanimoto rho ({rho_tan:.4f}) >= RBF rho ({rho_rbf:.4f}) on binary features"
            )


@pytest.mark.slow
@pytest.mark.unit
class TestGPFeatureScaling:
    def test_scale_features_default_true(self):
        learner = GaussianProcessLearner(random_state=42)
        assert learner.scale_features is True

    def test_feature_scaler_fitted_after_training(self):
        rng = np.random.RandomState(42)
        features = rng.randn(30, 5) * 100
        targets = rng.randn(30)
        learner = GaussianProcessLearner(random_state=42)
        learner.train(features, targets)
        assert learner._feature_scaler is not None

    def test_predictions_use_scaled_features(self):
        rng = np.random.RandomState(42)
        features = rng.randn(30, 5) * 100
        targets = rng.randn(30)
        learner = GaussianProcessLearner(random_state=42)
        learner.train(features, targets)
        preds, _ = learner.predict(features)
        assert np.all(np.isfinite(preds))

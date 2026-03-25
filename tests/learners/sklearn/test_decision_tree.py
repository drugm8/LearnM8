"""Tests for DecisionTreeLearner implementation."""

import numpy as np
import polars as pl
import pytest

from learnm8.exceptions import LearnerError
from learnm8.learners.sklearn.decision_tree import DecisionTreeLearner


@pytest.mark.integration
@pytest.mark.molecular
class TestDecisionTreeLearner:
    """Test DecisionTreeLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create DecisionTreeLearner instance for testing."""
        return DecisionTreeLearner(max_depth=10, random_state=42)

    def test_initialization_sets_tree_hyperparameters_and_untrained_state(self):
        """Test DecisionTreeLearner initializes with correct parameters."""
        learner = DecisionTreeLearner(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=123
        )

        assert learner.max_depth == 15
        assert learner.min_samples_split == 5
        assert learner.min_samples_leaf == 3
        assert learner.random_state == 123
        assert not learner.is_trained

    def test_predict_returns_finite_values_and_uncertainty_after_training(self, learner, small_real_compounds, small_real_morgan_features):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        targets = compounds['Activity'].to_numpy()

        learner.train(features, targets)
        assert learner.is_trained

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape == predictions.shape
        assert np.all(uncertainty >= 0)
        assert np.all(np.isfinite(predictions))

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

    def test_get_name_includes_decision_tree_identifier(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "DecisionTree" in name

    def test_feature_importance_returns_normalized_values_after_training(self, learner, small_real_compounds, small_real_morgan_features):
        """Test feature importance retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        assert learner.get_feature_importance() is None

        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())

        importance = learner.get_feature_importance()
        assert importance is not None
        assert importance.shape[0] == np.sum(learner._valid_feature_mask)
        assert np.all(importance >= 0)
        assert np.isclose(importance.sum(), 1.0)

    def test_supports_uncertainty_returns_true_for_default_criterion(self, learner):
        """Test DecisionTreeLearner supports uncertainty."""
        assert learner.supports_uncertainty() is True

    def test_train_with_1d_features(self, learner):
        """Test training with 1D features raises error."""
        features_1d = np.random.rand(10)
        targets = np.random.rand(10)

        with pytest.raises((ValueError, RuntimeError)):
            learner.train(features_1d, targets)

    def test_same_random_state_produces_identical_predictions(self, small_real_compounds, small_real_morgan_features):
        """Test predictions are deterministic with same random_state."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        targets = compounds['Activity'].to_numpy()

        learner1 = DecisionTreeLearner(random_state=42)
        learner1.train(features, targets)
        pred1, _ = learner1.predict(features)

        learner2 = DecisionTreeLearner(random_state=42)
        learner2.train(features, targets)
        pred2, _ = learner2.predict(features)

        np.testing.assert_array_equal(pred1, pred2)

    def test_max_depth_limits_tree_depth_relative_to_deeper_model(self, small_real_compounds, small_real_morgan_features):
        """Test max_depth parameter affects model complexity."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        targets = compounds['Activity'].to_numpy()

        learner_shallow = DecisionTreeLearner(max_depth=2, random_state=42)
        learner_shallow.train(features, targets)

        learner_deep = DecisionTreeLearner(max_depth=20, random_state=42)
        learner_deep.train(features, targets)

        assert learner_shallow.model.get_depth() <= 2
        assert learner_deep.model.get_depth() > learner_shallow.model.get_depth()

    def test_uncertainty_shape_matches_predictions(self, learner, small_real_compounds, small_real_morgan_features):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = learner.predict(features)
        assert uncertainty is not None
        assert uncertainty.shape == predictions.shape

    def test_uncertainty_non_negative(self, learner, small_real_compounds, small_real_morgan_features):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        _, uncertainty = learner.predict(features)
        assert uncertainty is not None
        assert np.all(uncertainty >= 0)
        assert np.all(np.isfinite(uncertainty))

    def test_repeated_predictions_return_identical_uncertainty_values(self, learner, small_real_compounds, small_real_morgan_features):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = small_real_morgan_features
        learner.train(features, compounds['Activity'].to_numpy())
        _, unc1 = learner.predict(features)
        _, unc2 = learner.predict(features)
        np.testing.assert_array_equal(unc1, unc2)

    def test_uncertainty_impurity_values(self):
        """Manual sqrt(impurity[apply(X)]) matches learner output."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = rng.randn(50)

        learner = DecisionTreeLearner(max_depth=5, random_state=42)
        learner.train(X, y)

        preprocessed = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        variance = np.var(preprocessed, axis=0)
        mask = variance > 1e-10
        preprocessed = preprocessed[:, mask]

        leaf_ids = learner.model.apply(preprocessed)
        expected = np.sqrt(np.maximum(learner.model.tree_.impurity[leaf_ids], 0.0))

        _, actual = learner.predict(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_uncertainty_non_squared_error(self):
        """criterion='absolute_error' → uncertainty is None AND supports_uncertainty() returns False."""
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        y = rng.randn(30)

        learner = DecisionTreeLearner(max_depth=5, random_state=42)
        learner.model = DecisionTreeRegressor(
            criterion='absolute_error', max_depth=5, random_state=42
        )
        learner.train(X, y)

        assert learner.supports_uncertainty() is False
        _, unc = learner.predict(X)
        assert unc is None

    def test_majority_zero_uncertainty_logs_debug_message(self, caplog):
        """DEBUG message when >50% zero uncertainty."""
        import logging
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = rng.randn(50)

        learner = DecisionTreeLearner(max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=42)
        learner.train(X, y)

        logging.getLogger('learnm8').propagate = True
        with caplog.at_level(logging.DEBUG, logger="learnm8.learners.sklearn.decision_tree"):
            learner.predict(X)

        has_zero_msg = any("zero uncertainty" in r.message for r in caplog.records)
        assert has_zero_msg

    def test_supports_uncertainty_reflects_trained_criterion(self):
        """True for squared_error, False for absolute_error after training."""
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        y = rng.randn(30)

        learner_sq = DecisionTreeLearner(max_depth=5, random_state=42)
        learner_sq.train(X, y)
        assert learner_sq.supports_uncertainty() is True

        learner_ae = DecisionTreeLearner(max_depth=5, random_state=42)
        learner_ae.model = DecisionTreeRegressor(
            criterion='absolute_error', max_depth=5, random_state=42
        )
        learner_ae.train(X, y)
        assert learner_ae.supports_uncertainty() is False

    @pytest.mark.parametrize("max_depth", [2, 5, 10, None])
    def test_predict_returns_finite_non_negative_uncertainty_across_tree_depths(self, max_depth):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = rng.randn(50)

        learner = DecisionTreeLearner(max_depth=max_depth, random_state=42)
        learner.train(X, y)
        preds, unc = learner.predict(X)

        assert unc is not None
        assert unc.shape == preds.shape
        assert np.all(unc >= 0)
        assert np.all(np.isfinite(unc))

    def test_zero_variance_feature_removal_preserves_uncertainty_output(self):
        rng = np.random.RandomState(42)
        X_good = rng.randn(30, 5)
        X_const = np.full((30, 3), 5.0)
        X = np.hstack([X_good, X_const])
        y = rng.randn(30)

        learner = DecisionTreeLearner(max_depth=5, random_state=42)
        learner.train(X, y)
        _, unc = learner.predict(X)

        assert unc is not None
        assert unc.shape == (30,)
        assert np.all(np.isfinite(unc))

    def test_tiny_dataset_returns_finite_predictions_and_uncertainty(self):
        rng = np.random.RandomState(42)
        X = rng.randn(3, 2)
        y = rng.randn(3)

        learner = DecisionTreeLearner(max_depth=2, min_samples_leaf=1, min_samples_split=2, random_state=42)
        learner.train(X, y)
        preds, unc = learner.predict(X)

        assert preds.shape == (3,)
        assert unc is not None
        assert unc.shape == (3,)
        assert np.all(np.isfinite(unc))

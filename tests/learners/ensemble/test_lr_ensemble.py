"""Tests for LREnsemble implementation."""

import numpy as np
import polars as pl
import pytest

from learnm8.exceptions import LearnerError
from learnm8.learners.ensemble.lr_ensemble import LREnsemble


@pytest.mark.slow
@pytest.mark.integration
class TestLREnsemble:
    """Test LREnsemble functionality with real molecular data."""

    @pytest.fixture
    def lr_ensemble(self):
        """Create LREnsemble instance for testing."""
        return LREnsemble()

    @pytest.fixture
    def compounds_20(self, small_real_compounds):
        return small_real_compounds.head(20)

    @pytest.fixture
    def features_20(self, small_real_morgan_features):
        return small_real_morgan_features[:20]

    def test_initialization_sets_default_regularization_strengths_random_states_and_untrained_state(self, lr_ensemble):
        """Test ensemble initialization with default parameters."""
        assert len(lr_ensemble.learners) == 3
        assert lr_ensemble.aggregation_method == 'mean'
        assert lr_ensemble.uncertainty_method == 'std'
        assert lr_ensemble.weights is None
        assert not lr_ensemble.is_trained
        assert lr_ensemble.supports_uncertainty() is True
        assert lr_ensemble.regularization_strengths == [0.1, 1.0, 10.0]
        assert lr_ensemble.random_states == [42, 123, 456]

    def test_initialization_custom_regularization(self):
        """Test ensemble initialization with custom regularization strengths."""
        custom_alphas = [0.01, 0.1, 1.0]
        custom_states = [10, 20, 30]

        ensemble = LREnsemble(
            regularization_strengths=custom_alphas,
            random_states=custom_states
        )

        assert len(ensemble.learners) == 3
        assert ensemble.regularization_strengths == custom_alphas
        assert ensemble.random_states == custom_states

    def test_initialization_with_weights(self):
        """Test ensemble initialization with custom weights."""
        weights = [0.5, 0.3, 0.2]
        ensemble = LREnsemble(weights=weights)

        assert ensemble.weights is not None
        assert np.allclose(ensemble.weights, weights)

    def test_diverse_regularization(self, lr_ensemble, compounds_20, features_20):
        """Test that ensemble learners have different regularization strengths."""
        assert len(lr_ensemble.learners) == 3

        alphas = [learner.alpha for learner in lr_ensemble.learners]
        assert len(set(alphas)) == 3
        assert alphas == [0.1, 1.0, 10.0]

        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = features_20
        lr_ensemble.train(features, compounds['Activity'].to_numpy())

        individual_preds = lr_ensemble.get_individual_predictions(features)
        assert len(individual_preds) == 3

        pred_arrays = [preds for preds in individual_preds.values() if preds is not None]
        assert len(pred_arrays) == 3

        for i in range(len(pred_arrays)):
            for j in range(i+1, len(pred_arrays)):
                assert not np.allclose(pred_arrays[i], pred_arrays[j], atol=1e-2)

    def test_get_name_includes_regularization_strengths(self, lr_ensemble):
        """Test name generation for LR ensemble."""
        name = lr_ensemble.get_name()
        assert "LREnsemble" in name
        assert "3xRidge" in name
        assert "0.1" in name
        assert "1.0" in name
        assert "10.0" in name

    def test_get_name_custom_alphas(self):
        """Test name generation with custom regularization strengths."""
        ensemble = LREnsemble(regularization_strengths=[0.01, 0.5, 5.0])
        name = ensemble.get_name()
        assert "LREnsemble" in name
        assert "0.0" in name
        assert "0.5" in name
        assert "5.0" in name

    def test_regularization_effect_on_predictions(self, compounds_20, features_20):
        """Test that different regularization strengths produce different predictions."""
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = features_20

        ensemble_weak = LREnsemble(regularization_strengths=[0.01, 0.01, 0.01])
        ensemble_strong = LREnsemble(regularization_strengths=[10.0, 10.0, 10.0])

        ensemble_weak.train(features, compounds['Activity'].to_numpy())
        ensemble_strong.train(features, compounds['Activity'].to_numpy())

        preds_weak, _ = ensemble_weak.predict(features)
        preds_strong, _ = ensemble_strong.predict(features)

        assert not np.allclose(preds_weak, preds_strong, atol=1e-1)

    def test_mismatched_array_lengths(self, lr_ensemble):
        """Test error handling with mismatched feature and target lengths."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises((ValueError, LearnerError)):
            lr_ensemble.train(features, targets)

    def test_add_learner_to_ensemble(self, lr_ensemble):
        """Test adding learners to LR ensemble."""
        from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner

        initial_count = len(lr_ensemble.learners)

        new_learner = LinearRegressionLearner(alpha=100.0, random_state=789)
        lr_ensemble.add_learner(new_learner)

        assert len(lr_ensemble.learners) == initial_count + 1
        assert not lr_ensemble.is_trained

    def test_remove_learner_from_ensemble(self, lr_ensemble):
        """Test removing learners from ensemble."""
        initial_count = len(lr_ensemble.learners)

        lr_ensemble.remove_learner(0)

        assert len(lr_ensemble.learners) == initial_count - 1
        assert not lr_ensemble.is_trained

    def test_prediction_consistency(self, lr_ensemble, compounds_20, features_20):
        """Test that predictions are consistent across multiple calls."""
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = features_20
        lr_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions1, uncertainty1 = lr_ensemble.predict(features)
        predictions2, uncertainty2 = lr_ensemble.predict(features)

        assert np.allclose(predictions1, predictions2)
        assert np.allclose(uncertainty1, uncertainty2)

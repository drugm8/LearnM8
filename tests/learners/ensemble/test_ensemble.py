"""Tests for EnsembleLearner implementation."""

import numpy as np
import polars as pl
import pytest

from learnm8.exceptions import LearnerError
from learnm8.learners.ensemble.ensemble import EnsembleLearner
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.learners.sklearn.random_forest import RandomForestLearner


@pytest.mark.integration
class TestEnsembleLearner:
    """Test EnsembleLearner functionality with real molecular data."""

    @pytest.fixture
    def base_learners(self):
        """Create base learners for ensemble testing."""
        return [
            RandomForestLearner(n_estimators=5, random_state=42),
            GaussianProcessLearner(random_state=42)
        ]

    @pytest.fixture
    def ensemble(self, base_learners):
        """Create EnsembleLearner instance for testing."""
        return EnsembleLearner(base_learners)

    @pytest.fixture
    def compounds_20(self, small_real_compounds):
        return small_real_compounds.head(20)

    @pytest.fixture
    def features_20(self, small_real_morgan_features):
        return small_real_morgan_features[:20]

    def test_initialization_sets_default_aggregation_uncertainty_and_untrained_state(self, ensemble, base_learners):
        """Test ensemble initialization."""
        assert len(ensemble.learners) == 2
        assert ensemble.aggregation_method == 'mean'
        assert ensemble.uncertainty_method == 'std'
        assert ensemble.weights is None
        assert not ensemble.is_trained
        assert ensemble.supports_uncertainty() is True

    def test_initialization_with_weights(self, base_learners):
        """Test ensemble initialization with weights."""
        weights = [0.7, 0.3]
        ensemble = EnsembleLearner(base_learners, weights=weights)

        assert ensemble.weights is not None
        assert np.allclose(ensemble.weights, weights)

    def test_invalid_weights(self, base_learners):
        """Test error handling with invalid weights."""
        with pytest.raises(LearnerError, match='Number of weights'):
            EnsembleLearner(base_learners, weights=[0.5])

        with pytest.raises(LearnerError, match=r'Weights must sum to 1\.0'):
            EnsembleLearner(base_learners, weights=[0.3, 0.4])

    def test_empty_learners_list(self):
        """Test error handling with empty learners list."""
        with pytest.raises(LearnerError, match='at least one base learner'):
            EnsembleLearner([])

    def test_get_name_includes_aggregation_and_component_separator(self, ensemble):
        """Test name generation."""
        name = ensemble.get_name()
        assert "Ensemble" in name
        assert "mean" in name
        assert "+" in name

    def test_failed_learner_handling(self, compounds_20, features_20):
        """Test handling of failed learners during training."""
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        from learnm8.core.interfaces import Learner

        class MockGoodLearner(Learner):
            def train(self, features, targets):
                pass
            def predict(self, features):
                return np.random.randn(len(features)), None
            def get_name(self):
                return "GoodLearner"
            def supports_uncertainty(self):
                return False

        class MockBadLearner(Learner):
            def train(self, features, targets):
                raise RuntimeError("Training failed")
            def predict(self, features):
                return np.random.randn(len(features)), None
            def get_name(self):
                return "BadLearner"
            def supports_uncertainty(self):
                return False

        good_learner = MockGoodLearner()
        bad_learner = MockBadLearner()

        ensemble = EnsembleLearner([good_learner, bad_learner])

        features = features_20
        with pytest.raises(LearnerError, match='BadLearner'):
            ensemble.train(features, compounds['Activity'].to_numpy())

    @pytest.mark.slow
    def test_uncertainty_diversity(self, base_learners, compounds_20, features_20):
        """Test that ensemble uncertainty captures model diversity."""
        compounds = compounds_20.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        ensemble = EnsembleLearner(base_learners)

        features = features_20
        ensemble.train(features, compounds['Activity'].to_numpy())
        _predictions, uncertainty = ensemble.predict(features)

        assert np.std(uncertainty) > 0
        assert np.all(uncertainty >= 0)

    def test_train_with_mismatched_shapes(self, base_learners):
        """Test error with mismatched feature/target shapes."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)
        ensemble = EnsembleLearner(base_learners)

        with pytest.raises(LearnerError, match='mismatched lengths'):
            ensemble.train(features, targets)

    def test_train_with_1d_features(self, base_learners):
        """Test error with 1D feature array."""
        features_1d = np.random.rand(10)
        targets = np.random.rand(10)
        ensemble = EnsembleLearner(base_learners)

        with pytest.raises(LearnerError, match='Features must be 2D'):
            ensemble.train(features_1d, targets)

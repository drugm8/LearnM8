"""Tests for DTEnsemble implementation."""

import pytest
import numpy as np
import polars as pl

from learnm8.exceptions import LearnerError
from learnm8.learners.ensemble.dt_ensemble import DTEnsemble
from learnm8.learners.sklearn.decision_tree import DecisionTreeLearner


@pytest.mark.integration
class TestDTEnsemble:
    """Test DTEnsemble functionality with real molecular data."""

    @pytest.fixture
    def dt_ensemble(self):
        """Create DTEnsemble instance for testing."""
        return DTEnsemble()

    def test_initialization_sets_default_depths_random_states_and_untrained_state(self, dt_ensemble):
        """Test DTEnsemble initialization with default parameters."""
        assert len(dt_ensemble.learners) == 3
        assert dt_ensemble.max_depths == [5, 10, 15]
        assert dt_ensemble.random_states == [42, 123, 456]
        assert dt_ensemble.aggregation_method == 'mean'
        assert dt_ensemble.uncertainty_method == 'std'
        assert not dt_ensemble.is_trained
        assert dt_ensemble.supports_uncertainty() is True

    def test_initialization_custom_depths(self):
        """Test DTEnsemble initialization with custom max depths."""
        custom_depths = [3, 8, 12]
        custom_states = [100, 200, 300]
        ensemble = DTEnsemble(max_depths=custom_depths, random_states=custom_states)

        assert len(ensemble.learners) == 3
        assert ensemble.max_depths == custom_depths
        assert ensemble.random_states == custom_states

        for i, learner in enumerate(ensemble.learners):
            assert isinstance(learner, DecisionTreeLearner)
            assert learner.max_depth == custom_depths[i]
            assert learner.random_state == custom_states[i]

    def test_n_models_parameter(self):
        """Test initialization with different number of models."""
        depths = [4, 8]
        states = [10, 20]
        ensemble = DTEnsemble(max_depths=depths, random_states=states)

        assert len(ensemble.learners) == 2
        assert ensemble.max_depths == depths
        assert ensemble.random_states == states

    def test_diverse_hyperparameters(self, dt_ensemble):
        """Test models have different max_depth values for diversity."""
        assert len(set(dt_ensemble.max_depths)) == len(dt_ensemble.max_depths)
        assert len(set(dt_ensemble.random_states)) == len(dt_ensemble.random_states)

        for i, learner in enumerate(dt_ensemble.learners):
            assert learner.max_depth == dt_ensemble.max_depths[i]
            assert learner.random_state == dt_ensemble.random_states[i]

    def test_get_name_includes_model_count_and_depth_list(self, dt_ensemble):
        """Test name generation for DTEnsemble."""
        name = dt_ensemble.get_name()
        assert "DTEnsemble" in name
        assert "3xDT" in name
        assert "depth=" in name
        assert "5,10,15" in name

    def test_get_name_custom_depths(self):
        """Test name generation with custom depths."""
        ensemble = DTEnsemble(max_depths=[3, 7, 11], random_states=[1, 2, 3])
        name = ensemble.get_name()
        assert "DTEnsemble" in name
        assert "3xDT" in name
        assert "3,7,11" in name

    def test_train_with_mismatched_arrays(self, dt_ensemble, small_real_morgan_features):
        """Test error handling with mismatched feature and target sizes."""
        features = small_real_morgan_features[:10]
        targets = np.random.beta(2, 5, 15)

        with pytest.raises(LearnerError):
            dt_ensemble.train(features, targets)

    def test_prediction_variance(self, small_real_compounds, small_real_morgan_features):
        """Test that different max_depth models produce diverse predictions."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        ensemble = DTEnsemble(max_depths=[3, 10, 20], random_states=[42, 42, 42])
        features = small_real_morgan_features
        ensemble.train(features, compounds['Activity'].to_numpy())

        individual_preds = ensemble.get_individual_predictions(features)
        assert len(individual_preds) == 3

        predictions_array = np.array([preds for preds in individual_preds.values() if preds is not None])
        assert predictions_array.shape[0] == 3
        assert predictions_array.shape[1] == len(compounds)

        variances = np.var(predictions_array, axis=0)
        assert np.mean(variances) > 0

    def test_consistency_with_fixed_random_state(self, small_real_compounds, small_real_morgan_features):
        """Test prediction consistency with fixed random states."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features

        ensemble1 = DTEnsemble(max_depths=[5, 10, 15], random_states=[42, 123, 456])
        ensemble1.train(features, compounds['Activity'].to_numpy())
        predictions1, _ = ensemble1.predict(features)

        ensemble2 = DTEnsemble(max_depths=[5, 10, 15], random_states=[42, 123, 456])
        ensemble2.train(features, compounds['Activity'].to_numpy())
        predictions2, _ = ensemble2.predict(features)

        assert np.allclose(predictions1, predictions2)

    def test_add_learner(self, dt_ensemble):
        """Test adding learners to DTEnsemble."""
        initial_count = len(dt_ensemble.learners)

        new_learner = DecisionTreeLearner(max_depth=20, random_state=999)
        dt_ensemble.add_learner(new_learner)

        assert len(dt_ensemble.learners) == initial_count + 1
        assert not dt_ensemble.is_trained

    def test_remove_learner(self, dt_ensemble):
        """Test removing learners from DTEnsemble."""
        initial_count = len(dt_ensemble.learners)

        dt_ensemble.remove_learner(0)

        assert len(dt_ensemble.learners) == initial_count - 1
        assert not dt_ensemble.is_trained

    def test_failed_learner_handling(self, small_real_compounds, small_real_morgan_features):
        """Test handling of failed learners during training."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        from learnm8.core.interfaces import Learner

        class MockBadLearner(Learner):
            def train(self, features, targets):
                raise RuntimeError("Training failed")
            def predict(self, features):
                return np.random.randn(len(features)), None
            def get_name(self):
                return "BadLearner"
            def supports_uncertainty(self):
                return False

        ensemble = DTEnsemble(max_depths=[5], random_states=[42])
        ensemble.add_learner(MockBadLearner())

        features = small_real_morgan_features
        ensemble.train(features, compounds['Activity'].to_numpy())

        assert ensemble.is_trained
        assert len(ensemble.learners) < 2

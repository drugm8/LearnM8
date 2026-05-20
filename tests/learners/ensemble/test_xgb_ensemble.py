"""Tests for XGBEnsemble implementation."""

import numpy as np
import polars as pl
import pytest

from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble


@pytest.mark.slow
class TestXGBEnsemble:
    """Test XGBEnsemble functionality with real molecular data."""

    @pytest.fixture
    def xgb_ensemble(self):
        """Create XGBEnsemble instance for testing."""
        return XGBEnsemble()

    @pytest.fixture
    def compounds_20(self, small_real_compounds):
        return small_real_compounds.head(20)

    @pytest.fixture
    def features_20(self, small_real_morgan_features):
        return small_real_morgan_features[:20]

    @pytest.fixture
    def custom_xgb_ensemble(self):
        """Create XGBEnsemble with custom hyperparameters."""
        return XGBEnsemble(
            learning_rates=[0.01, 0.05, 0.1],
            random_states=[10, 20, 30]
        )

    @pytest.fixture(scope="class")
    def trained_xgb(self, small_real_compounds, _small_real_morgan_features_raw):
        """Class-scoped trained XGBEnsemble — shared across read-only tests."""
        compounds = small_real_compounds.head(20).clone()
        if 'Activity' not in compounds.columns:
            rng = np.random.RandomState(42)
            compounds = compounds.with_columns(
                pl.Series('Activity', rng.beta(2, 5, len(compounds)))
            )
        features = _small_real_morgan_features_raw[:20].copy()
        targets = compounds['Activity'].to_numpy()
        ensemble = XGBEnsemble()
        ensemble.train(features, targets)
        return ensemble, features, compounds

    def test_initialization_sets_default_learning_rates_random_states_and_untrained_state(self, xgb_ensemble):
        """Test XGBEnsemble initialization with default parameters."""
        assert len(xgb_ensemble.learners) == 3
        assert xgb_ensemble.learning_rates == [0.05, 0.1, 0.2]
        assert xgb_ensemble.random_states == [42, 123, 356]
        assert xgb_ensemble.aggregation_method == 'mean'
        assert xgb_ensemble.uncertainty_method == 'std'
        assert not xgb_ensemble.is_trained
        assert xgb_ensemble.supports_uncertainty() is True

    def test_initialization_with_custom_parameters(self, custom_xgb_ensemble):
        """Test XGBEnsemble initialization with custom hyperparameters."""
        assert len(custom_xgb_ensemble.learners) == 3
        assert custom_xgb_ensemble.learning_rates == [0.01, 0.05, 0.1]
        assert custom_xgb_ensemble.random_states == [10, 20, 30]
        assert custom_xgb_ensemble.supports_uncertainty() is True

    def test_diverse_hyperparameters(self, xgb_ensemble):
        """Test that ensemble models have different learning rates."""
        learning_rates = []
        random_states = []

        for learner in xgb_ensemble.learners:
            learning_rates.append(learner.learning_rate)
            random_states.append(learner.model.random_state)

        assert len(set(learning_rates)) == 3
        assert learning_rates == [0.05, 0.1, 0.2]
        assert len(set(random_states)) == 3
        assert random_states == [42, 123, 356]

    def test_get_name_includes_model_count_and_learning_rates(self, xgb_ensemble):
        """Test name generation for XGBEnsemble."""
        name = xgb_ensemble.get_name()
        assert name == "XGBEnsemble(3xXGB,lr=[0.05,0.10,0.20])"
        assert "XGBEnsemble" in name
        assert "3xXGB" in name
        assert "lr=[" in name

    def test_get_name_custom_params(self, custom_xgb_ensemble):
        """Test name generation with custom parameters."""
        name = custom_xgb_ensemble.get_name()
        assert name == "XGBEnsemble(3xXGB,lr=[0.01,0.05,0.10])"

    def test_consistency_across_predictions(self, trained_xgb):
        """Test that predictions are consistent across multiple calls."""
        ensemble, features, _compounds = trained_xgb

        predictions1, uncertainty1 = ensemble.predict(features)
        predictions2, uncertainty2 = ensemble.predict(features)

        np.testing.assert_array_equal(predictions1, predictions2)
        np.testing.assert_array_equal(uncertainty1, uncertainty2)

    def test_mismatched_learning_rates_and_random_states(self):
        """Test behavior when learning rates and random states don't match."""
        ensemble = XGBEnsemble(learning_rates=[0.1, 0.2], random_states=[42])
        assert len(ensemble.learners) == 1

    def test_prediction_quality_improves_with_training_data(self, trained_xgb):
        """Test that predictions are reasonable with sufficient training data."""
        ensemble, features, compounds = trained_xgb
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.min() >= compounds['Activity'].min() - 0.5
        assert predictions.max() <= compounds['Activity'].max() + 0.5

        activity_std = compounds['Activity'].std()
        assert np.mean(uncertainty) < activity_std
        assert np.max(uncertainty) < 5 * activity_std

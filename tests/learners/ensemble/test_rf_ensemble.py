"""Tests for RFEnsemble implementation."""

import warnings

import numpy as np
import polars as pl
import pytest

from learnm8.exceptions import LearnerError
from learnm8.learners.ensemble.rf_ensemble import RFEnsemble
from learnm8.learners.sklearn.random_forest import RandomForestLearner


@pytest.mark.integration
class TestRFEnsemble:
    """Test RFEnsemble functionality with real molecular data."""

    @pytest.fixture
    def rf_ensemble(self):
        """Create RFEnsemble instance for testing."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return RFEnsemble(n_estimators=10)

    @pytest.fixture
    def custom_rf_ensemble(self):
        """Create RFEnsemble with custom parameters."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return RFEnsemble(n_estimators=20, random_states=[1, 2, 3])

    def test_deprecation_warning(self):
        """Test that RFEnsemble instantiation emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match='RFEnsemble is deprecated'):
            RFEnsemble(n_estimators=10)

    def test_initialization_sets_default_estimator_count_random_states_and_untrained_state(self, rf_ensemble):
        """Test RFEnsemble initialization with default parameters."""
        assert len(rf_ensemble.learners) == 3
        assert rf_ensemble.n_estimators == 10
        assert rf_ensemble.random_states == [42, 123, 356]
        assert rf_ensemble.aggregation_method == 'mean'
        assert rf_ensemble.uncertainty_method == 'std'
        assert not rf_ensemble.is_trained
        assert rf_ensemble.supports_uncertainty() is True

    def test_initialization_custom_random_states(self):
        """Test initialization with custom random states."""
        custom_states = [10, 20, 30, 40]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            ensemble = RFEnsemble(n_estimators=15, random_states=custom_states)

        assert len(ensemble.learners) == 4
        assert ensemble.n_estimators == 15
        assert ensemble.random_states == custom_states
        assert all(
            isinstance(learner, RandomForestLearner) for learner in ensemble.learners
        )

    def test_diverse_hyperparameters(self, rf_ensemble):
        """Test that ensemble models have different random states for diversity."""
        random_states = []
        for learner in rf_ensemble.learners:
            assert isinstance(learner, RandomForestLearner)
            random_states.append(learner.random_state)

        assert len(set(random_states)) == 3
        assert random_states == [42, 123, 356]

    def test_get_name_includes_model_count_and_estimator_count(self, rf_ensemble):
        """Test name generation."""
        name = rf_ensemble.get_name()
        assert 'RFEnsemble' in name
        assert '3xRF' in name
        assert 'n_est=10' in name

    def test_get_name_custom_ensemble(self, custom_rf_ensemble):
        """Test name generation with custom parameters."""
        name = custom_rf_ensemble.get_name()
        assert 'RFEnsemble' in name
        assert '3xRF' in name
        assert 'n_est=20' in name

    def test_ensemble_consistency(
        self, rf_ensemble, small_real_compounds, small_real_morgan_features
    ):
        """Test that ensemble predictions are consistent across multiple calls."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        rf_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions1, uncertainty1 = rf_ensemble.predict(features)
        predictions2, uncertainty2 = rf_ensemble.predict(features)

        np.testing.assert_array_almost_equal(predictions1, predictions2)
        np.testing.assert_array_almost_equal(uncertainty1, uncertainty2)

    def test_prediction_variance_across_models(
        self, rf_ensemble, small_real_compounds, small_real_morgan_features
    ):
        """Test that ensemble provides uncertainty estimates from model diversity."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        rf_ensemble.train(features, compounds['Activity'].to_numpy())
        _predictions, uncertainty = rf_ensemble.predict(features)

        assert uncertainty is not None
        assert len(uncertainty) == len(compounds)
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_large_ensemble(self, small_real_compounds, small_real_morgan_features):
        """Test RFEnsemble with larger number of models."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        random_states = list(range(10))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            ensemble = RFEnsemble(n_estimators=5, random_states=random_states)

        assert len(ensemble.learners) == 10

        features = small_real_morgan_features
        ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_mismatched_features_targets(self, rf_ensemble):
        """Test error handling with mismatched features and targets."""
        features = np.random.randn(10, 2048)
        targets = np.random.randn(5)

        with pytest.raises((ValueError, LearnerError)):
            rf_ensemble.train(features, targets)

    def test_uncertainty_magnitude(
        self, rf_ensemble, small_real_compounds, small_real_morgan_features
    ):
        """Test that uncertainty values are reasonable in magnitude."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features
        rf_ensemble.train(features, compounds['Activity'].to_numpy())
        _predictions, uncertainty = rf_ensemble.predict(features)

        activity_std = compounds['Activity'].std()
        assert np.mean(uncertainty) < activity_std
        assert np.max(uncertainty) < 5 * activity_std

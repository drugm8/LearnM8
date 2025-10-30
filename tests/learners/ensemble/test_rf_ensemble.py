"""Tests for RFEnsemble implementation."""

import pytest
import numpy as np
import pandas as pd

from learnm8.learners.ensemble.rf_ensemble import RFEnsemble
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.features.extraction import extract_features


class TestRFEnsemble:
    """Test RFEnsemble functionality with real molecular data."""

    @pytest.fixture
    def rf_ensemble(self):
        """Create RFEnsemble instance for testing."""
        return RFEnsemble(n_estimators=10)

    @pytest.fixture
    def custom_rf_ensemble(self):
        """Create RFEnsemble with custom parameters."""
        return RFEnsemble(n_estimators=20, random_states=[1, 2, 3])

    def test_initialization(self, rf_ensemble):
        """Test RFEnsemble initialization with default parameters."""
        assert len(rf_ensemble.learners) == 3
        assert rf_ensemble.n_estimators == 10
        assert rf_ensemble.random_states == [42, 123, 456]
        assert rf_ensemble.aggregation_method == 'mean'
        assert rf_ensemble.uncertainty_method == 'std'
        assert not rf_ensemble.is_trained
        assert rf_ensemble.supports_uncertainty() is True

    def test_initialization_custom_random_states(self):
        """Test initialization with custom random states."""
        custom_states = [10, 20, 30, 40]
        ensemble = RFEnsemble(n_estimators=15, random_states=custom_states)

        assert len(ensemble.learners) == 4
        assert ensemble.n_estimators == 15
        assert ensemble.random_states == custom_states
        assert all(isinstance(learner, RandomForestLearner) for learner in ensemble.learners)

    def test_diverse_hyperparameters(self, rf_ensemble):
        """Test that ensemble models have different random states for diversity."""
        random_states = []
        for learner in rf_ensemble.learners:
            assert isinstance(learner, RandomForestLearner)
            random_states.append(learner.random_state)

        assert len(set(random_states)) == 3
        assert random_states == [42, 123, 456]

    def test_train_predict_integration(self, rf_ensemble, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        rf_ensemble.train(features, compounds['Activity'].values)
        assert rf_ensemble.is_trained

        predictions, uncertainty = rf_ensemble.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_estimation(self, rf_ensemble, small_real_compounds, tmp_path):
        """Test that RFEnsemble provides uncertainty via std of predictions."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        rf_ensemble.train(features, compounds['Activity'].values)

        predictions, uncertainty = rf_ensemble.predict(features)

        assert uncertainty is not None
        assert len(uncertainty) == len(predictions)
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_supports_uncertainty(self, rf_ensemble):
        """Test that RFEnsemble supports uncertainty estimation."""
        assert rf_ensemble.supports_uncertainty() is True

    def test_train_with_empty_arrays(self, rf_ensemble):
        """Test error handling with empty training data."""
        empty_features = np.array([]).reshape(0, 2048)
        empty_targets = np.array([])

        with pytest.raises((ValueError, RuntimeError)):
            rf_ensemble.train(empty_features, empty_targets)

    def test_predict_without_training(self, rf_ensemble, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].tolist(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Ensemble must be trained before prediction"):
            rf_ensemble.predict(features)

    def test_get_name(self, rf_ensemble):
        """Test name generation."""
        name = rf_ensemble.get_name()
        assert "RFEnsemble" in name
        assert "3xRF" in name
        assert "n_est=10" in name

    def test_get_name_custom_ensemble(self, custom_rf_ensemble):
        """Test name generation with custom parameters."""
        name = custom_rf_ensemble.get_name()
        assert "RFEnsemble" in name
        assert "3xRF" in name
        assert "n_est=20" in name

    def test_ensemble_consistency(self, rf_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble predictions are consistent across multiple calls."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        rf_ensemble.train(features, compounds['Activity'].values)

        predictions1, uncertainty1 = rf_ensemble.predict(features)
        predictions2, uncertainty2 = rf_ensemble.predict(features)

        np.testing.assert_array_almost_equal(predictions1, predictions2)
        np.testing.assert_array_almost_equal(uncertainty1, uncertainty2)

    def test_individual_predictions(self, rf_ensemble, small_real_compounds, tmp_path):
        """Test individual learner predictions retrieval."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        rf_ensemble.train(features, compounds['Activity'].values)
        individual_preds = rf_ensemble.get_individual_predictions(features)

        assert len(individual_preds) >= 1
        for learner_name, preds in individual_preds.items():
            assert preds is not None
            assert len(preds) == len(compounds)
            assert np.all(np.isfinite(preds))

    def test_ensemble_statistics(self, rf_ensemble, small_real_compounds, tmp_path):
        """Test ensemble statistics retrieval."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        stats = rf_ensemble.get_ensemble_statistics()
        assert stats['n_learners'] == 3
        assert stats['is_trained'] is False

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        rf_ensemble.train(features, compounds['Activity'].values)
        stats = rf_ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert len(stats['learner_names']) == 3

    def test_edge_case_single_compound(self, rf_ensemble, tmp_path):
        """Test with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].tolist(), 'morgan', tmp_path)
        rf_ensemble.train(features, single_compound['Activity'].values)
        predictions, uncertainty = rf_ensemble.predict(features)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_prediction_variance_across_models(self, rf_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble provides uncertainty estimates from model diversity."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        rf_ensemble.train(features, compounds['Activity'].values)
        predictions, uncertainty = rf_ensemble.predict(features)

        assert uncertainty is not None
        assert len(uncertainty) == len(compounds)
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_weighted_ensemble(self, small_real_compounds, tmp_path):
        """Test weighted RFEnsemble aggregation."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        weights = [0.5, 0.3, 0.2]
        ensemble = RFEnsemble(n_estimators=10, weights=weights)

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].values)
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_different_aggregation_methods(self, small_real_compounds, tmp_path):
        """Test different aggregation methods with RFEnsemble."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)

        for method in ['mean', 'median']:
            ensemble = RFEnsemble(n_estimators=10, aggregation_method=method)
            ensemble.train(features, compounds['Activity'].values)
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_different_uncertainty_methods(self, small_real_compounds, tmp_path):
        """Test different uncertainty estimation methods with RFEnsemble."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)

        for method in ['std', 'mad', 'quantile']:
            ensemble = RFEnsemble(n_estimators=10, uncertainty_method=method)
            ensemble.train(features, compounds['Activity'].values)
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

    def test_large_ensemble(self, small_real_compounds, tmp_path):
        """Test RFEnsemble with larger number of models."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        random_states = list(range(10))
        ensemble = RFEnsemble(n_estimators=5, random_states=random_states)

        assert len(ensemble.learners) == 10

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].values)
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_mismatched_features_targets(self, rf_ensemble):
        """Test error handling with mismatched features and targets."""
        features = np.random.randn(10, 2048)
        targets = np.random.randn(5)

        with pytest.raises(ValueError):
            rf_ensemble.train(features, targets)

    def test_add_remove_learners(self, rf_ensemble):
        """Test adding and removing learners from ensemble."""
        initial_count = len(rf_ensemble.learners)

        new_learner = RandomForestLearner(n_estimators=5, random_state=999)
        rf_ensemble.add_learner(new_learner)

        assert len(rf_ensemble.learners) == initial_count + 1
        assert not rf_ensemble.is_trained

        rf_ensemble.remove_learner(0)
        assert len(rf_ensemble.learners) == initial_count
        assert not rf_ensemble.is_trained

    def test_uncertainty_magnitude(self, rf_ensemble, small_real_compounds, tmp_path):
        """Test that uncertainty values are reasonable in magnitude."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        rf_ensemble.train(features, compounds['Activity'].values)
        predictions, uncertainty = rf_ensemble.predict(features)

        assert np.mean(uncertainty) < np.std(compounds['Activity'])
        assert np.max(uncertainty) < 5 * np.std(compounds['Activity'])

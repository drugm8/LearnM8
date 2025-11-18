"""Tests for DTEnsemble implementation."""

import pytest
import numpy as np
import polars as pl

from learnm8.learners.ensemble.dt_ensemble import DTEnsemble
from learnm8.learners.sklearn.decision_tree import DecisionTreeLearner
from learnm8.features.extraction import extract_features


class TestDTEnsemble:
    """Test DTEnsemble functionality with real molecular data."""

    @pytest.fixture
    def dt_ensemble(self):
        """Create DTEnsemble instance for testing."""
        return DTEnsemble()

    def test_initialization(self, dt_ensemble):
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

    def test_train_predict_integration(self, dt_ensemble, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        dt_ensemble.train(features, compounds['Activity'].to_numpy())
        assert dt_ensemble.is_trained

        predictions, uncertainty = dt_ensemble.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_estimation(self, dt_ensemble, small_real_compounds, tmp_path):
        """Test ensemble SUPPORTS uncertainty estimation."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        dt_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions, uncertainty = dt_ensemble.predict(features)

        assert dt_ensemble.supports_uncertainty() is True
        assert uncertainty is not None
        assert len(uncertainty) == len(predictions)
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_diverse_hyperparameters(self, dt_ensemble):
        """Test models have different max_depth values for diversity."""
        assert len(set(dt_ensemble.max_depths)) == len(dt_ensemble.max_depths)
        assert len(set(dt_ensemble.random_states)) == len(dt_ensemble.random_states)

        for i, learner in enumerate(dt_ensemble.learners):
            assert learner.max_depth == dt_ensemble.max_depths[i]
            assert learner.random_state == dt_ensemble.random_states[i]

    def test_train_with_empty_arrays(self, dt_ensemble):
        """Test error handling with empty arrays."""
        empty_features = np.array([]).reshape(0, 2048)
        empty_targets = np.array([])

        with pytest.raises(ValueError):
            dt_ensemble.train(empty_features, empty_targets)

    def test_predict_without_training(self, dt_ensemble, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Ensemble must be trained before prediction"):
            dt_ensemble.predict(features)

    def test_get_name(self, dt_ensemble):
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

    def test_supports_uncertainty(self, dt_ensemble):
        """Test uncertainty support flag."""
        assert dt_ensemble.supports_uncertainty() is True

    def test_train_with_mismatched_arrays(self, dt_ensemble, small_real_compounds, tmp_path):
        """Test error handling with mismatched feature and target sizes."""
        features = extract_features(small_real_compounds['SMILES'].to_list()[:10], 'morgan', tmp_path)
        targets = np.random.beta(2, 5, 15)

        with pytest.raises(ValueError):
            dt_ensemble.train(features, targets)

    def test_prediction_variance(self, small_real_compounds, tmp_path):
        """Test that different max_depth models produce diverse predictions."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        ensemble = DTEnsemble(max_depths=[3, 10, 20], random_states=[42, 42, 42])
        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())

        individual_preds = ensemble.get_individual_predictions(features)
        assert len(individual_preds) == 3

        predictions_array = np.array([preds for preds in individual_preds.values() if preds is not None])
        assert predictions_array.shape[0] == 3
        assert predictions_array.shape[1] == len(compounds)

        variances = np.var(predictions_array, axis=0)
        assert np.mean(variances) > 0

    def test_ensemble_statistics(self, dt_ensemble, small_real_compounds, tmp_path):
        """Test ensemble statistics retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        stats = dt_ensemble.get_ensemble_statistics()
        assert stats['n_learners'] == 3
        assert stats['is_trained'] is False

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        dt_ensemble.train(features, compounds['Activity'].to_numpy())
        stats = dt_ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert len(stats['learner_names']) == 3

    def test_edge_case_single_compound(self, dt_ensemble, tmp_path):
        """Test with single compound."""
        single_compound = pl.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].to_list(), 'morgan', tmp_path)
        dt_ensemble.train(features, single_compound['Activity'].to_numpy())
        predictions, uncertainty = dt_ensemble.predict(features)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_aggregation_methods(self, small_real_compounds, tmp_path):
        """Test different aggregation methods with DTEnsemble."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['mean', 'median']:
            ensemble = DTEnsemble(aggregation_method=method)
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_uncertainty_methods(self, small_real_compounds, tmp_path):
        """Test different uncertainty estimation methods with DTEnsemble."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['std', 'mad', 'quantile']:
            ensemble = DTEnsemble(uncertainty_method=method)
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

    def test_weighted_ensemble(self, small_real_compounds, tmp_path):
        """Test weighted ensemble aggregation with DTEnsemble."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        weights = [0.5, 0.3, 0.2]
        ensemble = DTEnsemble(weights=weights)

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)
        assert np.allclose(ensemble.weights, weights)

    def test_individual_predictions(self, dt_ensemble, small_real_compounds, tmp_path):
        """Test individual learner predictions retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        dt_ensemble.train(features, compounds['Activity'].to_numpy())
        individual_preds = dt_ensemble.get_individual_predictions(features)

        assert len(individual_preds) == 3
        for learner_name, preds in individual_preds.items():
            assert preds is not None
            assert len(preds) == len(compounds)
            assert np.all(np.isfinite(preds))

    def test_consistency_with_fixed_random_state(self, small_real_compounds, tmp_path):
        """Test prediction consistency with fixed random states."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

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

    def test_failed_learner_handling(self, small_real_compounds, tmp_path):
        """Test handling of failed learners during training."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        from learnm8.core.interfaces import Learner

        class MockBadLearner(Learner):
            def train(self, features, targets):
                raise Exception("Training failed")
            def predict(self, features):
                return np.random.randn(len(features)), None
            def get_name(self):
                return "BadLearner"
            def supports_uncertainty(self):
                return False

        good_learner = DecisionTreeLearner(max_depth=5, random_state=42)
        bad_learner = MockBadLearner()

        ensemble = DTEnsemble(max_depths=[5], random_states=[42])
        ensemble.add_learner(bad_learner)

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())

        assert ensemble.is_trained
        assert len(ensemble.learners) < 2

    def test_uncertainty_diversity(self, dt_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble uncertainty captures model diversity."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        dt_ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = dt_ensemble.predict(features)

        assert np.std(uncertainty) > 0
        assert np.all(uncertainty >= 0)
        assert np.mean(uncertainty) > 0

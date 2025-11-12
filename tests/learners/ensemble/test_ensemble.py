"""Tests for EnsembleLearner implementation."""

import pytest
import numpy as np
import polars as pl
from unittest.mock import Mock

from learnm8.learners.ensemble.ensemble import EnsembleLearner
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.features.extraction import extract_features


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
    
    def test_initialization(self, ensemble, base_learners):
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
        # Wrong number of weights
        with pytest.raises(ValueError, match="Number of weights must match"):
            EnsembleLearner(base_learners, weights=[0.5])
        
        # Weights don't sum to 1
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            EnsembleLearner(base_learners, weights=[0.3, 0.4])
    
    def test_empty_learners_list(self):
        """Test error handling with empty learners list."""
        with pytest.raises(ValueError, match="At least one learner must be provided"):
            EnsembleLearner([])
    
    def test_train_predict_integration(self, ensemble, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        assert ensemble.is_trained

        predictions, uncertainty = ensemble.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_aggregation_methods(self, base_learners, small_real_compounds, tmp_path):
        """Test different aggregation methods."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['mean', 'median']:
            ensemble = EnsembleLearner(base_learners, aggregation_method=method)
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_uncertainty_methods(self, base_learners, small_real_compounds, tmp_path):
        """Test different uncertainty estimation methods."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['std', 'mad', 'quantile']:
            ensemble = EnsembleLearner(base_learners, uncertainty_method=method)
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

    def test_weighted_ensemble(self, base_learners, small_real_compounds, tmp_path):
        """Test weighted ensemble aggregation."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        weights = [0.8, 0.2]
        ensemble = EnsembleLearner(base_learners, weights=weights)

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_predict_without_training(self, ensemble, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Ensemble must be trained before prediction"):
            ensemble.predict(features)
    
    def test_get_name(self, ensemble):
        """Test name generation."""
        name = ensemble.get_name()
        assert "Ensemble" in name
        assert "mean" in name
        assert "+" in name  # Should show combined learner names
    
    def test_ensemble_statistics(self, ensemble, small_real_compounds, tmp_path):
        """Test ensemble statistics retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        stats = ensemble.get_ensemble_statistics()
        assert stats['n_learners'] == 2
        assert stats['is_trained'] is False

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        stats = ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert 'learners_with_uncertainty' in stats

    def test_individual_predictions(self, ensemble, small_real_compounds, tmp_path):
        """Test individual learner predictions retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        individual_preds = ensemble.get_individual_predictions(features)

        assert len(individual_preds) == 2
        for learner_name, preds in individual_preds.items():
            if preds is not None:
                assert len(preds) == len(compounds)

    def test_add_learner(self, ensemble):
        """Test adding learners to ensemble."""
        initial_count = len(ensemble.learners)

        new_learner = RandomForestLearner(n_estimators=3, random_state=123)
        ensemble.add_learner(new_learner)

        assert len(ensemble.learners) == initial_count + 1
        assert not ensemble.is_trained
    
    def test_remove_learner(self, ensemble):
        """Test removing learners from ensemble."""
        initial_count = len(ensemble.learners)

        ensemble.remove_learner(0)

        assert len(ensemble.learners) == initial_count - 1
        assert not ensemble.is_trained

    def test_failed_learner_handling(self, small_real_compounds, tmp_path):
        """Test handling of failed learners during training."""
        compounds = small_real_compounds.clone()
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
                raise Exception("Training failed")
            def predict(self, features):
                return np.random.randn(len(features)), None
            def get_name(self):
                return "BadLearner"
            def supports_uncertainty(self):
                return False

        good_learner = MockGoodLearner()
        bad_learner = MockBadLearner()

        ensemble = EnsembleLearner([good_learner, bad_learner])

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        assert len(ensemble.learners) == 1
        assert ensemble.is_trained

    def test_edge_case_single_compound(self, ensemble, tmp_path):
        """Test with single compound."""
        single_compound = pl.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, single_compound['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_uncertainty_diversity(self, base_learners, small_real_compounds, tmp_path):
        """Test that ensemble uncertainty captures model diversity."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        ensemble = EnsembleLearner(base_learners)

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert np.std(uncertainty) > 0
        assert np.all(uncertainty >= 0)

    def test_uncertainty_consistency(self, base_learners, small_real_compounds, tmp_path):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(pl.Series('Activity', np.random.beta(2, 5, len(compounds))))

        ensemble = EnsembleLearner(base_learners)

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())

        predictions, uncertainty = ensemble.predict(features)

        assert ensemble.supports_uncertainty() is True
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)

    def test_train_with_empty_arrays(self, base_learners):
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])
        ensemble = EnsembleLearner(base_learners)

        with pytest.raises(ValueError, match="Cannot train on empty dataset"):
            ensemble.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, base_learners):
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)
        ensemble = EnsembleLearner(base_learners)

        with pytest.raises(ValueError, match="Features and targets must have same length"):
            ensemble.train(features, targets)

    def test_train_with_1d_features(self, base_learners):
        features_1d = np.random.rand(10)
        targets = np.random.rand(10)
        ensemble = EnsembleLearner(base_learners)

        with pytest.raises((ValueError, RuntimeError)):
            ensemble.train(features_1d, targets)
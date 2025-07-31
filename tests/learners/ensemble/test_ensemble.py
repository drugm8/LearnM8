"""Tests for EnsembleLearner implementation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from learnm8.learners.ensemble.ensemble import EnsembleLearner
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.core.data_manager import DataManager


class TestEnsembleLearner:
    """Test EnsembleLearner functionality with real molecular data."""
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """Create DataManager for testing."""
        return DataManager(results_dir=tmp_path)
    
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
    
    def test_train_predict_integration(self, ensemble, small_real_compounds, data_manager):
        """Test training and prediction with real molecular data."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add activity column if missing
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Train ensemble
        ensemble.train(compounds, 'Activity', data_manager)
        assert ensemble.is_trained
        
        # Test prediction with uncertainty
        predictions, uncertainty = ensemble.predict(compounds, data_manager)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None  # Ensemble provides uncertainty
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)  # Uncertainty should be non-negative
    
    def test_aggregation_methods(self, base_learners, small_real_compounds, data_manager):
        """Test different aggregation methods."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        for method in ['mean', 'median']:
            ensemble = EnsembleLearner(base_learners, aggregation_method=method)
            ensemble.train(compounds, 'Activity', data_manager)
            predictions, uncertainty = ensemble.predict(compounds, data_manager)
            
            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))
    
    def test_uncertainty_methods(self, base_learners, small_real_compounds, data_manager):
        """Test different uncertainty estimation methods."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        for method in ['std', 'mad', 'quantile']:
            ensemble = EnsembleLearner(base_learners, uncertainty_method=method)
            ensemble.train(compounds, 'Activity', data_manager)
            predictions, uncertainty = ensemble.predict(compounds, data_manager)
            
            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)
    
    def test_weighted_ensemble(self, base_learners, small_real_compounds, data_manager):
        """Test weighted ensemble aggregation."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        weights = [0.8, 0.2]  # Favor first learner
        ensemble = EnsembleLearner(base_learners, weights=weights)
        ensemble.train(compounds, 'Activity', data_manager)
        predictions, uncertainty = ensemble.predict(compounds, data_manager)
        
        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)
    
    def test_train_with_missing_columns(self, ensemble, data_manager):
        """Test error handling with missing required columns."""
        invalid_compounds = pd.DataFrame({
            'compound_id': ['COMP_001', 'COMP_002'],
            'structure': ['CCO', 'CCC']
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            ensemble.train(invalid_compounds, 'Activity', data_manager)
    
    def test_predict_without_training(self, ensemble, small_real_compounds, data_manager):
        """Test error when predicting without training."""
        with pytest.raises(RuntimeError, match="Ensemble must be trained before prediction"):
            ensemble.predict(small_real_compounds, data_manager)
    
    def test_get_name(self, ensemble):
        """Test name generation."""
        name = ensemble.get_name()
        assert "Ensemble" in name
        assert "mean" in name
        assert "+" in name  # Should show combined learner names
    
    def test_ensemble_statistics(self, ensemble, small_real_compounds, data_manager):
        """Test ensemble statistics retrieval."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Before training
        stats = ensemble.get_ensemble_statistics()
        assert stats['n_learners'] == 2
        assert stats['is_trained'] is False
        
        # After training
        ensemble.train(compounds, 'Activity', data_manager)
        stats = ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert 'learners_with_uncertainty' in stats
    
    def test_individual_predictions(self, ensemble, small_real_compounds, data_manager):
        """Test individual learner predictions retrieval."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        ensemble.train(compounds, 'Activity', data_manager)
        individual_preds = ensemble.get_individual_predictions(compounds, data_manager)
        
        assert len(individual_preds) == 2
        for learner_name, preds in individual_preds.items():
            if preds is not None:  # Some predictions might fail
                assert len(preds) == len(compounds)
    
    def test_add_learner(self, ensemble, data_manager):
        """Test adding learners to ensemble."""
        initial_count = len(ensemble.learners)
        
        # Add a new learner
        new_learner = RandomForestLearner(n_estimators=3, random_state=123)
        ensemble.add_learner(new_learner)
        
        assert len(ensemble.learners) == initial_count + 1
        assert not ensemble.is_trained  # Should invalidate training
    
    def test_remove_learner(self, ensemble, data_manager):
        """Test removing learners from ensemble."""
        initial_count = len(ensemble.learners)
        
        # Remove a learner
        ensemble.remove_learner(0)
        
        assert len(ensemble.learners) == initial_count - 1
        assert not ensemble.is_trained  # Should invalidate training
    
    def test_failed_learner_handling(self, small_real_compounds, data_manager):
        """Test handling of failed learners during training."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Create mock learners that inherit from Learner base class
        from learnm8.core.interfaces import Learner
        
        class MockGoodLearner(Learner):
            def train(self, compounds, target_column, data_manager):
                pass
            def predict(self, compounds, data_manager):
                return np.random.randn(len(compounds)), None
            def get_name(self):
                return "GoodLearner"
            def supports_uncertainty(self):
                return False
        
        class MockBadLearner(Learner):
            def train(self, compounds, target_column, data_manager):
                raise Exception("Training failed")
            def predict(self, compounds, data_manager):
                return np.random.randn(len(compounds)), None
            def get_name(self):
                return "BadLearner"
            def supports_uncertainty(self):
                return False
        
        good_learner = MockGoodLearner()
        bad_learner = MockBadLearner()
        
        ensemble = EnsembleLearner([good_learner, bad_learner])
        
        # Training should continue with good learner only
        ensemble.train(compounds, 'Activity', data_manager)
        assert len(ensemble.learners) == 1  # Bad learner should be removed
        assert ensemble.is_trained
    
    def test_edge_case_single_compound(self, ensemble, data_manager):
        """Test with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        ensemble.train(single_compound, 'Activity', data_manager)
        predictions, uncertainty = ensemble.predict(single_compound, data_manager)
        
        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0
    
    def test_uncertainty_diversity(self, base_learners, small_real_compounds, data_manager):
        """Test that ensemble uncertainty captures model diversity."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        ensemble = EnsembleLearner(base_learners)
        ensemble.train(compounds, 'Activity', data_manager)
        predictions, uncertainty = ensemble.predict(compounds, data_manager)
        
        # Uncertainty should vary across predictions
        assert np.std(uncertainty) > 0  # Some variation in uncertainty
        assert np.all(uncertainty >= 0)  # Non-negative uncertainty
    
    def test_data_manager_integration(self, ensemble, small_real_compounds):
        """Test proper integration with DataManager."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Mock DataManager to test integration
        mock_dm = Mock()
        mock_dm.prepare_training_data.return_value = (
            np.random.randn(len(compounds), 100),  # Mock features
            compounds['Activity'].values
        )
        mock_dm.prepare_prediction_data.return_value = np.random.randn(len(compounds), 100)
        
        # Mock individual learners
        for learner in ensemble.learners:
            learner.train = Mock()
            learner.predict = Mock(return_value=(np.random.randn(len(compounds)), None))
        
        # Train and predict
        ensemble.train(compounds, 'Activity', mock_dm)
        predictions, uncertainty = ensemble.predict(compounds, mock_dm)
        
        # Verify learners were called
        for learner in ensemble.learners:
            learner.train.assert_called_once()
            learner.predict.assert_called_once()
        
        assert len(predictions) == len(compounds)
        assert len(uncertainty) == len(compounds)
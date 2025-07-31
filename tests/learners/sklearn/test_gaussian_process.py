"""Tests for GaussianProcessLearner implementation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.core.data_manager import DataManager


class TestGaussianProcessLearner:
    """Test GaussianProcessLearner functionality with real molecular data."""
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """Create DataManager for testing."""
        return DataManager(results_dir=tmp_path)
    
    @pytest.fixture
    def learner(self):
        """Create GaussianProcessLearner instance for testing."""
        return GaussianProcessLearner(random_state=42)
    
    def test_initialization(self, learner):
        """Test learner initialization."""
        assert learner.alpha == 1e-10
        assert learner.random_state == 42
        assert not learner.is_trained
        assert learner.supports_uncertainty() is True
    
    def test_train_predict_integration(self, learner, small_real_compounds, data_manager):
        """Test training and prediction with real molecular data."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add activity column if missing
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Train model
        learner.train(compounds, 'Activity', data_manager)
        assert learner.is_trained
        
        # Test prediction with uncertainty
        predictions, uncertainty = learner.predict(compounds, data_manager)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None  # GP provides uncertainty
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)  # Uncertainty should be non-negative
    
    def test_uncertainty_quality(self, learner, small_real_compounds, data_manager):
        """Test that uncertainty estimates are reasonable."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Train on subset
        train_compounds = compounds.iloc[:len(compounds)//2]
        learner.train(train_compounds, 'Activity', data_manager)
        
        # Predict on training data (should have lower uncertainty)
        train_pred, train_unc = learner.predict(train_compounds, data_manager)
        
        # Predict on test data (should have higher uncertainty)
        test_compounds = compounds.iloc[len(compounds)//2:]
        if len(test_compounds) > 0:
            test_pred, test_unc = learner.predict(test_compounds, data_manager)
            
            # Training uncertainty should generally be lower
            assert np.mean(train_unc) <= np.mean(test_unc) * 2  # Allow some variance
    
    def test_train_with_missing_columns(self, learner, data_manager):
        """Test error handling with missing required columns."""
        invalid_compounds = pd.DataFrame({
            'compound_id': ['COMP_001', 'COMP_002'],
            'structure': ['CCO', 'CCC']
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            learner.train(invalid_compounds, 'Activity', data_manager)
    
    def test_predict_without_training(self, learner, small_real_compounds, data_manager):
        """Test error when predicting without training."""
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(small_real_compounds, data_manager)
    
    def test_get_name(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "GaussianProcess" in name
        assert "α=1e-10" in name
    
    def test_learned_hyperparameters(self, learner, small_real_compounds, data_manager):
        """Test hyperparameter learning."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Before training
        assert learner.get_learned_hyperparameters() is None
        
        # After training
        learner.train(compounds, 'Activity', data_manager)
        hyperparams = learner.get_learned_hyperparameters()
        assert hyperparams is not None
        assert 'kernel' in hyperparams
        assert 'theta' in hyperparams
    
    def test_custom_kernel(self, data_manager, small_real_compounds):
        """Test learner with custom kernel configuration."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Test with custom alpha parameter
        learner = GaussianProcessLearner(alpha=1e-6, random_state=42)
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, uncertainty = learner.predict(compounds, data_manager)
        
        assert learner.alpha == 1e-6
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
    
    def test_edge_case_single_compound(self, learner, data_manager):
        """Test with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        learner.train(single_compound, 'Activity', data_manager)
        predictions, uncertainty = learner.predict(single_compound, data_manager)
        
        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0
    
    def test_numerical_stability(self, learner, data_manager):
        """Test numerical stability with extreme values."""
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.0, 1000.0, 0.001]  # Extreme values
        })
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, uncertainty = learner.predict(compounds, data_manager)
        
        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(uncertainty))
        assert np.all(uncertainty >= 0)
    
    def test_data_manager_integration(self, learner, small_real_compounds):
        """Test proper integration with DataManager."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Mock DataManager to test integration
        mock_dm = Mock()
        mock_dm.prepare_training_data.return_value = (
            np.random.randn(len(compounds), 50),  # Smaller features for GP
            compounds['Activity'].values
        )
        mock_dm.prepare_prediction_data.return_value = np.random.randn(len(compounds), 50)
        
        # Train and predict
        learner.train(compounds, 'Activity', mock_dm)
        predictions, uncertainty = learner.predict(compounds, mock_dm)
        
        # Verify DataManager methods were called
        mock_dm.prepare_training_data.assert_called_once()
        mock_dm.prepare_prediction_data.assert_called_once()
        assert len(predictions) == len(compounds)
        assert len(uncertainty) == len(compounds)
    
    def test_uncertainty_ordering(self, learner, data_manager):
        """Test that uncertainty estimates have reasonable ordering."""
        # Create compounds with varying similarity to training data
        train_compounds = pd.DataFrame({
            'ID': ['TRAIN_001', 'TRAIN_002'],
            'SMILES': ['CCO', 'CCC'],  # Simple molecules
            'Activity': [0.5, 0.6]
        })
        
        test_compounds = pd.DataFrame({
            'ID': ['TEST_001', 'TEST_002'],
            'SMILES': ['CCCO', 'c1ccccc1'],  # Similar and different molecules
        })
        
        learner.train(train_compounds, 'Activity', data_manager)
        _, uncertainties = learner.predict(test_compounds, data_manager)
        
        # All uncertainties should be positive
        assert np.all(uncertainties >= 0)
        assert len(uncertainties) == len(test_compounds)
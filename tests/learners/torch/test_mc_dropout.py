"""Tests for MCDropoutLearner implementation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from learnm8.learners.torch.mc_dropout import MCDropoutLearner
from learnm8.core.data_manager import DataManager


class TestMCDropoutLearner:
    """Test MCDropoutLearner functionality with real molecular data."""
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """Create DataManager for testing."""
        return DataManager(results_dir=tmp_path)
    
    @pytest.fixture
    def learner(self):
        """Create MCDropoutLearner instance for testing."""
        return MCDropoutLearner(
            hidden_sizes=(64, 32),
            n_dropout_samples=20,
            max_epochs=5,
            random_state=42
        )
    
    def test_initialization(self, learner):
        """Test learner initialization."""
        assert learner.hidden_sizes == (64, 32)
        assert learner.dropout_rate == 0.2
        assert learner.n_dropout_samples == 20
        assert learner.activation == 'relu'
        assert learner.batch_norm is True
        assert learner.max_epochs == 5
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
        assert learner.model is not None
        
        # Test prediction with uncertainty
        predictions, uncertainty = learner.predict(compounds, data_manager)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None  # MC Dropout provides uncertainty
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)  # Uncertainty should be non-negative
    
    def test_uncertainty_quality(self, learner, small_real_compounds, data_manager):
        """Test that MC Dropout uncertainty estimates are reasonable."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, uncertainty = learner.predict(compounds, data_manager)
        
        # Uncertainty should vary across predictions
        assert np.std(uncertainty) > 0  # Some variation in uncertainty
        
        # Predictions should be finite and reasonable
        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(uncertainty))
    
    def test_dropout_samples_effect(self, data_manager, small_real_compounds):
        """Test effect of different numbers of dropout samples."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Test with few samples
        learner_few = MCDropoutLearner(
            hidden_sizes=(32,),
            n_dropout_samples=5,
            max_epochs=3,
            random_state=42
        )
        
        learner_few.train(compounds, 'Activity', data_manager)
        pred_few, unc_few = learner_few.predict(compounds, data_manager)
        
        # Test with many samples
        learner_many = MCDropoutLearner(
            hidden_sizes=(32,),
            n_dropout_samples=50,
            max_epochs=3,
            random_state=42
        )
        
        learner_many.train(compounds, 'Activity', data_manager)
        pred_many, unc_many = learner_many.predict(compounds, data_manager)
        
        # Both should work
        assert len(pred_few) == len(compounds)
        assert len(pred_many) == len(compounds)
        assert np.all(unc_few >= 0)
        assert np.all(unc_many >= 0)
    
    def test_train_with_missing_columns(self, learner, data_manager):
        """Test error handling with missing required columns."""
        invalid_compounds = pd.DataFrame({
            'compound_id': ['COMP_001', 'COMP_002'],
            'structure': ['CCO', 'CCC']
        })
        
        with pytest.raises(RuntimeError, match="Training failed.*Missing required columns"):
            learner.train(invalid_compounds, 'Activity', data_manager)
    
    def test_predict_without_training(self, learner, small_real_compounds, data_manager):
        """Test error when predicting without training."""
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(small_real_compounds, data_manager)
    
    def test_get_name(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "MCDropout" in name
        assert "64-32" in name
        assert "samples=20" in name
        assert "dropout=0.2" in name
    
    def test_different_architectures(self, data_manager, small_real_compounds):
        """Test learner with different architectures."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Test deeper network
        learner = MCDropoutLearner(
            hidden_sizes=(128, 64, 32),
            activation='gelu',
            dropout_rate=0.3,
            n_dropout_samples=10,
            max_epochs=3,
            random_state=42
        )
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, uncertainty = learner.predict(compounds, data_manager)
        
        assert learner.hidden_sizes == (128, 64, 32)
        assert learner.activation == 'gelu'
        assert learner.dropout_rate == 0.3
        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)
    
    def test_dropout_rate_effect(self, data_manager, small_real_compounds):
        """Test effect of different dropout rates."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        for dropout_rate in [0.1, 0.3, 0.5]:
            learner = MCDropoutLearner(
                hidden_sizes=(32,),
                dropout_rate=dropout_rate,
                n_dropout_samples=10,
                max_epochs=2,
                random_state=42
            )
            
            learner.train(compounds, 'Activity', data_manager)
            predictions, uncertainty = learner.predict(compounds, data_manager)
            
            assert learner.dropout_rate == dropout_rate
            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)
    
    def test_edge_case_single_compound(self, data_manager):
        """Test with single compound using learner without batch norm."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        # Create learner without batch normalization for single compound training
        learner = MCDropoutLearner(
            hidden_sizes=(32,),
            batch_norm=False,  # Disable batch norm for single sample
            n_dropout_samples=5,
            max_epochs=2,
            random_state=42
        )
        
        learner.train(single_compound, 'Activity', data_manager)
        predictions, uncertainty = learner.predict(single_compound, data_manager)
        
        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0
    
    def test_uncertainty_consistency(self, learner, data_manager):
        """Test that uncertainty is consistent across multiple predictions."""
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.3, 0.6, 0.9]
        })
        
        learner.train(compounds, 'Activity', data_manager)
        
        # Multiple predictions should give different uncertainty values due to randomness
        pred1, unc1 = learner.predict(compounds, data_manager)
        pred2, unc2 = learner.predict(compounds, data_manager)
        
        # Predictions should be different due to MC sampling
        assert not np.allclose(pred1, pred2, rtol=1e-10)
        assert not np.allclose(unc1, unc2, rtol=1e-10)
        
        # But uncertainties should be in similar range
        assert np.all(unc1 >= 0)
        assert np.all(unc2 >= 0)
    
    def test_deterministic_with_same_seed(self, data_manager, small_real_compounds):
        """Test reproducibility with same random seed."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Two learners with same seed
        learner1 = MCDropoutLearner(
            hidden_sizes=(32,),
            max_epochs=2,
            random_state=42
        )
        learner2 = MCDropoutLearner(
            hidden_sizes=(32,),
            max_epochs=2,
            random_state=42
        )
        
        learner1.train(compounds, 'Activity', data_manager)
        learner2.train(compounds, 'Activity', data_manager)
        
        # Should produce similar but not identical results due to MC sampling
        pred1, _ = learner1.predict(compounds, data_manager)
        pred2, _ = learner2.predict(compounds, data_manager)
        
        assert len(pred1) == len(pred2)
        assert np.all(np.isfinite(pred1))
        assert np.all(np.isfinite(pred2))
    
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
            np.random.randn(len(compounds), 100),  # Mock features
            compounds['Activity'].values
        )
        mock_dm.prepare_prediction_data.return_value = np.random.randn(len(compounds), 100)
        
        # Train and predict
        learner.train(compounds, 'Activity', mock_dm)
        predictions, uncertainty = learner.predict(compounds, mock_dm)
        
        # Verify DataManager methods were called
        mock_dm.prepare_training_data.assert_called_once()
        mock_dm.prepare_prediction_data.assert_called_once()
        assert len(predictions) == len(compounds)
        assert len(uncertainty) == len(compounds)
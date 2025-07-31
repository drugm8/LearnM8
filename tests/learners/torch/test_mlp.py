"""Tests for MLPLearner implementation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from learnm8.learners.torch.mlp import MLPLearner
from learnm8.core.data_manager import DataManager


class TestMLPLearner:
    """Test MLPLearner functionality with real molecular data."""
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """Create DataManager for testing."""
        return DataManager(results_dir=tmp_path)
    
    @pytest.fixture
    def learner(self):
        """Create MLPLearner instance for testing."""
        return MLPLearner(
            hidden_sizes=(64, 32),
            max_epochs=5,
            random_state=42
        )
    
    def test_initialization(self, learner):
        """Test learner initialization."""
        assert learner.hidden_sizes == (64, 32)
        assert learner.activation == 'relu'
        assert learner.dropout_rate == 0.2
        assert learner.batch_norm is True
        assert learner.max_epochs == 5
        assert not learner.is_trained
        assert learner.supports_uncertainty() is False
    
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
        
        # Test prediction
        predictions, uncertainty = learner.predict(compounds, data_manager)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is None  # MLP doesn't support uncertainty
        assert np.all(np.isfinite(predictions))
    
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
        assert "MLP" in name
        assert "64-32" in name
        assert "relu" in name
        assert "dropout=0.2" in name
    
    def test_different_architectures(self, data_manager, small_real_compounds):
        """Test learner with different architectures."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Test deeper network
        learner = MLPLearner(
            hidden_sizes=(128, 64, 32, 16),
            activation='gelu',
            dropout_rate=0.3,
            max_epochs=3,
            random_state=42
        )
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, _ = learner.predict(compounds, data_manager)
        
        assert learner.hidden_sizes == (128, 64, 32, 16)
        assert learner.activation == 'gelu'
        assert predictions.shape[0] == len(compounds)
    
    def test_activation_functions(self, data_manager, small_real_compounds):
        """Test different activation functions."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        for activation in ['relu', 'tanh', 'gelu']:
            learner = MLPLearner(
                hidden_sizes=(32,),
                activation=activation,
                max_epochs=2,
                random_state=42
            )
            
            learner.train(compounds, 'Activity', data_manager)
            predictions, _ = learner.predict(compounds, data_manager)
            
            assert predictions.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))
    
    def test_batch_normalization(self, data_manager, small_real_compounds):
        """Test with and without batch normalization."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        for batch_norm in [True, False]:
            learner = MLPLearner(
                hidden_sizes=(32,),
                batch_norm=batch_norm,
                max_epochs=2,
                random_state=42
            )
            
            learner.train(compounds, 'Activity', data_manager)
            predictions, _ = learner.predict(compounds, data_manager)
            
            assert predictions.shape[0] == len(compounds)
    
    def test_dropout_regularization(self, data_manager, small_real_compounds):
        """Test different dropout rates."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        for dropout_rate in [0.0, 0.1, 0.5]:
            learner = MLPLearner(
                hidden_sizes=(32,),
                dropout_rate=dropout_rate,
                max_epochs=2,
                random_state=42
            )
            
            learner.train(compounds, 'Activity', data_manager)
            predictions, _ = learner.predict(compounds, data_manager)
            
            assert learner.dropout_rate == dropout_rate
            assert predictions.shape[0] == len(compounds)
    
    def test_edge_case_single_compound(self, data_manager):
        """Test with single compound using learner without batch norm."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        # Create learner without batch normalization for single compound training
        learner = MLPLearner(
            hidden_sizes=(32,),
            batch_norm=False,  # Disable batch norm for single sample
            max_epochs=2,
            random_state=42
        )
        
        learner.train(single_compound, 'Activity', data_manager)
        predictions, _ = learner.predict(single_compound, data_manager)
        
        assert len(predictions) == 1
        assert np.isfinite(predictions[0])
    
    def test_training_history(self, learner, small_real_compounds, data_manager):
        """Test training history tracking."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Before training
        assert len(learner.get_training_history()) == 0
        
        # After training
        learner.train(compounds, 'Activity', data_manager)
        history = learner.get_training_history()
        
        assert len(history) > 0
        assert 'epoch' in history[0]
        assert 'train_loss' in history[0]
        assert 'val_loss' in history[0]
    
    def test_early_stopping(self, data_manager, small_real_compounds):
        """Test early stopping mechanism."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        learner = MLPLearner(
            hidden_sizes=(32,),
            max_epochs=100,
            early_stopping_patience=2,
            random_state=42
        )
        
        learner.train(compounds, 'Activity', data_manager)
        history = learner.get_training_history()
        
        # Should stop early due to small dataset
        assert len(history) < 100
    
    def test_gpu_cpu_compatibility(self, data_manager, small_real_compounds):
        """Test device compatibility."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Test CPU explicitly
        learner = MLPLearner(
            hidden_sizes=(16,),
            device='cpu',
            max_epochs=2,
            random_state=42
        )
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, _ = learner.predict(compounds, data_manager)
        
        assert str(learner.device) == 'cpu'
        assert predictions.shape[0] == len(compounds)
    
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
        predictions, _ = learner.predict(compounds, mock_dm)
        
        # Verify DataManager methods were called
        mock_dm.prepare_training_data.assert_called_once()
        mock_dm.prepare_prediction_data.assert_called_once()
        assert len(predictions) == len(compounds)
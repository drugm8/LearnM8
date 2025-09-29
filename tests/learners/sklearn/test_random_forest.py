"""Tests for RandomForestLearner implementation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.core.data_manager import DataManager


class TestRandomForestLearner:
    """Test RandomForestLearner functionality with real molecular data."""
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """Create DataManager for testing."""
        return DataManager(results_dir=tmp_path)
    
    @pytest.fixture
    def learner(self):
        """Create RandomForestLearner instance for testing."""
        return RandomForestLearner(featurizer_type='morgan', n_estimators=10, random_state=42)
    
    def test_initialization(self, learner):
        """Test learner initialization."""
        assert learner.n_estimators == 10
        assert learner.random_state == 42
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
        
        # Test prediction
        predictions, uncertainty = learner.predict(compounds, data_manager)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is None  # RF doesn't support uncertainty
        assert np.all(np.isfinite(predictions))
    
    def test_train_with_missing_columns(self, learner, data_manager):
        """Test error handling with missing required columns."""
        invalid_compounds = pd.DataFrame({
            'compound_id': ['COMP_001', 'COMP_002'],
            'structure': ['CCO', 'CCC']
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            learner.train(invalid_compounds, 'Activity', data_manager)
    
    def test_train_with_empty_dataframe(self, learner, data_manager):
        """Test error handling with empty DataFrame."""
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES', 'Activity'])
        
        with pytest.raises(ValueError, match="compounds DataFrame is empty"):
            learner.train(empty_compounds, 'Activity', data_manager)
    
    def test_predict_without_training(self, learner, small_real_compounds, data_manager):
        """Test error when predicting without training."""
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(small_real_compounds, data_manager)
    
    def test_get_name(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "RandomForest" in name
        assert "n_estimators=10" in name
    
    def test_feature_importance(self, learner, small_real_compounds, data_manager):
        """Test feature importance retrieval."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Before training
        assert learner.get_feature_importance() is None
        
        # After training
        learner.train(compounds, 'Activity', data_manager)
        importance = learner.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0
        assert np.all(importance >= 0)
    
    def test_oob_score(self, learner, small_real_compounds, data_manager):
        """Test out-of-bag score retrieval."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        # Before training
        assert learner.get_oob_score() is None
        
        # After training
        learner.train(compounds, 'Activity', data_manager)
        oob_score = learner.get_oob_score()
        assert oob_score is not None
        assert isinstance(oob_score, float)
    
    def test_different_hyperparameters(self, data_manager, small_real_compounds):
        """Test learner with different hyperparameters."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))
        
        learner = RandomForestLearner(
            featurizer_type='morgan',
            n_estimators=5,
            max_depth=3,
            min_samples_split=5,
            random_state=42
        )
        
        learner.train(compounds, 'Activity', data_manager)
        predictions, _ = learner.predict(compounds, data_manager)
        
        assert learner.max_depth == 3
        assert predictions.shape[0] == len(compounds)
    
    def test_edge_case_single_compound(self, learner, data_manager):
        """Test with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })
        
        learner.train(single_compound, 'Activity', data_manager)
        predictions, _ = learner.predict(single_compound, data_manager)
        
        assert len(predictions) == 1
        assert np.isfinite(predictions[0])
    
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
            compounds,  # valid compounds DataFrame
            np.random.randn(len(compounds), 100),  # Mock features
            compounds['Activity'].values
        )
        mock_dm.prepare_prediction_data.return_value = (
            compounds,  # valid compounds DataFrame
            np.random.randn(len(compounds), 100)
        )
        
        # Train and predict
        learner.train(compounds, 'Activity', mock_dm)
        predictions, _ = learner.predict(compounds, mock_dm)
        
        # Verify DataManager methods were called
        mock_dm.prepare_training_data.assert_called_once()
        mock_dm.prepare_prediction_data.assert_called_once()
        assert len(predictions) == len(compounds)
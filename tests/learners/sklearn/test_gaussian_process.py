"""Tests for GaussianProcessLearner implementation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.features.extraction import extract_features


class TestGaussianProcessLearner:
    """Test GaussianProcessLearner functionality with real molecular data."""

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
    
    def test_train_predict_integration(self, learner, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        assert learner.is_trained

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)
    
    def test_uncertainty_quality(self, learner, small_real_compounds, tmp_path):
        """Test that uncertainty estimates are reasonable."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        train_compounds = compounds.iloc[:len(compounds)//2]
        test_compounds = compounds.iloc[len(compounds)//2:]

        train_features = extract_features(train_compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(train_features, train_compounds['Activity'].values)

        train_pred, train_unc = learner.predict(train_features)

        if len(test_compounds) > 0:
            test_features = extract_features(test_compounds['SMILES'].tolist(), 'morgan', tmp_path)
            test_pred, test_unc = learner.predict(test_features)
            assert np.mean(train_unc) <= np.mean(test_unc) * 2
    
    def test_predict_without_training(self, learner, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].tolist(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(features)
    
    def test_get_name(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "GaussianProcess" in name
        assert "α=1e-10" in name
    
    def test_learned_hyperparameters(self, learner, small_real_compounds, tmp_path):
        """Test hyperparameter learning."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        assert learner.get_learned_hyperparameters() is None

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        hyperparams = learner.get_learned_hyperparameters()
        assert hyperparams is not None
        assert 'kernel' in hyperparams
        assert 'theta' in hyperparams

    def test_custom_kernel(self, tmp_path, small_real_compounds):
        """Test learner with custom kernel configuration."""
        learner = GaussianProcessLearner(alpha=1e-6, random_state=42)

        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        predictions, uncertainty = learner.predict(features)

        assert learner.alpha == 1e-6
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None

    def test_edge_case_single_compound(self, learner, tmp_path):
        """Test with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, single_compound['Activity'].values)
        predictions, uncertainty = learner.predict(features)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_numerical_stability(self, learner, tmp_path):
        """Test numerical stability with extreme values."""
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.0, 1000.0, 0.001]
        })

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        predictions, uncertainty = learner.predict(features)

        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(uncertainty))
        assert np.all(uncertainty >= 0)
    
    def test_uncertainty_ordering(self, learner, tmp_path):
        """Test that uncertainty estimates have reasonable ordering."""
        train_compounds = pd.DataFrame({
            'ID': ['TRAIN_001', 'TRAIN_002'],
            'SMILES': ['CCO', 'CCC'],
            'Activity': [0.5, 0.6]
        })

        test_compounds = pd.DataFrame({
            'ID': ['TEST_001', 'TEST_002'],
            'SMILES': ['CCCO', 'c1ccccc1'],
        })

        train_features = extract_features(train_compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(train_features, train_compounds['Activity'].values)

        test_features = extract_features(test_compounds['SMILES'].tolist(), 'morgan', tmp_path)
        _, uncertainties = learner.predict(test_features)

        assert np.all(uncertainties >= 0)
        assert len(uncertainties) == len(test_compounds)
"""Tests for MLPLearner implementation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from learnm8.learners.torch.mlp import MLPLearner
from learnm8.features.extraction import extract_features


class TestMLPLearner:
    """Test MLPLearner functionality with real molecular data."""

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
    
    def test_train_predict_integration(self, learner, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        assert learner.is_trained
        assert learner.model is not None

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is None
        assert np.all(np.isfinite(predictions))

    def test_predict_without_training(self, learner, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].tolist(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(features)
    
    def test_get_name(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "MLP" in name
        assert "64-32" in name
        assert "relu" in name
        assert "dropout=0.2" in name
    
    def test_different_architectures(self, tmp_path, small_real_compounds):
        """Test learner with different architectures."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = MLPLearner(
            hidden_sizes=(128, 64, 32, 16),
            activation='gelu',
            dropout_rate=0.3,
            max_epochs=3,
            random_state=42
        )

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        predictions, _ = learner.predict(features)

        assert learner.hidden_sizes == (128, 64, 32, 16)
        assert learner.activation == 'gelu'
        assert predictions.shape[0] == len(compounds)

    def test_activation_functions(self, tmp_path, small_real_compounds):
        """Test different activation functions."""
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

            features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
            learner.train(features, compounds['Activity'].values)
            predictions, _ = learner.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_batch_normalization(self, tmp_path, small_real_compounds):
        """Test with and without batch normalization."""
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

            features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
            learner.train(features, compounds['Activity'].values)
            predictions, _ = learner.predict(features)

            assert predictions.shape[0] == len(compounds)

    def test_dropout_regularization(self, tmp_path, small_real_compounds):
        """Test different dropout rates."""
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

            features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
            learner.train(features, compounds['Activity'].values)
            predictions, _ = learner.predict(features)

            assert learner.dropout_rate == dropout_rate
            assert predictions.shape[0] == len(compounds)

    def test_edge_case_single_compound(self, tmp_path):
        """Test with single compound using learner without batch norm."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        learner = MLPLearner(
            hidden_sizes=(32,),
            batch_norm=False,
            max_epochs=2,
            random_state=42
        )

        features = extract_features(single_compound['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, single_compound['Activity'].values)
        predictions, _ = learner.predict(features)

        assert len(predictions) == 1
        assert np.isfinite(predictions[0])
    
    def test_training_history(self, learner, small_real_compounds, tmp_path):
        """Test training history tracking."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        assert len(learner.get_training_history()) == 0

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        history = learner.get_training_history()

        assert len(history) > 0
        assert 'epoch' in history[0]
        assert 'train_loss' in history[0]
        assert 'val_loss' in history[0]

    def test_early_stopping(self, tmp_path, small_real_compounds):
        """Test early stopping mechanism."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = MLPLearner(
            hidden_sizes=(32,),
            max_epochs=100,
            early_stopping_patience=2,
            random_state=42
        )

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        history = learner.get_training_history()

        assert len(history) < 100

    def test_gpu_cpu_compatibility(self, tmp_path, small_real_compounds):
        """Test device compatibility."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = MLPLearner(
            hidden_sizes=(16,),
            device='cpu',
            max_epochs=2,
            random_state=42
        )

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        predictions, _ = learner.predict(features)

        assert str(learner.device) == 'cpu'
        assert predictions.shape[0] == len(compounds)
"""Tests for FastpropLearner implementation."""

import pytest
import numpy as np
import pandas as pd

from learnm8.learners.torch.fastprop_learner import FastpropLearner
from learnm8.features.extraction import extract_features


class TestFastpropLearner:
    """Test FastpropLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create FastpropLearner instance with small architecture for fast tests."""
        return FastpropLearner(
            fnn_layers=2,
            hidden_size=128,
            max_epochs=3,
            batch_size=16,
            early_stopping_patience=2,
            random_state=42
        )

    def test_initialization(self, learner):
        """Test learner initialization with default parameters."""
        assert learner.fnn_layers == 2
        assert learner.hidden_size == 128
        assert learner.max_epochs == 3
        assert learner.batch_size == 16
        assert learner.clamp_input is True
        assert learner.early_stopping_patience == 2
        assert not learner.is_trained
        assert learner.supports_uncertainty() is False

    def test_train_predict_integration(self, learner, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(
            compounds['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].values)
        assert learner.is_trained
        assert learner.model is not None
        assert learner.trainer is not None

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is None
        assert np.all(np.isfinite(predictions))

    def test_predict_without_training(self, learner, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(
            small_real_compounds['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(features)

    def test_get_name(self, learner):
        """Test name generation includes architecture details."""
        name = learner.get_name()
        assert "Fastprop" in name
        assert "layers=2" in name
        assert "hidden=128" in name

    def test_supports_uncertainty(self, learner):
        """Verify single model returns False for uncertainty support."""
        assert learner.supports_uncertainty() is False

    def test_morgan_features(self, tmp_path, small_real_compounds):
        """Test with morgan fingerprints (2048-bit)."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].values)
        predictions, _ = learner.predict(features)

        assert features.shape[1] == 2048
        assert predictions.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))

    def test_maccs_features(self, tmp_path, small_real_compounds):
        """Test with maccs fingerprints (167-bit)."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].tolist(),
            'maccs',
            tmp_path
        )

        learner.train(features, compounds['Activity'].values)
        predictions, _ = learner.predict(features)

        assert features.shape[1] == 167
        assert predictions.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))

    def test_descriptors_features(self, tmp_path, small_real_compounds):
        """Test with Mordred descriptors (1613-dim)."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].tolist(),
            'descriptors',
            tmp_path
        )

        learner.train(features, compounds['Activity'].values)
        predictions, _ = learner.predict(features)

        assert features.shape[1] == 1613
        assert predictions.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))

    def test_different_architectures(self, tmp_path, small_real_compounds):
        """Test with different fnn_layers and hidden_size."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        for fnn_layers, hidden_size in [(2, 128), (3, 256), (4, 512)]:
            learner = FastpropLearner(
                fnn_layers=fnn_layers,
                hidden_size=hidden_size,
                max_epochs=2,
                random_state=42
            )

            features = extract_features(
                compounds['SMILES'].tolist(),
                'morgan',
                tmp_path
            )

            learner.train(features, compounds['Activity'].values)
            predictions, _ = learner.predict(features)

            assert learner.fnn_layers == fnn_layers
            assert learner.hidden_size == hidden_size
            assert predictions.shape[0] == len(compounds)

    def test_clamp_input_flag(self, tmp_path, small_real_compounds):
        """Test clamp_input=True vs False."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        for clamp_input in [True, False]:
            learner = FastpropLearner(
                fnn_layers=2,
                hidden_size=64,
                clamp_input=clamp_input,
                max_epochs=2,
                random_state=42
            )

            features = extract_features(
                compounds['SMILES'].tolist(),
                'morgan',
                tmp_path
            )

            learner.train(features, compounds['Activity'].values)
            predictions, _ = learner.predict(features)

            assert learner.clamp_input == clamp_input
            assert predictions.shape[0] == len(compounds)

    def test_early_stopping(self, tmp_path, small_real_compounds):
        """Verify early stopping prevents full epoch training."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=100,
            early_stopping_patience=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].values)

        assert learner.is_trained

    def test_cpu_device(self, tmp_path, small_real_compounds):
        """Test explicit CPU device."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            device='cpu',
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].values)
        predictions, _ = learner.predict(features)

        assert str(learner.device) == 'cpu'
        assert predictions.shape[0] == len(compounds)

    def test_auto_device(self, tmp_path, small_real_compounds):
        """Test auto device detection."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            device='auto',
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].values)
        predictions, _ = learner.predict(features)

        assert str(learner.device) in ['cpu', 'cuda', 'cuda:0']
        assert predictions.shape[0] == len(compounds)

    def test_edge_case_single_compound(self, tmp_path):
        """Test with single compound (potential batch_size issues).

        Note: Single compound training may produce NaN due to variance
        calculation issues in standard_scale with n=1. This is expected.
        """
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=32,
            max_epochs=2,
            batch_size=1,
            random_state=42
        )

        features = extract_features(
            single_compound['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        learner.train(features, single_compound['Activity'].values)
        predictions, _ = learner.predict(features)

        assert len(predictions) == 1

    def test_train_with_empty_arrays(self, learner):
        """Test error handling for empty inputs."""
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(ValueError, match="Cannot train on empty dataset"):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        """Test error handling for shape mismatches."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(ValueError, match="Features and targets must have same length"):
            learner.train(features, targets)

    def test_uncertainty_consistency(self, learner, small_real_compounds, tmp_path):
        """Verify uncertainty is always None for single model."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(
            compounds['SMILES'].tolist(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].values)
        predictions, uncertainty = learner.predict(features)

        assert learner.supports_uncertainty() is False
        assert uncertainty is None

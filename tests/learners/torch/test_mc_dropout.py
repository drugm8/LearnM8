"""Tests for MCDropoutLearner implementation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from learnm8.learners.torch.mc_dropout import MCDropoutLearner
from learnm8.features.extraction import extract_features


class TestMCDropoutLearner:
    """Test MCDropoutLearner functionality with real molecular data."""

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
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_quality(self, learner, small_real_compounds, tmp_path):
        """Test that MC Dropout uncertainty estimates are reasonable."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        predictions, uncertainty = learner.predict(features)

        assert np.std(uncertainty) > 0
        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(uncertainty))

    def test_dropout_samples_effect(self, tmp_path, small_real_compounds):
        """Test effect of different numbers of dropout samples."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)

        learner_few = MCDropoutLearner(
            hidden_sizes=(32,),
            n_dropout_samples=5,
            max_epochs=3,
            random_state=42
        )

        learner_few.train(features, compounds['Activity'].values)
        pred_few, unc_few = learner_few.predict(features)

        learner_many = MCDropoutLearner(
            hidden_sizes=(32,),
            n_dropout_samples=50,
            max_epochs=3,
            random_state=42
        )

        learner_many.train(features, compounds['Activity'].values)
        pred_many, unc_many = learner_many.predict(features)

        assert len(pred_few) == len(compounds)
        assert len(pred_many) == len(compounds)
        assert np.all(unc_few >= 0)
        assert np.all(unc_many >= 0)

    def test_predict_without_training(self, learner, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].tolist(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(features)
    
    def test_get_name(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "MCDropout" in name
        assert "64-32" in name
        assert "samples=20" in name
        assert "dropout=0.2" in name
    
    def test_different_architectures(self, tmp_path, small_real_compounds):
        """Test learner with different architectures."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

        learner = MCDropoutLearner(
            hidden_sizes=(128, 64, 32),
            activation='gelu',
            dropout_rate=0.3,
            n_dropout_samples=10,
            max_epochs=3,
            random_state=42
        )

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)
        predictions, uncertainty = learner.predict(features)

        assert learner.hidden_sizes == (128, 64, 32)
        assert learner.activation == 'gelu'
        assert learner.dropout_rate == 0.3
        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_dropout_rate_effect(self, tmp_path, small_real_compounds):
        """Test effect of different dropout rates."""
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

            features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
            learner.train(features, compounds['Activity'].values)
            predictions, uncertainty = learner.predict(features)

            assert learner.dropout_rate == dropout_rate
            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

    def test_edge_case_single_compound(self, tmp_path):
        """Test with single compound using learner without batch norm."""
        single_compound = pd.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        learner = MCDropoutLearner(
            hidden_sizes=(32,),
            batch_norm=False,
            n_dropout_samples=5,
            max_epochs=2,
            random_state=42
        )

        features = extract_features(single_compound['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, single_compound['Activity'].values)
        predictions, uncertainty = learner.predict(features)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_uncertainty_consistency(self, learner, tmp_path):
        """Test that uncertainty is consistent across multiple predictions."""
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.3, 0.6, 0.9]
        })

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].values)

        pred1, unc1 = learner.predict(features)
        pred2, unc2 = learner.predict(features)

        assert not np.allclose(pred1, pred2, rtol=1e-10)
        assert not np.allclose(unc1, unc2, rtol=1e-10)

        assert np.all(unc1 >= 0)
        assert np.all(unc2 >= 0)

    def test_deterministic_with_same_seed(self, tmp_path, small_real_compounds):
        """Test reproducibility with same random seed."""
        compounds = small_real_compounds.copy()
        if 'Activity' not in compounds.columns:
            compounds['Activity'] = np.random.beta(2, 5, len(compounds))

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

        features = extract_features(compounds['SMILES'].tolist(), 'morgan', tmp_path)
        learner1.train(features, compounds['Activity'].values)
        learner2.train(features, compounds['Activity'].values)

        pred1, _ = learner1.predict(features)
        pred2, _ = learner2.predict(features)

        assert len(pred1) == len(pred2)
        assert np.all(np.isfinite(pred1))
        assert np.all(np.isfinite(pred2))
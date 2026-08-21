"""Tests for MLPLearner implementation."""

import pytest
import numpy as np
import polars as pl
from unittest.mock import Mock

from learnm8.learners.torch.mlp import MLPLearner
from learnm8.features.extraction import extract_features


@pytest.mark.slow
@pytest.mark.integration
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
    
    def test_initialization_sets_network_defaults_and_untrained_state(self, learner):
        """Test learner initialization."""
        assert learner.hidden_sizes == (64, 32)
        assert learner.activation == 'relu'
        assert learner.dropout_rate == 0.2
        assert learner.batch_norm is True
        assert learner.max_epochs == 5
        assert not learner.is_trained
        assert learner.supports_uncertainty() is False
    
    def test_predict_returns_finite_values_without_uncertainty_after_training(self, learner, small_real_compounds, small_real_morgan_features):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())
        assert learner.is_trained
        assert learner.model is not None

        predictions, uncertainty = learner.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is None
        assert np.all(np.isfinite(predictions))

    def test_predict_without_training(self, learner, small_real_morgan_features):
        """Test error when predicting without training."""
        from learnm8.exceptions import LearnerError

        features = small_real_morgan_features.copy()
        with pytest.raises(LearnerError, match="must be trained before prediction"):
            learner.predict(features)
    
    def test_get_name_includes_architecture_activation_and_dropout(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "MLP" in name
        assert "64-32" in name
        assert "relu" in name
        assert "dropout=0.2" in name
    
    def test_custom_hidden_layer_configuration_trains_and_predicts(self, small_real_compounds, small_real_morgan_features):
        """Test learner with different architectures."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = MLPLearner(
            hidden_sizes=(128, 64, 32, 16),
            activation='gelu',
            dropout_rate=0.3,
            max_epochs=3,
            random_state=42
        )

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert learner.hidden_sizes == (128, 64, 32, 16)
        assert learner.activation == 'gelu'
        assert predictions.shape[0] == len(compounds)

    def test_supported_activation_functions_train_and_predict_finite_values(self, small_real_compounds, small_real_morgan_features):
        """Test different activation functions."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        for activation in ['relu', 'tanh', 'gelu']:
            learner = MLPLearner(
                hidden_sizes=(32,),
                activation=activation,
                max_epochs=2,
                random_state=42
            )

            features = small_real_morgan_features.copy()
            learner.train(features, compounds['Activity'].to_numpy())
            predictions, _ = learner.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_batch_norm_toggle_trains_and_predicts(self, small_real_compounds, small_real_morgan_features):
        """Test with and without batch normalization."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        for batch_norm in [True, False]:
            learner = MLPLearner(
                hidden_sizes=(32,),
                batch_norm=batch_norm,
                max_epochs=2,
                random_state=42
            )

            features = small_real_morgan_features.copy()
            learner.train(features, compounds['Activity'].to_numpy())
            predictions, _ = learner.predict(features)

            assert predictions.shape[0] == len(compounds)

    def test_dropout_rate_is_preserved_across_training_runs(self, small_real_compounds, small_real_morgan_features):
        """Test different dropout rates."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        for dropout_rate in [0.0, 0.1, 0.5]:
            learner = MLPLearner(
                hidden_sizes=(32,),
                dropout_rate=dropout_rate,
                max_epochs=2,
                random_state=42
            )

            features = small_real_morgan_features.copy()
            learner.train(features, compounds['Activity'].to_numpy())
            predictions, _ = learner.predict(features)

            assert learner.dropout_rate == dropout_rate
            assert predictions.shape[0] == len(compounds)

    def test_single_compound_without_batch_norm_trains_and_predicts(self, tmp_path):
        """Test with single compound using learner without batch norm."""
        from learnm8.exceptions import LearnerError

        single_compound = pl.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        learner = MLPLearner(
            hidden_sizes=(32,),
            batch_norm=False,
            max_epochs=2,
            random_state=42,
            remove_zero_variance=False
        )

        features = extract_features(single_compound['SMILES'].to_list(), 'morgan', tmp_path)
        learner.train(features, single_compound['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert len(predictions) == 1
        assert np.isfinite(predictions[0])
    
    def test_training_history_contains_epoch_and_loss_entries_after_training(self, learner, small_real_compounds, small_real_morgan_features):
        """Test training history tracking."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        assert len(learner.get_training_history()) == 0

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())
        history = learner.get_training_history()

        assert len(history) > 0
        assert 'epoch' in history[0]
        assert 'train_loss' in history[0]
        assert 'val_loss' in history[0]

    def test_early_stopping_configuration_is_respected_during_training(self, small_real_compounds, small_real_morgan_features):
        """Test early stopping mechanism."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = MLPLearner(
            hidden_sizes=(32,),
            max_epochs=100,
            early_stopping_patience=2,
            random_state=42
        )

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())
        history = learner.get_training_history()

        assert hasattr(learner, 'early_stopping_patience')
        assert learner.early_stopping_patience == 2
        assert len(history) > 0

    def test_cpu_device_configuration_trains_and_predicts(self, small_real_compounds, small_real_morgan_features):
        """Test device compatibility."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = MLPLearner(
            hidden_sizes=(16,),
            device='cpu',
            max_epochs=2,
            random_state=42
        )

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert str(learner.device) == 'cpu'
        assert predictions.shape[0] == len(compounds)

    def test_predict_returns_no_uncertainty_when_uncertainty_support_is_disabled(self, learner, small_real_compounds, small_real_morgan_features):
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())

        predictions, uncertainty = learner.predict(features)

        assert learner.supports_uncertainty() is False
        assert uncertainty is None

    def test_train_with_empty_arrays(self, learner):
        from learnm8.exceptions import LearnerError

        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(LearnerError, match="empty dataset"):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        from learnm8.exceptions import LearnerError

        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(LearnerError, match="mismatched lengths"):
            learner.train(features, targets)

    def test_train_with_1d_features(self, learner):
        features_1d = np.random.rand(10)
        targets = np.random.rand(10)

        learner.train(features_1d.reshape(-1, 1), targets)
        predictions, _ = learner.predict(features_1d.reshape(-1, 1))
        assert len(predictions) == 10

    def test_preferred_feature_dtype_is_float32(self, learner):
        assert learner.preferred_feature_dtype() == 'float32'

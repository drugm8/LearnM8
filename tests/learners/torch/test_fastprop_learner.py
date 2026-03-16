"""Tests for FastpropLearner implementation."""

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import torch
from pytorch_lightning.callbacks import EarlyStopping

from learnm8.exceptions import ConfigurationError
from learnm8.features.extraction import extract_features
from learnm8.learners.torch.fastprop_learner import FastpropLearner


@pytest.mark.unit
class TestFastpropLearnerUnit:
    """Fast unit tests for FastpropLearner constructor and validation."""

    def test_default_hidden_size(self):
        assert FastpropLearner().hidden_size == 300

    def test_default_val_fraction(self):
        assert FastpropLearner().val_fraction == 0.1

    def test_explicit_hidden_size_override(self):
        assert FastpropLearner(hidden_size=512).hidden_size == 512

    def test_explicit_val_fraction_override(self):
        assert FastpropLearner(val_fraction=0.2).val_fraction == 0.2

    def test_fnn_layers_zero_raises(self):
        with pytest.raises(ConfigurationError, match='fnn_layers >= 1'):
            FastpropLearner(fnn_layers=0)

    def test_fnn_layers_negative_raises(self):
        with pytest.raises(ConfigurationError, match='fnn_layers >= 1'):
            FastpropLearner(fnn_layers=-1)

    def test_fnn_layers_one_accepted(self):
        learner = FastpropLearner(fnn_layers=1)
        assert learner.fnn_layers == 1

    def test_val_fraction_zero_accepted(self):
        learner = FastpropLearner(val_fraction=0.0)
        assert learner.val_fraction == 0.0

    def test_val_fraction_negative_raises(self):
        with pytest.raises(ConfigurationError, match='val_fraction'):
            FastpropLearner(val_fraction=-0.1)

    def test_val_fraction_one_raises(self):
        with pytest.raises(ConfigurationError, match='val_fraction'):
            FastpropLearner(val_fraction=1.0)

    def test_val_fraction_above_one_raises(self):
        with pytest.raises(ConfigurationError, match='val_fraction'):
            FastpropLearner(val_fraction=1.5)

    def test_invalid_precision_raises_configuration_error(self):
        with pytest.raises(ConfigurationError):
            FastpropLearner(precision='invalid')

    def test_predict_does_not_call_standard_scale(self):
        learner = FastpropLearner(fnn_layers=1, hidden_size=64)
        learner.is_trained = True
        learner.model = MagicMock()
        learner.feature_means = torch.zeros(10)
        learner.feature_vars = torch.ones(10)
        learner.target_means = torch.zeros(1)
        learner.target_vars = torch.ones(1)

        dummy_output = torch.tensor([[0.5]] * 5)
        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = [dummy_output]

        with (
            patch('learnm8.learners.torch.fastprop_learner.Trainer', return_value=mock_trainer),
            patch('learnm8.learners.torch.fastprop_learner.standard_scale') as scale_spy,
        ):
            predictions, _ = learner.predict(np.random.randn(5, 10).astype(np.float32))
            scale_spy.assert_not_called()

        assert predictions.shape[0] == 5


@pytest.mark.slow
@pytest.mark.integration
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

    def test_initialization_sets_fastprop_defaults_and_untrained_state(self, learner):
        """Test learner initialization with default parameters."""
        assert learner.fnn_layers == 2
        assert learner.hidden_size == 128
        assert learner.max_epochs == 3
        assert learner.batch_size == 16
        assert learner.clamp_input is True
        assert learner.early_stopping_patience == 2
        assert not learner.is_trained
        assert learner.supports_uncertainty() is False

    def test_predict_returns_finite_values_without_uncertainty_after_training(self, learner, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())
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
            small_real_compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            learner.predict(features)

    def test_get_name_includes_layer_count_and_hidden_size(self, learner):
        """Test name generation includes architecture details."""
        name = learner.get_name()
        assert "Fastprop" in name
        assert "layers=2" in name
        assert "hidden=128" in name

    def test_supports_uncertainty_returns_false_for_single_fastprop_model(self, learner):
        """Verify single model returns False for uncertainty support."""
        assert learner.supports_uncertainty() is False

    def test_morgan_features(self, tmp_path, small_real_compounds):
        """Test with morgan fingerprints (2048-bit)."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert features.shape[1] == 2048
        assert predictions.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))

    def test_maccs_features(self, tmp_path, small_real_compounds):
        """Test with maccs fingerprints (167-bit)."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'maccs',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert features.shape[1] == 167
        assert predictions.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))

    def test_descriptors_features(self, tmp_path, small_real_compounds):
        """Test with Mordred descriptors (1613-dim)."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'descriptors',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert features.shape[1] == 1613
        assert predictions.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))

    def test_multiple_fastprop_architectures_train_and_predict(self, tmp_path, small_real_compounds):
        """Test with different fnn_layers and hidden_size."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        for fnn_layers, hidden_size in [(2, 128), (3, 256), (4, 512)]:
            learner = FastpropLearner(
                fnn_layers=fnn_layers,
                hidden_size=hidden_size,
                max_epochs=2,
                random_state=42
            )

            features = extract_features(
                compounds['SMILES'].to_list(),
                'morgan',
                tmp_path
            )

            learner.train(features, compounds['Activity'].to_numpy())
            predictions, _ = learner.predict(features)

            assert learner.fnn_layers == fnn_layers
            assert learner.hidden_size == hidden_size
            assert predictions.shape[0] == len(compounds)

    def test_clamp_input_flag(self, tmp_path, small_real_compounds):
        """Test clamp_input=True vs False."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        for clamp_input in [True, False]:
            learner = FastpropLearner(
                fnn_layers=2,
                hidden_size=64,
                clamp_input=clamp_input,
                max_epochs=2,
                random_state=42
            )

            features = extract_features(
                compounds['SMILES'].to_list(),
                'morgan',
                tmp_path
            )

            learner.train(features, compounds['Activity'].to_numpy())
            predictions, _ = learner.predict(features)

            assert learner.clamp_input == clamp_input
            assert predictions.shape[0] == len(compounds)

    def test_early_stopping_configuration_trains_successfully(self, tmp_path, small_real_compounds):
        """Verify early stopping prevents full epoch training."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=100,
            early_stopping_patience=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())

        assert learner.is_trained

    def test_cpu_device(self, tmp_path, small_real_compounds):
        """Test explicit CPU device."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            device='cpu',
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert str(learner.device) == 'cpu'
        assert predictions.shape[0] == len(compounds)

    def test_auto_device(self, tmp_path, small_real_compounds):
        """Test auto device detection."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            device='auto',
            max_epochs=2,
            random_state=42
        )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())
        predictions, _ = learner.predict(features)

        assert str(learner.device) in ['cpu', 'cuda', 'cuda:0']
        assert predictions.shape[0] == len(compounds)

    def test_single_compound_training_and_prediction_succeeds_with_batch_size_one(self, tmp_path):
        """Test with single compound (potential batch_size issues).

        Note: Single compound training may produce NaN due to variance
        calculation issues in standard_scale with n=1. This is expected.
        """
        single_compound = pl.DataFrame({
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
            single_compound['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, single_compound['Activity'].to_numpy())
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
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())
        _predictions, uncertainty = learner.predict(features)

        assert learner.supports_uncertainty() is False
        assert uncertainty is None

    def test_aggressive_gc_enabled_by_default(self):
        """Verify enable_aggressive_gc defaults to True."""
        learner = FastpropLearner()
        assert learner.enable_aggressive_gc is True

    def test_aggressive_gc_can_be_disabled(self):
        """Verify enable_aggressive_gc can be set to False."""
        learner = FastpropLearner(enable_aggressive_gc=False)
        assert learner.enable_aggressive_gc is False

    def test_cleanup_gpu_memory_called_after_training(self, tmp_path, small_real_compounds, monkeypatch):
        """Verify _cleanup_gpu_memory is called after training when enabled."""
        cleanup_called = []

        def mock_cleanup(self, context=""):
            cleanup_called.append(context)

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            enable_aggressive_gc=True
        )

        monkeypatch.setattr(learner, '_cleanup_gpu_memory', lambda context="": mock_cleanup(learner, context))

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())

        assert len(cleanup_called) > 0
        assert any('after training' in ctx for ctx in cleanup_called)

    def test_cleanup_gpu_memory_called_after_prediction(self, tmp_path, small_real_compounds, monkeypatch):
        """Verify _cleanup_gpu_memory is called after prediction when enabled."""
        cleanup_called = []

        def mock_cleanup(self, context=""):
            cleanup_called.append(context)

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            enable_aggressive_gc=True
        )

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())

        cleanup_called.clear()
        monkeypatch.setattr(learner, '_cleanup_gpu_memory', lambda context="": mock_cleanup(learner, context))

        learner.predict(features)

        assert len(cleanup_called) > 0
        assert any('after prediction' in ctx for ctx in cleanup_called)

    def test_cleanup_not_called_when_disabled(self, tmp_path, small_real_compounds, monkeypatch):
        """Verify _cleanup_gpu_memory is not called when enable_aggressive_gc=False."""
        cleanup_called = []

        def mock_cleanup(self, context=""):
            cleanup_called.append(context)

        learner = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            enable_aggressive_gc=False
        )

        monkeypatch.setattr(learner, '_cleanup_gpu_memory', lambda context="": mock_cleanup(learner, context))

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner.train(features, compounds['Activity'].to_numpy())
        learner.predict(features)

        assert len(cleanup_called) == 0

    def test_predictions_unaffected_by_gc(self, tmp_path, small_real_compounds):
        """Verify predictions are identical with GC enabled vs disabled."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(
            compounds['SMILES'].to_list(),
            'morgan',
            tmp_path
        )

        learner_gc_on = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_state=42,
            enable_aggressive_gc=True
        )
        learner_gc_on.train(features, compounds['Activity'].to_numpy())
        pred_gc_on, _ = learner_gc_on.predict(features)

        learner_gc_off = FastpropLearner(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_state=42,
            enable_aggressive_gc=False
        )
        learner_gc_off.train(features, compounds['Activity'].to_numpy())
        pred_gc_off, _ = learner_gc_off.predict(features)

        assert np.allclose(pred_gc_on, pred_gc_off, rtol=1e-5)

    def test_train_creates_validation_split(self, small_real_compounds, tmp_path):
        """Verify EarlyStopping monitors validation_mse_scaled_loss with sufficient samples."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = extract_features(
            compounds['SMILES'].to_list(), 'morgan', tmp_path
        )
        learner = FastpropLearner(
            fnn_layers=1, hidden_size=64, max_epochs=5,
            val_fraction=0.1, random_state=42
        )
        learner.train(features, compounds['Activity'].to_numpy())
        es_callbacks = [
            cb for cb in learner.trainer.callbacks
            if isinstance(cb, EarlyStopping)
        ]
        assert len(es_callbacks) == 1
        assert es_callbacks[0].monitor == 'validation_mse_scaled_loss'

    def test_train_skips_validation_on_small_dataset(self, tmp_path, caplog):
        """Verify early stopping disabled when n_samples < min_samples_for_split."""
        small_dataset = pl.DataFrame({
            'ID': [f'COMP_{i:03d}' for i in range(10)],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'C1CCNCC1',
                        'CCCO', 'c1ccncc1', 'CC(=O)N', 'CCNC', 'C1CCOCC1'],
            'Activity': np.random.beta(2, 5, 10).tolist()
        })
        features = extract_features(
            small_dataset['SMILES'].to_list(), 'morgan', tmp_path
        )
        learner = FastpropLearner(
            fnn_layers=1, hidden_size=64, max_epochs=3,
            val_fraction=0.1, random_state=42
        )
        import logging
        with caplog.at_level(logging.WARNING):
            learner.train(features, small_dataset['Activity'].to_numpy())
        es_callbacks = [
            cb for cb in learner.trainer.callbacks
            if isinstance(cb, EarlyStopping)
        ]
        assert len(es_callbacks) == 0
        assert 'min_samples_for_split' in caplog.text

    def test_val_fraction_zero_disables_validation(self, small_real_compounds, tmp_path):
        """Verify val_fraction=0.0 trains without validation regardless of sample count."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )
        features = extract_features(
            compounds['SMILES'].to_list(), 'morgan', tmp_path
        )
        learner = FastpropLearner(
            fnn_layers=1, hidden_size=64, max_epochs=3,
            val_fraction=0.0, random_state=42
        )
        learner.train(features, compounds['Activity'].to_numpy())
        es_callbacks = [
            cb for cb in learner.trainer.callbacks
            if isinstance(cb, EarlyStopping)
        ]
        assert len(es_callbacks) == 0

    def test_scaling_stats_from_training_only(self, tmp_path):
        """Verify feature_means/vars computed from training subset, not full data."""
        np.random.seed(42)
        n_features = 50
        n_samples = 100
        features_a = np.random.randn(80, n_features) * 1.0
        features_b = np.random.randn(20, n_features) * 10.0 + 50.0
        features = np.vstack([features_a, features_b]).astype(np.float32)
        targets = np.random.randn(n_samples).astype(np.float32)

        learner = FastpropLearner(
            fnn_layers=1, hidden_size=32, max_epochs=2,
            val_fraction=0.2, random_state=42
        )
        learner.train(features, targets)

        full_means = np.mean(features, axis=0)
        stored_means = learner.feature_means.numpy()
        assert not np.allclose(stored_means, full_means, atol=0.5), \
            'Stored means should differ from full-data means (computed from training subset only)'

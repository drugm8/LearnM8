"""Tests for FastpropEnsemble implementation."""

import gc

import numpy as np
import polars as pl
import pytest
import torch

from learnm8.exceptions import ConfigurationError, LearnerError
from learnm8.learners.ensemble.fastprop_ensemble import FastpropEnsemble


def _cleanup_gpu_memory():
    """Release GPU memory and force garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.mark.unit
class TestFastpropEnsembleUnit:
    """Fast unit tests for FastpropEnsemble constructor and validation."""

    def test_fnn_layers_zero_raises(self):
        with pytest.raises(ConfigurationError, match='fnn_layers >= 1'):
            FastpropEnsemble(fnn_layers=0, hidden_size=64, max_epochs=3, device='cpu')

    def test_val_fraction_forwarded(self):
        ensemble = FastpropEnsemble(
            val_fraction=0.2, fnn_layers=1, hidden_size=64, device='cpu'
        )
        for learner in ensemble.learners:
            assert learner.val_fraction == 0.2

    def test_default_hidden_size(self):
        ensemble = FastpropEnsemble()
        assert ensemble.hidden_size == 300
        for learner in ensemble.learners:
            assert learner.hidden_size == 300

    def test_default_val_fraction(self):
        ensemble = FastpropEnsemble()
        assert ensemble.val_fraction == 0.1
        for learner in ensemble.learners:
            assert learner.val_fraction == 0.1


@pytest.mark.slow
@pytest.mark.integration
class TestFastpropEnsemble:
    """Test FastpropEnsemble functionality with real molecular data."""

    @pytest.fixture
    def fastprop_ensemble(self):
        """Create FastpropEnsemble instance for testing."""
        return FastpropEnsemble(
            fnn_layers=1, hidden_size=64, max_epochs=3, device='cpu'
        )

    def test_initialization_sets_default_architecture_random_states_and_untrained_state(
        self, fastprop_ensemble
    ):
        """Test ensemble initialization with default parameters."""
        assert len(fastprop_ensemble.learners) == 3
        assert fastprop_ensemble.aggregation_method == 'mean'
        assert fastprop_ensemble.uncertainty_method == 'std'
        assert fastprop_ensemble.weights is None
        assert not fastprop_ensemble.is_trained
        assert fastprop_ensemble.supports_uncertainty() is True
        assert fastprop_ensemble.fnn_layers == 1
        assert fastprop_ensemble.hidden_size == 64
        assert fastprop_ensemble.random_states == [42, 123, 356]

    def test_initialization_custom_parameters(self):
        """Test ensemble initialization with custom parameters."""
        custom_states = [10, 20, 30]

        ensemble = FastpropEnsemble(
            fnn_layers=2,
            hidden_size=128,
            max_epochs=5,
            random_states=custom_states,
            device='cpu',
        )

        assert len(ensemble.learners) == 3
        assert ensemble.fnn_layers == 2
        assert ensemble.hidden_size == 128
        assert ensemble.random_states == custom_states

    def test_initialization_with_weights(self):
        """Test ensemble initialization with custom weights."""
        weights = [0.5, 0.3, 0.2]
        ensemble = FastpropEnsemble(
            fnn_layers=1, hidden_size=64, max_epochs=3, weights=weights, device='cpu'
        )

        assert ensemble.weights is not None
        assert np.allclose(ensemble.weights, weights)

    def test_predict_returns_finite_values_and_uncertainty_after_training(
        self,
        trained_fastprop_ensemble,
        small_real_compounds,
        small_real_morgan_features,
    ):
        """Test prediction with pre-trained session ensemble (no per-test retrain)."""
        assert trained_fastprop_ensemble.is_trained
        features = small_real_morgan_features

        predictions, uncertainty = trained_fastprop_ensemble.predict(features)
        assert predictions.shape[0] == len(small_real_compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(small_real_compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_estimation_returns_finite_nonconstant_nonnegative_values(
        self,
        trained_fastprop_ensemble,
        small_real_compounds,
        small_real_morgan_features,
    ):
        """Test that ensemble provides uncertainty estimates (uses session fixture)."""
        features = small_real_morgan_features
        _predictions, uncertainty = trained_fastprop_ensemble.predict(features)

        assert uncertainty is not None
        assert len(uncertainty) == len(small_real_compounds)
        assert np.all(np.isfinite(uncertainty))
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_train_with_empty_arrays(self, fastprop_ensemble):
        """Test error handling when training with empty arrays."""
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(ValueError):
            fastprop_ensemble.train(empty_features, empty_targets)

    def test_predict_without_training(
        self, fastprop_ensemble, small_real_morgan_features
    ):
        """Test error when predicting without training."""
        with pytest.raises(
            LearnerError, match='Ensemble must be trained before prediction'
        ):
            fastprop_ensemble.predict(small_real_morgan_features)

    def test_get_name_includes_model_count_layers_and_hidden_size(
        self, fastprop_ensemble
    ):
        """Test name generation for Fastprop ensemble."""
        name = fastprop_ensemble.get_name()
        assert 'FastpropEnsemble' in name
        assert '3xFastprop' in name
        assert 'layers=1' in name
        assert 'hidden=64' in name

    def test_get_name_custom_architecture(self):
        """Test name generation with custom architecture."""
        ensemble = FastpropEnsemble(
            fnn_layers=3, hidden_size=256, max_epochs=3, device='cpu'
        )
        name = ensemble.get_name()
        assert 'FastpropEnsemble' in name
        assert 'layers=3' in name
        assert 'hidden=256' in name

    def test_supports_uncertainty_returns_true_for_fastprop_ensemble(
        self, fastprop_ensemble
    ):
        """Test that Fastprop ensemble supports uncertainty estimation."""
        assert fastprop_ensemble.supports_uncertainty() is True

    def test_aggregation_methods(
        self, small_real_compounds, small_real_morgan_features
    ):
        """Test different aggregation methods with Fastprop ensemble."""
        compounds = small_real_compounds.head(10).clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features[:10]

        for method in ['mean', 'median']:
            ensemble = FastpropEnsemble(
                fnn_layers=1,
                hidden_size=64,
                max_epochs=3,
                aggregation_method=method,
                device='cpu',
            )
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

            del ensemble
            _cleanup_gpu_memory()

    def test_uncertainty_methods(
        self, small_real_compounds, small_real_morgan_features
    ):
        """Test different uncertainty estimation methods."""
        compounds = small_real_compounds.head(10).clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features[:10]

        for method in ['std', 'mad', 'quantile']:
            ensemble = FastpropEnsemble(
                fnn_layers=1,
                hidden_size=64,
                max_epochs=3,
                uncertainty_method=method,
                device='cpu',
            )
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

            del ensemble
            _cleanup_gpu_memory()

    def test_weighted_ensemble(self, small_real_compounds, small_real_morgan_features):
        """Test weighted ensemble aggregation."""
        compounds = small_real_compounds.head(10).clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        weights = [0.6, 0.3, 0.1]
        ensemble = FastpropEnsemble(
            fnn_layers=1, hidden_size=64, max_epochs=3, weights=weights, device='cpu'
        )

        features = small_real_morgan_features[:10]
        ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_small_dataset_trains_and_predicts_finite_values(self, tmp_path):
        """Test with small dataset of 5 diverse compounds."""
        ensemble = FastpropEnsemble(
            fnn_layers=1, hidden_size=64, max_epochs=3, device='cpu'
        )

        small_dataset = pl.DataFrame(
            {
                'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004', 'COMP_005'],
                'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'C1CCNCC1'],
                'Activity': [0.5, 0.3, 0.8, 0.1, 0.6],
            }
        )

        from learnm8.features.extraction import extract_features

        features = extract_features(
            small_dataset['SMILES'].to_list(), 'morgan', tmp_path
        )
        ensemble.train(features, small_dataset['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert len(predictions) == 5
        assert len(uncertainty) == 5
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_diversity(
        self, trained_fastprop_ensemble, small_real_morgan_features
    ):
        """Test that ensemble uncertainty captures model diversity (uses session fixture)."""
        features = small_real_morgan_features
        _predictions, uncertainty = trained_fastprop_ensemble.predict(features)

        assert np.std(uncertainty) > 0
        assert np.all(uncertainty >= 0)

    def test_mismatched_array_lengths(self, fastprop_ensemble):
        """Test error handling with mismatched feature and target lengths."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(ValueError):
            fastprop_ensemble.train(features, targets)

    def test_prediction_consistency(
        self, trained_fastprop_ensemble, small_real_morgan_features
    ):
        """Test that predictions are consistent across multiple calls (uses session fixture)."""
        features = small_real_morgan_features
        predictions1, uncertainty1 = trained_fastprop_ensemble.predict(features)
        predictions2, uncertainty2 = trained_fastprop_ensemble.predict(features)

        assert np.allclose(predictions1, predictions2)
        assert np.allclose(uncertainty1, uncertainty2)

    def test_multiple_fastprop_architectures_train_and_predict(
        self, small_real_compounds, small_real_morgan_features
    ):
        """Test ensemble with different architecture configurations."""
        compounds = small_real_compounds.head(10).clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features[:10]

        architectures = [
            {'fnn_layers': 1, 'hidden_size': 64},
            {'fnn_layers': 2, 'hidden_size': 128},
            {'fnn_layers': 3, 'hidden_size': 256},
        ]

        for arch in architectures:
            ensemble = FastpropEnsemble(
                fnn_layers=arch['fnn_layers'],
                hidden_size=arch['hidden_size'],
                max_epochs=3,
                device='cpu',
            )
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

            del ensemble
            _cleanup_gpu_memory()

    def test_aggressive_gc_enabled_by_default(self):
        """Verify enable_aggressive_gc defaults to True."""
        ensemble = FastpropEnsemble()
        assert ensemble.enable_aggressive_gc is True
        for learner in ensemble.learners:
            assert learner.enable_aggressive_gc is True

    def test_aggressive_gc_can_be_disabled(self):
        """Verify enable_aggressive_gc can be set to False."""
        ensemble = FastpropEnsemble(enable_aggressive_gc=False)
        assert ensemble.enable_aggressive_gc is False
        for learner in ensemble.learners:
            assert learner.enable_aggressive_gc is False

    def test_predictions_unaffected_by_gc(
        self, small_real_compounds, small_real_morgan_features
    ):
        """Verify predictions are identical with GC enabled vs disabled."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features

        ensemble_gc_on = FastpropEnsemble(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_states=[42, 123, 456],
            enable_aggressive_gc=True,
        )
        ensemble_gc_on.train(features, compounds['Activity'].to_numpy())
        pred_gc_on, _ = ensemble_gc_on.predict(features)

        ensemble_gc_off = FastpropEnsemble(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_states=[42, 123, 456],
            enable_aggressive_gc=False,
        )
        ensemble_gc_off.train(features, compounds['Activity'].to_numpy())
        pred_gc_off, _ = ensemble_gc_off.predict(features)

        assert np.allclose(pred_gc_on, pred_gc_off, rtol=1e-5)

        del ensemble_gc_on, ensemble_gc_off
        _cleanup_gpu_memory()

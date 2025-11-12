"""Tests for FastpropEnsemble implementation."""

import pytest
import numpy as np
import polars as pl

from learnm8.learners.ensemble.fastprop_ensemble import FastpropEnsemble
from learnm8.features.extraction import extract_features


class TestFastpropEnsemble:
    """Test FastpropEnsemble functionality with real molecular data."""

    @pytest.fixture
    def fastprop_ensemble(self):
        """Create FastpropEnsemble instance for testing."""
        return FastpropEnsemble(
            fnn_layers=1,
            hidden_size=64,
            max_epochs=3,
            device='cpu'
        )

    def test_initialization(self, fastprop_ensemble):
        """Test ensemble initialization with default parameters."""
        assert len(fastprop_ensemble.learners) == 3
        assert fastprop_ensemble.aggregation_method == 'mean'
        assert fastprop_ensemble.uncertainty_method == 'std'
        assert fastprop_ensemble.weights is None
        assert not fastprop_ensemble.is_trained
        assert fastprop_ensemble.supports_uncertainty() is True
        assert fastprop_ensemble.fnn_layers == 1
        assert fastprop_ensemble.hidden_size == 64
        assert fastprop_ensemble.random_states == [42, 123, 456]

    def test_initialization_custom_parameters(self):
        """Test ensemble initialization with custom parameters."""
        custom_states = [10, 20, 30]

        ensemble = FastpropEnsemble(
            fnn_layers=2,
            hidden_size=128,
            max_epochs=5,
            random_states=custom_states,
            device='cpu'
        )

        assert len(ensemble.learners) == 3
        assert ensemble.fnn_layers == 2
        assert ensemble.hidden_size == 128
        assert ensemble.random_states == custom_states

    def test_initialization_with_weights(self):
        """Test ensemble initialization with custom weights."""
        weights = [0.5, 0.3, 0.2]
        ensemble = FastpropEnsemble(
            fnn_layers=1,
            hidden_size=64,
            max_epochs=3,
            weights=weights,
            device='cpu'
        )

        assert ensemble.weights is not None
        assert np.allclose(ensemble.weights, weights)

    def test_train_predict_integration(self, fastprop_ensemble, small_real_compounds, tmp_path):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        fastprop_ensemble.train(features, compounds['Activity'].to_numpy())
        assert fastprop_ensemble.is_trained

        predictions, uncertainty = fastprop_ensemble.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_estimation(self, fastprop_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble provides uncertainty estimates."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        fastprop_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions, uncertainty = fastprop_ensemble.predict(features)

        assert uncertainty is not None
        assert len(uncertainty) == len(compounds)
        assert np.all(np.isfinite(uncertainty))
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_diverse_random_states(self, fastprop_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble learners have different random states."""
        assert len(fastprop_ensemble.learners) == 3

        random_states = [learner.random_state for learner in fastprop_ensemble.learners]
        assert len(set(random_states)) == 3
        assert random_states == [42, 123, 456]

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        fastprop_ensemble.train(features, compounds['Activity'].to_numpy())

        individual_preds = fastprop_ensemble.get_individual_predictions(features)
        assert len(individual_preds) == 3

        pred_arrays = [preds for preds in individual_preds.values() if preds is not None]
        assert len(pred_arrays) == 3

        for i in range(len(pred_arrays)):
            for j in range(i+1, len(pred_arrays)):
                assert not np.allclose(pred_arrays[i], pred_arrays[j], rtol=1e-3)

    def test_train_with_empty_arrays(self, fastprop_ensemble):
        """Test error handling when training with empty arrays."""
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(ValueError):
            fastprop_ensemble.train(empty_features, empty_targets)

    def test_predict_without_training(self, fastprop_ensemble, small_real_compounds, tmp_path):
        """Test error when predicting without training."""
        features = extract_features(small_real_compounds['SMILES'].to_list(), 'morgan', tmp_path)
        with pytest.raises(RuntimeError, match="Ensemble must be trained before prediction"):
            fastprop_ensemble.predict(features)

    def test_get_name(self, fastprop_ensemble):
        """Test name generation for Fastprop ensemble."""
        name = fastprop_ensemble.get_name()
        assert "FastpropEnsemble" in name
        assert "3xFastprop" in name
        assert "layers=1" in name
        assert "hidden=64" in name

    def test_get_name_custom_architecture(self):
        """Test name generation with custom architecture."""
        ensemble = FastpropEnsemble(
            fnn_layers=3,
            hidden_size=256,
            max_epochs=3,
            device='cpu'
        )
        name = ensemble.get_name()
        assert "FastpropEnsemble" in name
        assert "layers=3" in name
        assert "hidden=256" in name

    def test_supports_uncertainty(self, fastprop_ensemble):
        """Test that Fastprop ensemble supports uncertainty estimation."""
        assert fastprop_ensemble.supports_uncertainty() is True

    def test_aggregation_methods(self, small_real_compounds, tmp_path):
        """Test different aggregation methods with Fastprop ensemble."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['mean', 'median']:
            ensemble = FastpropEnsemble(
                fnn_layers=1,
                hidden_size=64,
                max_epochs=3,
                aggregation_method=method,
                device='cpu'
            )
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_uncertainty_methods(self, small_real_compounds, tmp_path):
        """Test different uncertainty estimation methods."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        for method in ['std', 'mad', 'quantile']:
            ensemble = FastpropEnsemble(
                fnn_layers=1,
                hidden_size=64,
                max_epochs=3,
                uncertainty_method=method,
                device='cpu'
            )
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

    def test_weighted_ensemble(self, small_real_compounds, tmp_path):
        """Test weighted ensemble aggregation."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        weights = [0.6, 0.3, 0.1]
        ensemble = FastpropEnsemble(
            fnn_layers=1,
            hidden_size=64,
            max_epochs=3,
            weights=weights,
            device='cpu'
        )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_ensemble_statistics(self, fastprop_ensemble, small_real_compounds, tmp_path):
        """Test ensemble statistics retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        stats = fastprop_ensemble.get_ensemble_statistics()
        assert stats['n_learners'] == 3
        assert stats['is_trained'] is False

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        fastprop_ensemble.train(features, compounds['Activity'].to_numpy())
        stats = fastprop_ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert 'learners_with_uncertainty' in stats
        assert len(stats['learner_names']) == 3

    def test_individual_predictions(self, fastprop_ensemble, small_real_compounds, tmp_path):
        """Test individual learner predictions retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        fastprop_ensemble.train(features, compounds['Activity'].to_numpy())
        individual_preds = fastprop_ensemble.get_individual_predictions(features)

        assert len(individual_preds) == 3
        for learner_name, preds in individual_preds.items():
            assert preds is not None
            assert len(preds) == len(compounds)
            assert np.all(np.isfinite(preds))

    def test_edge_case_single_compound(self, tmp_path):
        """Test with single compound."""
        ensemble = FastpropEnsemble(
            fnn_layers=1,
            hidden_size=64,
            max_epochs=3,
            device='cpu'
        )

        single_compound = pl.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        features = extract_features(single_compound['SMILES'].to_list(), 'morgan', tmp_path)
        ensemble.train(features, single_compound['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_uncertainty_diversity(self, fastprop_ensemble, small_real_compounds, tmp_path):
        """Test that ensemble uncertainty captures model diversity."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        fastprop_ensemble.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = fastprop_ensemble.predict(features)

        assert np.std(uncertainty) > 0
        assert np.all(uncertainty >= 0)

    def test_mismatched_array_lengths(self, fastprop_ensemble):
        """Test error handling with mismatched feature and target lengths."""
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(ValueError):
            fastprop_ensemble.train(features, targets)

    def test_prediction_consistency(self, fastprop_ensemble, small_real_compounds, tmp_path):
        """Test that predictions are consistent across multiple calls."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        fastprop_ensemble.train(features, compounds['Activity'].to_numpy())

        predictions1, uncertainty1 = fastprop_ensemble.predict(features)
        predictions2, uncertainty2 = fastprop_ensemble.predict(features)

        assert np.allclose(predictions1, predictions2)
        assert np.allclose(uncertainty1, uncertainty2)

    def test_different_architectures(self, small_real_compounds, tmp_path):
        """Test ensemble with different architecture configurations."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)

        architectures = [
            {'fnn_layers': 0, 'hidden_size': 64},
            {'fnn_layers': 1, 'hidden_size': 64},
            {'fnn_layers': 2, 'hidden_size': 128}
        ]

        for arch in architectures:
            ensemble = FastpropEnsemble(
                fnn_layers=arch['fnn_layers'],
                hidden_size=arch['hidden_size'],
                max_epochs=3,
                device='cpu'
            )
            ensemble.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = ensemble.predict(features)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

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

        ensemble_gc_on = FastpropEnsemble(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_states=[42, 123, 456],
            enable_aggressive_gc=True
        )
        ensemble_gc_on.train(features, compounds['Activity'].to_numpy())
        pred_gc_on, _ = ensemble_gc_on.predict(features)

        ensemble_gc_off = FastpropEnsemble(
            fnn_layers=2,
            hidden_size=64,
            max_epochs=2,
            random_states=[42, 123, 456],
            enable_aggressive_gc=False
        )
        ensemble_gc_off.train(features, compounds['Activity'].to_numpy())
        pred_gc_off, _ = ensemble_gc_off.predict(features)

        assert np.allclose(pred_gc_on, pred_gc_off, rtol=1e-5)

"""Tests for ChempropEnsemble implementation."""

import pytest
import numpy as np
import polars as pl

try:
    from learnm8.learners.ensemble.chemprop_ensemble import ChempropEnsemble
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CHEMPROP_AVAILABLE,
    reason="Chemprop not available"
)


class TestChempropEnsemble:
    """Test ChempropEnsemble functionality with real molecular data."""

    @pytest.fixture
    def chemprop_ensemble(self):
        """Create ChempropEnsemble instance for testing."""
        return ChempropEnsemble(
            message_hidden_dim=300,
            depth=2,
            max_epochs=3
        )

    def test_initialization(self, chemprop_ensemble):
        """Test ensemble initialization with default parameters."""
        assert len(chemprop_ensemble.learners) == 3
        assert chemprop_ensemble.aggregation_method == 'mean'
        assert chemprop_ensemble.uncertainty_method == 'std'
        assert chemprop_ensemble.weights is None
        assert not chemprop_ensemble.is_trained
        assert chemprop_ensemble.supports_uncertainty() is True
        assert chemprop_ensemble.requires_smiles() is True
        assert chemprop_ensemble.message_hidden_dim == 300
        assert chemprop_ensemble.depth == 2
        assert chemprop_ensemble.random_states == [42, 123, 456]

    def test_initialization_custom_parameters(self):
        """Test ensemble initialization with custom parameters."""
        custom_states = [10, 20, 30]

        ensemble = ChempropEnsemble(
            message_hidden_dim=500,
            depth=5,
            ffn_hidden_dim=400,
            max_epochs=5,
            random_states=custom_states
        )

        assert len(ensemble.learners) == 3
        assert ensemble.message_hidden_dim == 500
        assert ensemble.depth == 5
        assert ensemble.ffn_hidden_dim == 400
        assert ensemble.random_states == custom_states

    def test_initialization_with_weights(self):
        """Test ensemble initialization with custom weights."""
        weights = [0.5, 0.3, 0.2]
        ensemble = ChempropEnsemble(
            depth=2,
            max_epochs=3,
            weights=weights
        )

        assert ensemble.weights is not None
        assert np.allclose(ensemble.weights, weights)

    def test_train_predict_integration(self, chemprop_ensemble, small_real_compounds):
        """Test training and prediction with real molecular data."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        chemprop_ensemble.train(features=None, targets=targets, smiles=smiles)
        assert chemprop_ensemble.is_trained

        predictions, uncertainty = chemprop_ensemble.predict(features=None, smiles=smiles)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_estimation(self, chemprop_ensemble, small_real_compounds):
        """Test that ensemble provides uncertainty estimates."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        chemprop_ensemble.train(features=None, targets=targets, smiles=smiles)

        predictions, uncertainty = chemprop_ensemble.predict(features=None, smiles=smiles)

        assert uncertainty is not None
        assert len(uncertainty) == len(compounds)
        assert np.all(np.isfinite(uncertainty))
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_diverse_random_states(self, chemprop_ensemble, small_real_compounds):
        """Test that ensemble learners have different random states."""
        assert len(chemprop_ensemble.learners) == 3

        random_states = [learner.random_state for learner in chemprop_ensemble.learners]
        assert len(set(random_states)) == 3
        assert random_states == [42, 123, 456]

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        chemprop_ensemble.train(features=None, targets=targets, smiles=smiles)

        individual_preds = chemprop_ensemble.get_individual_predictions(features=None, smiles=smiles)

        assert len(individual_preds) == 3

        pred_arrays = [preds for preds in individual_preds.values() if preds is not None]
        assert len(pred_arrays) == 3

        for i in range(len(pred_arrays)):
            for j in range(i+1, len(pred_arrays)):
                assert not np.allclose(pred_arrays[i], pred_arrays[j], rtol=1e-3)

    def test_train_with_empty_arrays(self, chemprop_ensemble):
        """Test error handling when training with empty arrays."""
        empty_smiles = []
        empty_targets = np.array([])

        with pytest.raises(ValueError):
            chemprop_ensemble.train(features=None, targets=empty_targets, smiles=empty_smiles)

    def test_predict_without_training(self, chemprop_ensemble, small_real_compounds):
        """Test error when predicting without training."""
        smiles = small_real_compounds['SMILES'].to_list()[:5]
        with pytest.raises(RuntimeError, match="Ensemble must be trained before prediction"):
            chemprop_ensemble.predict(features=None, smiles=smiles)

    def test_get_name(self, chemprop_ensemble):
        """Test name generation for Chemprop ensemble."""
        name = chemprop_ensemble.get_name()
        assert "ChempropEnsemble" in name
        assert "3xChemprop" in name
        assert "depth=2" in name
        assert "hidden=300" in name

    def test_get_name_custom_architecture(self):
        """Test name generation with custom architecture."""
        ensemble = ChempropEnsemble(
            message_hidden_dim=500,
            depth=5,
            max_epochs=3
        )
        name = ensemble.get_name()
        assert "ChempropEnsemble" in name
        assert "depth=5" in name
        assert "hidden=500" in name

    def test_supports_uncertainty(self, chemprop_ensemble):
        """Test that Chemprop ensemble supports uncertainty estimation."""
        assert chemprop_ensemble.supports_uncertainty() is True

    def test_requires_smiles(self, chemprop_ensemble):
        """Test that Chemprop ensemble requires SMILES."""
        assert chemprop_ensemble.requires_smiles() is True

    def test_aggregation_methods(self, small_real_compounds):
        """Test different aggregation methods with Chemprop ensemble."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        for method in ['mean', 'median']:
            ensemble = ChempropEnsemble(
                depth=2,
                max_epochs=3,
                aggregation_method=method
            )
            ensemble.train(features=None, targets=targets, smiles=smiles)
            predictions, uncertainty = ensemble.predict(features=None, smiles=smiles)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_uncertainty_methods(self, small_real_compounds):
        """Test different uncertainty estimation methods."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        for method in ['std', 'mad', 'quantile']:
            ensemble = ChempropEnsemble(
                depth=2,
                max_epochs=3,
                uncertainty_method=method
            )
            ensemble.train(features=None, targets=targets, smiles=smiles)
            predictions, uncertainty = ensemble.predict(features=None, smiles=smiles)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

    def test_weighted_ensemble(self, small_real_compounds):
        """Test weighted ensemble aggregation."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        weights = [0.6, 0.3, 0.1]
        ensemble = ChempropEnsemble(
            depth=2,
            max_epochs=3,
            weights=weights
        )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        ensemble.train(features=None, targets=targets, smiles=smiles)
        predictions, uncertainty = ensemble.predict(features=None, smiles=smiles)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_ensemble_statistics(self, chemprop_ensemble, small_real_compounds):
        """Test ensemble statistics retrieval."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        stats = chemprop_ensemble.get_ensemble_statistics()
        assert stats['n_learners'] == 3
        assert stats['is_trained'] is False

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        chemprop_ensemble.train(features=None, targets=targets, smiles=smiles)
        stats = chemprop_ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert 'learners_with_uncertainty' in stats
        assert len(stats['learner_names']) == 3

    def test_edge_case_single_compound(self):
        """Test with single compound."""
        ensemble = ChempropEnsemble(
            depth=2,
            max_epochs=3
        )

        single_compound = pl.DataFrame({
            'ID': ['COMP_001'],
            'SMILES': ['CCO'],
            'Activity': [0.5]
        })

        smiles = single_compound['SMILES'].to_list()
        targets = single_compound['Activity'].to_numpy()

        ensemble.train(features=None, targets=targets, smiles=smiles)
        predictions, uncertainty = ensemble.predict(features=None, smiles=smiles)

        assert len(predictions) == 1
        assert len(uncertainty) == 1
        assert np.isfinite(predictions[0])
        assert uncertainty[0] >= 0

    def test_uncertainty_diversity(self, chemprop_ensemble, small_real_compounds):
        """Test that ensemble uncertainty captures model diversity."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        chemprop_ensemble.train(features=None, targets=targets, smiles=smiles)
        predictions, uncertainty = chemprop_ensemble.predict(features=None, smiles=smiles)

        assert np.std(uncertainty) > 0
        assert np.all(uncertainty >= 0)

    def test_mismatched_array_lengths(self, chemprop_ensemble):
        """Test error handling with mismatched SMILES and target lengths."""
        smiles = ['CCO', 'CC', 'CCC']
        targets = np.random.randn(5)

        with pytest.raises(ValueError):
            chemprop_ensemble.train(features=None, targets=targets, smiles=smiles)

    def test_prediction_consistency(self, chemprop_ensemble, small_real_compounds):
        """Test that predictions are consistent across multiple calls."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        chemprop_ensemble.train(features=None, targets=targets, smiles=smiles)

        predictions1, uncertainty1 = chemprop_ensemble.predict(features=None, smiles=smiles)
        predictions2, uncertainty2 = chemprop_ensemble.predict(features=None, smiles=smiles)

        assert np.allclose(predictions1, predictions2)
        assert np.allclose(uncertainty1, uncertainty2)

    def test_different_architectures(self, small_real_compounds):
        """Test ensemble with different architecture configurations."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        architectures = [
            {'depth': 2, 'message_hidden_dim': 300},
            {'depth': 3, 'message_hidden_dim': 500},
            {'depth': 4, 'message_hidden_dim': 400, 'ffn_num_layers': 2}
        ]

        for arch in architectures:
            ensemble = ChempropEnsemble(
                **arch,
                max_epochs=3
            )
            ensemble.train(features=None, targets=targets, smiles=smiles)
            predictions, uncertainty = ensemble.predict(features=None, smiles=smiles)

            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(np.isfinite(predictions))

    def test_model_parameters_propagation(self):
        """Test that model parameters propagate to all learners."""
        ensemble = ChempropEnsemble(
            message_hidden_dim=500,
            depth=5,
            atom_messages=True,
            batch_norm=True,
            max_epochs=3
        )

        for learner in ensemble.learners:
            assert learner.message_hidden_dim == 500
            assert learner.depth == 5
            assert learner.atom_messages is True
            assert learner.batch_norm is True


class TestChempropEnsembleWithDescriptors:
    """Test ChempropEnsemble with extra descriptors (x_d)."""

    def test_train_predict_with_descriptors(self, small_real_compounds, tmp_path):
        """Test ensemble training and prediction with descriptors."""
        from learnm8.features.extraction import extract_features

        ensemble = ChempropEnsemble(max_epochs=5)
        compounds = small_real_compounds.clone()

        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        features = extract_features(smiles, 'morgan', tmp_path)

        ensemble.train(features=features, targets=targets, smiles=smiles)

        predictions, uncertainties = ensemble.predict(features=features, smiles=smiles)

        assert predictions.shape == (10,)
        assert uncertainties.shape == (10,)
        assert np.all(uncertainties >= 0)
        assert np.all(np.isfinite(predictions))

    def test_train_predict_without_descriptors(self, small_real_compounds):
        """Test ensemble backward compatibility without descriptors."""
        ensemble = ChempropEnsemble(max_epochs=5)
        compounds = small_real_compounds.clone()

        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        ensemble.train(features=None, targets=targets, smiles=smiles)

        predictions, uncertainties = ensemble.predict(features=None, smiles=smiles)

        assert predictions.shape == (10,)
        assert uncertainties.shape == (10,)
        assert np.all(uncertainties >= 0)

    def test_uncertainty_with_vs_without_descriptors(self, small_real_compounds, tmp_path):
        """Test that uncertainty is provided in both modes."""
        from learnm8.features.extraction import extract_features

        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        ensemble_with = ChempropEnsemble(max_epochs=5)
        features = extract_features(smiles, 'morgan', tmp_path)
        ensemble_with.train(features=features, targets=targets, smiles=smiles)
        _, unc_with = ensemble_with.predict(features=features, smiles=smiles)

        ensemble_without = ChempropEnsemble(max_epochs=5, random_states=[42, 123, 456])
        ensemble_without.train(features=None, targets=targets, smiles=smiles)
        _, unc_without = ensemble_without.predict(features=None, smiles=smiles)

        assert unc_with.shape == (10,)
        assert unc_without.shape == (10,)
        assert np.all(unc_with >= 0)
        assert np.all(unc_without >= 0)

    def test_aggressive_gc_enabled_by_default(self):
        """Verify enable_aggressive_gc defaults to True."""
        ensemble = ChempropEnsemble()
        assert ensemble.enable_aggressive_gc is True
        for learner in ensemble.learners:
            assert learner.enable_aggressive_gc is True

    def test_aggressive_gc_can_be_disabled(self):
        """Verify enable_aggressive_gc can be set to False."""
        ensemble = ChempropEnsemble(enable_aggressive_gc=False)
        assert ensemble.enable_aggressive_gc is False
        for learner in ensemble.learners:
            assert learner.enable_aggressive_gc is False

    def test_cleanup_gpu_memory_called_after_training(self, small_real_compounds, monkeypatch):
        """Verify _cleanup_gpu_memory is called after training when enabled."""
        cleanup_called = []

        def mock_cleanup(self, context=""):
            cleanup_called.append(context)

        ensemble = ChempropEnsemble(
            max_epochs=2,
            enable_aggressive_gc=True
        )

        monkeypatch.setattr(ensemble, '_cleanup_gpu_memory', lambda context="": mock_cleanup(ensemble, context))

        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        ensemble.train(features=None, targets=targets, smiles=smiles)

        assert len(cleanup_called) > 0
        assert any('after ensemble training' in ctx for ctx in cleanup_called)

    def test_cleanup_gpu_memory_called_after_prediction(self, small_real_compounds, monkeypatch):
        """Verify _cleanup_gpu_memory is called after prediction when enabled."""
        cleanup_called = []

        def mock_cleanup(self, context=""):
            cleanup_called.append(context)

        ensemble = ChempropEnsemble(
            max_epochs=2,
            enable_aggressive_gc=True
        )

        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        ensemble.train(features=None, targets=targets, smiles=smiles)

        cleanup_called.clear()
        monkeypatch.setattr(ensemble, '_cleanup_gpu_memory', lambda context="": mock_cleanup(ensemble, context))

        ensemble.predict(features=None, smiles=smiles)

        assert len(cleanup_called) > 0
        assert any('after ensemble prediction' in ctx for ctx in cleanup_called)

    def test_cleanup_not_called_when_disabled(self, small_real_compounds, monkeypatch):
        """Verify _cleanup_gpu_memory is not called when enable_aggressive_gc=False."""
        cleanup_called = []

        def mock_cleanup(self, context=""):
            cleanup_called.append(context)

        ensemble = ChempropEnsemble(
            max_epochs=2,
            enable_aggressive_gc=False
        )

        monkeypatch.setattr(ensemble, '_cleanup_gpu_memory', lambda context="": mock_cleanup(ensemble, context))

        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        ensemble.train(features=None, targets=targets, smiles=smiles)
        ensemble.predict(features=None, smiles=smiles)

        assert len(cleanup_called) == 0

    def test_predictions_unaffected_by_gc(self, small_real_compounds):
        """Verify predictions are identical with GC enabled vs disabled."""
        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        ensemble_gc_on = ChempropEnsemble(
            max_epochs=2,
            random_states=[42, 123, 456],
            enable_aggressive_gc=True
        )
        ensemble_gc_on.train(features=None, targets=targets, smiles=smiles)
        pred_gc_on, _ = ensemble_gc_on.predict(features=None, smiles=smiles)

        ensemble_gc_off = ChempropEnsemble(
            max_epochs=2,
            random_states=[42, 123, 456],
            enable_aggressive_gc=False
        )
        ensemble_gc_off.train(features=None, targets=targets, smiles=smiles)
        pred_gc_off, _ = ensemble_gc_off.predict(features=None, smiles=smiles)

        assert np.allclose(pred_gc_on, pred_gc_off, rtol=1e-5)

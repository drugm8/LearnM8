import pytest
import numpy as np
from pathlib import Path

from learnm8.learners.torch.chemprop_learner import ChempropLearner


@pytest.mark.slow
@pytest.mark.integration
class TestChempropLearnerBasic:

    def test_initialization_sets_message_passing_hyperparameters(self):
        learner = ChempropLearner(
            message_hidden_dim=300,
            depth=3,
            max_epochs=2
        )
        assert learner.message_hidden_dim == 300
        assert learner.depth == 3
        assert learner.max_epochs == 2
        assert learner.is_trained is False

    def test_requires_smiles_returns_true(self):
        learner = ChempropLearner(max_epochs=2)
        assert learner.requires_smiles() is True

    def test_supports_uncertainty_returns_false(self):
        learner = ChempropLearner(max_epochs=2)
        assert learner.supports_uncertainty() is False

    def test_get_name_includes_depth_and_hidden_size(self):
        learner = ChempropLearner(depth=5, message_hidden_dim=500)
        name = learner.get_name()
        assert 'Chemprop' in name
        assert 'depth=5' in name
        assert 'hidden=500' in name

    def test_train_raises_when_smiles_are_missing(self):
        learner = ChempropLearner(max_epochs=2)
        features = np.random.rand(10, 100)
        targets = np.random.rand(10)

        with pytest.raises(ValueError, match="requires SMILES"):
            learner.train(features, targets, smiles=None)

    def test_predict_raises_when_smiles_are_missing_after_training(self, small_real_compounds):
        learner = ChempropLearner(max_epochs=2)

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()
        learner.train(features=None, targets=targets, smiles=smiles)

        features = np.random.rand(5, 100)
        with pytest.raises(ValueError, match="requires SMILES"):
            learner.predict(features, smiles=None)

    def test_predict_raises_before_training(self):
        learner = ChempropLearner(max_epochs=2)

        with pytest.raises(RuntimeError, match="must be trained"):
            learner.predict(features=None, smiles=['CC', 'CCO'])


@pytest.mark.slow
@pytest.mark.integration
class TestChempropLearnerTrainPredict:

    def test_train_and_predict_return_finite_values_without_uncertainty(self, small_real_compounds):
        learner = ChempropLearner(max_epochs=5)

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        learner.train(features=None, targets=targets, smiles=smiles)

        assert learner.is_trained is True
        assert learner.model is not None

        predictions, uncertainties = learner.predict(features=None, smiles=smiles)

        assert predictions.shape == (len(smiles),)
        assert uncertainties is None
        assert np.all(np.isfinite(predictions))

    def test_predict_returns_finite_values_for_unseen_compounds(self, small_real_compounds):
        learner = ChempropLearner(max_epochs=5)

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        n_train = len(compounds) // 2
        train_smiles = compounds['SMILES'][:n_train].to_list()
        train_targets = compounds['Activity'][:n_train].to_numpy()
        test_smiles = compounds['SMILES'][n_train:].to_list()

        learner.train(features=None, targets=train_targets, smiles=train_smiles)
        predictions, uncertainties = learner.predict(features=None, smiles=test_smiles)

        assert predictions.shape == (len(test_smiles),)
        assert uncertainties is None
        assert np.all(np.isfinite(predictions))


@pytest.mark.slow
@pytest.mark.integration
class TestChempropLearnerModelParameters:

    def test_custom_architecture_parameters_train_and_predict(self, small_real_compounds):
        learner = ChempropLearner(
            message_hidden_dim=500,
            depth=5,
            ffn_hidden_dim=400,
            ffn_num_layers=2,
            max_epochs=3
        )

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        learner.train(features=None, targets=targets, smiles=smiles)
        predictions, _ = learner.predict(features=None, smiles=smiles)

        assert predictions.shape == (len(smiles),)
        assert np.all(np.isfinite(predictions))

    def test_atom_messages_mode_trains_and_predicts(self, small_real_compounds):
        learner = ChempropLearner(
            atom_messages=True,
            max_epochs=3,
            early_stopping=False
        )

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        learner.train(features=None, targets=targets, smiles=smiles)
        predictions, _ = learner.predict(features=None, smiles=smiles)

        assert predictions.shape == (len(smiles),)

    def test_batch_norm_enabled_trains_and_predicts(self, small_real_compounds):
        learner = ChempropLearner(
            batch_norm=True,
            max_epochs=3,
            early_stopping=False
        )

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        learner.train(features=None, targets=targets, smiles=smiles)
        predictions, _ = learner.predict(features=None, smiles=smiles)

        assert predictions.shape == (len(smiles),)

    def test_supported_aggregation_modes_train_and_predict(self, small_real_compounds):
        for agg in ['mean', 'sum', 'norm']:
            learner = ChempropLearner(
                aggregation=agg,
                max_epochs=2,
                early_stopping=False
            )

            compounds = small_real_compounds.clone()
            if 'Activity' not in compounds.columns:
                compounds = compounds.with_columns(
                    pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
                )

            smiles = compounds['SMILES'].to_list()[:10]
            targets = compounds['Activity'].to_numpy()[:10]

            learner.train(features=None, targets=targets, smiles=smiles)
            predictions, _ = learner.predict(features=None, smiles=smiles)

            assert predictions.shape == (len(smiles),)


@pytest.mark.slow
@pytest.mark.integration
class TestChempropLearnerEarlyStopping:

    def test_early_stopping_enabled_by_default(self):
        learner = ChempropLearner(max_epochs=50)
        assert learner.early_stopping is True
        assert learner.early_stopping_patience == 10
        assert learner.val_fraction == 0.1

    def test_early_stopping_stops_before_max_epochs(self, small_real_compounds):
        learner = ChempropLearner(
            max_epochs=100,
            early_stopping=True,
            early_stopping_patience=2,
            val_fraction=0.2
        )

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        learner.train(features=None, targets=targets, smiles=smiles)

        assert learner.is_trained is True
        assert learner.trainer.current_epoch < 100

    def test_disabling_early_stopping_runs_full_epoch_count(self, small_real_compounds):
        max_epochs = 5
        learner = ChempropLearner(
            max_epochs=max_epochs,
            early_stopping=False
        )

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()[:20]
        targets = compounds['Activity'].to_numpy()[:20]

        learner.train(features=None, targets=targets, smiles=smiles)

        assert learner.is_trained is True
        assert learner.trainer.current_epoch == max_epochs

    def test_validation_split_configuration_trains_successfully(self, small_real_compounds):
        learner = ChempropLearner(
            max_epochs=2,
            early_stopping=True,
            val_fraction=0.2
        )

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()
        n_samples = len(smiles)

        learner.train(features=None, targets=targets, smiles=smiles)

        assert learner.is_trained is True

    def test_early_stopping_with_custom_parameters(self, small_real_compounds):
        learner = ChempropLearner(
            max_epochs=50,
            early_stopping=True,
            early_stopping_patience=5,
            early_stopping_min_delta=0.001,
            val_fraction=0.15
        )

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        learner.train(features=None, targets=targets, smiles=smiles)

        assert learner.is_trained is True
        assert learner.early_stopping_patience == 5
        assert learner.early_stopping_min_delta == 0.001
        assert learner.val_fraction == 0.15


@pytest.mark.slow
@pytest.mark.integration
class TestChempropLearnerWithDescriptors:
    """Test ChempropLearner with extra descriptors (x_d)."""

    def test_train_predict_with_morgan_descriptors(self, small_real_compounds, tmp_path):
        """Test training and prediction with Morgan fingerprints as x_d."""
        from learnm8.features.extraction import extract_features

        learner = ChempropLearner(max_epochs=5)
        compounds = small_real_compounds.clone()

        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        features = extract_features(smiles, 'morgan', tmp_path)

        learner.train(features=features, targets=targets, smiles=smiles)

        assert learner.is_trained is True
        assert learner.training_feature_dim == 2048

        predictions, uncertainty = learner.predict(features=features, smiles=smiles)

        assert predictions.shape == (10,)
        assert uncertainty is None
        assert np.all(np.isfinite(predictions))

    def test_train_predict_without_descriptors_backward_compat(self, small_real_compounds):
        """Test backward compatibility - training without descriptors."""
        learner = ChempropLearner(max_epochs=5)
        compounds = small_real_compounds.clone()

        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        learner.train(features=None, targets=targets, smiles=smiles)

        assert learner.is_trained is True
        assert learner.training_feature_dim is None

        predictions, uncertainty = learner.predict(features=None, smiles=smiles)

        assert predictions.shape == (10,)
        assert uncertainty is None
        assert np.all(np.isfinite(predictions))

    def test_descriptor_dimension_validation_train_without_predict_with(self, small_real_compounds, tmp_path):
        """Test error when trained without descriptors but predict with descriptors."""
        from learnm8.features.extraction import extract_features

        learner = ChempropLearner(max_epochs=5)
        compounds = small_real_compounds.clone()

        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        learner.train(features=None, targets=targets, smiles=smiles)

        features = extract_features(smiles, 'morgan', tmp_path)

        with pytest.raises(ValueError, match="trained without descriptors"):
            learner.predict(features=features, smiles=smiles)

    def test_descriptor_dimension_validation_train_with_predict_without(self, small_real_compounds, tmp_path):
        """Test error when trained with descriptors but predict without descriptors."""
        from learnm8.features.extraction import extract_features

        learner = ChempropLearner(max_epochs=5)
        compounds = small_real_compounds.clone()

        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        features = extract_features(smiles, 'morgan', tmp_path)
        learner.train(features=features, targets=targets, smiles=smiles)

        with pytest.raises(ValueError, match="trained with descriptors"):
            learner.predict(features=None, smiles=smiles)

    def test_descriptor_dimension_mismatch(self, small_real_compounds, tmp_path):
        """Test error on dimension mismatch between training and prediction."""
        from learnm8.features.extraction import extract_features

        learner = ChempropLearner(max_epochs=5)
        compounds = small_real_compounds.clone()

        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        train_features = extract_features(smiles, 'morgan', tmp_path)
        learner.train(features=train_features, targets=targets, smiles=smiles)

        test_features = extract_features(smiles, 'maccs', tmp_path)

        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            learner.predict(features=test_features, smiles=smiles)

    def test_multiple_descriptor_featurizers_train_and_predict(self, small_real_compounds, tmp_path):
        """Test compatibility with different featurizer types."""
        from learnm8.features.extraction import extract_features

        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        for featurizer in ['morgan', 'maccs', 'ecfp6']:
            learner = ChempropLearner(max_epochs=5)
            features = extract_features(smiles, featurizer, tmp_path)

            learner.train(features=features, targets=targets, smiles=smiles)
            predictions, _ = learner.predict(features=features, smiles=smiles)

            assert predictions.shape == (10,)
            assert np.all(np.isfinite(predictions))

    def test_early_stopping_with_descriptors(self, small_real_compounds, tmp_path):
        """Test early stopping works correctly with descriptors."""
        from learnm8.features.extraction import extract_features

        learner = ChempropLearner(
            max_epochs=20,
            early_stopping=True,
            early_stopping_patience=3,
            val_fraction=0.2
        )
        compounds = small_real_compounds.clone()

        smiles = compounds['SMILES'].to_list()[:20]
        targets = compounds['Activity'].to_numpy()[:20]
        features = extract_features(smiles, 'morgan', tmp_path)

        learner.train(features=features, targets=targets, smiles=smiles)

        assert learner.is_trained is True
        assert learner.training_feature_dim == 2048

        predictions, _ = learner.predict(features=features, smiles=smiles)
        assert predictions.shape == (20,)
        assert np.all(np.isfinite(predictions))

    def test_aggressive_gc_enabled_by_default(self):
        """Verify enable_aggressive_gc defaults to True."""
        learner = ChempropLearner()
        assert learner.enable_aggressive_gc is True

    def test_aggressive_gc_can_be_disabled(self):
        """Verify enable_aggressive_gc can be set to False."""
        learner = ChempropLearner(enable_aggressive_gc=False)
        assert learner.enable_aggressive_gc is False

    def test_cleanup_gpu_memory_called_after_training(self, small_real_compounds, monkeypatch):
        """Verify _cleanup_gpu_memory is called after training when enabled."""
        cleanup_called = []

        def mock_cleanup(self, context=""):
            cleanup_called.append(context)

        learner = ChempropLearner(
            max_epochs=2,
            enable_aggressive_gc=True
        )

        monkeypatch.setattr(learner, '_cleanup_gpu_memory', lambda context="": mock_cleanup(learner, context))

        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        learner.train(features=None, targets=targets, smiles=smiles)

        assert len(cleanup_called) > 0
        assert any('after training' in ctx for ctx in cleanup_called)

    def test_cleanup_gpu_memory_called_after_prediction(self, small_real_compounds, monkeypatch):
        """Verify _cleanup_gpu_memory is called after prediction when enabled."""
        cleanup_called = []

        def mock_cleanup(self, context=""):
            cleanup_called.append(context)

        learner = ChempropLearner(
            max_epochs=2,
            enable_aggressive_gc=True
        )

        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        learner.train(features=None, targets=targets, smiles=smiles)

        cleanup_called.clear()
        monkeypatch.setattr(learner, '_cleanup_gpu_memory', lambda context="": mock_cleanup(learner, context))

        learner.predict(features=None, smiles=smiles)

        assert len(cleanup_called) > 0
        assert any('after prediction' in ctx for ctx in cleanup_called)

    def test_cleanup_not_called_when_disabled(self, small_real_compounds, monkeypatch):
        """Verify _cleanup_gpu_memory is a no-op when enable_aggressive_gc=False."""
        learner = ChempropLearner(
            max_epochs=2,
            enable_aggressive_gc=False
        )

        assert learner.enable_aggressive_gc is False

        cleanup_did_work = []
        original_cleanup = ChempropLearner._cleanup_gpu_memory

        def tracking_cleanup(self, context=""):
            original_cleanup(self, context)
            if self.enable_aggressive_gc:
                cleanup_did_work.append(context)

        monkeypatch.setattr(ChempropLearner, '_cleanup_gpu_memory', tracking_cleanup)

        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        learner.train(features=None, targets=targets, smiles=smiles)
        learner.predict(features=None, smiles=smiles)

        assert len(cleanup_did_work) == 0

    def test_predictions_unaffected_by_gc(self, small_real_compounds):
        """Verify predictions are identical with GC enabled vs disabled."""
        compounds = small_real_compounds.clone()
        smiles = compounds['SMILES'].to_list()[:10]
        targets = compounds['Activity'].to_numpy()[:10]

        learner_gc_on = ChempropLearner(
            max_epochs=2,
            random_state=42,
            enable_aggressive_gc=True
        )
        learner_gc_on.train(features=None, targets=targets, smiles=smiles)
        pred_gc_on, _ = learner_gc_on.predict(features=None, smiles=smiles)

        learner_gc_off = ChempropLearner(
            max_epochs=2,
            random_state=42,
            enable_aggressive_gc=False
        )
        learner_gc_off.train(features=None, targets=targets, smiles=smiles)
        pred_gc_off, _ = learner_gc_off.predict(features=None, smiles=smiles)

        assert np.allclose(pred_gc_on, pred_gc_off, rtol=1e-2, atol=1e-3)


@pytest.mark.slow
@pytest.mark.integration
class TestChempropLearnerPerformanceConfig:

    def test_default_predict_batch_size_is_four_times_training_batch_size(self):
        learner = ChempropLearner(batch_size=32)
        assert learner.predict_batch_size == 128

    def test_custom_predict_batch_size_overrides_default(self):
        learner = ChempropLearner(batch_size=32, predict_batch_size=256)
        assert learner.predict_batch_size == 256

    def test_precision_auto_cpu(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        learner = ChempropLearner(precision='auto')
        assert learner.precision == '32-true'

    def test_explicit_precision_values_are_preserved(self):
        learner = ChempropLearner(precision='16-mixed')
        assert learner.precision == '16-mixed'

        learner = ChempropLearner(precision='32-true')
        assert learner.precision == '32-true'

    def test_precision_invalid(self):
        with pytest.raises(ValueError, match="Invalid precision"):
            ChempropLearner(precision='invalid')

    def test_pin_memory_defaults_to_true(self):
        learner = ChempropLearner()
        assert learner.pin_memory is True

    def test_pin_memory_can_be_disabled(self):
        learner = ChempropLearner(pin_memory=False)
        assert learner.pin_memory is False

    def test_train_predict_with_performance_params(self, small_real_compounds):
        import polars as pl
        learner = ChempropLearner(
            max_epochs=5,
            batch_size=8,
            predict_batch_size=16,
            precision='32-true',
            pin_memory=True
        )

        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        smiles = compounds['SMILES'].to_list()
        targets = compounds['Activity'].to_numpy()

        learner.train(features=None, targets=targets, smiles=smiles)
        predictions, _ = learner.predict(features=None, smiles=smiles)

        assert predictions.shape == (len(smiles),)
        assert np.all(np.isfinite(predictions))

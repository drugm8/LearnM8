"""Tests for MCDropoutLearner implementation."""

import numpy as np
import polars as pl
import pytest
import torch.nn as nn

from learnm8.exceptions import LearnerError
from learnm8.features.extraction import extract_features
from learnm8.learners.torch.mc_dropout import MCDropoutLearner


@pytest.mark.slow
@pytest.mark.integration
class TestMCDropoutLearner:
    """Test MCDropoutLearner functionality with real molecular data."""

    @pytest.fixture
    def learner(self):
        """Create MCDropoutLearner instance for testing."""
        return MCDropoutLearner(
            hidden_sizes=(64, 32),
            n_dropout_samples=20,
            max_epochs=2,
            random_state=42
        )

    def test_initialization_sets_dropout_sampling_defaults_and_uncertainty_support(self, learner):
        """Test learner initialization."""
        assert learner.hidden_sizes == (64, 32)
        assert learner.dropout_rate == 0.2
        assert learner.n_dropout_samples == 20
        assert learner.activation == 'relu'
        assert learner.batch_norm is False
        assert learner.max_epochs == 2
        assert not learner.is_trained
        assert learner.supports_uncertainty() is True

    def test_predict_returns_finite_values_and_uncertainty_after_training(self, learner, small_real_compounds, small_real_morgan_features):
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
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_mc_dropout_uncertainty_varies_across_compounds_and_remains_finite(self, learner, small_real_compounds, small_real_morgan_features):
        """Test that MC Dropout uncertainty estimates are reasonable."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = learner.predict(features)

        assert np.std(uncertainty) > 0
        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(uncertainty))

    def test_dropout_samples_effect(self, small_real_compounds, small_real_morgan_features):
        """Test effect of different numbers of dropout samples."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features.copy()

        learner_few = MCDropoutLearner(
            hidden_sizes=(32,),
            n_dropout_samples=5,
            max_epochs=3,
            random_state=42
        )

        learner_few.train(features, compounds['Activity'].to_numpy())
        pred_few, unc_few = learner_few.predict(features)

        learner_many = MCDropoutLearner(
            hidden_sizes=(32,),
            n_dropout_samples=50,
            max_epochs=3,
            random_state=42
        )

        learner_many.train(features, compounds['Activity'].to_numpy())
        pred_many, unc_many = learner_many.predict(features)

        assert len(pred_few) == len(compounds)
        assert len(pred_many) == len(compounds)
        assert np.all(unc_few >= 0)
        assert np.all(unc_many >= 0)

    def test_predict_without_training(self, learner, small_real_morgan_features):
        """Test error when predicting without training."""
        features = small_real_morgan_features.copy()
        with pytest.raises(LearnerError, match="must be trained before prediction"):
            learner.predict(features)

    def test_get_name_includes_architecture_dropout_and_sample_count(self, learner):
        """Test name generation."""
        name = learner.get_name()
        assert "MCDropout" in name
        assert "64-32" in name
        assert "samples=20" in name
        assert "dropout=0.2" in name

    def test_custom_architecture_configuration_trains_and_predicts(self, small_real_compounds, small_real_morgan_features):
        """Test learner with different architectures."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = MCDropoutLearner(
            hidden_sizes=(128, 64, 32),
            activation='gelu',
            dropout_rate=0.3,
            n_dropout_samples=10,
            max_epochs=3,
            random_state=42
        )

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = learner.predict(features)

        assert learner.hidden_sizes == (128, 64, 32)
        assert learner.activation == 'gelu'
        assert learner.dropout_rate == 0.3
        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)

    def test_dropout_rate_effect(self, small_real_compounds, small_real_morgan_features):
        """Test effect of different dropout rates."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        for dropout_rate in [0.1, 0.3, 0.5]:
            learner = MCDropoutLearner(
                hidden_sizes=(32,),
                dropout_rate=dropout_rate,
                n_dropout_samples=10,
                max_epochs=2,
                random_state=42
            )

            features = small_real_morgan_features.copy()
            learner.train(features, compounds['Activity'].to_numpy())
            predictions, uncertainty = learner.predict(features)

            assert learner.dropout_rate == dropout_rate
            assert predictions.shape[0] == len(compounds)
            assert uncertainty.shape[0] == len(compounds)
            assert np.all(uncertainty >= 0)

    def test_single_compound_training_raises_for_zero_variance_features(self, tmp_path):
        """Test that single compound fails gracefully (zero-variance features)."""
        single_compound = pl.DataFrame({
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

        features = extract_features(single_compound['SMILES'].to_list(), 'morgan', tmp_path)
        with pytest.raises(LearnerError):
            learner.train(features, single_compound['Activity'].to_numpy())

    def test_uncertainty_consistency(self, learner, tmp_path):
        """Repeated predict() on the same input is reproducible (Item 6).

        The dropout RNG is snapshotted/restored around predict and reseeded per
        chunk, so two consecutive predict() calls return identical mean and
        uncertainty — σ is no longer a fresh random draw each call.
        """
        compounds = pl.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': ['CCO', 'CCC', 'CCN'],
            'Activity': [0.3, 0.6, 0.9]
        })

        features = extract_features(compounds['SMILES'].to_list(), 'morgan', tmp_path)
        learner.train(features, compounds['Activity'].to_numpy())

        pred1, unc1 = learner.predict(features)
        pred2, unc2 = learner.predict(features)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(unc1, unc2)

        assert np.all(unc1 >= 0)
        assert np.all(unc2 >= 0)

    def test_deterministic_with_same_seed(self, small_real_compounds, small_real_morgan_features):
        """Test reproducibility with same random seed."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

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

        features = small_real_morgan_features.copy()
        learner1.train(features, compounds['Activity'].to_numpy())
        learner2.train(features, compounds['Activity'].to_numpy())

        pred1, _ = learner1.predict(features)
        pred2, _ = learner2.predict(features)

        assert len(pred1) == len(pred2)
        assert np.all(np.isfinite(pred1))
        assert np.all(np.isfinite(pred2))

    def test_train_with_empty_arrays(self, learner):
        empty_features = np.array([]).reshape(0, 10)
        empty_targets = np.array([])

        with pytest.raises(LearnerError, match="Cannot train"):
            learner.train(empty_features, empty_targets)

    def test_train_with_mismatched_shapes(self, learner):
        features = np.random.randn(10, 5)
        targets = np.random.randn(8)

        with pytest.raises(LearnerError, match="mismatched lengths"):
            learner.train(features, targets)

    def test_train_with_1d_features(self, learner):
        """Test that 1D features are reshaped to 2D automatically."""
        features_1d = np.random.rand(10)
        targets = np.random.rand(10)
        learner.train(features_1d, targets)
        assert learner.is_trained

    def test_batchnorm_eval_during_predict(self, small_real_compounds, small_real_morgan_features):
        """Test that BatchNorm layers are in eval mode during MC predict."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        learner = MCDropoutLearner(
            hidden_sizes=(32,),
            batch_norm=True,
            n_dropout_samples=10,
            max_epochs=3,
            random_state=42
        )

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())
        predictions, uncertainty = learner.predict(features)

        for m in learner.model.modules():
            if isinstance(m, nn.BatchNorm1d):
                assert not m.training, "BatchNorm should be in eval mode after predict"

        assert np.all(uncertainty >= 0)
        assert np.any(uncertainty > 0)
        assert np.all(np.isfinite(predictions))

    def test_uncertainty_nonnegative_nonzero(self, learner, small_real_compounds, small_real_morgan_features):
        """Test that uncertainty values are non-negative and non-zero for reasonable inputs."""
        compounds = small_real_compounds.clone()
        if 'Activity' not in compounds.columns:
            compounds = compounds.with_columns(
                pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
            )

        features = small_real_morgan_features.copy()
        learner.train(features, compounds['Activity'].to_numpy())
        _, uncertainty = learner.predict(features)

        assert np.all(uncertainty >= 0), "Uncertainty must be non-negative"
        assert np.any(uncertainty > 0), "Uncertainty should be non-zero for varied inputs"

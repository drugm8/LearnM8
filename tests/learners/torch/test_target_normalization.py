"""T020 - TorchLearner target normalization tests.

Tests verify that MLP and MC Dropout learners standardize training targets
before the training loop and inverse-transform predictions to the original
target scale. Written before implementation per TDD.
"""

import numpy as np
import pytest

from learnm8.learners.torch.mlp import MLPLearner


@pytest.mark.unit
class TestTorchLearnerTargetNormalization:
    """FR-006, FR-007, FR-008, FR-009: Target normalization for TorchLearner."""

    def test_target_scaler_fitted_after_train(self):
        """FR-006: _target_scaler is a fitted StandardScaler after train()."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20).astype(np.float32)
        y = (rng.randn(50) * 500 + 3000).astype(np.float32)

        learner = MLPLearner(
            hidden_sizes=(32,),
            max_epochs=10,
            random_state=42,
            batch_norm=False,
        )
        learner.train(X, y)

        assert learner._target_scaler is not None, (
            "_target_scaler should be set after training on high-magnitude targets"
        )
        assert hasattr(learner._target_scaler, 'mean_'), (
            "_target_scaler should be a fitted StandardScaler (has mean_ attribute)"
        )
        assert hasattr(learner._target_scaler, 'scale_'), (
            "_target_scaler should be a fitted StandardScaler (has scale_ attribute)"
        )

    def test_target_scaler_captures_correct_statistics(self):
        """FR-006: _target_scaler.mean_ and scale_ match the training target distribution."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20).astype(np.float32)
        target_mean = 3000.0
        target_std = 500.0
        y = (rng.randn(50) * target_std + target_mean).astype(np.float32)

        learner = MLPLearner(
            hidden_sizes=(32,),
            max_epochs=5,
            random_state=42,
            batch_norm=False,
        )
        learner.train(X, y)

        assert learner._target_scaler is not None
        np.testing.assert_allclose(
            learner._target_scaler.mean_[0],
            np.mean(y),
            rtol=1e-3,
            err_msg="_target_scaler.mean_ should match the mean of training targets",
        )
        np.testing.assert_allclose(
            learner._target_scaler.scale_[0],
            np.std(y, ddof=1) if len(y) > 1 else np.std(y),
            rtol=0.1,
            err_msg="_target_scaler.scale_ should be close to std of training targets",
        )

    def test_predictions_inverse_transformed_to_original_scale(self):
        """FR-007, SC-004: Predictions are in original target scale, not normalized scale."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20).astype(np.float32)
        target_mean = 3000.0
        target_std = 500.0
        y = (rng.randn(50) * target_std + target_mean).astype(np.float32)

        learner = MLPLearner(
            hidden_sizes=(32,),
            max_epochs=10,
            random_state=42,
            batch_norm=False,
        )
        learner.train(X, y)
        predictions, _ = learner.predict(X)

        pred_mean = np.mean(predictions)
        pred_min = np.min(predictions)
        pred_max = np.max(predictions)

        assert pred_mean > 100, (
            f"Predictions mean={pred_mean:.2f} should be in original scale (~3000), "
            f"not normalized scale (~0)"
        )
        assert pred_min > -1000, (
            f"Predictions min={pred_min:.2f} should be in original target range, "
            f"not normalized (~-3)"
        )
        assert pred_max < 10000, (
            f"Predictions max={pred_max:.2f} should be in original target range"
        )

    def test_zero_variance_targets_skip_normalization(self):
        """FR-009: Zero-variance targets must not trigger division-by-zero in scaler."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20).astype(np.float32)
        y = np.full(50, 2.5, dtype=np.float32)

        learner = MLPLearner(
            hidden_sizes=(32,),
            max_epochs=5,
            random_state=42,
            batch_norm=False,
        )
        learner.train(X, y)

        assert learner._target_scaler is None, (
            "_target_scaler should be None when targets have zero variance "
            "(normalization skipped to avoid division by zero)"
        )

        predictions, _ = learner.predict(X)
        assert np.all(np.isfinite(predictions)), (
            "Predictions should be finite even when target normalization is skipped"
        )

    def test_predictions_finite_on_large_magnitude_targets(self):
        """End-to-end: training + predicting on large-magnitude targets produces finite output."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20).astype(np.float32)
        y = (rng.randn(50) * 500 + 3000).astype(np.float32)

        learner = MLPLearner(
            hidden_sizes=(32,),
            max_epochs=10,
            random_state=42,
            batch_norm=False,
        )
        learner.train(X, y)
        predictions, _ = learner.predict(X)

        assert predictions.shape[0] == 50
        assert np.all(np.isfinite(predictions)), (
            "All predictions should be finite after training on large-magnitude targets"
        )

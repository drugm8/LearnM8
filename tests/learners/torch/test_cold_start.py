"""T014 - TorchLearner cold-start model recreation tests.

Tests verify that each call to train() creates a fresh model with new random
weights, regardless of whether input dimensions changed. Written before
implementation per TDD.
"""

import hashlib

import numpy as np
import pytest
import torch

from learnm8.learners.torch.mlp import MLPLearner


def _state_dict_checksum(model: torch.nn.Module) -> str:
    """Compute a deterministic hash of all model parameter tensors."""
    h = hashlib.md5()
    for key in sorted(model.state_dict().keys()):
        tensor = model.state_dict()[key]
        h.update(tensor.cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.unit
class TestTorchLearnerColdStart:
    """FR-004, FR-005: Cold-start model recreation between AL cycles."""

    def test_model_parameters_differ_between_consecutive_train_calls(self):
        """FR-004: Each train() produces fresh weights; consecutive trains differ."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20).astype(np.float32)
        y = rng.randn(50).astype(np.float32)

        learner = MLPLearner(
            hidden_sizes=(16,),
            max_epochs=3,
            random_state=42,
            batch_norm=False,
        )

        learner.train(X, y)
        checksum_cycle1 = _state_dict_checksum(learner.model)

        X2 = rng.randn(50, 20).astype(np.float32)
        y2 = rng.randn(50).astype(np.float32)
        learner.train(X2, y2)
        checksum_cycle2 = _state_dict_checksum(learner.model)

        assert checksum_cycle1 != checksum_cycle2, (
            "Model weights should differ between consecutive train() calls "
            "(cold-start recreates model from scratch each cycle)"
        )

    def test_model_recreated_when_input_dimensions_unchanged(self):
        """FR-004: Model is recreated even when input_dim stays the same."""
        rng = np.random.RandomState(10)
        X = rng.randn(50, 20).astype(np.float32)
        y = rng.randn(50).astype(np.float32)

        learner = MLPLearner(
            hidden_sizes=(16,),
            max_epochs=3,
            random_state=42,
            batch_norm=False,
        )

        learner.train(X, y)
        model_id_after_cycle1 = id(learner.model)

        learner.train(X, y)
        model_id_after_cycle2 = id(learner.model)

        assert model_id_after_cycle1 != model_id_after_cycle2, (
            "A new model object should be created on each train() call (cold-start)"
        )

    def test_three_cycle_parameter_checksums_all_differ(self):
        """SC-003: Three consecutive train() calls each produce distinct parameter states."""
        rng = np.random.RandomState(7)

        learner = MLPLearner(
            hidden_sizes=(16,),
            max_epochs=3,
            random_state=42,
            batch_norm=False,
        )

        checksums = []
        for _ in range(3):
            X = rng.randn(50, 20).astype(np.float32)
            y = rng.randn(50).astype(np.float32)
            learner.train(X, y)
            checksums.append(_state_dict_checksum(learner.model))

        assert len(set(checksums)) == 3, (
            f"All three cycles should produce different model states. "
            f"Got checksums: {checksums}"
        )

    def test_optimizer_recreated_each_cycle(self):
        """FR-004: Optimizer object is recreated on each train() call."""
        rng = np.random.RandomState(99)
        X = rng.randn(50, 20).astype(np.float32)
        y = rng.randn(50).astype(np.float32)

        learner = MLPLearner(
            hidden_sizes=(16,),
            max_epochs=3,
            random_state=42,
            batch_norm=False,
        )

        learner.train(X, y)
        optimizer_id_1 = id(learner.optimizer)

        learner.train(X, y)
        optimizer_id_2 = id(learner.optimizer)

        assert optimizer_id_1 != optimizer_id_2, (
            "Optimizer should be a new object after each train() call (cold-start)"
        )

    def test_feature_scaler_reset_each_cycle(self):
        """Cold-start resets the feature scaler so stale scaler state cannot persist."""
        rng = np.random.RandomState(55)

        learner = MLPLearner(
            hidden_sizes=(16,),
            max_epochs=3,
            random_state=42,
            batch_norm=False,
        )

        X1 = rng.randn(50, 20).astype(np.float32) * 10 + 100
        y1 = rng.randn(50).astype(np.float32)
        learner.train(X1, y1)
        mean1 = learner.scaler.mean_.copy()

        X2 = rng.randn(50, 20).astype(np.float32) * 0.01
        y2 = rng.randn(50).astype(np.float32)
        learner.train(X2, y2)
        mean2 = learner.scaler.mean_.copy()

        assert not np.allclose(mean1, mean2), (
            "Feature scaler should be re-fit on each cycle (cold-start resets scaler)"
        )

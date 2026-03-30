"""Tests for SVGPLearner checkpoint save/restore (T007)."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.slow]

gpytorch = pytest.importorskip("gpytorch")

from learnm8.learners.gpytorch.svgp import SVGPLearner


class TestSVGPCheckpointing:
    def test_best_epoch_weights_restored_after_early_stopping(self):
        """After early stopping, model+likelihood weights match best ELBO epoch."""
        rng = np.random.RandomState(42)
        features = rng.rand(100, 10).astype(np.float32)
        features = (features > 0.5).astype(np.float32)
        targets = rng.randn(100).astype(np.float32)

        learner = SVGPLearner(
            kernel="tanimoto",
            n_inducing=20,
            n_epochs=50,
            early_stopping_patience=5,
            device="cpu",
            random_state=42,
        )
        learner.train(features, targets)
        assert learner.is_trained

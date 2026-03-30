"""Tests for SVGPLearner feature scaling (T030)."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.slow]

gpytorch = pytest.importorskip("gpytorch")

from learnm8.learners.gpytorch.svgp import SVGPLearner


class TestSVGPScaling:
    def test_rbf_kernel_applies_feature_scaling(self):
        rng = np.random.RandomState(42)
        features = rng.randn(50, 5).astype(np.float32) * 100
        targets = rng.randn(50).astype(np.float32)

        learner = SVGPLearner(kernel="rbf", n_inducing=10, device="cpu", random_state=42)
        learner._feature_type = "continuous"
        learner.train(features, targets)

        assert learner.is_trained
        assert hasattr(learner, "_feature_scaler")
        assert learner._feature_scaler is not None

    def test_tanimoto_kernel_skips_feature_scaling(self):
        rng = np.random.RandomState(42)
        features = (rng.rand(50, 10) > 0.5).astype(np.float32)
        targets = rng.randn(50).astype(np.float32)

        learner = SVGPLearner(kernel="tanimoto", n_inducing=10, device="cpu", random_state=42)
        learner._feature_type = "binary"
        learner.train(features, targets)

        assert learner.is_trained
        assert learner._feature_scaler is None

    def test_auto_kernel_continuous_selects_rbf_and_scales(self):
        rng = np.random.RandomState(42)
        features = rng.randn(50, 5).astype(np.float32) * 100
        targets = rng.randn(50).astype(np.float32)

        learner = SVGPLearner(kernel="auto", n_inducing=10, device="cpu", random_state=42)
        learner._feature_type = "continuous"
        learner.train(features, targets)

        assert learner.is_trained
        assert learner._kernel_name == "rbf"
        assert learner._feature_scaler is not None

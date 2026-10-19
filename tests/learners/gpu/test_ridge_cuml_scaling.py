"""Tests for RidgeCumlLearner feature scaling."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.slow]

cuml = pytest.importorskip('cuml')

# Import below needs importorskip first.
from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner  # noqa: E402


class TestRidgeCumlScaling:
    def test_feature_scaling_applied_during_training(self):
        rng = np.random.RandomState(42)
        features = rng.randn(50, 5).astype(np.float32) * 100
        targets = rng.randn(50).astype(np.float32)

        learner = RidgeCumlLearner(alpha=0.1, random_state=42)
        learner._feature_type = 'continuous'
        learner.train(features, targets)

        assert learner.is_trained
        assert hasattr(learner, '_feature_scaler')
        assert learner._feature_scaler is not None

    def test_predictions_use_scaled_features(self):
        rng = np.random.RandomState(42)
        features = rng.randn(50, 5).astype(np.float32) * 100
        targets = rng.randn(50).astype(np.float32)

        learner = RidgeCumlLearner(alpha=0.1, random_state=42)
        learner._feature_type = 'continuous'
        learner.train(features, targets)

        pred_features = rng.randn(10, 5).astype(np.float32) * 100
        preds, _uncert = learner.predict(pred_features)
        assert np.all(np.isfinite(preds))

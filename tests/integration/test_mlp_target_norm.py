import numpy as np
import pytest

from learnm8.learners.torch.mlp import MLPLearner

pytestmark = pytest.mark.unit

class TestMLPTargetNormIntegration:
    @pytest.mark.slow
    def test_predictions_in_original_scale_for_high_magnitude_targets(self):
        """MLP trained on targets with mean>1000 returns predictions in original scale."""
        rng = np.random.RandomState(42)
        features = rng.randn(100, 20).astype(np.float32)
        # Targets with large magnitude: range ~[2000, 4000]
        targets = (rng.randn(100) * 500 + 3000).astype(np.float32)

        learner = MLPLearner(
            hidden_sizes=(32, 16), max_epochs=50,
            learning_rate=0.01, random_state=42
        )
        learner.train(features, targets)

        preds, _ = learner.predict(features)

        # Predictions should be in the original scale, not normalized
        # They should be roughly in the target range, not near [-2, 2]
        assert preds.mean() > 100  # Not in normalized scale
        assert np.all(np.isfinite(preds))

    def test_predictions_in_original_scale_for_small_targets(self):
        """MLP with small targets also returns correct scale."""
        rng = np.random.RandomState(42)
        features = rng.randn(100, 20).astype(np.float32)
        targets = rng.randn(100).astype(np.float32)  # mean~0, std~1

        learner = MLPLearner(
            hidden_sizes=(32,), max_epochs=20, random_state=42
        )
        learner.train(features, targets)
        preds, _ = learner.predict(features)

        # Should be in reasonable range
        assert np.abs(preds.mean()) < 10
        assert np.all(np.isfinite(preds))

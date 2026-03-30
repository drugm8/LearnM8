import numpy as np
import pytest
from learnm8.learners.sklearn.random_forest import RandomForestLearner

pytestmark = pytest.mark.unit


class TestNaNHandlingIntegration:
    def test_continuous_features_nan_imputed_with_medians(self):
        rng = np.random.RandomState(42)
        features = rng.randn(50, 10)
        nan_mask = rng.random((50, 10)) < 0.15
        features[nan_mask] = np.nan
        targets = rng.randn(50)

        learner = RandomForestLearner(n_estimators=5, random_state=42)
        learner._feature_type = 'continuous'
        learner.train(features, targets)

        assert learner._feature_imputer is not None
        assert learner.is_trained

        pred_features = rng.randn(10, 10)
        pred_features[0, 0] = np.nan
        preds, _ = learner.predict(pred_features)
        assert np.all(np.isfinite(preds))

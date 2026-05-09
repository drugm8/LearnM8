"""Numeric equivalence test: tree learners on uint8 vs float32 features (T010).

A 1k-compound deterministic Morgan matrix passed to the same RF/XGB/DT
configuration MUST produce identical predictions whether the input dtype is
``uint8`` or ``float32``. Threshold ``rtol=1e-6, atol=1e-6`` ensures we catch
any silent dtype-induced cast asymmetry in preprocessing.

Tests must FAIL on current main because the uint8 input path doesn't exist —
``_preprocess_features`` calls ``np.isnan`` on uint8 which raises ``TypeError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from learnm8.learners.sklearn.decision_tree import DecisionTreeLearner
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner

TREE_LEARNER_FACTORIES = [
    pytest.param(lambda: RandomForestLearner(n_estimators=20, random_state=42), id='rf'),
    pytest.param(lambda: XGBoostLearner(n_estimators=20, random_state=42), id='xgb'),
    pytest.param(lambda: DecisionTreeLearner(max_depth=8, random_state=42), id='dt'),
]


def _make_binary_dataset(seed: int = 42, n_samples: int = 1000, n_features: int = 256):
    rng = np.random.default_rng(seed)
    X_uint8 = rng.integers(0, 2, size=(n_samples, n_features), dtype=np.uint8)
    coef = rng.standard_normal(n_features).astype(np.float32)
    y = (X_uint8.astype(np.float32) @ coef).astype(np.float32)
    return X_uint8, y


@pytest.mark.unit
@pytest.mark.parametrize('factory', TREE_LEARNER_FACTORIES)
def test_tree_uint8_vs_float32_predictions_match(factory):
    X_uint8, y = _make_binary_dataset()
    X_float = X_uint8.astype(np.float32)

    learner_u8 = factory()
    learner_u8._feature_type = 'binary'
    learner_u8.train(X_uint8.copy(), y)
    preds_u8, _ = learner_u8.predict(X_uint8.copy())

    learner_f32 = factory()
    learner_f32._feature_type = 'binary'
    learner_f32.train(X_float.copy(), y)
    preds_f32, _ = learner_f32.predict(X_float.copy())

    np.testing.assert_allclose(preds_u8, preds_f32, rtol=1e-6, atol=1e-6)

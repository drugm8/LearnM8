"""``Learner.preferred_feature_dtype()`` contract tests (T009, spec 017).

Tree learners route to ``'uint8'`` when their bound features are binary;
all non-tree learners (GP, MLP, MCDropout, Chemprop, Fastprop, GPyTorch,
GPU GP, cuML Ridge, FIL RF) MUST return ``'float32'`` regardless of feature
type so gradient/kernel/scaler paths stay numerically faithful.

Tests must FAIL on current main because the method does not yet exist.
"""

from __future__ import annotations

import pytest

from learnm8.learners.sklearn.decision_tree import DecisionTreeLearner
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.learners.sklearn.linear_regression import LinearRegressionLearner
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner

TREE_LEARNERS = [
    RandomForestLearner,
    XGBoostLearner,
    DecisionTreeLearner,
]

NON_TREE_LEARNERS = [
    GaussianProcessLearner,
    LinearRegressionLearner,
]


@pytest.mark.unit
@pytest.mark.parametrize('cls', TREE_LEARNERS)
def test_tree_learner_uint8_when_binary(cls):
    learner = cls()
    learner._feature_type = 'binary'
    assert learner.preferred_feature_dtype() == 'uint8'


@pytest.mark.unit
@pytest.mark.parametrize('cls', TREE_LEARNERS)
def test_tree_learner_float32_when_continuous(cls):
    learner = cls()
    learner._feature_type = 'continuous'
    assert learner.preferred_feature_dtype() == 'float32'


@pytest.mark.unit
@pytest.mark.parametrize('cls', NON_TREE_LEARNERS)
def test_non_tree_learner_float32(cls):
    learner = cls()
    learner._feature_type = 'binary'
    assert learner.preferred_feature_dtype() == 'float32'
    learner._feature_type = 'continuous'
    assert learner.preferred_feature_dtype() == 'float32'


@pytest.mark.unit
def test_torch_learners_always_float32():
    """MLP / MCDropout / Chemprop / Fastprop need float32 for gradients."""
    from learnm8.learners.torch.mc_dropout import MCDropoutLearner
    from learnm8.learners.torch.mlp import MLPLearner

    for cls in (MLPLearner, MCDropoutLearner):
        learner = cls()
        for ft in ('binary', 'continuous'):
            learner._feature_type = ft
            assert learner.preferred_feature_dtype() == 'float32', (
                f"{cls.__name__} with _feature_type={ft!r} should be float32"
            )


@pytest.mark.unit
def test_default_interface_returns_float32():
    """Custom Learner subclasses inherit the float32 default (REQ-14)."""
    import numpy as np

    from learnm8.core.interfaces import Learner

    class _Custom(Learner):
        def train(self, features, targets, smiles=None):
            pass

        def predict(self, features, smiles=None):
            return np.zeros(features.shape[0]), None

        def get_name(self) -> str:
            return 'custom'

    assert _Custom().preferred_feature_dtype() == 'float32'

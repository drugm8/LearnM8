"""Pre-trained model fixtures for LearnM8 tests.

Class-scoped fixtures providing trained models for read-only assertions.
Tests that mutate models (retrain, add/remove learners) MUST use function-scoped fixtures.
"""

import pytest
import numpy as np

from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
from learnm8.learners.ensemble.rf_ensemble import RFEnsemble
from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble


@pytest.fixture(scope="class")
def trained_rf(small_real_morgan_features, small_real_compounds):
    """Pre-trained RandomForest for read-only tests.

    WARNING: Shared across class. Do NOT call train(), add_learner(),
    or remove_learner(). Tests that mutate MUST use a function-scoped fixture.
    """
    learner = RandomForestLearner(n_estimators=10, random_state=42)
    features = small_real_morgan_features.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    learner.train(features, targets)
    return learner


@pytest.fixture(scope="class")
def trained_gp(small_real_morgan_features, small_real_compounds):
    """Pre-trained GaussianProcess for read-only tests.

    WARNING: Shared across class. Do NOT call train().
    """
    learner = GaussianProcessLearner(random_state=42)
    features = small_real_morgan_features.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    learner.train(features, targets)
    return learner


@pytest.fixture(scope="class")
def trained_xgb(small_real_morgan_features, small_real_compounds):
    """Pre-trained XGBoost for read-only tests.

    WARNING: Shared across class. Do NOT call train().
    """
    learner = XGBoostLearner(n_estimators=10, random_state=42)
    features = small_real_morgan_features.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    learner.train(features, targets)
    return learner


@pytest.fixture(scope="class")
def trained_rf_ensemble(small_real_morgan_features, small_real_compounds):
    """Pre-trained RFEnsemble for read-only tests.

    WARNING: Shared across class. Do NOT call train(), add_learner(),
    or remove_learner().
    """
    ensemble = RFEnsemble(n_estimators=10, random_state=42)
    features = small_real_morgan_features.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    ensemble.train(features, targets)
    return ensemble


@pytest.fixture(scope="class")
def trained_mixed_ensemble(small_real_morgan_features, small_real_compounds):
    """Pre-trained MixedEnsemble for read-only tests.

    WARNING: Shared across class. Do NOT call train(), add_learner(),
    or remove_learner().
    """
    ensemble = MixedEnsemble(random_state=42)
    features = small_real_morgan_features.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    ensemble.train(features, targets)
    return ensemble

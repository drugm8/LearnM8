"""Pre-trained model fixtures for LearnM8 tests.

Session-scoped fixtures providing trained models for read-only assertions.
Tests that mutate models (retrain, add/remove learners) MUST use function-scoped fixtures.
"""

import pytest

from learnm8.learners.ensemble.chemprop_ensemble import ChempropEnsemble
from learnm8.learners.ensemble.dt_ensemble import DTEnsemble
from learnm8.learners.ensemble.fastprop_ensemble import FastpropEnsemble
from learnm8.learners.ensemble.lr_ensemble import LREnsemble
from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
from learnm8.learners.ensemble.rf_ensemble import RFEnsemble
from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.learners.sklearn.random_forest import RandomForestLearner
from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner


@pytest.fixture(scope="session")
def trained_rf(_small_real_morgan_features_raw, small_real_compounds):
    """Pre-trained RandomForest for read-only tests.

    WARNING: Shared across session. Do NOT call train(), add_learner(),
    or remove_learner(). Tests that mutate MUST use a function-scoped fixture.
    """
    learner = RandomForestLearner(n_estimators=10, random_state=42)
    features = _small_real_morgan_features_raw.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    learner.train(features, targets)
    return learner


@pytest.fixture(scope="session")
def trained_gp(_small_real_morgan_features_raw, small_real_compounds):
    """Pre-trained GaussianProcess for read-only tests.

    WARNING: Shared across session. Do NOT call train().
    """
    learner = GaussianProcessLearner(random_state=42)
    features = _small_real_morgan_features_raw.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    learner.train(features, targets)
    return learner


@pytest.fixture(scope="session")
def trained_xgb(_small_real_morgan_features_raw, small_real_compounds):
    """Pre-trained XGBoost for read-only tests.

    WARNING: Shared across session. Do NOT call train().
    """
    learner = XGBoostLearner(n_estimators=10, random_state=42)
    features = _small_real_morgan_features_raw.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    learner.train(features, targets)
    return learner


@pytest.fixture(scope="session")
def trained_rf_ensemble(_small_real_morgan_features_raw, small_real_compounds):
    """Pre-trained RFEnsemble for read-only tests.

    WARNING: Shared across session. Do NOT call train(), add_learner(),
    or remove_learner().
    """
    ensemble = RFEnsemble(n_estimators=10, random_state=42)
    features = _small_real_morgan_features_raw.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    ensemble.train(features, targets)
    return ensemble


@pytest.fixture(scope="session")
def trained_mixed_ensemble(_small_real_morgan_features_raw, small_real_compounds):
    """Pre-trained MixedEnsemble for read-only tests.

    WARNING: Shared across session. Do NOT call train(), add_learner(),
    or remove_learner().
    """
    ensemble = MixedEnsemble(random_state=42)
    features = _small_real_morgan_features_raw.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    ensemble.train(features, targets)
    return ensemble


@pytest.fixture(scope="session")
def trained_lr_ensemble(_small_real_morgan_features_raw, small_real_compounds):
    """Pre-trained LREnsemble for read-only tests.

    WARNING: Shared across session. Do NOT call train(), add_learner(),
    or remove_learner().
    """
    ensemble = LREnsemble(regularization_strengths=[0.1, 1.0], random_states=[42, 123])
    features = _small_real_morgan_features_raw.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    ensemble.train(features, targets)
    return ensemble


@pytest.fixture(scope="session")
def trained_dt_ensemble(_small_real_morgan_features_raw, small_real_compounds):
    """Pre-trained DTEnsemble for read-only tests.

    WARNING: Shared across session. Do NOT call train(), add_learner(),
    or remove_learner().
    """
    ensemble = DTEnsemble(max_depths=[5, 10], random_states=[42, 123])
    features = _small_real_morgan_features_raw.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    ensemble.train(features, targets)
    return ensemble


@pytest.fixture(scope="session")
def trained_xgb_ensemble(_small_real_morgan_features_raw, small_real_compounds):
    """Pre-trained XGBEnsemble for read-only tests.

    WARNING: Shared across session. Do NOT call train(), add_learner(),
    or remove_learner().
    """
    ensemble = XGBEnsemble(learning_rates=[0.05, 0.1], random_states=[42, 123])
    features = _small_real_morgan_features_raw.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    ensemble.train(features, targets)
    return ensemble


@pytest.fixture(scope="session")
def trained_chemprop_ensemble(small_real_compounds):
    """Pre-trained ChempropEnsemble for read-only tests.

    Trains a 3-model ensemble on small_real_compounds once per session,
    eliminating ~5s of per-test training in read-only consumers.

    WARNING: Shared across session. Do NOT call train(), add_learner(),
    or remove_learner(). Tests that need an untrained ensemble or that
    verify is_trained transitions MUST construct their own instance.
    """
    ensemble = ChempropEnsemble(message_hidden_dim=300, depth=2, max_epochs=3)
    smiles = small_real_compounds['SMILES'].to_list()
    targets = small_real_compounds['Activity'].to_numpy()
    ensemble.train(features=None, targets=targets, smiles=smiles)
    return ensemble


@pytest.fixture(scope="session")
def trained_fastprop_ensemble(_small_real_morgan_features_raw, small_real_compounds):
    """Pre-trained FastpropEnsemble for read-only tests.

    Same scoping rules as ``trained_chemprop_ensemble``.
    """
    ensemble = FastpropEnsemble(fnn_layers=1, hidden_size=64, max_epochs=3, device='cpu')
    features = _small_real_morgan_features_raw.copy()
    targets = small_real_compounds['Activity'].to_numpy()
    ensemble.train(features, targets)
    return ensemble

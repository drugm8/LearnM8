import numpy as np
import pytest

from learnm8.core.interfaces import Learner
from learnm8.exceptions import LearnerError
from learnm8.learners.ensemble.ensemble import EnsembleLearner


class MockLearner(Learner):
    def __init__(self, name='mock', fail_on_train=False, fail_on_predict=False):
        self.name = name
        self.fail_on_train = fail_on_train
        self.fail_on_predict = fail_on_predict
        self.is_trained = False

    def train(self, features, targets, smiles=None):
        if self.fail_on_train:
            raise LearnerError(f'{self.name} training failed')
        self.is_trained = True

    def predict(self, features, smiles=None):
        if self.fail_on_predict:
            raise LearnerError(f'{self.name} prediction failed')
        return np.random.randn(len(features)), None

    def supports_uncertainty(self):
        return False

    def get_name(self):
        return self.name


@pytest.fixture
def features():
    return np.random.randn(20, 10).astype(np.float32)


@pytest.fixture
def targets():
    return np.random.randn(20).astype(np.float32)


@pytest.mark.unit
def test_train_raises_on_any_member_failure(features, targets):
    members = [
        MockLearner(name='member_0'),
        MockLearner(name='member_1', fail_on_train=True),
        MockLearner(name='member_2'),
    ]
    ensemble = EnsembleLearner(members)
    with pytest.raises(LearnerError):
        ensemble.train(features, targets)


@pytest.mark.unit
def test_predict_raises_on_any_member_failure(features, targets):
    members = [
        MockLearner(name='member_0'),
        MockLearner(name='member_1', fail_on_predict=True),
        MockLearner(name='member_2'),
    ]
    ensemble = EnsembleLearner(members)
    ensemble.train(features, targets)
    ensemble.is_trained = True
    for m in ensemble.learners:
        m.is_trained = True
    with pytest.raises(LearnerError):
        ensemble.predict(features)


@pytest.mark.unit
def test_get_individual_predictions_raises_on_member_failure(features, targets):
    members = [
        MockLearner(name='member_0'),
        MockLearner(name='member_1', fail_on_predict=True),
        MockLearner(name='member_2'),
    ]
    ensemble = EnsembleLearner(members)
    ensemble.train(features, targets)
    ensemble.is_trained = True
    for m in ensemble.learners:
        m.is_trained = True
    with pytest.raises(LearnerError):
        ensemble.get_individual_predictions(features)


@pytest.mark.unit
def test_all_members_succeed_returns_normal_results(features, targets):
    members = [
        MockLearner(name='member_0'),
        MockLearner(name='member_1'),
        MockLearner(name='member_2'),
    ]
    ensemble = EnsembleLearner(members)
    ensemble.train(features, targets)
    preds, uncertainty = ensemble.predict(features)
    assert preds.shape == (len(features),)
    assert uncertainty is not None
    assert uncertainty.shape == (len(features),)


@pytest.mark.unit
def test_fail_fast_error_identifies_failed_member(features, targets):
    members = [
        MockLearner(name='member_0'),
        MockLearner(name='failing_member', fail_on_train=True),
        MockLearner(name='member_2'),
    ]
    ensemble = EnsembleLearner(members)
    with pytest.raises(LearnerError, match='failing_member'):
        ensemble.train(features, targets)

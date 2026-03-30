import numpy as np
import pytest

from learnm8.exceptions import LearnerError
from learnm8.learners.base import _validate_predict_inputs, _validate_train_inputs

pytestmark = pytest.mark.unit


def test_validate_train_inputs_rejects_empty_features():
    features = np.empty((0, 10), dtype=np.float32)
    targets = np.empty((0,), dtype=np.float32)
    with pytest.raises(LearnerError):
        _validate_train_inputs(features, targets, 'test_learner')


def test_validate_train_inputs_rejects_mismatched_shapes():
    features = np.ones((10, 5), dtype=np.float32)
    targets = np.ones((8,), dtype=np.float32)
    with pytest.raises(LearnerError):
        _validate_train_inputs(features, targets, 'test_learner')


def test_validate_train_inputs_rejects_wrong_ndim():
    features_1d = np.ones((10,), dtype=np.float32)
    targets = np.ones((10,), dtype=np.float32)
    with pytest.raises(LearnerError):
        _validate_train_inputs(features_1d, targets, 'test_learner')

    features_3d = np.ones((10, 5, 2), dtype=np.float32)
    with pytest.raises(LearnerError):
        _validate_train_inputs(features_3d, targets, 'test_learner')


def test_validate_predict_inputs_rejects_untrained_learner():
    with pytest.raises(LearnerError):
        _validate_predict_inputs(False, 'test_learner')


def test_validate_train_inputs_accepts_valid_inputs():
    features = np.ones((20, 10), dtype=np.float32)
    targets = np.ones((20,), dtype=np.float32)
    _validate_train_inputs(features, targets, 'test_learner')

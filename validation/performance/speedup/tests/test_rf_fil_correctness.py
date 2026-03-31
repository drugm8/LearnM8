from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('cuml', reason='RAPIDS cuML not available')

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module')
def trained_rf_fil():
    from learnm8.learners.gpu.rf_fil import RfFilLearner
    rng = np.random.default_rng(42)
    X = rng.random((1000, 50)).astype(np.float32)
    y = rng.random(1000).astype(np.float32)
    learner = RfFilLearner(n_estimators=10, random_state=42)
    learner.train(X, y)
    return learner, X


def test_finite_predictions(trained_rf_fil):
    learner, X = trained_rf_fil
    preds, _unc = learner.predict(X[:100])
    assert np.all(np.isfinite(preds)), 'VR-001: predictions must be finite'


def test_predictions_shape(trained_rf_fil):
    learner, X = trained_rf_fil
    preds, _unc = learner.predict(X[:100])
    assert preds.shape == (100,), 'VR-002: predictions shape must match n_samples'


def test_finite_non_negative_uncertainty(trained_rf_fil):
    learner, X = trained_rf_fil
    _preds, unc = learner.predict(X[:100])
    assert unc is not None, 'VR-003: uncertainty must not be None'
    assert np.all(np.isfinite(unc)), 'VR-003: uncertainty must be finite'
    assert np.all(unc >= 0.0), 'VR-003: uncertainty must be non-negative'


def test_uncertainty_shape(trained_rf_fil):
    learner, X = trained_rf_fil
    preds, unc = learner.predict(X[:100])
    assert unc.shape == preds.shape, 'uncertainty shape must match predictions shape'


def test_prediction_determinism(trained_rf_fil):
    learner, X = trained_rf_fil
    preds1, _ = learner.predict(X[:50])
    preds2, _ = learner.predict(X[:50])
    np.testing.assert_array_equal(preds1, preds2, err_msg='VR-004: predictions must be deterministic')


def test_uncertainty_determinism(trained_rf_fil):
    learner, X = trained_rf_fil
    _, unc1 = learner.predict(X[:50])
    _, unc2 = learner.predict(X[:50])
    np.testing.assert_array_equal(unc1, unc2, err_msg='VR-007: uncertainty must be deterministic')


def test_supports_uncertainty():
    from learnm8.learners.gpu.rf_fil import RfFilLearner
    learner = RfFilLearner(n_estimators=10)
    assert learner.supports_uncertainty() is True


def test_get_name():
    from learnm8.learners.gpu.rf_fil import RfFilLearner
    learner = RfFilLearner(n_estimators=10)
    assert learner.get_name() == 'rf_fil'


def test_predict_before_train_raises():
    from learnm8.exceptions import LearnerError
    from learnm8.learners.gpu.rf_fil import RfFilLearner
    learner = RfFilLearner(n_estimators=10)
    X = np.random.rand(10, 5).astype(np.float32)
    with pytest.raises(LearnerError):
        learner.predict(X)


def test_uncertainty_nonzero_for_diverse_data():
    from learnm8.learners.gpu.rf_fil import RfFilLearner
    rng = np.random.default_rng(0)
    X_train = rng.random((500, 30)).astype(np.float32)
    y_train = (X_train[:, 0] * 3.0 + rng.normal(0, 0.1, 500)).astype(np.float32)
    X_test = rng.random((100, 30)).astype(np.float32)
    learner = RfFilLearner(n_estimators=20, random_state=0)
    learner.train(X_train, y_train)
    _, unc = learner.predict(X_test)
    assert np.any(unc > 0.0), 'uncertainty should be non-zero for diverse data'

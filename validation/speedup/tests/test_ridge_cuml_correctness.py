from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip('cuml', reason='RAPIDS cuML not available')

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module')
def trained_ridge_cuml():
    from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
    rng = np.random.default_rng(42)
    X = rng.random((1000, 50))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 1000)
    learner = RidgeCumlLearner(alpha=0.1, random_state=42)
    learner.train(X, y)
    return learner, X, y


def test_finite_predictions(trained_ridge_cuml):
    learner, X, _ = trained_ridge_cuml
    preds, unc = learner.predict(X[:100])
    assert np.all(np.isfinite(preds)), 'VR-001: predictions must be finite'


def test_predictions_shape(trained_ridge_cuml):
    learner, X, _ = trained_ridge_cuml
    preds, unc = learner.predict(X[:100])
    assert preds.shape == (100,), 'VR-002: predictions shape must match n_samples'


def test_finite_non_negative_uncertainty(trained_ridge_cuml):
    learner, X, _ = trained_ridge_cuml
    preds, unc = learner.predict(X[:100])
    assert unc is not None, 'VR-003: uncertainty must not be None'
    assert np.all(np.isfinite(unc)), 'VR-003: uncertainty must be finite'
    assert np.all(unc >= 0.0), 'VR-003: uncertainty must be non-negative'


def test_uncertainty_shape(trained_ridge_cuml):
    learner, X, _ = trained_ridge_cuml
    preds, unc = learner.predict(X[:100])
    assert unc.shape == preds.shape, 'uncertainty shape must match predictions shape'


def test_prediction_determinism(trained_ridge_cuml):
    learner, X, _ = trained_ridge_cuml
    preds1, _ = learner.predict(X[:50])
    preds2, _ = learner.predict(X[:50])
    np.testing.assert_array_equal(preds1, preds2, err_msg='VR-004: predictions must be deterministic')


def test_uncertainty_determinism(trained_ridge_cuml):
    learner, X, _ = trained_ridge_cuml
    _, unc1 = learner.predict(X[:50])
    _, unc2 = learner.predict(X[:50])
    np.testing.assert_array_equal(unc1, unc2, err_msg='VR-007: uncertainty must be deterministic')


def test_supports_uncertainty():
    from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
    learner = RidgeCumlLearner(alpha=0.1)
    assert learner.supports_uncertainty() is True


def test_get_name():
    from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
    learner = RidgeCumlLearner(alpha=0.1)
    assert learner.get_name() == 'ridge_cuml'


def test_none_alpha_raises():
    from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
    from learnm8.exceptions import LearnerError
    with pytest.raises(LearnerError, match='alpha'):
        RidgeCumlLearner(alpha=None)


def test_predict_before_train_raises():
    from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
    from learnm8.exceptions import LearnerError
    learner = RidgeCumlLearner(alpha=0.1)
    X = np.random.rand(10, 5)
    with pytest.raises(LearnerError):
        learner.predict(X)


def test_leverage_uncertainty_increases_for_ood():
    from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
    rng = np.random.default_rng(7)
    X_train = rng.random((500, 20))
    y_train = X_train[:, 0] * 5.0 + rng.normal(0, 0.1, 500)
    X_in = rng.random((50, 20))
    X_ood = rng.uniform(10.0, 20.0, (50, 20))
    learner = RidgeCumlLearner(alpha=1.0, random_state=7)
    learner.train(X_train, y_train)
    _, unc_in = learner.predict(X_in)
    _, unc_ood = learner.predict(X_ood)
    assert np.median(unc_ood) > np.median(unc_in), \
        'leverage uncertainty should be higher for out-of-distribution data'


def test_train_sets_is_trained():
    from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
    rng = np.random.default_rng(1)
    X = rng.random((200, 10))
    y = rng.random(200)
    learner = RidgeCumlLearner(alpha=0.1)
    assert learner.is_trained is False
    learner.train(X, y)
    assert learner.is_trained is True


def test_gpu_leverage_computed_during_training():
    from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
    rng = np.random.default_rng(99)
    X = rng.random((500, 30))
    y = X[:, 0] * 3.0 + rng.normal(0, 0.1, 500)
    learner = RidgeCumlLearner(alpha=0.1, random_state=99)
    learner.train(X, y)
    assert learner._gram_chol_gpu is not None, \
        'GPU Cholesky factor must be computed during training when cuML is available'


def test_gpu_cpu_leverage_numerical_agreement(trained_ridge_cuml):
    learner, X, _ = trained_ridge_cuml
    X_test = X[:200]

    t0 = time.perf_counter_ns()
    _, unc_gpu = learner.predict(X_test)
    t_gpu = time.perf_counter_ns() - t0

    gram_chol_gpu_backup = learner._gram_chol_gpu
    learner._gram_chol_gpu = None
    try:
        t0 = time.perf_counter_ns()
        _, unc_cpu = learner.predict(X_test)
        t_cpu = time.perf_counter_ns() - t0
    finally:
        learner._gram_chol_gpu = gram_chol_gpu_backup

    print(f'\n  GPU leverage time: {t_gpu / 1e6:.2f} ms')
    print(f'  CPU leverage time: {t_cpu / 1e6:.2f} ms')

    np.testing.assert_allclose(
        unc_gpu, unc_cpu, atol=1e-10,
        err_msg='GPU and CPU leverage uncertainty must agree within atol=1e-10',
    )

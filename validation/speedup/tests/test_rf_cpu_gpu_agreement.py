from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import spearmanr

pytest.importorskip('cuml', reason='RAPIDS cuML not available')

try:
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA GPU not available', allow_module_level=True)
except ImportError:
    pytest.skip('torch not available', allow_module_level=True)

pytestmark = pytest.mark.gpu

_FAIL_THRESHOLD = 1e-3
_WARN_THRESHOLD = 1e-5
_MIN_SPEARMAN = 0.99


@pytest.fixture(scope='module')
def cpu_gpu_rf_pair():
    from learnm8.learners.sklearn.random_forest import RandomForestLearner
    from learnm8.learners.gpu.rf_fil import RfFilLearner

    rng = np.random.default_rng(42)
    X = rng.random((2000, 100)).astype(np.float32)
    y = (X[:, :5] @ rng.random(5) + rng.normal(0, 0.05, 2000)).astype(np.float32)

    n_estimators = 50
    random_state = 42

    cpu_learner = RandomForestLearner(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    cpu_learner.train(X[:1000], y[:1000])

    gpu_learner = RfFilLearner(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    gpu_learner.train(X[:1000].astype(np.float32), y[:1000].astype(np.float32))

    X_test = X[1000:]
    return cpu_learner, gpu_learner, X_test


def test_vr008_max_abs_diff_below_fail_threshold(cpu_gpu_rf_pair):
    cpu_learner, gpu_learner, X_test = cpu_gpu_rf_pair
    cpu_preds, _ = cpu_learner.predict(X_test)
    gpu_preds, _ = gpu_learner.predict(X_test)
    max_diff = float(np.max(np.abs(cpu_preds - gpu_preds)))
    assert max_diff < _FAIL_THRESHOLD, (
        f'VR-008 fail threshold exceeded: max abs diff = {max_diff:.2e} >= {_FAIL_THRESHOLD:.2e}'
    )


def test_vr008_warn_threshold(cpu_gpu_rf_pair):
    cpu_learner, gpu_learner, X_test = cpu_gpu_rf_pair
    cpu_preds, _ = cpu_learner.predict(X_test)
    gpu_preds, _ = gpu_learner.predict(X_test)
    max_diff = float(np.max(np.abs(cpu_preds - gpu_preds)))
    if max_diff >= _WARN_THRESHOLD:
        warnings.warn(
            f'VR-008 warn threshold exceeded: max abs diff = {max_diff:.2e} >= {_WARN_THRESHOLD:.2e}. '
            f'float32 training introduces minor numerical divergence.',
            stacklevel=2,
        )


def test_vr008_spearman_correlation(cpu_gpu_rf_pair):
    cpu_learner, gpu_learner, X_test = cpu_gpu_rf_pair
    cpu_preds, _ = cpu_learner.predict(X_test)
    gpu_preds, _ = gpu_learner.predict(X_test)
    rho = float(spearmanr(cpu_preds, gpu_preds).statistic)
    assert np.isfinite(rho), 'Spearman rho must be finite'
    assert rho > _MIN_SPEARMAN, (
        f'VR-008 Spearman rho too low: {rho:.4f} < {_MIN_SPEARMAN}'
    )


def test_gpu_predictions_finite(cpu_gpu_rf_pair):
    _, gpu_learner, X_test = cpu_gpu_rf_pair
    gpu_preds, _ = gpu_learner.predict(X_test)
    assert np.all(np.isfinite(gpu_preds)), 'GPU predictions must be finite'


def test_gpu_uncertainty_finite_non_negative(cpu_gpu_rf_pair):
    _, gpu_learner, X_test = cpu_gpu_rf_pair
    _, unc = gpu_learner.predict(X_test)
    assert unc is not None
    assert np.all(np.isfinite(unc)), 'GPU uncertainty must be finite'
    assert np.all(unc >= 0.0), 'GPU uncertainty must be non-negative'


def test_mean_abs_diff_reasonable(cpu_gpu_rf_pair):
    cpu_learner, gpu_learner, X_test = cpu_gpu_rf_pair
    cpu_preds, _ = cpu_learner.predict(X_test)
    gpu_preds, _ = gpu_learner.predict(X_test)
    mean_diff = float(np.mean(np.abs(cpu_preds - gpu_preds)))
    assert mean_diff < _FAIL_THRESHOLD, (
        f'Mean abs diff between CPU and GPU RF too large: {mean_diff:.2e}'
    )


def test_float32_training_consistency():
    from learnm8.learners.gpu.rf_fil import RfFilLearner
    rng = np.random.default_rng(99)
    X = rng.random((500, 20)).astype(np.float32)
    y = rng.random(500).astype(np.float32)
    learner = RfFilLearner(n_estimators=10, random_state=42)
    learner.train(X, y)
    preds1, _ = learner.predict(X[:100])
    preds2, _ = learner.predict(X[:100])
    np.testing.assert_array_equal(preds1, preds2, err_msg='FIL RF must produce identical results on repeated predict')

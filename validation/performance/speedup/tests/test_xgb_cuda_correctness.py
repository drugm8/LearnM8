from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA GPU not available', allow_module_level=True)
except ImportError:
    pytest.skip('torch not available', allow_module_level=True)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module')
def trained_xgb_cuda():
    from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
    rng = np.random.default_rng(42)
    X = rng.random((1000, 50)).astype(np.float32)
    y = rng.random(1000)
    learner = XGBoostLearner(n_estimators=20, random_state=42, device='cuda')
    learner.train(X, y)
    return learner, X


@pytest.fixture(scope='module')
def trained_xgb_cpu():
    from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
    rng = np.random.default_rng(42)
    X = rng.random((1000, 50)).astype(np.float32)
    y = rng.random(1000)
    learner = XGBoostLearner(n_estimators=20, random_state=42, device='cpu')
    learner.train(X, y)
    return learner, X


def test_finite_predictions(trained_xgb_cuda):
    learner, X = trained_xgb_cuda
    preds, _unc = learner.predict(X[:100])
    assert np.all(np.isfinite(preds)), 'VR-001: predictions must be finite'


def test_predictions_shape(trained_xgb_cuda):
    learner, X = trained_xgb_cuda
    preds, _unc = learner.predict(X[:100])
    assert preds.shape == (100,), 'VR-002: predictions shape must match n_samples'


def test_no_uncertainty(trained_xgb_cuda):
    learner, X = trained_xgb_cuda
    _, unc = learner.predict(X[:100])
    assert unc is None, 'XGBoostLearner should return None uncertainty'


def test_prediction_determinism(trained_xgb_cuda):
    learner, X = trained_xgb_cuda
    preds1, _ = learner.predict(X[:50])
    preds2, _ = learner.predict(X[:50])
    np.testing.assert_array_equal(preds1, preds2, err_msg='VR-004: predictions must be deterministic')


def test_get_device():
    from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
    learner = XGBoostLearner(device='cuda')
    assert learner.device == 'cuda'


def test_supports_uncertainty_false():
    from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
    learner = XGBoostLearner(device='cuda')
    assert learner.supports_uncertainty() is False


def test_predict_before_train_raises():
    from learnm8.exceptions import LearnerError
    from learnm8.learners.sklearn.xgboost_learner import XGBoostLearner
    learner = XGBoostLearner(n_estimators=5, device='cuda')
    X = np.random.rand(10, 5).astype(np.float32)
    with pytest.raises(LearnerError):
        learner.predict(X)


def test_cuda_cpu_ranking_agreement(trained_xgb_cuda, trained_xgb_cpu):
    from scipy.stats import spearmanr
    cuda_learner, X = trained_xgb_cuda
    cpu_learner, _ = trained_xgb_cpu
    cuda_preds, _ = cuda_learner.predict(X[:200])
    cpu_preds, _ = cpu_learner.predict(X[:200])
    rho = float(spearmanr(cuda_preds, cpu_preds).statistic)
    assert np.isfinite(rho), 'Spearman rho between CPU and CUDA XGB must be finite'
    assert rho > 0.90, f'CPU vs CUDA XGB ranking agreement too low: rho={rho:.4f}'

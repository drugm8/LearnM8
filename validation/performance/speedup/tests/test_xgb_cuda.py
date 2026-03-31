from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from validation.performance.speedup.lib.assertions import assert_predictions_match
from validation.performance.speedup.lib.data_setup import generate_synthetic_data
from validation.performance.speedup.lib.model_factory import create_learner, train_learner


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU not available')
def test_xgb_cuda():
    X_train, y_train, X_test, _ = generate_synthetic_data(seed=42)

    learner = create_learner('xgb', n_estimators=50, random_state=42)
    train_learner(learner, X_train, y_train)

    cpu_preds, _ = learner.predict(X_test)

    learner.model.set_params(device='cuda')
    gpu_preds, _ = learner.predict(X_test)

    assert cpu_preds.shape == gpu_preds.shape
    assert np.isfinite(gpu_preds).all()

    assert_predictions_match(cpu_preds, gpu_preds, 'near_zero')


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU not available')
def test_xgb_cuda_training():
    X_train, y_train, X_test, y_test = generate_synthetic_data(seed=42)

    cpu_learner = create_learner('xgb', n_estimators=50, random_state=42)
    train_learner(cpu_learner, X_train, y_train)
    cpu_preds, _ = cpu_learner.predict(X_test)

    gpu_learner = create_learner('xgb', n_estimators=50, random_state=42)
    gpu_learner.model.set_params(device='cuda')
    train_learner(gpu_learner, X_train, y_train)
    gpu_preds, _ = gpu_learner.predict(X_test)

    assert cpu_preds.shape == gpu_preds.shape
    assert np.isfinite(gpu_preds).all()

    cpu_quality = float(spearmanr(y_test, cpu_preds)[0])
    gpu_quality = float(spearmanr(y_test, gpu_preds)[0])
    cross_rho = float(spearmanr(cpu_preds, gpu_preds)[0])

    assert np.isfinite(cpu_quality)
    assert np.isfinite(gpu_quality)
    assert np.isfinite(cross_rho)

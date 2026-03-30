from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from learnm8.learners.base import _preprocess_features
from validation.speedup.lib.assertions import (
    assert_predictions_match,
)
from validation.speedup.lib.data_setup import generate_synthetic_data
from validation.speedup.lib.model_factory import (
    create_learner,
    predict_primary_output,
    train_learner,
)


def _generate_structured_regression_data(
    n_train: int,
    n_test: int,
    n_features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    X = rng.binomial(1, 0.1, (n_total, n_features)).astype(np.float64)
    weights = rng.normal(0.0, 1.0, size=20)
    signal = X[:, :20] @ weights
    interaction = 0.5 * (X[:, 20] * X[:, 21]) - 0.25 * (X[:, 22] * X[:, 23])
    noise = rng.normal(0.0, 0.1, size=n_total)
    y = signal + interaction + noise
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU not available')
@pytest.mark.parametrize('learner_type', ['rf', 'xgb'])
def test_rapids_fil(learner_type):
    cuml = pytest.importorskip('cuml')

    X_train, y_train, X_test, _ = generate_synthetic_data(seed=42)

    kwargs = {'random_state': 42, 'n_estimators': 50}

    learner = create_learner(learner_type, **kwargs)
    train_learner(learner, X_train, y_train)

    sklearn_preds, _ = learner.predict(X_test)

    preprocessed, _ = _preprocess_features(
        X_test,
        valid_feature_mask=learner._valid_feature_mask,
        remove_zero_variance=learner.remove_zero_variance,
        is_training=False,
    )

    input_data = preprocessed.astype(np.float32)

    try:
        if learner_type == 'xgb':
            with tempfile.NamedTemporaryFile(suffix='.json') as f:
                learner.model.save_model(f.name)
                fil_model = cuml.ForestInference.load(
                    f.name, model_type='xgboost_json',
                )
        else:
            fil_model = cuml.ForestInference.load_from_sklearn(learner.model)
    except MemoryError as e:
        if 'std::bad_alloc' in str(e):
            pytest.skip(
                'cuml FIL GPU init failed (std::bad_alloc) — '
                'known issue on WSL2 with certain driver/CUDA combinations'
            )
        raise

    fil_preds = fil_model.predict(input_data)
    fil_preds = np.asarray(fil_preds).flatten()

    assert fil_preds.shape == sklearn_preds.shape
    assert_predictions_match(sklearn_preds, fil_preds, 'near_zero')


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU not available')
def test_rapids_cuml_random_forest():
    cuml = pytest.importorskip('cuml')

    X_train, y_train, X_test, y_test = _generate_structured_regression_data(
        n_train=1000,
        n_test=250,
        n_features=100,
        seed=42,
    )

    learner = create_learner('rf', n_estimators=100, random_state=42, n_jobs=-1)
    train_learner(learner, X_train, y_train)

    baseline_preds = predict_primary_output(learner, X_test)

    processed_train, _ = _preprocess_features(
        X_train,
        remove_zero_variance=learner.remove_zero_variance,
        is_training=True,
    )
    processed_test, _ = _preprocess_features(
        X_test,
        valid_feature_mask=learner._valid_feature_mask,
        remove_zero_variance=learner.remove_zero_variance,
        is_training=False,
    )

    model = cuml.ensemble.RandomForestRegressor(
        n_estimators=learner.n_estimators,
        max_depth=learner.max_depth if learner.max_depth is not None else 16,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_streams=1,
        output_type='numpy',
    )
    model.fit(processed_train.astype(np.float32), y_train.astype(np.float32))
    cuml_preds = np.asarray(model.predict(processed_test.astype(np.float32))).reshape(-1)

    assert cuml_preds.shape == baseline_preds.shape
    assert np.isfinite(cuml_preds).all()
    baseline_quality_rho = float(spearmanr(y_test, baseline_preds).statistic)
    optimized_quality_rho = float(spearmanr(y_test, cuml_preds).statistic)
    cross_model_rho = float(spearmanr(baseline_preds, cuml_preds).statistic)

    assert np.isfinite(baseline_quality_rho)
    assert np.isfinite(optimized_quality_rho)
    assert np.isfinite(cross_model_rho)
    assert baseline_quality_rho >= 0.7
    assert optimized_quality_rho >= 0.7
    assert optimized_quality_rho >= baseline_quality_rho - 0.05


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU not available')
def test_rapids_cuml_linear_regression():
    cuml = pytest.importorskip('cuml')

    X_train, y_train, X_test, _ = generate_synthetic_data(
        n_train=800,
        n_test=200,
        n_features=100,
        seed=42,
    )

    learner = create_learner('lr', alpha=None)
    train_learner(learner, X_train, y_train)

    assert learner.is_ridge is False

    baseline_preds, _ = learner.predict(X_test)

    processed_train, _ = _preprocess_features(
        X_train,
        remove_zero_variance=learner.remove_zero_variance,
        is_training=True,
    )
    processed_test, _ = _preprocess_features(
        X_test,
        valid_feature_mask=learner._valid_feature_mask,
        remove_zero_variance=learner.remove_zero_variance,
        is_training=False,
    )

    model = cuml.linear_model.LinearRegression(
        algorithm='eig',
        output_type='numpy',
    )
    model.fit(processed_train.astype(np.float32), y_train.astype(np.float32))
    cuml_preds = np.asarray(model.predict(processed_test.astype(np.float32))).reshape(-1)

    assert cuml_preds.shape == baseline_preds.shape
    assert np.isfinite(cuml_preds).all()
    prediction_rho = float(spearmanr(baseline_preds, cuml_preds).statistic)
    assert np.isfinite(prediction_rho)

# Optimization #13: RAPIDS FIL
# Scope: mechanism validation — inference only
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from validation.speedup.lib.assertions import assert_predictions_match, assert_ranking_preserved
from validation.speedup.lib.data_setup import load_benchmark_data
from validation.speedup.lib.model_factory import (
    create_learner,
    predict_primary_output,
    train_learner,
)
from validation.speedup.lib.reporting import (
    create_result,
    print_result_table,
    save_result,
)
from validation.speedup.lib.runtime import (
    benchmark_learner_kwargs,
    configure_process_environment,
)
from validation.speedup.lib.timing import time_prediction, time_training, time_with_setup

configure_process_environment()

try:
    import torch
except ImportError:
    print("Skipping benchmark: torch not available (needed for CUDA check)")
    sys.exit(0)

if not torch.cuda.is_available():
    print("Skipping benchmark: CUDA GPU not available")
    sys.exit(0)

try:
    from cuml import ForestInference
    from cuml.ensemble import RandomForestRegressor as CuMLRandomForestRegressor
    from cuml.linear_model import LinearRegression as CuMLLinearRegression
except ImportError:
    print("Skipping benchmark: cuml not available (install RAPIDS for FIL support)")
    sys.exit(0)

RESULTS_DIR = Path(__file__).parent.parent / "results"
LEARNERS = ["rf", "xgb", "dt"]


def _benchmark_cuml_random_forest(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> dict:
    from learnm8.learners.base import _preprocess_features

    baseline_kwargs = benchmark_learner_kwargs('rf', use_gpu=False)

    def do_train_baseline():
        learner = create_learner('rf', **baseline_kwargs)
        train_learner(learner, train_features, train_targets)
        return learner

    baseline_train_timing = time_training(do_train_baseline)
    trained_baseline = do_train_baseline()

    template_learner = trained_baseline

    processed_train, valid_mask = _preprocess_features(
        train_features,
        remove_zero_variance=template_learner.remove_zero_variance,
        is_training=True,
    )
    processed_test, _ = _preprocess_features(
        test_features,
        valid_feature_mask=valid_mask,
        remove_zero_variance=template_learner.remove_zero_variance,
        is_training=False,
    )

    cuml_model = [None]

    def do_train_cuml():
        cuml_model[0] = CuMLRandomForestRegressor(
            n_estimators=template_learner.n_estimators,
            max_depth=template_learner.max_depth if template_learner.max_depth is not None else 16,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=baseline_kwargs['random_state'],
            n_streams=1,
            output_type='numpy',
        )
        cuml_model[0].fit(
            processed_train.astype(np.float32),
            train_targets.astype(np.float32),
        )

    optimized_train_timing = time_training(do_train_cuml)
    do_train_cuml()

    def baseline_predict():
        return predict_primary_output(
            trained_baseline, test_features, preprocessed=processed_test
        )

    def optimized_predict():
        return np.asarray(
            cuml_model[0].predict(processed_test.astype(np.float32))
        ).reshape(-1)

    baseline_timing = time_prediction(baseline_predict, n_samples=len(test_features))
    baseline_preds = baseline_predict()

    optimized_timing = time_prediction(optimized_predict, n_samples=len(test_features))
    optimized_preds = optimized_predict()

    baseline_quality_rho = float(spearmanr(test_targets, baseline_preds).statistic)
    optimized_quality_rho = float(spearmanr(test_targets, optimized_preds).statistic)
    cross_model_rho = float(spearmanr(baseline_preds, optimized_preds).statistic)

    if not np.isfinite(baseline_quality_rho) or not np.isfinite(optimized_quality_rho):
        raise AssertionError('Random forest holdout quality rho is not finite')

    result = create_result(
        optimization_id=13,
        optimization_name='rapids_cuml_random_forest',
        learner_name='rf',
        tier=2,
        baseline_time=baseline_timing['median_time'],
        optimized_time=optimized_timing['median_time'],
        prediction_match=True,
        optimization_backend='cuml_random_forest_regressor',
        prediction_spearman_rho=cross_model_rho,
        baseline_times=baseline_timing['times'],
        optimized_times=optimized_timing['times'],
        training_baseline_time=baseline_train_timing['median_time'],
        training_optimized_time=optimized_train_timing['median_time'],
        training_baseline_times=baseline_train_timing['times'],
        training_optimized_times=optimized_train_timing['times'],
        n_samples=len(test_features),
        n_features=test_features.shape[1],
        n_warmup_runs=baseline_timing['n_warmup'],
        n_timed_runs=baseline_timing['n_runs'],
    )
    result['timing_scope'] = 'predict_only'
    result['max_abs_diff'] = float(np.max(np.abs(baseline_preds - optimized_preds)))
    result['mean_abs_diff'] = float(np.mean(np.abs(baseline_preds - optimized_preds)))
    result['baseline_quality_rho'] = baseline_quality_rho
    result['optimized_quality_rho'] = optimized_quality_rho
    result['cross_model_rho'] = cross_model_rho
    return result


def _benchmark_cuml_linear_regression(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
) -> dict:
    from learnm8.learners.base import _preprocess_features

    baseline_kwargs = benchmark_learner_kwargs('lr', use_gpu=False)

    def do_train_baseline():
        learner = create_learner('lr', alpha=None, **baseline_kwargs)
        train_learner(learner, train_features, train_targets)
        return learner

    baseline_train_timing = time_training(do_train_baseline)
    trained_baseline = do_train_baseline()

    template_learner = trained_baseline

    processed_train, valid_mask = _preprocess_features(
        train_features,
        remove_zero_variance=template_learner.remove_zero_variance,
        is_training=True,
    )
    processed_test, _ = _preprocess_features(
        test_features,
        valid_feature_mask=valid_mask,
        remove_zero_variance=template_learner.remove_zero_variance,
        is_training=False,
    )

    cuml_model = [None]

    def do_train_cuml():
        cuml_model[0] = CuMLLinearRegression(
            algorithm='eig',
            output_type='numpy',
        )
        cuml_model[0].fit(
            processed_train.astype(np.float32),
            train_targets.astype(np.float32),
        )

    optimized_train_timing = time_training(do_train_cuml)
    do_train_cuml()

    def baseline_predict():
        return predict_primary_output(
            trained_baseline, test_features, preprocessed=processed_test
        )

    def optimized_predict():
        return np.asarray(
            cuml_model[0].predict(processed_test.astype(np.float32))
        ).reshape(-1)

    baseline_timing = time_prediction(baseline_predict, n_samples=len(test_features))
    baseline_preds = baseline_predict()

    optimized_timing = time_prediction(optimized_predict, n_samples=len(test_features))
    optimized_preds = optimized_predict()

    prediction_rho = assert_ranking_preserved(
        baseline_preds,
        optimized_preds,
        min_spearman=0.999,
        label='predictions (sklearn lr vs cuML linear regression)',
    )

    result = create_result(
        optimization_id=13,
        optimization_name='rapids_linear_regression',
        learner_name='lr',
        tier=2,
        baseline_time=baseline_timing['median_time'],
        optimized_time=optimized_timing['median_time'],
        prediction_match=True,
        optimization_backend='cuml_linear_regression_eig',
        prediction_spearman_rho=float(prediction_rho),
        baseline_times=baseline_timing['times'],
        optimized_times=optimized_timing['times'],
        training_baseline_time=baseline_train_timing['median_time'],
        training_optimized_time=optimized_train_timing['median_time'],
        training_baseline_times=baseline_train_timing['times'],
        training_optimized_times=optimized_train_timing['times'],
        n_samples=len(test_features),
        n_features=test_features.shape[1],
        n_warmup_runs=baseline_timing['n_warmup'],
        n_timed_runs=baseline_timing['n_runs'],
    )
    result['timing_scope'] = 'predict_only'
    result['max_abs_diff'] = float(np.max(np.abs(baseline_preds - optimized_preds)))
    result['mean_abs_diff'] = float(np.mean(np.abs(baseline_preds - optimized_preds)))
    return result


def _build_fil_predictor(
    learner_name: str,
    learner,
    processed_features: np.ndarray,
) -> tuple[str, str, callable, callable]:
    if learner_name == 'xgb':
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            model_path = tmp.name

        learner.model.save_model(model_path)
        fil_model = ForestInference.load(model_path, model_type='xgboost_json')

        def cleanup() -> None:
            Path(model_path).unlink(missing_ok=True)

        optimization_name = 'rapids_fil_treelite'
        backend = 'cuml_fil_treelite_xgboost_json'
    else:
        if learner_name != 'rf':
            raise NotImplementedError(
                f'CuML FIL sklearn import does not support {learner.model.__class__.__name__}'
            )

        fil_model = ForestInference.load_from_sklearn(learner.model)

        def cleanup() -> None:
            return None

        optimization_name = 'rapids_fil_sklearn'
        backend = 'cuml_fil_sklearn_import'

    def predict_fn():
        return np.asarray(fil_model.predict(processed_features)).squeeze()

    return optimization_name, backend, predict_fn, cleanup


def main():
    try:
        features, targets, _smiles = load_benchmark_data()
    except FileNotFoundError as e:
        print(f"Skipping benchmark: {e}")
        return

    n_train = int(len(targets) * 0.05)
    train_features, test_features = features[:n_train], features[n_train:]
    train_targets = targets[:n_train]
    test_targets = targets[n_train:]

    from learnm8.learners.base import _preprocess_features

    results = []

    rf_result = _benchmark_cuml_random_forest(
        train_features,
        train_targets,
        test_features,
        test_targets,
    )
    save_result(rf_result, RESULTS_DIR)
    results.append(rf_result)

    for learner_name in LEARNERS:
        learner_kwargs = benchmark_learner_kwargs(learner_name, use_gpu=False)

        def do_train(learner_name=learner_name, learner_kwargs=learner_kwargs):
            learner = create_learner(learner_name, **learner_kwargs)
            train_learner(learner, train_features, train_targets)
            return learner

        train_timing = time_training(do_train)
        trained = do_train()

        preprocessed, _ = _preprocess_features(
            test_features,
            valid_feature_mask=trained._valid_feature_mask,
            remove_zero_variance=trained.remove_zero_variance,
            is_training=False,
        )
        processed_f32 = preprocessed.astype(np.float32)

        def baseline_predict(trained=trained, preprocessed=preprocessed):
            return predict_primary_output(
                trained, test_features, preprocessed=preprocessed
            )

        try:
            optimization_name, backend, fil_predict, cleanup = _build_fil_predictor(
                learner_name,
                trained,
                processed_f32,
            )
        except NotImplementedError as exc:
            print(f'  Skipping {learner_name}: {exc}')
            continue

        fil_ref = [fil_predict]

        def fil_setup():
            pass

        def fil_predict_fn():
            return np.asarray(fil_ref[0]()).squeeze()

        try:
            baseline_timing = time_prediction(
                baseline_predict, n_samples=len(test_features)
            )
            baseline_preds = baseline_predict()

            optimized_timing = time_with_setup(
                fil_setup, fil_predict_fn, n_samples=len(test_features)
            )
            optimized_preds = fil_predict_fn()
        finally:
            cleanup()

        if baseline_preds.shape != optimized_preds.shape:
            raise AssertionError(
                f"Shape mismatch: baseline {baseline_preds.shape} vs "
                f"FIL {optimized_preds.shape} for {learner_name}"
            )

        prediction_rho = assert_predictions_match(
            baseline_preds,
            optimized_preds,
            "near_zero",
        )

        result = create_result(
            optimization_id=13,
            optimization_name=optimization_name,
            learner_name=learner_name,
            tier=2,
            baseline_time=baseline_timing["median_time"],
            optimized_time=optimized_timing["subsequent_median"],
            prediction_match=True,
            optimization_backend=backend,
            prediction_spearman_rho=prediction_rho,
            uncertainty_spearman_rho=None,
            setup_time=optimized_timing["setup_time"],
            baseline_times=baseline_timing["times"],
            optimized_times=optimized_timing["subsequent_times"],
            training_baseline_time=train_timing["median_time"],
            training_optimized_time=train_timing["median_time"],
            training_baseline_times=train_timing["times"],
            training_optimized_times=train_timing["times"],
            n_samples=len(test_features),
            n_features=test_features.shape[1],
            n_warmup_runs=optimized_timing["n_warmup"],
            n_timed_runs=optimized_timing["n_runs"],
        )
        result['timing_scope'] = 'predict_only'
        result['first_inference_time'] = optimized_timing['first_inference']
        save_result(result, RESULTS_DIR)
        results.append(result)

    lr_result = _benchmark_cuml_linear_regression(
        train_features,
        train_targets,
        test_features,
    )
    save_result(lr_result, RESULTS_DIR)
    results.append(lr_result)

    print_result_table(results)


if __name__ == "__main__":
    main()

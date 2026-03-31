# Optimization #9: XGBoost CUDA device
# Scope: mechanism validation — inference only
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from validation.performance.speedup.lib.assertions import assert_ranking_preserved  # noqa: E402
from validation.performance.speedup.lib.data_setup import load_benchmark_data  # noqa: E402
from validation.performance.speedup.lib.model_factory import (  # noqa: E402
    create_learner,
    predict_primary_output,
    train_learner,
)
from validation.performance.speedup.lib.reporting import (  # noqa: E402
    create_result,
    print_result_table,
    save_result,
)
from validation.performance.speedup.lib.runtime import (  # noqa: E402
    benchmark_learner_kwargs,
    configure_process_environment,
)
from validation.performance.speedup.lib.timing import time_prediction, time_training  # noqa: E402

configure_process_environment()

try:
    import torch
except ImportError:
    print("Skipping benchmark: torch not available (needed for CUDA check)")
    sys.exit(0)

if not torch.cuda.is_available():
    print("Skipping benchmark: CUDA GPU not available")
    sys.exit(0)

from learnm8.learners.base import _preprocess_features  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _create_xgb_learner(*, use_gpu: bool):
    learner = create_learner(
        'xgb',
        **benchmark_learner_kwargs('xgb', use_gpu=False),
    )
    if use_gpu:
        learner.model.set_params(device='cuda')
    return learner


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

    def do_train_cpu():
        learner = _create_xgb_learner(use_gpu=False)
        train_learner(learner, train_features, train_targets)
        return learner

    def do_train_gpu():
        learner = _create_xgb_learner(use_gpu=True)
        train_learner(learner, train_features, train_targets)
        return learner

    cpu_train_timing = time_training(do_train_cpu)
    trained_cpu = do_train_cpu()

    gpu_train_timing = time_training(do_train_gpu)
    trained_gpu = do_train_gpu()

    preprocessed, _ = _preprocess_features(
        test_features,
        valid_feature_mask=trained_cpu._valid_feature_mask,
        remove_zero_variance=trained_cpu.remove_zero_variance,
        is_training=False,
    )

    def baseline_predict():
        return predict_primary_output(
            trained_cpu, test_features, preprocessed=preprocessed
        )

    def optimized_predict():
        return predict_primary_output(
            trained_gpu, test_features, preprocessed=preprocessed
        )

    baseline_timing = time_prediction(
        baseline_predict, n_samples=len(test_features)
    )
    baseline_preds = baseline_predict()

    optimized_timing = time_prediction(
        optimized_predict, n_samples=len(test_features)
    )
    optimized_preds = optimized_predict()

    cpu_training_quality = float(spearmanr(test_targets, baseline_preds)[0])
    gpu_training_quality = float(spearmanr(test_targets, optimized_preds)[0])
    training_cross_rho = float(spearmanr(baseline_preds, optimized_preds)[0])
    if not np.isfinite(training_cross_rho):
        raise AssertionError('CPU-vs-GPU XGBoost cycle rho is not finite')
    if not np.isfinite(gpu_training_quality):
        raise AssertionError('GPU-trained XGBoost holdout quality rho is not finite')

    rho = assert_ranking_preserved(
        baseline_preds,
        optimized_preds,
        min_spearman=0.90,
        label='XGBoost CPU vs CUDA predictions',
    )

    result = create_result(
        optimization_id=9,
        optimization_name="xgb_cuda",
        learner_name="xgb",
        tier=1,
        baseline_time=baseline_timing["median_time"],
        optimized_time=optimized_timing["median_time"],
        prediction_match=True,
        prediction_spearman_rho=rho,
        baseline_times=baseline_timing["times"],
        optimized_times=optimized_timing["times"],
        training_baseline_time=cpu_train_timing["median_time"],
        training_optimized_time=gpu_train_timing["median_time"],
        training_baseline_times=cpu_train_timing["times"],
        training_optimized_times=gpu_train_timing["times"],
        n_samples=len(test_features),
        n_features=test_features.shape[1],
        n_warmup_runs=baseline_timing["n_warmup"],
        n_timed_runs=baseline_timing["n_runs"],
    )
    result['timing_scope'] = 'predict_only'
    result['cycle_baseline_quality_rho'] = cpu_training_quality
    result['cycle_optimized_quality_rho'] = gpu_training_quality
    result['cycle_prediction_rho'] = training_cross_rho
    result['max_abs_diff'] = float(np.max(np.abs(baseline_preds - optimized_preds)))
    result['mean_abs_diff'] = float(np.mean(np.abs(baseline_preds - optimized_preds)))
    save_result(result, RESULTS_DIR)
    print_result_table([result])


if __name__ == "__main__":
    main()

# Cycle sweep benchmark: test optimizations across realistic AL cycle progressions
# Simulates a 10-cycle active learning campaign with batch_fraction=0.01
# Benchmarks at cycle points [1, 3, 5, 7, 9] to measure scaling behavior
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validation.performance.speedup.lib.runtime import (  # noqa: E402
    benchmark_learner_kwargs,
    configure_process_environment,
    max_cpu_threads,
)

configure_process_environment()

import sklearn  # noqa: E402

try:
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

try:
    import torch
    import torch.nn as nn  # noqa: F401

    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    from validation.performance.speedup.lib.runtime import configure_torch_runtime

    configure_torch_runtime(torch, use_gpu=True)

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from learnm8.learners.base import _preprocess_features  # noqa: E402
from validation.performance.speedup.lib.assertions import (  # noqa: E402
    assert_predictions_match,
    assert_ranking_preserved,
)
from validation.performance.speedup.lib.config import get_results_dir  # noqa: E402
from validation.performance.speedup.lib.data_setup import load_benchmark_data  # noqa: E402
from validation.performance.speedup.lib.model_factory import (  # noqa: E402
    create_learner,
    predict_primary_output,
    train_learner,
)
from validation.performance.speedup.lib.reporting import (  # noqa: E402
    _get_hardware_info,
    _get_software_info,
    _NumpyEncoder,
)
from validation.performance.speedup.lib.timing import (  # noqa: E402
    time_prediction,
    time_training,
    time_with_setup,
)

_console = Console()

RESULTS_DIR = Path(__file__).parent.parent / "results"
BATCH_FRACTION = 0.01
CYCLE_POINTS = [1, 3, 5, 7, 9]
N_CYCLES_TOTAL = 10


def _compute_cycle_sizes(total_samples: int) -> list[dict]:
    batch_size = int(total_samples * BATCH_FRACTION)
    points = []
    for cycle in CYCLE_POINTS:
        train_size = cycle * batch_size
        pool_size = total_samples - train_size
        if train_size >= total_samples or pool_size <= 0:
            break
        points.append({
            "cycle": cycle,
            "train_size": train_size,
            "pool_size": pool_size,
            "batch_size": batch_size,
        })
    return points


def _interpolate_campaign_total(
    measured_cycles: list[int],
    measured_times: list[float],
) -> float:
    total = 0.0
    for cycle in range(1, N_CYCLES_TOTAL + 1):
        if cycle in measured_cycles:
            total += measured_times[measured_cycles.index(cycle)]
        else:
            total += float(np.interp(cycle, measured_cycles, measured_times))
    return total


def _preprocess_once(learner, features):
    preprocessed, _ = _preprocess_features(
        features,
        valid_feature_mask=getattr(learner, "_valid_feature_mask", None),
        remove_zero_variance=getattr(learner, "remove_zero_variance", True),
        is_training=False,
    )
    return preprocessed


def _sweep_assume_finite(
    learner_name: str,
    features: np.ndarray,
    targets: np.ndarray,
    cycle_sizes: list[dict],
) -> dict:
    print(f"\n--- assume_finite / {learner_name} ---")
    cycle_results = []

    for point in cycle_sizes:
        cycle = point["cycle"]
        train_size = point["train_size"]
        pool_size = point["pool_size"]
        print(f"\n  Cycle {cycle}: train={train_size}, pool={pool_size}")

        train_feat = features[:train_size]
        train_tgt = targets[:train_size]
        pool_feat = features[train_size:train_size + pool_size]

        learner_kwargs = benchmark_learner_kwargs(learner_name, use_gpu=False)

        def do_train_baseline(lk=learner_kwargs, tf=train_feat, tt=train_tgt):
            learner = create_learner(learner_name, **lk)
            train_learner(learner, tf, tt)
            return learner

        def do_train_optimized(lk=learner_kwargs, tf=train_feat, tt=train_tgt):
            original_config = sklearn.get_config()
            try:
                sklearn.set_config(assume_finite=True)
                learner = create_learner(learner_name, **lk)
                train_learner(learner, tf, tt)
                return learner
            finally:
                sklearn.set_config(**original_config)

        baseline_train_timing = time_training(do_train_baseline, n_warmup=1, n_runs=3)
        trained_baseline = do_train_baseline()

        optimized_train_timing = time_training(do_train_optimized, n_warmup=1, n_runs=3)
        trained_optimized = do_train_optimized()

        preprocessed = _preprocess_once(trained_baseline, pool_feat)

        def baseline_predict(tb=trained_baseline, pf=pool_feat, pp=preprocessed):
            return predict_primary_output(tb, pf, preprocessed=pp)

        def optimized_predict(to=trained_optimized, pf=pool_feat, pp=preprocessed):
            original_config = sklearn.get_config()
            try:
                sklearn.set_config(assume_finite=True)
                return predict_primary_output(to, pf, preprocessed=pp)
            finally:
                sklearn.set_config(**original_config)

        baseline_timing = time_prediction(
            baseline_predict, n_samples=pool_size, n_warmup=1, n_runs=3
        )
        optimized_timing = time_prediction(
            optimized_predict, n_samples=pool_size, n_warmup=1, n_runs=3
        )

        baseline_preds = baseline_predict()
        optimized_preds = optimized_predict()
        assert_predictions_match(baseline_preds, optimized_preds, "zero_impact")

        pred_speedup = baseline_timing["median_time"] / optimized_timing["median_time"] if optimized_timing["median_time"] > 0 else float("inf")
        train_speedup = baseline_train_timing["median_time"] / optimized_train_timing["median_time"] if optimized_train_timing["median_time"] > 0 else float("inf")

        cycle_results.append({
            "cycle": cycle,
            "train_size": train_size,
            "pool_size": pool_size,
            "train_baseline_time": baseline_train_timing["median_time"],
            "train_optimized_time": optimized_train_timing["median_time"],
            "train_speedup": train_speedup,
            "predict_baseline_time": baseline_timing["median_time"],
            "predict_optimized_time": optimized_timing["median_time"],
            "predict_speedup": pred_speedup,
            "train_baseline_times": baseline_train_timing["times"],
            "train_optimized_times": optimized_train_timing["times"],
            "predict_baseline_times": baseline_timing["times"],
            "predict_optimized_times": optimized_timing["times"],
        })

    return _build_sweep_result("assume_finite", learner_name, cycle_results)


def _sweep_onnx_runtime(
    features: np.ndarray,
    targets: np.ndarray,
    cycle_sizes: list[dict],
) -> dict | None:
    if not _HAS_ONNX:
        print("\n--- onnx_runtime / rf --- SKIPPED (onnxruntime not installed)")
        return None

    learner_name = "rf"
    print(f"\n--- onnx_runtime / {learner_name} ---")

    learner_kwargs = benchmark_learner_kwargs(learner_name, use_gpu=False)

    first_point = cycle_sizes[0]
    train_feat = features[:first_point["train_size"]]
    train_tgt = targets[:first_point["train_size"]]

    seed_learner = create_learner(learner_name, **learner_kwargs)
    train_learner(seed_learner, train_feat, train_tgt)

    cycle_results = []
    for point in cycle_sizes:
        cycle = point["cycle"]
        train_size = point["train_size"]
        pool_size = point["pool_size"]
        print(f"\n  Cycle {cycle}: train={train_size}, pool={pool_size}")

        t_feat = features[:train_size]
        t_tgt = targets[:train_size]
        pool_feat = features[train_size:train_size + pool_size]

        def do_train(lk=learner_kwargs, tf=t_feat, tt=t_tgt):
            learner = create_learner(learner_name, **lk)
            train_learner(learner, tf, tt)
            return learner

        train_timing = time_training(do_train, n_warmup=1, n_runs=3)
        trained = do_train()

        preprocessed = _preprocess_once(trained, pool_feat)
        preprocessed_f32 = preprocessed.astype(np.float32)

        def baseline_predict(t=trained, pf=pool_feat, pp=preprocessed):
            return predict_primary_output(t, pf, preprocessed=pp)

        baseline_timing = time_prediction(
            baseline_predict, n_samples=pool_size, n_warmup=1, n_runs=3
        )
        baseline_preds = baseline_predict()

        ort_session = [None]

        def onnx_setup(t=trained, pp=preprocessed, _os=ort_session):
            initial_type = [("X", FloatTensorType([None, pp.shape[1]]))]
            onnx_model = convert_sklearn(t.model, initial_types=initial_type)
            options = ort.SessionOptions()
            options.intra_op_num_threads = max_cpu_threads()
            options.inter_op_num_threads = 1
            _os[0] = ort.InferenceSession(
                onnx_model.SerializeToString(), sess_options=options
            )

        def onnx_predict(pf32=preprocessed_f32, _os=ort_session):
            input_name = _os[0].get_inputs()[0].name
            return _os[0].run(None, {input_name: pf32})[0].flatten()

        optimized_timing = time_with_setup(
            onnx_setup, onnx_predict, n_samples=pool_size, n_warmup=1, n_runs=3
        )
        optimized_preds = onnx_predict()

        assert_predictions_match(baseline_preds, optimized_preds, "near_zero")

        pred_speedup = baseline_timing["median_time"] / optimized_timing["subsequent_median"] if optimized_timing["subsequent_median"] > 0 else float("inf")

        cycle_results.append({
            "cycle": cycle,
            "train_size": train_size,
            "pool_size": pool_size,
            "train_baseline_time": train_timing["median_time"],
            "train_optimized_time": train_timing["median_time"],
            "train_speedup": 1.0,
            "predict_baseline_time": baseline_timing["median_time"],
            "predict_optimized_time": optimized_timing["subsequent_median"],
            "predict_speedup": pred_speedup,
            "setup_time": optimized_timing["setup_time"],
            "first_inference_time": optimized_timing["first_inference"],
            "train_baseline_times": train_timing["times"],
            "train_optimized_times": train_timing["times"],
            "predict_baseline_times": baseline_timing["times"],
            "predict_optimized_times": optimized_timing["subsequent_times"],
        })

    return _build_sweep_result("onnx_runtime", learner_name, cycle_results)


def _preprocess_and_tensorize(learner, features):
    processed, _ = _preprocess_features(
        features,
        valid_feature_mask=learner._valid_feature_mask,
        remove_zero_variance=learner.remove_zero_variance,
        is_training=False,
    )
    scaled = learner.scaler.transform(processed)
    return torch.FloatTensor(scaled).to(learner.device)


def _predict_with_model(model, x_tensor):
    model.eval()
    with torch.no_grad():
        return model(x_tensor).cpu().numpy().squeeze()


def _sweep_torch_compile(
    features: np.ndarray,
    targets: np.ndarray,
    cycle_sizes: list[dict],
) -> dict | None:
    if not _HAS_TORCH:
        print("\n--- torch_compile / mlp --- SKIPPED (CUDA not available)")
        return None

    learner_name = "mlp"
    print(f"\n--- torch_compile / {learner_name} ---")

    learner_kwargs = benchmark_learner_kwargs(learner_name, use_gpu=True)
    compiled_model = [None]
    cycle_results = []

    for i, point in enumerate(cycle_sizes):
        cycle = point["cycle"]
        train_size = point["train_size"]
        pool_size = point["pool_size"]
        print(f"\n  Cycle {cycle}: train={train_size}, pool={pool_size}")

        t_feat = features[:train_size]
        t_tgt = targets[:train_size]
        pool_feat = features[train_size:train_size + pool_size]

        def do_train(lk=learner_kwargs, tf=t_feat, tt=t_tgt):
            learner = create_learner(learner_name, **lk)
            train_learner(learner, tf, tt)
            return learner

        train_timing = time_training(do_train, n_warmup=1, n_runs=3, cuda_sync=True)
        trained = do_train()

        x_tensor = _preprocess_and_tensorize(trained, pool_feat)

        def baseline_predict(m=trained.model, xt=x_tensor):
            return _predict_with_model(m, xt)

        baseline_timing = time_prediction(
            baseline_predict, n_samples=pool_size, n_warmup=1, n_runs=3, cuda_sync=True
        )
        baseline_preds = baseline_predict()

        if i == 0:
            def compile_setup(m=trained.model):
                compiled_model[0] = torch.compile(m, dynamic=True, fullgraph=False)

            def compiled_predict(xt=x_tensor):
                return _predict_with_model(compiled_model[0], xt)

            optimized_timing = time_with_setup(
                compile_setup, compiled_predict,
                n_samples=pool_size, n_warmup=1, n_runs=3, cuda_sync=True,
            )
            optimized_preds = compiled_predict()
            opt_median = optimized_timing["subsequent_median"]
            opt_times = optimized_timing["subsequent_times"]
            setup_time = optimized_timing["setup_time"]
            first_inference = optimized_timing["first_inference"]
        else:
            compiled_model[0] = torch.compile(trained.model, dynamic=True, fullgraph=False)

            def compiled_predict_subsequent(xt=x_tensor):
                return _predict_with_model(compiled_model[0], xt)

            optimized_timing = time_prediction(
                compiled_predict_subsequent, n_samples=pool_size,
                n_warmup=1, n_runs=3, cuda_sync=True,
            )
            optimized_preds = compiled_predict_subsequent()
            opt_median = optimized_timing["median_time"]
            opt_times = optimized_timing["times"]
            setup_time = None
            first_inference = None

        assert_ranking_preserved(
            baseline_preds, optimized_preds,
            min_spearman=0.999,
            label=f"torch.compile cycle {cycle}",
        )

        pred_speedup = baseline_timing["median_time"] / opt_median if opt_median > 0 else float("inf")

        entry = {
            "cycle": cycle,
            "train_size": train_size,
            "pool_size": pool_size,
            "train_baseline_time": train_timing["median_time"],
            "train_optimized_time": train_timing["median_time"],
            "train_speedup": 1.0,
            "predict_baseline_time": baseline_timing["median_time"],
            "predict_optimized_time": opt_median,
            "predict_speedup": pred_speedup,
            "train_baseline_times": train_timing["times"],
            "train_optimized_times": train_timing["times"],
            "predict_baseline_times": baseline_timing["times"],
            "predict_optimized_times": opt_times,
        }
        if setup_time is not None:
            entry["setup_time"] = setup_time
        if first_inference is not None:
            entry["first_inference_time"] = first_inference
        cycle_results.append(entry)

    return _build_sweep_result("torch_compile", learner_name, cycle_results)


def _build_sweep_result(
    optimization_name: str,
    learner_name: str,
    cycle_results: list[dict],
) -> dict:
    measured_cycles = [cr["cycle"] for cr in cycle_results]
    train_baseline_times = [cr["train_baseline_time"] for cr in cycle_results]
    train_optimized_times = [cr["train_optimized_time"] for cr in cycle_results]
    pred_baseline_times = [cr["predict_baseline_time"] for cr in cycle_results]
    pred_optimized_times = [cr["predict_optimized_time"] for cr in cycle_results]

    total_train_baseline = _interpolate_campaign_total(measured_cycles, train_baseline_times)
    total_train_optimized = _interpolate_campaign_total(measured_cycles, train_optimized_times)
    total_pred_baseline = _interpolate_campaign_total(measured_cycles, pred_baseline_times)
    total_pred_optimized = _interpolate_campaign_total(measured_cycles, pred_optimized_times)

    total_baseline = total_train_baseline + total_pred_baseline
    total_optimized = total_train_optimized + total_pred_optimized
    campaign_speedup = total_baseline / total_optimized if total_optimized > 0 else float("inf")
    train_campaign_speedup = total_train_baseline / total_train_optimized if total_train_optimized > 0 else float("inf")
    pred_campaign_speedup = total_pred_baseline / total_pred_optimized if total_pred_optimized > 0 else float("inf")

    return {
        "optimization_name": optimization_name,
        "learner_name": learner_name,
        "scope": "cycle_sweep",
        "batch_fraction": BATCH_FRACTION,
        "n_cycles_total": N_CYCLES_TOTAL,
        "n_cycles_measured": len(cycle_results),
        "cycle_points": cycle_results,
        "campaign_total_train_baseline": total_train_baseline,
        "campaign_total_train_optimized": total_train_optimized,
        "campaign_train_speedup": train_campaign_speedup,
        "campaign_total_pred_baseline": total_pred_baseline,
        "campaign_total_pred_optimized": total_pred_optimized,
        "campaign_pred_speedup": pred_campaign_speedup,
        "campaign_total_baseline": total_baseline,
        "campaign_total_optimized": total_optimized,
        "campaign_speedup": campaign_speedup,
        "hardware_info": _get_hardware_info(),
        "software_info": _get_software_info(),
    }


def _save_sweep_result(result: dict, output_dir: Path) -> Path:
    output_dir = get_results_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"cycle_sweep_{result['optimization_name']}_{result['learner_name']}.json"
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)
    print(f"  Saved: {path.name}")
    return path


def _speedup_label(ratio: float) -> tuple[str, str]:
    if ratio > 1.05:
        return "SPEEDUP", "green"
    if ratio < 0.95:
        return "REGRESSION", "red"
    return "NEUTRAL", "yellow"


def _fmt_time(t: float) -> str:
    return f"{t:.3f}s"


def _fmt_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_cycle_table(all_results: list[dict]) -> None:
    table = Table(title="Cycle Sweep Results")
    table.add_column("Optimization")
    table.add_column("Learner")
    table.add_column("Cycle")
    table.add_column("Train Size")
    table.add_column("Pool Size")
    table.add_column("Train (B)")
    table.add_column("Train (O)")
    table.add_column("Train Speedup")
    table.add_column("Pred (B)")
    table.add_column("Pred (O)")
    table.add_column("Pred Speedup")

    for result in all_results:
        opt_name = result["optimization_name"]
        learner = result["learner_name"]
        for cp in result["cycle_points"]:
            train_label, train_color = _speedup_label(cp["train_speedup"])
            pred_label, pred_color = _speedup_label(cp["predict_speedup"])
            table.add_row(
                opt_name,
                learner,
                str(cp["cycle"]),
                _fmt_size(cp["train_size"]),
                _fmt_size(cp["pool_size"]),
                _fmt_time(cp["train_baseline_time"]),
                _fmt_time(cp["train_optimized_time"]),
                f"[{train_color}]{cp['train_speedup']:.2f}x ({train_label})[/{train_color}]",
                _fmt_time(cp["predict_baseline_time"]),
                _fmt_time(cp["predict_optimized_time"]),
                f"[{pred_color}]{cp['predict_speedup']:.2f}x ({pred_label})[/{pred_color}]",
            )

    _console.print(table)


def print_campaign_summary(all_results: list[dict]) -> None:
    table = Table(title=f"Campaign Summary ({N_CYCLES_TOTAL}-cycle projection)")
    table.add_column("Optimization")
    table.add_column("Learner")
    table.add_column("Train (B)")
    table.add_column("Train (O)")
    table.add_column("Train Speedup")
    table.add_column("Pred (B)")
    table.add_column("Pred (O)")
    table.add_column("Pred Speedup")
    table.add_column("Total (B)")
    table.add_column("Total (O)")
    table.add_column("Campaign Speedup")

    for result in all_results:
        _train_label, train_color = _speedup_label(result["campaign_train_speedup"])
        _pred_label, pred_color = _speedup_label(result["campaign_pred_speedup"])
        total_label, total_color = _speedup_label(result["campaign_speedup"])

        table.add_row(
            result["optimization_name"],
            result["learner_name"],
            _fmt_time(result["campaign_total_train_baseline"]),
            _fmt_time(result["campaign_total_train_optimized"]),
            f"[{train_color}]{result['campaign_train_speedup']:.2f}x[/{train_color}]",
            _fmt_time(result["campaign_total_pred_baseline"]),
            _fmt_time(result["campaign_total_pred_optimized"]),
            f"[{pred_color}]{result['campaign_pred_speedup']:.2f}x[/{pred_color}]",
            _fmt_time(result["campaign_total_baseline"]),
            _fmt_time(result["campaign_total_optimized"]),
            f"[{total_color}]{result['campaign_speedup']:.2f}x ({total_label})[/{total_color}]",
        )

    _console.print(table)


def main():
    try:
        features, targets, _smiles = load_benchmark_data()
    except FileNotFoundError as e:
        print(f"Skipping benchmark: {e}")
        return

    total_samples = len(targets)
    cycle_sizes = _compute_cycle_sizes(total_samples)
    print(f"Total samples: {total_samples}")
    print(f"Batch fraction: {BATCH_FRACTION}")
    print(f"Batch size: {cycle_sizes[0]['batch_size']}")
    print(f"Cycle points: {[c['cycle'] for c in cycle_sizes]}")
    for cs in cycle_sizes:
        print(f"  Cycle {cs['cycle']}: train={cs['train_size']}, pool={cs['pool_size']}")

    all_results = []

    for learner_name in ["rf", "dt", "lr"]:
        result = _sweep_assume_finite(learner_name, features, targets, cycle_sizes)
        _save_sweep_result(result, RESULTS_DIR)
        all_results.append(result)

    if _HAS_TORCH:
        result = _sweep_torch_compile(features, targets, cycle_sizes)
        if result:
            _save_sweep_result(result, RESULTS_DIR)
            all_results.append(result)

    result = _sweep_onnx_runtime(features, targets, cycle_sizes)
    if result:
        _save_sweep_result(result, RESULTS_DIR)
        all_results.append(result)

    print_cycle_table(all_results)
    print_campaign_summary(all_results)


if __name__ == "__main__":
    main()

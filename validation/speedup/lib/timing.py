from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from validation.speedup.lib.config import get_timing_runs


def _try_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _infer_n_samples(*args: Any) -> int | None:
    for arg in reversed(args):
        shape = getattr(arg, "shape", None)
        if shape is not None and len(shape) > 0:
            try:
                return int(shape[0])
            except (TypeError, ValueError):
                pass

        try:
            return len(arg)
        except TypeError:
            continue

    return None


def _summarize(times: list[float]) -> tuple[float, float]:
    times_arr = np.array(times)
    median_time = float(np.median(times_arr))
    q1 = float(np.percentile(times_arr, 25))
    q3 = float(np.percentile(times_arr, 75))
    iqr = q3 - q1
    return median_time, iqr


def time_prediction(
    predict_fn: Callable,
    *args,
    n_warmup: int | None = None,
    n_runs: int | None = None,
    n_samples: int | None = None,
    cuda_sync: bool = False,
    **kwargs,
) -> dict:
    n_warmup, n_runs = get_timing_runs(n_warmup, n_runs)
    fn_name = getattr(predict_fn, "__name__", str(predict_fn))
    sample_count = n_samples if n_samples is not None else _infer_n_samples(*args)
    sample_suffix = f" on {sample_count} samples" if sample_count is not None else ""
    print(
        f"  Timing {fn_name}{sample_suffix} ({n_warmup} warmup, {n_runs} timed runs)..."
    )

    for _ in range(n_warmup):
        predict_fn(*args, **kwargs)
        if cuda_sync:
            _try_cuda_sync()

    gc.disable()
    try:
        times = []
        for _ in range(n_runs):
            if cuda_sync:
                _try_cuda_sync()
            start = time.perf_counter_ns()
            predict_fn(*args, **kwargs)
            if cuda_sync:
                _try_cuda_sync()
            end = time.perf_counter_ns()
            times.append((end - start) / 1e9)
    finally:
        gc.enable()

    median_time, iqr = _summarize(times)
    print(f"  -> median={median_time:.4f}s, IQR={iqr:.4f}s")

    return {
        "median_time": median_time,
        "iqr": iqr,
        "times": times,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
    }


def time_training(
    train_fn: Callable,
    *args,
    n_warmup: int | None = None,
    n_runs: int | None = None,
    cuda_sync: bool = False,
    **kwargs,
) -> dict:
    n_warmup, n_runs = get_timing_runs(n_warmup, n_runs)
    fn_name = getattr(train_fn, "__name__", str(train_fn))
    print(f"  Timing {fn_name} training ({n_warmup} warmup, {n_runs} timed runs)...")

    for _ in range(n_warmup):
        train_fn(*args, **kwargs)
        if cuda_sync:
            _try_cuda_sync()

    gc.disable()
    try:
        times = []
        for _ in range(n_runs):
            if cuda_sync:
                _try_cuda_sync()
            start = time.perf_counter_ns()
            train_fn(*args, **kwargs)
            if cuda_sync:
                _try_cuda_sync()
            end = time.perf_counter_ns()
            times.append((end - start) / 1e9)
    finally:
        gc.enable()

    median_time, iqr = _summarize(times)
    print(f"  -> median={median_time:.4f}s, IQR={iqr:.4f}s")

    return {
        "median_time": median_time,
        "iqr": iqr,
        "times": times,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
    }


def time_with_setup(
    setup_fn: Callable,
    predict_fn: Callable,
    n_warmup: int | None = None,
    n_runs: int | None = None,
    n_samples: int | None = None,
    cuda_sync: bool = False,
) -> dict:
    n_warmup, n_runs = get_timing_runs(n_warmup, n_runs)
    sample_suffix = f" on {n_samples} samples" if n_samples is not None else ""
    print(
        f"  Timing setup + predict{sample_suffix} ({n_warmup} warmup, {n_runs} timed runs)..."
    )

    if cuda_sync:
        _try_cuda_sync()
    start = time.perf_counter_ns()
    setup_fn()
    if cuda_sync:
        _try_cuda_sync()
    setup_time = (time.perf_counter_ns() - start) / 1e9

    if cuda_sync:
        _try_cuda_sync()
    start = time.perf_counter_ns()
    predict_fn()
    if cuda_sync:
        _try_cuda_sync()
    first_inference = (time.perf_counter_ns() - start) / 1e9

    for _ in range(n_warmup):
        predict_fn()
        if cuda_sync:
            _try_cuda_sync()

    gc.disable()
    try:
        subsequent_times = []
        for _ in range(n_runs):
            if cuda_sync:
                _try_cuda_sync()
            start = time.perf_counter_ns()
            predict_fn()
            if cuda_sync:
                _try_cuda_sync()
            end = time.perf_counter_ns()
            subsequent_times.append((end - start) / 1e9)
    finally:
        gc.enable()

    if len(subsequent_times) > 0:
        subsequent_median, subsequent_iqr = _summarize(subsequent_times)
    else:
        subsequent_median, subsequent_iqr = 0.0, 0.0

    print(
        f"  -> setup={setup_time:.4f}s, first={first_inference:.4f}s, "
        f"subsequent_median={subsequent_median:.4f}s, IQR={subsequent_iqr:.4f}s"
    )

    return {
        "setup_time": setup_time,
        "first_inference": first_inference,
        "subsequent_median": subsequent_median,
        "subsequent_iqr": subsequent_iqr,
        "subsequent_times": subsequent_times,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
    }


def time_cycle(
    train_fn: Callable,
    predict_fn: Callable,
    n_warmup: int | None = None,
    n_runs: int | None = None,
    n_samples: int | None = None,
    cuda_sync: bool = False,
) -> dict:
    n_warmup, n_runs = get_timing_runs(n_warmup, n_runs)
    sample_suffix = f" on {n_samples} samples" if n_samples is not None else ""
    print(
        f"  Timing train + predict{sample_suffix} "
        f"({n_warmup} warmup, {n_runs} timed runs)..."
    )

    for _ in range(n_warmup):
        train_fn()
        if cuda_sync:
            _try_cuda_sync()
        predict_fn()
        if cuda_sync:
            _try_cuda_sync()

    gc.disable()
    try:
        train_times = []
        predict_times = []
        for _ in range(n_runs):
            if cuda_sync:
                _try_cuda_sync()
            t0 = time.perf_counter_ns()
            train_fn()
            if cuda_sync:
                _try_cuda_sync()
            t1 = time.perf_counter_ns()
            predict_fn()
            if cuda_sync:
                _try_cuda_sync()
            t2 = time.perf_counter_ns()
            train_times.append((t1 - t0) / 1e9)
            predict_times.append((t2 - t1) / 1e9)
    finally:
        gc.enable()

    train_median, train_iqr = _summarize(train_times)
    predict_median, predict_iqr = _summarize(predict_times)
    total_times = [t + p for t, p in zip(train_times, predict_times, strict=True)]
    total_median, total_iqr = _summarize(total_times)

    print(
        f"  -> train={train_median:.4f}s, predict={predict_median:.4f}s, "
        f"total={total_median:.4f}s"
    )

    return {
        "train_median": train_median,
        "train_iqr": train_iqr,
        "train_times": train_times,
        "predict_median": predict_median,
        "predict_iqr": predict_iqr,
        "predict_times": predict_times,
        "total_median": total_median,
        "total_iqr": total_iqr,
        "total_times": total_times,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
    }

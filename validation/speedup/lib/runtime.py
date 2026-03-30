from __future__ import annotations

import contextlib
import os

CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

TORCH_LEARNERS = {"mlp", "mc_dropout", "chemprop", "fastprop"}
CPU_PARALLEL_LEARNERS = {"rf", "xgb", "lr"}


def max_cpu_threads() -> int:
    override = os.environ.get("BENCHMARK_CPU_THREADS")
    if override:
        return max(1, int(override))
    return os.cpu_count() or 1


def configure_process_environment() -> int:
    threads = max_cpu_threads()
    for env_var in CPU_THREAD_ENV_VARS:
        os.environ[env_var] = str(threads)
    return threads


def configure_torch_runtime(torch_module, use_gpu: bool) -> int:
    threads = max_cpu_threads()
    torch_module.set_num_threads(threads)
    with contextlib.suppress(RuntimeError):
        torch_module.set_num_interop_threads(1)
    return threads


def benchmark_learner_kwargs(
    learner_name: str,
    *,
    use_gpu: bool,
    random_state: int = 42,
    **overrides,
) -> dict:
    kwargs: dict = {"random_state": random_state}

    if learner_name in CPU_PARALLEL_LEARNERS:
        kwargs["n_jobs"] = -1

    if learner_name in TORCH_LEARNERS:
        kwargs["device"] = "cuda" if use_gpu else "cpu"

    kwargs.update(overrides)
    return kwargs

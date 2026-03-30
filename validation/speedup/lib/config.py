from __future__ import annotations

import os
from pathlib import Path

DATASET_ENV_VAR = 'BENCHMARK_DATASET_PATH'
WARMUP_ENV_VAR = 'BENCHMARK_N_WARMUP_RUNS'
TIMED_ENV_VAR = 'BENCHMARK_N_TIMED_RUNS'
RESULTS_DIR_ENV_VAR = 'BENCHMARK_RESULTS_DIR'

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / 'AmpC_screen_100K.csv'
)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'


def get_benchmark_dataset_path(data_path: str | None = None) -> Path:
    path = data_path or os.environ.get(DATASET_ENV_VAR)
    if path:
        return Path(path).expanduser().resolve()
    return DEFAULT_DATASET_PATH


def get_results_dir(output_dir: str | Path | None = None) -> Path:
    path = os.environ.get(RESULTS_DIR_ENV_VAR) or output_dir
    if path:
        return Path(path).expanduser().resolve()
    return DEFAULT_RESULTS_DIR


def get_timing_runs(
    n_warmup: int | None = None,
    n_runs: int | None = None,
    *,
    default_warmup: int = 2,
    default_runs: int = 5,
) -> tuple[int, int]:
    resolved_warmup = n_warmup
    if resolved_warmup is None:
        resolved_warmup = int(os.environ.get(WARMUP_ENV_VAR, default_warmup))

    resolved_runs = n_runs
    if resolved_runs is None:
        resolved_runs = int(os.environ.get(TIMED_ENV_VAR, default_runs))

    if resolved_warmup < 0:
        raise ValueError('n_warmup must be non-negative')
    if resolved_runs <= 0:
        raise ValueError('n_runs must be positive')

    return resolved_warmup, resolved_runs

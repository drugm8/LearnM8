# performance

GPU/CPU performance benchmarks for LearnM8 learners. Contains a top-level GP
benchmark script and the `speedup/` sub-package for RAPIDS/CUDA acceleration
tests (requires the `learnm8-speedup2` conda environment with cuML/XGBoost CUDA).

## Scripts

### Top-level benchmarks (`scripts/`)

| Script | Description |
|---|---|
| `benchmark_gp_active_learning.py` | Benchmarks GPyTorch GP learner across dataset sizes and cycle counts |

### speedup/ sub-package (`speedup/scripts/`)

| Script | Description |
|---|---|
| `benchmark_rapids_fil.py` | Benchmarks RAPIDS Forest Inference Library (FIL) vs CPU RF |
| `benchmark_xgb_cuda.py` | Benchmarks XGBoost with CUDA tree method vs CPU |
| `benchmark_gpu_learners.py` | Batched prediction throughput benchmark across GPU-capable learners |
| `benchmark_gpu_cycle.py` | End-to-end active learning cycle timing on GPU |
| `benchmark_cycle_sweep.py` | Sweeps dataset size and cycle count for GPU vs CPU cycle timing |

The `speedup/tests/` directory contains pytest-based correctness tests for GPU
learner outputs (FIL, XGBoost CUDA, Ridge cuML, ensemble GPU).

## How to Run

Top-level benchmark (standard conda env):

```bash
PYTHONPATH=. python validation/performance/scripts/benchmark_gp_active_learning.py
```

speedup benchmarks (requires `learnm8-speedup2` env):

```bash
conda activate learnm8-speedup2
PYTHONPATH=. python validation/performance/speedup/scripts/benchmark_rapids_fil.py
PYTHONPATH=. python validation/performance/speedup/scripts/benchmark_xgb_cuda.py
PYTHONPATH=. python validation/performance/speedup/scripts/benchmark_gpu_learners.py
PYTHONPATH=. python validation/performance/speedup/scripts/benchmark_gpu_cycle.py
PYTHONPATH=. python validation/performance/speedup/scripts/benchmark_cycle_sweep.py
```

Run the full category via the orchestrator (opt-in):

```bash
PYTHONPATH=. python validation/run_all_validations.py --category performance
```

## Outputs

Results are written to:

```
validation/reports/performance/<script_name>/
```

Each benchmark produces timing CSVs and a summary JSON with throughput and
speedup ratios relative to the CPU baseline.

# Running Experiments

This tutorial covers the basic Python API and CLI workflow. For the full parameter list see the [API Reference](../user-guide/api-reference.md) and [CLI Reference](../user-guide/cli-reference.md).

## Basic Python API Workflow

```python
import polars as pl
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',   # needs ID and SMILES columns
    oracle=None,                     # None = use the pool CSV as a CSV oracle
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01,
)
```

`oracle` has no default. Pass `None` to use the compound pool file as the oracle (benchmark mode), a CSV path, a `module.py:function` string, a Polars DataFrame, or an `Oracle` instance. Cycle 0 selects an initial random batch; cycles 1 to `n_cycles - 1` train the learner and select with `strategy` (default `'greedy'`). `batch_fraction` is relative to the original pool size, so every cycle labels the same number of compounds.

### Results dictionary

```python
compounds_df = results['compounds_df']
labeled = compounds_df.filter(pl.col('status') == 'labeled')
```

- `compounds_df`: final Polars DataFrame with columns `ID`, `SMILES`, `status`, `labeled_cycle`, `selected_cycle`, `pruned_cycle` and the target column. `status` is one of `unlabeled`, `labeled`, `pruned`, `failed`.
- `cycle_metrics`: list of per-cycle metric dictionaries.
- `aggregate_metrics`: metrics aggregated across cycles.
- `validation_result`: `ValidationResult` with the valid and invalid compounds.
- `output_dir`, `saved_files`, `prediction_files`: where results were written.
- `labeled_count`, `unlabeled_count`: compound counts at the end of the run.

Call `compounds_df.to_pandas()` if you need a pandas DataFrame.

### Output directory

Results go to `output_dir`, or to `learnm8_results_<timestamp>/` in the working directory if none is given:

```
learnm8_results_20250118_143022/
├── compounds_final.csv          # final master DataFrame (Parquet for large pools)
├── cycle_metrics.csv            # one row per cycle
├── selection_history.csv        # compounds selected in each cycle
├── validation_report.csv        # invalid compounds (only if any)
├── config.json                  # experiment configuration
├── learnm8.log                  # execution log
├── prediction_cycle_N.parquet   # per-cycle predictions
└── .cache/                      # feature cache (unless cache_dir is set)
```

`output_format` (`'auto'`, `'csv'`, `'parquet'`) selects the file format; `'auto'` uses Parquet for pools of 1M compounds or more. Pass `cache_dir` to share the feature cache between runs.

## Basic CLI Workflow

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10
```

Add a second positional argument to use an explicit oracle, either a CSV file or `module.py:function`:

```bash
learnm8 run library.csv scoring_module.py:calculate_affinity \
  --target binding_score --learner rf --featurizer morgan --n-cycles 20
```

`-o DIR` sets the output directory. Cycle schedules are covered in [Building Custom Cycles](building-custom-cycles.md); parameters stored in a file in [Configuration Files](../advanced/configuration-files.md).

## Choosing a learner

All learners are selected by name via `learner=` (CLI `--learner`). "Uncertainty" means the learner reports predictive uncertainty, which UCB, EI, PI, Thompson and entropy acquisition need.

| Learner | Uncertainty | Needs featurizer | GPU | Use when |
| --- | --- | --- | --- | --- |
| [`rf`](../components/learners/random-forest.md) | Yes (spread across trees) | Yes | No | General-purpose default |
| [`gp`](../components/learners/gaussian-process.md) | Yes | Yes | No | Small labeled sets; exact GP training scales as O(n^3) |
| [`xgb`](../components/learners/xgboost.md) | No | Yes | Optional (CUDA) | Gradient boosting with greedy or random acquisition |
| [`dt`](../components/learners/decision-tree.md) | Yes | Yes | No | Simple, interpretable baseline |
| [`lr`](../components/learners/linear-regression.md) | Yes | Yes | No | Linear baseline |
| [`gpu_gp`](../components/learners/gpu-gp.md) | Yes | Yes | Yes (GPyTorch) | Exact GP on a GPU |
| [`svgp`](../components/learners/svgp.md) | Yes | Yes | Yes (GPyTorch) | Sparse variational GP for larger labeled sets |
| [`rf_fil`](../components/learners/rf-fil.md) | Yes | Yes | Yes (RAPIDS cuML) | Random forest with GPU inference |
| [`ridge_cuml`](../components/learners/ridge-cuml.md) | Yes | Yes | Yes (RAPIDS cuML) | Ridge regression on a GPU |
| [`mlp`](../components/learners/mlp.md) | No | Yes | Optional (CUDA) | Feedforward network without uncertainty |
| [`mc_dropout`](../components/learners/mc-dropout.md) | Yes (MC dropout) | Yes | Optional (CUDA) | Neural network with uncertainty |
| [`fastprop`](../components/learners/fastprop.md) | No | Yes | Optional (CUDA) | Descriptor-based neural network |
| [`chemprop`](../components/learners/chemprop.md) | No | Optional (hybrid mode) | Optional (CUDA) | Graph neural network learning from SMILES |
| `lr_ensemble`, `xgb_ensemble`, `dt_ensemble`, `fastprop_ensemble` | Yes (member spread) | Yes | Follows members | Uncertainty from a model that has none, or more robust estimates; see [Ensembles](../components/learners/ensembles.md) |
| [`mixed_ensemble`](../components/learners/mixed-ensemble.md) | Yes | Yes | Optional | RF, linear and XGBoost members combined |
| [`chemprop_ensemble`](../components/learners/chemprop-ensemble.md) | Yes | Optional (hybrid mode) | Optional (CUDA) | Chemprop with uncertainty |

`gpu_gp` and `svgp` need GPyTorch and GAUCHE; `rf_fil` and `ridge_cuml` need RAPIDS cuML (a missing package raises `LearnerError` when the model trains). See the [learner overview](../components/learners/overview.md) for details.

## Choosing an acquisition strategy

Set `strategy=` for cycles 1 and later, and `initial_strategy=` (default `'random'`) for cycle 0.

```python
results = run_active_learning(
    compound_pool='compounds.csv', oracle=None, learner='gp',
    target_col='Activity', featurizer='morgan', n_cycles=10,
    strategy='ucb', acquisition_params={'beta': 2.0},
)
```

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --strategy ucb --acquisition-params '{"beta": 2.0}'
```

`greedy` and `topk` exploit predictions, `random` ignores the model, and `ucb`, `ei`, `pi`, `thompson` and `entropy` use uncertainty. See [Acquisition](../components/acquisition/overview.md) for each strategy and its parameters.

## Benchmark mode vs run mode

LearnM8 has two modes. There is no `mode` parameter: the mode is derived from the oracle type and cannot be overridden.

| | Benchmark mode | Run mode |
| --- | --- | --- |
| Oracle | CSV oracle (`CSVOracle`, a CSV path, a DataFrame, or `oracle=None`) | Python oracle (`PythonOracle`, `module.py:function`, or another non-CSV `Oracle`) |
| Ground truth | Known for the whole pool | Not known; measured only for selected compounds |
| Metrics | Basic metrics plus discovery and ranking metrics | Basic metrics only |

**Oracle derivation.** `oracle=None` loads the compound pool file as a CSV oracle, so `compound_pool` must be a file path (not a DataFrame) and contain the target column. A CSV path, a DataFrame or a `CSVOracle` instance also gives benchmark mode. Anything else (a `module.py:function` string, `PythonOracle`, or a custom `Oracle` subclass) gives run mode. See [Oracles](../components/oracles/overview.md).

**Metrics.** Both modes record the batch size, cumulative labeled count, average target value of the selected compounds, the best value found so far, uncertainty statistics (when the learner reports them) and the diversity metrics. Benchmark mode additionally computes, against the ground truth in the oracle:

- discovery metrics: top-K discovery rates (`top_10_discovery`, `top_100_discovery`, ... `top_10_pct_discovery`), `cumulative_ef`, `batch_hit_rate`, `batch_ef` (these three need an `Activity` column), and the cumulative and batch score-improvement ratios;
- ranking metrics on the still-unlabeled compounds: `unlabeled_top_100_overlap`, `unlabeled_top_1000_overlap`, `unlabeled_ef_1_0`, `unlabeled_ef_5_0` (also need `Activity`), and `unlabeled_spearman_correlation`;
- ground-truth reference values (`ground_truth_avg_score`, `ground_truth_ef_*`) and `selected_percentile`.

`aggregate_metrics` correspondingly gains `final_top_10_discovery`, `final_cumulative_ef`, `final_unlabeled_spearman` and similar keys only in benchmark mode.

**When to use which.**

- Benchmark mode: you already have measured values for the whole library and want to compare learners, featurizers or acquisition strategies.
- Run mode: measurements are produced on demand by a scoring function, docking program or assay. For assays that take days to months, where the oracle cannot answer inside one process, see [Lab Campaigns](lab-campaigns.md).

A typical workflow is to choose a configuration in benchmark mode on a related dataset, then reuse it with a Python oracle.

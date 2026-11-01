# Configuration Files

`learnm8 run --config experiment.yaml` reads run parameters from a YAML file so an experiment can be stored, shared and version-controlled. JSON with the same keys also works (`.json`).

## Schema

Keys are the `learnm8 run` options with dashes replaced by underscores, plus the `run_active_learning()` names `output_dir`, `smiles_column`, `id_column` and `large_features_ack`, which are accepted as aliases. A `cycles` list can be given as in the Python API.

```yaml
# experiment.yaml
compound_pool: compound_library.csv
oracle: scoring_module.py:calculate_affinity   # omit to use compound_pool as a CSV oracle
target_col: binding_score
learner: rf
featurizer: morgan
score_direction: lower
random_state: 123

cycles:
  - strategy: random
    n_cycles: 1
    batch_fraction: 0.02
  - strategy: ucb
    n_cycles: 5
    batch_fraction: 0.01
    acquisition_params:
      beta: 2.0
  - strategy: greedy
    n_cycles: 4
    batch_fraction: 0.01
    pruning_strategy: score
    pruning_params:
      pruning_fraction: 0.2

output_dir: results/ensemble_screening
cache_dir: .cache/shared
```

Run it with:

```bash
learnm8 run --config experiment.yaml
```

Each entry under `cycles` takes the fields of `CycleConfig` (`strategy`, `n_cycles`, `batch_fraction`, `acquisition_params`, `pruning_strategy`, `pruning_params`); see [Building Custom Cycles](../tutorials/building-custom-cycles.md). To use the simple API instead, replace `cycles` with `n_cycles`, `batch_fraction`, `strategy` and `initial_strategy`.

## Parameters

| Key | Default | Meaning |
| --- | --- | --- |
| `compound_pool` | required | Compound file (CSV, SDF or SMI) with ID and SMILES |
| `oracle` | none | CSV file or `module.py:function`; if omitted the pool is the oracle |
| `target_col` | required | Target property column |
| `learner` | `rf` | Learner name |
| `featurizer` | none | Featurizer name; required except for `chemprop` and `chemprop_ensemble` |
| `smiles_col`, `id_col` | auto-detect | Column names (aliases `smiles_column`, `id_column`) |
| `score_direction` | `higher` | `higher` or `lower` is better |
| `n_cycles` | `10` | Number of cycles (simple API) |
| `batch_fraction` | `0.01` | Fraction of the original pool per cycle (simple API) |
| `strategy` | `greedy` | Acquisition strategy for cycles 1+ (simple API) |
| `initial_strategy` | `random` | Strategy for cycle 0 (simple API) |
| `cycles` | none | List of cycle blocks; takes precedence over the simple-API keys |
| `acquisition_params` | none | Mapping of strategy parameters |
| `pruning_fraction`, `pruning_strategy`, `pruning_params` | none | See [Pruning](pruning.md) |
| `output` | timestamped directory | Output directory (alias `output_dir`) |
| `cache_dir` | `<output>/.cache` | Feature cache directory |
| `output_format` | `auto` | `auto`, `csv` or `parquet` |
| `random_state` | `42` | Random seed |
| `n_jobs`, `device` | `-1`, `auto` | CPU parallelism and compute device |
| `memory_safety_factor` | `0.7` | Fraction of available memory used for prediction batching |
| `allow_large_features` | `false` | Bypass the large-feature guard (alias `large_features_ack`) |
| `force_uncertainty` | `false` | Always compute uncertainty |
| `skip_validation` | `false` | Skip input validation |
| `streaming` | `auto` | `auto`, `always` or `never` |
| `quiet` | `false` | Suppress progress output |

Defaults are those of `learnm8 run`; see the [CLI Reference](../user-guide/cli-reference.md) for the full flag descriptions. Keys the CLI does not recognize are not rejected, so check spelling against this table. Chemprop fine-tuning ([Chemprop Fine-Tuning](chemprop-fine-tuning.md)) is not configurable from a file.

## Precedence

The file supplies defaults. A flag typed on the command line overrides the same key in the file, and LearnM8 prints a note for each key it ignores for that reason:

```bash
learnm8 run --config experiment.yaml --learner gp --output results/gp_run
```

`--cycles` and `--config` cannot be passed together on the command line; put `cycles` in the file instead.

## See also

- [CLI Reference](../user-guide/cli-reference.md)
- [Running Experiments](../tutorials/running-experiments.md)

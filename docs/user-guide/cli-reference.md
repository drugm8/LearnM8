# CLI Reference

The `learnm8` command has three subcommands: `run`, `validate` and `suggest`.

```bash
learnm8 --help
learnm8 --version
learnm8 run --help
```

| Subcommand | Purpose |
|------------|---------|
| `run` | Execute an active learning experiment |
| `validate` | Check a compound pool before running |
| `suggest` | Write the next batch for a slow lab assay, see [Lab Campaigns](../tutorials/lab-campaigns.md) |

## learnm8 run

```bash
learnm8 run [COMPOUND_POOL] [ORACLE] --target COLUMN --featurizer TYPE --learner MODEL [OPTIONS]
```

### Positional arguments

| Argument | Description |
|----------|-------------|
| `COMPOUND_POOL` | Compound file (`.csv`, `.sdf`, `.smi`) with ID and SMILES. May instead be set in `--config` |
| `ORACLE` | Oracle specification. If omitted, the pool file is used as a CSV oracle (benchmark mode) |

`ORACLE` is either a CSV file (benchmark mode) or `module.py:function` (run mode). See [Benchmark mode vs run mode](../tutorials/running-experiments.md#benchmark-mode-vs-run-mode).

### Core configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--target COLUMN`, `--target-column COLUMN` | none | Target property column. Required unless `target_col` is set in `--config` |
| `--featurizer TYPE` | none | Featurizer name, see [Featurizers](../components/featurizers/overview.md). Required unless the learner works on SMILES directly (`chemprop`, `chemprop_ensemble`) |
| `--learner MODEL` | `rf` | Learner name, see [Learners](../components/learners/overview.md) |
| `--score-direction {higher,lower}` | `higher` | Optimization direction |
| `--smiles-col NAME` | auto-detect | SMILES column in the pool |
| `--id-col NAME` | auto-detect | ID column in the pool |

`--learner` accepts all 19 learner names; `--featurizer` accepts every featurizer name. `gpu_gp` and `svgp` need GPyTorch (`ConfigurationError` when the learner is created without it) and `rf_fil` and `ridge_cuml` need RAPIDS cuML (`LearnerError` at train time). List the learner names with:

```bash
python -c "from learnm8.api import list_available_learners; print(list_available_learners())"
```

### Cycle control

`--cycles` and `--config` are mutually exclusive. Without either, the simple parameters below are used.

| Option | Default | Description |
|--------|---------|-------------|
| `--cycles SPEC` | none | Cycle specification string, e.g. `"random:0.02 greedy:0.01*5"` |
| `--config PATH` | none | YAML or JSON config file, see [Configuration Files](../advanced/configuration-files.md) |
| `--n-cycles N` | 10 | Number of cycles, including cycle 0 |
| `--batch-fraction F` | 0.01 | Fraction of the original pool selected per cycle |
| `--strategy {greedy,random,topk,ucb,ei,pi,thompson,entropy}` | `greedy` | Acquisition strategy for cycles 1 and later |

Cycle 0 is always a random batch; there is no option to change it. An uncertainty strategy (`ucb`, `ei`, `pi`, `thompson`, `entropy`) with a learner that has no uncertainty (`xgb`, `mlp`, `fastprop`, `chemprop`) is rejected before cycle 0.

A `--cycles` specification is whitespace-separated `strategy:fraction` entries. Append `*n` to repeat an entry `n` times: `greedy:0.01*5` is five greedy cycles at batch fraction 0.01.

```bash
learnm8 run compounds.csv --target Activity --featurizer morgan --learner rf \
  --cycles "random:0.02 ucb:0.01*5 greedy:0.01*3"
```

See [Building Custom Cycles](../tutorials/building-custom-cycles.md).

### Pruning

| Option | Default | Description |
|--------|---------|-------------|
| `--pruning-fraction F` | none | Fraction of the pool to prune per cycle (0.0 to 0.9) |
| `--pruning-strategy {score}` | `score` if `--pruning-fraction` is set | Pruning strategy |

See [Pruning](../advanced/pruning.md).

### Acquisition parameters

`--acquisition-params JSON` passes a JSON object to the acquisition function.

```bash
learnm8 run compounds.csv --target Activity --featurizer morgan --learner gp \
  --cycles "ucb:0.01*10" --acquisition-params '{"beta": 2.5}'
```

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `-o PATH`, `--output PATH` | auto-generated | Output directory |
| `--cache-dir PATH` | `{output}/.cache` | Feature cache directory. Share one across experiments on the same compounds |
| `--output-format {auto,csv,parquet}` | `auto` | `auto` writes Parquet for more than 1,000,000 rows, CSV otherwise |
| `--quiet` | off | Suppress progress output |

The output directory contains `compounds_final` (`.csv` or `.parquet`), `cycle_metrics.csv`, `selection_history` (`.csv` or `.parquet`), `config.json`, `validation_report.csv` (when invalid compounds were found), and one `prediction_cycle_N.parquet` per cycle.

### Resources and advanced options

| Option | Default | Description |
|--------|---------|-------------|
| `--n-jobs N` | -1 | Parallel CPU jobs. `-1` uses all cores |
| `--device DEVICE` | `auto` | `auto`, `cpu`, `cuda` or `cuda:N` |
| `--random-state SEED` | 42 | Random seed |
| `--memory-safety-factor F` | 0.7 | Fraction of available memory used for prediction batching |
| `--allow-large-features` | off | Bypass the guard that blocks descriptor-based featurizers on pools above 10M compounds |
| `--force-uncertainty` | off | Compute uncertainty every cycle even if the acquisition function does not need it. No effect for learners without uncertainty support |
| `--skip-validation` | off | Skip SMILES validation, duplicate-ID checks and target-type checks. Oracle-pool reconciliation still runs |
| `--enable-chemprop-fine-tuning` | off | Fine-tune Chemprop from the previous cycle's checkpoint instead of retraining from scratch (`chemprop` and `chemprop_ensemble` only). See [Chemprop Fine-Tuning](../advanced/chemprop-fine-tuning.md) |
| `--disable-molecular-similarity [METRIC ...]` | off | Skip the diversity metrics. With no names, all of them are skipped; with names (e.g. `scaffold_diversity_index_batch`), only those. Unknown names are a configuration error |
| `--streaming {auto,always,never}` | `auto` | Streaming predict-then-select path. `auto` uses it when a cycle is eligible (streamable strategy, an output directory, pruning none or `score`) and otherwise falls back to the in-memory path. `always` errors if a cycle is ineligible. `never` forces the in-memory path. Stochastic strategies (`thompson`, `random`, `topk`) draw different but equivalent samples under streaming |

## learnm8 validate

Validates SMILES with datamol and reports the valid fraction.

```bash
learnm8 validate COMPOUND_POOL [--smiles-col NAME] [--id-col NAME] [--n-jobs N] [-o DIR]
```

| Option | Default | Description |
|--------|---------|-------------|
| `COMPOUND_POOL` | required | Compound file (`.csv`, `.sdf`, `.smi`) |
| `--smiles-col NAME` | auto-detect | SMILES column |
| `--id-col NAME` | auto-detect | ID column |
| `--n-jobs N` | -1 | Parallel CPU jobs |
| `-o PATH`, `--output PATH` | none | Write `validation_report.csv` (invalid compounds with an `error` column) to this directory |

The console output lists the number of valid and invalid compounds and the first five errors.

```bash
learnm8 validate compounds.csv -o validation/
```

## learnm8 suggest

Trains on the results entered so far, writes the next batch file and exits. See the [Lab Campaigns tutorial](../tutorials/lab-campaigns.md) for the workflow and file format.

```bash
learnm8 suggest COMPOUND_POOL BATCH_DIR --target COLUMN [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `COMPOUND_POOL` | required | Compound file (`.csv`, `.sdf`, `.smi`) |
| `BATCH_DIR` | required | Folder holding the `batch_cycle_N.csv` files |
| `--target COLUMN`, `--target-column COLUMN` | required | Column the lab fills in |
| `--learner MODEL` | `rf` | Learner name |
| `--featurizer TYPE` | none | Featurizer name. Required unless the learner works on SMILES directly |
| `--batch-size N` | 96 | Compounds in the new batch |
| `--strategy NAME` | `greedy` | Acquisition strategy for rounds 1 and later. Round 0 is always random |
| `--score-direction {higher,lower}` | `higher` | Optimization direction |
| `--acquisition-params JSON` | none | JSON object of strategy parameters |
| `--smiles-col NAME`, `--id-col NAME` | auto-detect | Column names in the pool |
| `--cache-dir PATH` | `BATCH_DIR/.learnm8/cache` | Feature cache directory |
| `--n-jobs N` | -1 | Parallel CPU jobs |
| `--device DEVICE` | `auto` | `auto`, `cpu`, `cuda` or `cuda:N` |
| `--random-state N` | 42 | Random seed |
| `--pruning-fraction F` | 0.0 | Fraction of the unlabeled pool removed for good in each pruning round, 0.0 to 0.9. 0 = off. Capped so at least `--batch-size` unlabeled compounds remain; this cap is specific to `learnm8 suggest` (`learnm8 run` pruning does not cap). See [Pruning](../tutorials/lab-campaigns.md#pruning) |
| `--pruning-start N` | 3 | First round (cycle number) that may prune |

Settings are not stored between calls; pass the same flags every round.

## Examples

Benchmark experiment, pool file used as the oracle:

```bash
learnm8 run ADA.csv --target Activity --featurizer morgan --learner rf --n-cycles 15
```

Run mode with a Python oracle, a shared cache and an explicit output directory:

```bash
learnm8 run compound_library.csv docking_module.py:calculate_binding \
  --target binding_score --featurizer morgan --learner mc_dropout \
  --cycles "random:0.01 ucb:0.005*10 greedy:0.01*5" \
  --cache-dir .shared_cache -o screening_results/
```

Large library with score pruning:

```bash
learnm8 run large_library.csv oracle.py:score --target Activity \
  --featurizer descriptors --learner rf \
  --cycles "random:0.01 greedy:0.005*19" \
  --pruning-fraction 0.3 --pruning-strategy score -o large_screen/
```

Chemprop works directly on SMILES, so no featurizer is needed. Adding `--featurizer` switches it to hybrid mode, where the features are used as extra descriptors alongside the graph:

```bash
learnm8 run compounds.csv --target Activity --learner chemprop --n-cycles 10
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Data or runtime error: invalid input data (`ValidationError`), feature extraction, oracle, learner or persistence failure, malformed `--cycles` or `--acquisition-params`, any other unexpected exception |
| 2 | Configuration error (`ConfigurationError`), for example `suggest` re-run before any result is entered; compound pool file not found; missing `--target`; missing `--featurizer` for a learner that needs one; unknown strategy, learner or featurizer. Command-line usage errors reported by argparse (unknown flag, invalid choice, missing argument) also exit with 2 |

## See also

- [API Reference](api-reference.md)
- [Configuration Files](../advanced/configuration-files.md)
- [Running Experiments](../tutorials/running-experiments.md)

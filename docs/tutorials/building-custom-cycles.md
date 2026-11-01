# Building Custom Cycles

This tutorial explains how to control which acquisition strategy and batch size each cycle uses.

## Cycle numbering and batch size

Cycle 0 is the initial batch and trains no model. Cycles 1 and later train on the labeled compounds and select with the cycle's strategy. `n_cycles=10` runs cycles 0 to 9.

`batch_fraction` is always relative to the **original** pool size, not to the compounds still unlabeled. The batch size is `min(max(1, ceil(original_pool_size * batch_fraction)), len(selection_pool))`: at least one compound, capped at the number of compounds left to select from. For a 10,000-compound pool, `batch_fraction=0.01` selects 100 compounds per cycle.

## Simple API

The simple API uses one batch fraction for every cycle, one strategy for cycle 0 and one for cycles 1 and later:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    initial_strategy='random',   # cycle 0 (default)
    strategy='greedy',           # cycles 1+ (default)
    n_cycles=10,
    batch_fraction=0.01,
)
```

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --initial-strategy random --strategy greedy --n-cycles 10 --batch-fraction 0.01
```

## CycleConfig

For different strategies, batch sizes or parameters per phase, pass a list of `CycleConfig` objects as `cycles`. When `cycles` is given, `n_cycles`, `batch_fraction`, `strategy` and `initial_strategy` are not used.

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('ucb', n_cycles=3, batch_fraction=0.01,
                    acquisition_params={'beta': 3.0}),
        CycleConfig('greedy', n_cycles=4, batch_fraction=0.005,
                    pruning_strategy='score',
                    pruning_params={'pruning_fraction': 0.2}),
    ],
)
```

This runs 8 cycles: one random cycle at 2%, three UCB cycles at 1% and four greedy cycles at 0.5%. Fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `strategy` | required | Acquisition strategy name |
| `n_cycles` | `1` | Number of consecutive cycles with this configuration |
| `batch_fraction` | required | Fraction of the original pool to select per cycle, in (0, 1] |
| `pruning_strategy` | `None` | Pruning strategy (`'score'`); see [Pruning](../advanced/pruning.md) |
| `pruning_params` | `None` | Pruning parameters, e.g. `{'pruning_fraction': 0.3}` |
| `acquisition_params` | `None` | Strategy parameters, e.g. `{'beta': 1.5}` for UCB |

Top-level `pruning_strategy`, `pruning_fraction`, `pruning_params` and `acquisition_params` act as defaults for every `CycleConfig` that does not set its own. A value set on a `CycleConfig` wins.

## CLI cycle specification

The CLI takes a space-separated string of `strategy:fraction` or `strategy:fraction*count` entries, where `count` repeats the entry:

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --cycles "random:0.02 ucb:0.01*3 greedy:0.005*4"
```

This is the same schedule as the `CycleConfig` example above without the per-block parameters. The string form carries no per-cycle parameters; `--acquisition-params` (a JSON string) and `--pruning-fraction` apply to every cycle that uses them. For per-cycle parameters, use a [configuration file](../advanced/configuration-files.md) with a `cycles` list. `--cycles` cannot be combined with `--config` on the command line, and `--n-cycles`, `--batch-fraction` and `--strategy` are ignored when `--cycles` is given.

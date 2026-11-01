# Core Concepts

This page introduces the ideas behind LearnM8's approach to active learning for molecular screening.

## Active learning

Active learning lets the algorithm choose which data points to label next instead of labeling random ones. For molecular screening, that means choosing which compounds to measure (synthesize, dock, assay) so that a limited measurement budget goes to the most useful compounds.

### The cycle

Learning proceeds in iterative cycles, each with four phases:

| Phase | Action | Input | Output |
| --- | --- | --- | --- |
| **Train** | Fit the learner | Labeled compounds and their features | Trained model |
| **Predict** | Score the unlabeled pool | Unlabeled compounds and features | Predictions, and uncertainties if needed |
| **Acquire** | Select a batch | Predictions and uncertainties | Selected compound IDs |
| **Measure** | Ask the oracle | Selected compound IDs | Measured values |

**Cycle 0** is a random initial batch and trains no model. Cycles 1 and later run the full train, predict, acquire, measure loop. `n_cycles=10` runs cycles 0 to 9.

### Pool-based

LearnM8 is pool-based: you start with a fixed pool of candidate compounds, and each compound has a `status` in the master DataFrame:

- `unlabeled`: not yet measured (initial state)
- `labeled`: measured
- `pruned`: removed from the pool by [pruning](../advanced/pruning.md)
- `failed`: the oracle returned no usable value

### Batches

Compounds are selected in batches. `batch_fraction` is a fraction of the **original** pool size, so every cycle with the same fraction selects the same number of compounds (at least one, and no more than remain). See [Building Custom Cycles](../tutorials/building-custom-cycles.md).

## Functional API

Everything runs through one function call; there are no experiment objects to manage:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
)
```

For different strategies or batch sizes per phase, pass a list of `CycleConfig` objects instead of `n_cycles`:

```python
from learnm8 import CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01),
        CycleConfig('ucb', n_cycles=4, batch_fraction=0.01),
    ],
)
```

## Components

### Compound pool

All candidate molecules. It needs a unique `ID` column and a `SMILES` column. SMILES are validated before the run starts; you can check a file ahead of time with:

```bash
learnm8 validate compounds.csv
```

### Oracle

The oracle provides measurements for the compounds you select.

- A **CSV oracle** reads known values from a CSV file. Use it to test strategies on data you already have (benchmark mode).
- A **Python oracle** calls your function, for example a docking run or an assay result lookup (run mode).

The oracle type determines the operating mode; see [Benchmark mode vs run mode](../tutorials/running-experiments.md#benchmark-mode-vs-run-mode). For assays that take days or weeks, see [Lab Campaigns](../tutorials/lab-campaigns.md). Details: [Oracles](../components/oracles/overview.md).

### Learner

The learner is the model that predicts the target from molecular features. Learners are chosen by name (`learner='rf'`) and are compared in [Running Experiments](../tutorials/running-experiments.md#choosing-a-learner); the full list is in [Learners](../components/learners/overview.md).

A learner can report predictive uncertainty or not. Uncertainty is computed only when the acquisition strategy needs it (or when you pass `force_uncertainty=True`).

`chemprop` and `chemprop_ensemble` learn directly from SMILES and do not need a featurizer; passing one switches them to hybrid mode, which adds the features to the graph representation.

### Acquisition function

The acquisition function decides which unlabeled compounds to measure next.

| Strategy | Uses uncertainty | Idea |
| --- | --- | --- |
| `random` | No | Select at random, ignoring the model |
| `greedy` | No | Select the best predicted values |
| `topk` | No | Draw from the top-scoring compounds |
| `ucb` | Yes | Prediction plus a multiple of the uncertainty |
| `ei` | Yes | Expected improvement over the best value so far |
| `pi` | Yes | Probability of improving on the best value so far |
| `thompson` | Yes | Sample from the predictive distribution |
| `entropy` | Yes | Prefer high-uncertainty compounds |

See [Acquisition](../components/acquisition/overview.md) for details and parameters.

### Featurizer

The featurizer converts SMILES into numeric vectors for the learner (`featurizer='morgan'`). 39 featurizers are available, covering circular, key-based, topological and hashed fingerprints, descriptors, and 3D representations; see [Featurizers](../components/featurizers/overview.md). Features are cached on disk (HDF5), by default in `<output_dir>/.cache`, and can be shared between runs with `cache_dir`.

## Tracking progress

Each cycle appends a metrics dictionary to `results['cycle_metrics']`, also written to `cycle_metrics.csv`:

```python
for m in results['cycle_metrics']:
    print(m['cycle'], m['cumulative_labeled'], m['best_so_far'], m['avg_score_selected'])
```

Benchmark mode adds discovery and ranking metrics computed against the known values. See the [metrics section](../tutorials/running-experiments.md#benchmark-mode-vs-run-mode) of Running Experiments.

## Next steps

- [Quickstart](quickstart.md)
- [Building Custom Cycles](../tutorials/building-custom-cycles.md)
- [Running Experiments](../tutorials/running-experiments.md)

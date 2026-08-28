<div align="center" markdown>
  ![LearnM8](assets/learnm8_dark_full.png#only-light){ width="420" }
  ![LearnM8](assets/learnm8_white_full.png#only-dark){ width="420" }
</div>

# LearnM8: Active Learning Framework for Molecular Screening

LearnM8 is an active learning framework for molecular property prediction and compound screening. Built with a pure functional architecture, it enables researchers to efficiently explore chemical space through intelligent compound selection, uncertainty-guided decision-making, and state-of-the-art machine learning models.

The framework addresses a fundamental challenge in computational chemistry: how to select the most informative molecules for experimental testing when resources are limited. Through iterative cycles of prediction, selection, and measurement, LearnM8 helps researchers identify promising compounds faster than traditional screening approaches while minimizing experimental costs.

## Key Features

**Comprehensive model suite**

- 21 models spanning scikit-learn, PyTorch, Chemprop GNNs and ensembles — see [Learners](components/learners/overview.md)
- Uncertainty quantification for 17 of the 21
- Optional GPU acceleration via GPyTorch and RAPIDS cuML

**Rich acquisition strategies**

- 9 strategies: basic, uncertainty-based and optimization-based — see [Acquisition](components/acquisition/overview.md)
- Uncertainty computation is skipped automatically when the active strategy does not need it

**Flexible featurization**

- 39 registered featurizers, 2D and 3D — see [Featurizers](components/featurizers/available-featurizers.md)
- HDF5 feature cache gives roughly 100× speedup on repeated extraction

**Built for scale**

- Automatic parallelization for feature extraction
- Vectorized Polars operations throughout
- Streaming parquet output keeps RAM constant on pools beyond 1M compounds

**Two operating modes**

- **Benchmark mode** (CSV oracle): full discovery, enrichment and ranking metrics against ground truth
- **Production mode** (Python oracle): drives real assays or docking software
- Auto-detected from the oracle type; no manual flag required

## Usage Examples

### Benchmark mode

Ground truth lives in the same CSV, so passing `oracle=None` lets LearnM8 detect it automatically.

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)

print(f"Best compound: {results['aggregate_metrics']['best_compound_value']:.3f}")
print(f"Top-10 discovery rate: {results['aggregate_metrics']['final_top_10_discovery']:.1%}")
```

The same run from the command line:

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan --n-cycles 10
```

### Production mode

Point LearnM8 at your own scoring function and it will call it each cycle.

```python
from learnm8 import run_active_learning
from learnm8.oracles import PythonOracle

oracle = PythonOracle(
    module_path='scoring_module.py',
    function_name='calculate_binding_affinity'
)

results = run_active_learning(
    compound_pool='compound_library.csv',
    oracle=oracle,
    learner='mc_dropout',
    target_col='binding_affinity',
    featurizer='morgan',
    n_cycles=20,
    batch_fraction=0.005
)
```

### Custom cycle schedules

Vary the strategy, the batch size and the pruning per phase — explore first, then exploit.

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    target_col='Activity',
    learner='rf_ensemble',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('ucb', n_cycles=5, batch_fraction=0.01,
                    acquisition_params={'beta': 3.0}),
        CycleConfig('greedy', n_cycles=4, batch_fraction=0.01,
                    pruning_strategy='score',
                    pruning_params={'pruning_fraction': 0.3}),
    ]
)
```

`acquisition_params` passes strategy-specific arguments through to the acquisition function. Here `beta` raises UCB's exploration weight above its default of `2.0`, since `prediction + beta * uncertainty` favours uncertain compounds more as `beta` grows.

`pruning_params` works the same way for the pruner. Score-based pruning runs before selection on every cycle of the phase, discarding the worst-predicted 30% of the compounds that are _still unlabeled_, so the design space shrinks geometrically as confidence in the model grows.

## Next Steps

**Getting started**

- **[Installation Guide](getting-started/installation.md)**: Set up LearnM8 with conda and optional dependencies
- **[Quickstart Tutorial](getting-started/quickstart.md)**: Run your first experiment
- **[Core Concepts](getting-started/concepts.md)**: Understand active learning and LearnM8's design

**Components**

- **[Learners](components/learners/overview.md)**: All 21 models, with a decision matrix by dataset size and use case
- **[Acquisition Functions](components/acquisition/overview.md)**: All 9 strategies and when to reach for each
- **[Featurizers](components/featurizers/available-featurizers.md)**: All 39 featurizers with storage formats and dimensions
- **[Oracles](components/oracles/overview.md)**: Benchmark and production measurement sources

**Reference**

- **[CLI Reference](user-guide/cli-reference.md)**: Full command-line documentation
- **[API Reference](user-guide/api-reference.md)**: Complete Python API reference

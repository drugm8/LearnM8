# LearnM8: Active Learning Framework for Molecular Screening

LearnM8 is a comprehensive active learning framework designed for molecular property prediction and compound screening. Built with a pure functional architecture, it enables researchers to efficiently explore chemical space through intelligent compound selection, uncertainty-guided decision-making, and state-of-the-art machine learning models.

The framework addresses a fundamental challenge in computational chemistry: how to select the most informative molecules for experimental testing when resources are limited. Through iterative cycles of prediction, selection, and measurement, LearnM8 helps researchers identify promising compounds faster than traditional screening approaches while minimizing experimental costs.

LearnM8 combines modern software engineering practices with cutting-edge molecular machine learning. The framework provides both benchmark capabilities for algorithm development and production-ready tools for real-world screening campaigns. Whether you're testing new active learning strategies on known datasets or deploying predictive models in drug discovery pipelines, LearnM8 offers the flexibility and performance you need.

## Key Features

**Pure Functional Architecture**

- Simple `run_active_learning()` function as the main entry point
- No complex state management or hidden side effects
- Explicit cycle control through `CycleConfig` dataclass
- Dependency injection for all components
- Composition over inheritance for maximum flexibility

**Comprehensive Model Suite**

- 15+ machine learning models with uncertainty quantification
- Scikit-learn models (Random Forest, Gaussian Process, XGBoost, Decision Tree, Linear Regression)
- PyTorch neural networks (MLP, MC Dropout, FastProp)
- Graph neural networks (Chemprop) that work directly with SMILES
- Ensemble methods for robust uncertainty estimation
- GPU support for accelerated training

**Rich Acquisition Strategies**

- 11+ selection strategies from basic to sophisticated
- Exploitation strategies (greedy, top-k)
- Exploration strategies (random, entropy, Thompson sampling)
- Uncertainty-based methods (UCB, Expected Improvement, Probability of Improvement)
- Diversity-focused selection (BitBIRCH, simulated annealing)
- Custom acquisition function support

**Performance Optimizations**

- HDF5-based feature caching (100x speedup on repeated extraction)
- Automatic parallelization (5-10x faster feature extraction)
- Vectorized Polars operations (10x faster DataFrame updates)
- Early validation with datamol (catch invalid SMILES before experiments)
- Persistent cache directory for cross-experiment reuse

**Two Operating Modes**

- Benchmark mode: CSVOracle with full dataset prediction for accurate metrics
- Production mode: PythonOracle with unlabeled pool prediction for efficiency
- Automatic mode detection based on oracle type
- Support for custom scoring functions

**Molecular-Specific Design**

- RDKit integration for molecular featurization
- Multiple fingerprint types (Morgan, MACCS, ECFP6)
- Mordred descriptors for comprehensive chemical features
- SMILES-based validation before experiments
- Chemical scaffold and clustering-based diversity

## Quick Example

Get started with LearnM8 in seconds using the command-line interface:

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10
```

Or use the Python API for more control:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer_type='morgan',
    n_cycles=10
)
```

## Architecture Highlights

**Explicit Cycle Control**

LearnM8 uses explicit cycle specifications instead of hidden hyperparameters. Users specify exactly what happens in each cycle:

```python
from learnm8 import CycleConfig

cycles = [
    CycleConfig('random', n_cycles=1, batch_fraction=0.02),
    CycleConfig('ucb', n_cycles=5, batch_fraction=0.01),
    CycleConfig('diverse', n_cycles=4, batch_fraction=0.01)
]
```

**Dependency Injection**

Components receive dependencies explicitly rather than managing hidden state:

```python
from learnm8.learners import GaussianProcessLearner
from learnm8.oracles import PythonOracle

learner = GaussianProcessLearner(alpha=1e-6)
oracle = PythonOracle('scoring_module.py', 'calculate_affinity')

results = run_active_learning(
    compound_pool=df,
    oracle=oracle,
    learner=learner,
    target_col='binding_score',
    featurizer_type='morgan'
)
```

**Functional Data Flow**

Every cycle returns metrics and updated DataFrames without mutating global state:

```python
results = run_active_learning(...)

compounds_df = results['compounds_df']
cycle_metrics = results['cycle_metrics']
validation_result = results['validation_result']
```

## Component Overview

**Learners**: 15+ models including Random Forest, Gaussian Process, XGBoost, neural networks, graph neural networks, and ensembles

**Acquisition Functions**: 11+ strategies including greedy, UCB, Expected Improvement, Thompson sampling, BitBIRCH clustering, and simulated annealing

**Featurizers**: Morgan fingerprints, MACCS keys, ECFP6, and Mordred descriptors with automatic caching

**Oracles**: CSVOracle for benchmarking and PythonOracle for custom scoring functions

**Pruning**: Score-based pruning for design space reduction in large compound libraries

## Next Steps

- **[Installation Guide](getting-started/installation.md)**: Set up LearnM8 with conda and optional dependencies
- **[Quickstart Tutorial](getting-started/quickstart.md)**: Run your first experiment in under 5 minutes
- **[Core Concepts](getting-started/concepts.md)**: Understand active learning fundamentals and LearnM8 architecture
- **[CLI Reference](user-guide/cli-reference.md)**: Complete command-line documentation
- **[API Reference](user-guide/api-reference.md)**: Detailed Python API documentation

## Links

- **GitHub Repository**: [https://github.com/volkamerlab/LearnM8](https://github.com/volkamerlab/LearnM8)
- **Issue Tracker**: [https://github.com/volkamerlab/LearnM8/issues](https://github.com/volkamerlab/LearnM8/issues)
- **License**: See [LICENSE](https://github.com/volkamerlab/LearnM8/blob/main/LICENSE)

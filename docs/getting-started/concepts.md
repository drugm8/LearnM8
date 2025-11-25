# Core Concepts

This guide introduces the fundamental concepts behind LearnM8's approach to active learning for molecular screening.

## Active Learning Fundamentals

### What is Active Learning?

Active learning is a machine learning paradigm where the algorithm strategically selects which data points to label next, rather than passively accepting random samples. For molecular screening, this means intelligently choosing which compounds to synthesize and test, dramatically reducing the experimental burden while maximizing the information gained.

In traditional screening campaigns, you might synthesize and test thousands of compounds at random. With active learning, you can achieve similar or better results by testing far fewer compounds—often reducing the number of experiments by 10-100x.

### The Cycle-Based Approach

Active learning operates in iterative cycles, each consisting of four phases:

1. **Train**: Build a predictive model using currently labeled compounds
2. **Predict**: Score the entire unlabeled compound pool
3. **Acquire**: Select the most informative compounds based on predictions and uncertainty
4. **Measure**: Obtain experimental/computational measurements for selected compounds

Each cycle refines the model's understanding of the chemical space, focusing resources on the most promising or informative regions.

### Pool-Based Active Learning

LearnM8 implements **pool-based active learning**, where you start with a fixed pool of candidate compounds. The framework maintains a master dataset tracking each compound's status:

- **Unlabeled**: Compounds awaiting evaluation (initial state)
- **Acquired**: Compounds selected for measurement
- **Labeled**: Compounds with known measurements

This contrasts with stream-based active learning, where compounds arrive one at a time. Pool-based selection allows global optimization over the entire candidate space, enabling sophisticated acquisition strategies that consider the entire chemical landscape.

## LearnM8 Architecture

### Pure Functional Design

LearnM8 embraces a **pure functional architecture** that eliminates complex state management:

```python
# Simple, explicit function call
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)
```

Rather than managing experiment objects and internal state, you provide inputs to a single function and receive comprehensive results. This approach offers several advantages:

- **Simplicity**: No class hierarchies or state machines to understand
- **Reproducibility**: Identical inputs produce identical outputs
- **Transparency**: All configuration is explicit in the function call
- **Testability**: Pure functions are easier to test and debug

### Explicit Cycle Control

LearnM8 gives you complete control over what happens in each cycle through the `cycles` parameter:

```python
from learnm8 import CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01),
        CycleConfig('diverse', n_cycles=4, batch_fraction=0.01)
    ]
)
```

Each `CycleConfig` specifies:
- Which acquisition strategy to use
- How many compounds to select (batch size)
- Optional pruning and strategy parameters

This explicit specification eliminates hidden defaults and gives you precise control over the exploration/exploitation tradeoff.

### Dependency Injection

Components receive their dependencies explicitly rather than maintaining internal state. For example, learners don't handle feature extraction—they receive pre-computed features:

```python
class MyLearner(Learner):
    def train(self, features: np.ndarray, targets: np.ndarray, smiles=None):
        # Receives features directly, doesn't need to know about featurization
        self.model.fit(features, targets)

    def predict(self, features: np.ndarray, smiles=None):
        return self.model.predict(features), None
```

This pattern makes components:
- **Modular**: Easy to swap implementations
- **Testable**: Dependencies can be mocked
- **Reusable**: Components work in different contexts
- **Maintainable**: Clear boundaries between concerns

## Core Components

### Compound Pool

The compound pool is your starting dataset—all candidate molecules for screening. At minimum, it requires:

- **ID**: Unique identifier for each compound
- **SMILES**: Chemical structure in SMILES notation

Example CSV format:

```csv
ID,SMILES
CHEMBL1,CCO
CHEMBL2,CCC
CHEMBL3,CCN
```

LearnM8 validates SMILES strings using datamol before starting experiments, catching errors early:

```bash
learnm8 validate compounds.csv
```

### Oracle (Measurement Function)

The oracle provides ground truth measurements for compounds. LearnM8 supports two oracle types corresponding to different use cases:

**CSVOracle (Benchmark Mode)**:
```python
# Oracle is a CSV file with known measurements
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',  # Contains ID, Activity columns
    target_col='Activity',
    learner='gp',
    featurizer='morgan'
)
```

Use this when you have complete ground truth data and want to benchmark active learning strategies against random selection.

**PythonOracle (Production Mode)**:
```python
# Oracle is a custom function (e.g., docking, experimental measurement)
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='scoring_module.py:calculate_binding',
    target_col='binding_score',
    learner='gp',
    featurizer='morgan'
)
```

Use this for real screening campaigns where measurements are expensive and you want to minimize experimental costs.

### Learner (ML Model)

The learner is the predictive model that learns structure-activity relationships from labeled compounds. LearnM8 provides 15+ models across multiple categories:

**Scikit-learn Models**:
- `RandomForestLearner`: Fast baseline, good for most tasks
- `GaussianProcessLearner`: Best uncertainty quantification
- `XGBoostLearner`: High-performance gradient boosting

**PyTorch Neural Networks**:
- `MLPLearner`: Standard feedforward network
- `MCDropoutLearner`: Neural network with uncertainty via dropout

**Graph Neural Networks**:
- `ChempropLearner`: State-of-the-art, works directly with SMILES

**Ensemble Models**:
- `EnsembleLearner`: Combines multiple models for robust uncertainty

Key considerations when choosing a learner:

| Dataset Size | Uncertainty Needed | Best Choice |
|--------------|-------------------|-------------|
| < 1,000 | Yes | `GaussianProcessLearner` or `ChempropLearner` |
| < 1,000 | No | `RandomForestLearner` |
| 1,000-10,000 | Yes | `EnsembleLearner` or `ChempropLearner` |
| 1,000-10,000 | No | `XGBoostLearner` |
| > 10,000 | Yes | `ChempropLearner` with ensemble |
| > 10,000 | No | `XGBoostLearner` or `MLPLearner` |

### Acquisition Function (Selection Strategy)

The acquisition function determines which compounds to measure next based on model predictions and uncertainty. LearnM8 offers several strategy categories:

**Exploitation (Greedy)**:
```python
# Select compounds with highest predicted values
cycles=[('greedy', 0.01)]
```

**Exploration (Diverse)**:
```python
# Select chemically diverse compounds
cycles=[('bitbirch', 0.01)]
```

**Balanced (Uncertainty-Based)**:
```python
# Select based on prediction + uncertainty tradeoff
cycles=[('ucb', 0.01)]  # Upper Confidence Bound
```

Common strategies:

| Strategy | Type | Requires Uncertainty | When to Use |
|----------|------|---------------------|-------------|
| `random` | Baseline | No | Initial sampling, control experiments |
| `greedy` | Exploitation | No | When you want highest predicted values |
| `ucb` | Balanced | Yes | General-purpose active learning |
| `ei` | Balanced | Yes | Optimization campaigns |
| `thompson` | Balanced | Yes | Stochastic exploration |
| `bitbirch` | Diversity | No | Large chemical libraries (1M+ compounds) |

### Featurizer (Molecular Representation)

The featurizer converts SMILES strings into numerical vectors that machine learning models can process. LearnM8 provides several molecular representations:

| Featurizer | Type | Dimensions | Best For |
|------------|------|------------|----------|
| `morgan` | Fingerprint | 2048 | General-purpose molecular similarity |
| `maccs` | Fingerprint | 167 | Structural keys, fast computation |
| `ecfp6` | Fingerprint | 2048 | Larger molecular contexts |
| `descriptors` | Numerical | ~200 | Physicochemical properties |

**Note**: Graph neural networks (Chemprop) work directly with SMILES and don't require a featurizer:

```python
# No featurizer needed for Chemprop
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',  # Works directly with SMILES
    target_col='Activity'
)

# Optional: Hybrid mode combines graph + descriptors
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',
    featurizer='descriptors',  # Additional features
    target_col='Activity'
)
```

## The Active Learning Cycle

Each active learning cycle follows this sequence:

| Phase | Action | Input | Output |
|-------|--------|-------|--------|
| **Train** | Build predictive model | Labeled compounds + features | Trained model |
| **Predict** | Score compound pool | Unlabeled compounds + features | Predictions + uncertainties |
| **Acquire** | Select informative compounds | Predictions + uncertainties | Acquired compound IDs |
| **Measure** | Obtain ground truth | Acquired compound IDs | Measured values |

### Cycle 0: Initialization

The first cycle (cycle 0) uses random sampling to create an initial training set. This provides the model with an unbiased view of the chemical space before applying more sophisticated acquisition strategies.

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.02),    # Cycle 0: Initial random sample (2%)
        ('greedy', 0.01),    # Cycle 1: Greedy selection (1%)
        ('greedy', 0.01),    # Cycle 2: Greedy selection (1%)
        ('diverse', 0.01)    # Cycle 3: Diverse exploration (1%)
    ]
)
```

### Batch-Based Selection

LearnM8 selects compounds in batches rather than one at a time. Batch size is specified as a fraction of the original pool size:

```python
# batch_fraction=0.01 means select 1% of original pool each cycle
# For 10,000 compound pool: 100 compounds per cycle
cycles=[('greedy', 0.01)]
```

Batch selection enables:
- **Parallelization**: Measure multiple compounds simultaneously
- **Efficiency**: Amortize setup costs across multiple measurements
- **Diversity**: Select diverse compounds within each batch

### Tracking Progress

Each cycle generates comprehensive metrics tracking model performance and selection behavior:

```python
results = run_active_learning(...)

# Access per-cycle metrics
for cycle_data in results['cycle_metrics']:
    print(f"Cycle {cycle_data['cycle']}: "
          f"Strategy={cycle_data['strategy']}, "
          f"Labeled={cycle_data['n_labeled']}, "
          f"R²={cycle_data['r2_score']:.3f}")
```

## Two Operating Modes

LearnM8 adapts its behavior based on your use case:

### Benchmark Mode

**When to use**: Testing algorithms, comparing acquisition strategies, research

**Characteristics**:
- Oracle is a CSV file with complete ground truth
- Model predicts on the full dataset each cycle
- Enables accurate enrichment factor calculations
- Higher computational cost (predicts all compounds)

**Example**:
```python
results = run_active_learning(
    compound_pool='benchmark_compounds.csv',
    oracle='benchmark_compounds.csv',  # Same file = benchmark mode
    target_col='Activity',
    learner='gp',
    featurizer='morgan'
)
```

**Use benchmark mode to**:
- Compare different acquisition strategies
- Evaluate model performance over cycles
- Validate active learning algorithms
- Generate publication-quality enrichment curves

### Production Mode

**When to use**: Real screening campaigns, expensive experiments

**Characteristics**:
- Oracle is a Python function performing measurements
- Model predicts only on unlabeled compounds
- Minimizes computational cost
- Designed for expensive oracles (synthesis, docking, etc.)

**Example**:
```python
results = run_active_learning(
    compound_pool='virtual_library.csv',
    oracle='docking.py:score_binding',  # Custom function
    target_col='binding_affinity',
    learner='ensemble',
    featurizer='morgan'
)
```

**Use production mode to**:
- Guide experimental synthesis campaigns
- Optimize computational screening (docking, MD)
- Minimize expensive measurements
- Maximize screening efficiency

### Mode Auto-Detection

LearnM8 automatically detects the mode based on the oracle type. You can override this if needed:

```python
# Explicit mode specification
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    mode='benchmark',  # Force benchmark mode
    learner='gp',
    target_col='Activity',
    featurizer='morgan'
)
```

## Next Steps

Now that you understand LearnM8's core concepts, you can:

- **Run your first experiment**: Follow the [Quickstart Guide](quickstart.md) for a hands-on introduction
- **Learn cycle specifications**: See [Building Custom Cycles](../tutorials/building-custom-cycles.md) for advanced cycle control
- **Explore components**: Browse the [Components](../components/learners/overview.md) section for detailed component documentation
- **Understand modes deeply**: Read [Benchmark vs Production](../tutorials/benchmark-vs-production.md) for mode-specific guidance

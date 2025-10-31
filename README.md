# LearnM8: Active Learning Framework for Molecular Screening

LearnM8 is a comprehensive active learning framework designed for molecular property prediction and compound screening. It provides a pure functional API with sophisticated machine learning models, acquisition strategies, and design space pruning techniques specifically tailored for chemical space exploration.

## 🚀 Quick Start

```bash
# Install dependencies
conda env create -f environment.yml
conda activate learnm8
pip install -e .

# Validate compounds before running
learnm8 validate compounds.csv --featurizer morgan

# Validate with custom cache directory
learnm8 validate compounds.csv --featurizer morgan --cache-dir .shared_cache

# Basic usage (auto-detect oracle from compound_pool)
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10

# With custom cache directory (reuse cache across runs)
learnm8 run compounds.csv compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10 --cache-dir .shared_cache

# List available components
learnm8 list learners
learnm8 list acquisition
learnm8 list schedules

# Use predefined schedules
learnm8 run compounds.csv compounds.csv --target Activity --learner ensemble --featurizer morgan --schedule intensive

# Config file support
learnm8 run --config experiment.yaml
```

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Machine Learning Models](#machine-learning-models)
- [Acquisition Strategies](#acquisition-strategies)
- [Pruning Methods](#pruning-methods)
- [Molecular Featurizers](#molecular-featurizers)
- [Compatibility Matrix](#compatibility-matrix)
- [CLI Usage](#cli-usage)
- [API Reference](#api-reference)
- [Installation](#installation)
- [Examples](#examples)

## 🏗️ Architecture Overview

LearnM8 features a **modern modular architecture** with early validation and performance optimizations:

```
┌──────────────────────────────────────────────────────────────┐
│                     LearnM8 API (api.py)                     │
│              Main entry point: run_active_learning()         │
└──────────────────────────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────────┐
│   Validation    │  │    Config   │  │  Initialization │
│ validate_pool() │  │ CycleConfig │  │ init_master_df()│
└─────────────────┘  └─────────────┘  └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                   ┌───────────────────┐
                   │  Cycle Execution  │
                   │  execute_cycle()  │
                   └───────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐  ┌──────────────┐  ┌────────────────┐
│  DataFrame Ops  │  │ Persistence  │  │    Features    │
│  vectorized ops │  │ save_results │  │ extract_features│
└─────────────────┘  └──────────────┘  └────────────────┘
                                                │
                                                ▼
                                        ┌───────────────┐
                                        │  HDF5 Cache   │
                                        │  100x faster  │
                                        └───────────────┘
```

### Core Principles

1. **Early Validation**: Catch invalid compounds before running experiments
2. **Performance First**: 5-100x improvements through caching and parallelization
3. **Modular Design**: 7 core modules with clear separation of concerns
4. **Functional API**: Simple `run_active_learning()` function as main entry point
5. **Flexible Configuration**: CycleConfig dataclass for explicit cycle control

## 🤖 Machine Learning Models

### Scikit-learn Based Models

| Model | Class | Uncertainty | GPU | Best Use Case |
|-------|-------|-------------|-----|---------------|
| Random Forest | `RandomForestLearner` | ❌ | ❌ | Fast prototyping, baseline |
| Gaussian Process | `GaussianProcessLearner` | ✅ | ❌ | Small datasets, best uncertainty |
| XGBoost | `XGBoostLearner` | ❌ | ❌ | Large datasets, high performance |
| Decision Tree | `DecisionTreeLearner` | ❌ | ❌ | Interpretable models |
| Linear Regression | `LinearRegressionLearner` | ❌ | ❌ | Simple linear relations |
| Advanced RF | `AdvancedRandomForestLearner` | ✅ | ❌ | RF with uncertainty estimation |

### PyTorch Based Models

| Model | Class | Uncertainty | GPU | Best Use Case |
|-------|-------|-------------|-----|---------------|
| Multi-Layer Perceptron | `MLPLearner` | ❌ | ✅ | Complex non-linear patterns |
| MC Dropout | `MCDropoutLearner` | ✅ | ✅ | Neural nets with uncertainty |

### Ensemble Models

| Model | Class | Uncertainty | Components | Best Use Case |
|-------|-------|-------------|------------|---------------|
| General Ensemble | `EnsembleLearner` | ✅ | Mixed models | Production, best overall performance |
| RF Ensemble | `RFEnsemble` | ✅ | Random Forest variants | Forest-based ensemble |
| Linear Ensemble | `LREnsemble` | ✅ | Linear model variants | Linear relationships |
| XGBoost Ensemble | `XGBEnsemble` | ✅ | XGBoost variants | Gradient boosting ensemble |
| Decision Tree Ensemble | `DTEnsemble` | ✅ | Decision tree variants | Tree-based methods |
| Mixed Ensemble | `MixedEnsemble` | ✅ | All model types | Maximum diversity |

### Model Selection Guide

**For Small Datasets (< 1000 compounds):**
- `GaussianProcessLearner` - Best uncertainty quantification
- `MCDropoutLearner` - If GPU available

**For Medium Datasets (1000-10000 compounds):**
- `EnsembleLearner` - Best overall performance
- `XGBoostLearner` - Fast and accurate

**For Large Datasets (> 10000 compounds):**
- `XGBoostLearner` - Scalable performance
- `MLPLearner` - If GPU available

## 🎯 Acquisition Strategies

### Basic Strategies

| Strategy | Class | Uncertainty Required | Description |
|----------|-------|---------------------|-------------|
| Greedy | `GreedyAcquisition` | ❌ | Select highest predicted values |
| Random | `RandomAcquisition` | ❌ | Random baseline selection |
| Top-K | `TopKAcquisition` | ❌ | Select from top-K predictions |

### Uncertainty-Based Strategies

| Strategy | Class | Dependencies | Description |
|----------|-------|--------------|-------------|
| Upper Confidence Bound | `UCBAcquisition` | ❌ | Prediction + β × uncertainty |
| Expected Improvement | `ExpectedImprovementAcquisition` | scipy | Expected improvement over current best |
| Probability Improvement | `ProbabilityImprovementAcquisition` | scipy | Probability of improvement |
| Thompson Sampling | `ThompsonSamplingAcquisition` | ❌ | Sample from posterior distribution |
| Entropy | `EntropyAcquisition` | ❌ | Maximum information gain |

### Diversity-Based Strategies

| Strategy | Class | Dependencies | Description |
|----------|-------|--------------|-------------|
| UMAP + DBSCAN | `UMAPDBSCANAcquisition` | umap, sklearn | UMAP dimensionality reduction + clustering |
| UMAP + K-Means | `UMAPKMeansAcquisition` | umap, sklearn | UMAP + K-means clustering |
| t-SNE + DBSCAN | `TSNEDBSCANAcquisition` | sklearn | t-SNE dimensionality reduction + clustering |
| t-SNE + K-Means | `TSNEKMeansAcquisition` | sklearn | t-SNE + K-means clustering |
| Kennard-Stone | `KennardStoneAcquisition` | astartes | Optimal diverse sampling |
| Sphere Exclusion | `SphereExclusionAcquisition` | astartes | Geometric diversity sampling |
| Scaffold-Based | `ScaffoldAcquisition` | rdkit | Chemical scaffold diversity |
| BitBIRCH | `BitBIRCHAcquisition` | bitbirch | Molecular-native clustering |
| Butina Clustering | `ButinaClusteringAcquisition` | rdkit | RDKit-based molecular clustering |

### Optimization-Based Strategies

| Strategy | Class | Dependencies | Description |
|----------|-------|--------------|-------------|
| Simulated Annealing | `SimulatedAnnealingAcquisition` | ❌ | Annealing-based optimization |

## ✂️ Pruning Methods

Design space pruning reduces computational costs by removing unpromising compounds:

### Probabilistic Pruning

| Method | Class | Uncertainty Required | Description |
|--------|-------|---------------------|-------------|
| Basic Probabilistic | `ProbabilisticPruner` | ❌ | Remove low-probability compounds |
| Uncertainty Threshold | `UncertaintyThresholdPruner` | ✅ | Remove high-uncertainty compounds |
| Prediction Threshold | `PredictionThresholdPruner` | ❌ | Remove low-prediction compounds |
| Confidence Interval | `ConfidenceIntervalPruner` | ✅ | Remove based on confidence intervals |

### Adaptive Pruning

| Method | Class | Description |
|--------|-------|-------------|
| Cycle Budget | `CycleBudgetPruner` | Adaptive pruning based on cycle budget |
| Performance-Based | `PerformanceBasedPruner` | Prune based on model performance |

## 🧬 Molecular Featurizers

| Featurizer | Type | Size | Description |
|------------|------|------|-------------|
| Morgan | Fingerprint | 2048 | Circular fingerprints (radius=2) |
| MACCS | Fingerprint | 167 | MACCS structural keys |
| ECFP6 | Fingerprint | 2048 | Extended-connectivity (radius=3) |
| Descriptors | Numerical | ~200 | Mordred molecular descriptors |

## 🔄 Compatibility Matrix

### Model-Acquisition Compatibility

| Model Type | Basic | Uncertainty-Based | Diversity-Based | Optimization |
|------------|-------|------------------|-----------------|--------------|
| RandomForestLearner | ✅ | ❌ | ✅ | ✅ |
| GaussianProcessLearner | ✅ | ✅ | ✅ | ✅ |
| XGBoostLearner | ✅ | ❌ | ✅ | ✅ |
| MLPLearner | ✅ | ❌ | ✅ | ✅ |
| MCDropoutLearner | ✅ | ✅ | ✅ | ✅ |
| EnsembleLearner | ✅ | ✅ | ✅ | ✅ |
| All Ensemble Types | ✅ | ✅ | ✅ | ✅ |

### Acquisition-Featurizer Compatibility

| Acquisition Category | Morgan | MACCS | ECFP6 | Descriptors |
|---------------------|--------|-------|-------|-------------|
| Basic | ✅ | ✅ | ✅ | ✅ |
| Uncertainty-Based | ✅ | ✅ | ✅ | ✅ |
| Diversity-Based | ✅ | ✅ | ✅ | ✅ |
| Optimization | ✅ | ✅ | ✅ | ✅ |

### Model-Pruning Compatibility

| Model Type | Probabilistic | Uncertainty Threshold | Confidence Interval |
|------------|---------------|----------------------|-------------------|
| Models without uncertainty | ✅ | ❌ | ❌ |
| Models with uncertainty | ✅ | ✅ | ✅ |

## 💻 CLI Usage

### CLI Subcommands

LearnM8 provides three main subcommands:

```bash
learnm8 run       # Execute active learning experiment
learnm8 validate  # Validate compounds before running
learnm8 list      # List available components
```

### Validation Subcommand

```bash
# Validate compounds early
learnm8 validate compounds.csv --featurizer morgan

# With custom cache directory
learnm8 validate compounds.csv --featurizer morgan --cache-dir .cache

# Outputs:
# - Valid compounds count
# - Invalid compounds with error messages
# - Success rate statistics
```

### List Subcommand

```bash
# List available components
learnm8 list learners      # Show all machine learning models
learnm8 list acquisition   # Show acquisition strategies
learnm8 list featurizers   # Show molecular featurizers
learnm8 list schedules     # Show predefined schedules
```

### Run Subcommand

#### Basic Usage

```bash
# Benchmark mode (CSV oracle auto-detected)
learnm8 run data.csv --target Activity --learner gp --featurizer morgan --n-cycles 10

# Custom oracle mode
learnm8 run compounds.csv oracle.py:calculate_score --target binding_affinity --learner ensemble --featurizer morgan

# Ensemble learning
learnm8 run data.csv --target Activity --learner mixed_ensemble --featurizer morgan --cycles "random:0.01 greedy:0.005*10"
```

#### Predefined Schedules

```bash
# Quick exploration (5 cycles)
learnm8 run data.csv --target Activity --learner gp --featurizer morgan --schedule quick

# Standard screening (10 cycles)
learnm8 run data.csv --target Activity --learner gp --featurizer morgan --schedule standard

# Intensive optimization (20 cycles)
learnm8 run data.csv --target Activity --learner ensemble --featurizer morgan --schedule intensive

# Diversity-focused (10 cycles with diverse strategies)
learnm8 run compounds.csv oracle.py:score --target binding --learner mc_dropout --featurizer morgan --schedule diverse
```

#### Config File Support

```bash
# Use YAML config file
learnm8 run --config experiment.yaml

# Override specific parameters
learnm8 run --config experiment.yaml --learner ensemble --n-cycles 15
```

Example config file (`experiment.yaml`):
```yaml
compound_pool: compounds.csv
oracle: oracle.csv
target: Activity
learner: gp
featurizer: morgan
n_cycles: 10
batch_fraction: 0.01
random_state: 42
output: results/
```

**Config File Key Mapping:**
- `target` in YAML → `--target` in CLI → `target_col` in Python API
- `output` in YAML → `--output` in CLI → `output_dir` in Python API
- All other keys match their CLI flag names (without `--` prefix)

**Config Precedence:**
1. Config file values are used as defaults
2. Explicitly provided CLI flags override config values
3. API calls use explicit parameter names (e.g., `target_col`, `output_dir`)

#### Advanced Cycle Specification

```bash
# Explicit cycle specification (use actual acquisition strategy names)
learnm8 run compounds.csv oracle.py:score --target binding \
  --cycles "random:0.01 ucb:0.005*5 thompson:0.01*3" \
  --learner ensemble \
  --featurizer morgan \
  --pruning-strategy uncertainty_threshold \
  --pruning-params '{"threshold": 0.1}'
```

### CLI Parameters

#### Run Subcommand Arguments

**Required:**
- `compound_pool`: CSV file with ID, SMILES columns
- `oracle`: Oracle specification (CSV file or Python module:function)

**Targeting:**
- `--target`: Target property column name (required)

**Model Selection:**
- `--learner`: Choose model (`rf`, `gp`, `xgb`, `mlp`, `mc_dropout`, `ensemble`, `rf_ensemble`, `lr_ensemble`, `xgb_ensemble`, `dt_ensemble`, `mixed_ensemble`)

**Cycle Control:**
- `--cycles`: Explicit cycle specification (e.g., `"random:0.01 greedy:0.005*5"`)
- `--schedule`: Predefined schedules (`quick`, `standard`, `intensive`, `diverse`)
- `--n-cycles`: Number of cycles (default: 10)
- `--batch-fraction`: Batch fraction (default: 0.01)

**Configuration:**
- `--featurizer`: **Required** - Molecular features (`morgan`, `descriptors`, `maccs`, `ecfp6`)
- `--config`: Load parameters from YAML/JSON file
- `--pruning-strategy`: Pruning method
- `--random-state`: Random seed (default: 42)
- `--output-dir`: Output directory (default: `learnm8_output_<timestamp>`)
- `--export-csv`: Enable comprehensive CSV export

## 🔧 API Reference

### Core Function

```python
from learnm8 import run_active_learning

# Simple API
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer_type='morgan',
    n_cycles=10,
    batch_fraction=0.01
)

# Advanced API with CycleConfig
from learnm8 import CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer_type='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01,
                    pruning_strategy='score',
                    pruning_params={'pruning_fraction': 0.3})
    ]
)

# Results structure
# Returns dict with keys:
#   - compounds_df: Master DataFrame with all data
#   - cycle_metrics: List of metrics per cycle
#   - validation_result: ValidationResult with compound stats
#   - output_dir: Path to output directory
#   - saved_files: Dict of saved file paths
#   - labeled_data: Final labeled compounds
#   - unlabeled_data: Remaining unlabeled compounds
```

### Validation API

```python
from learnm8 import validate_compound_pool
from pathlib import Path

# Validate compounds before running
result = validate_compound_pool(
    compound_pool=df,
    featurizer_type='morgan',
    cache_dir=Path('.cache')
)

print(f"Valid: {len(result.valid_compounds)}")
print(f"Invalid: {len(result.invalid_compounds)}")
print(f"Success rate: {result.success_rate:.1%}")

# Access error details
for compound_id, error in result.invalid_compounds.items():
    print(f"{compound_id}: {error}")
```

### Feature Extraction API

```python
from learnm8 import extract_features
from pathlib import Path

# Extract features with automatic caching
features = extract_features(
    smiles_list=['CCO', 'CCC', 'CCN'],
    featurizer_type='morgan',
    cache_dir=Path('.cache'),
    n_jobs=-1  # Auto-detect optimal parallelization
)

# Features automatically cached in HDF5 for 100x speedup on reuse
```

### Component Creation

```python
# Learners
from learnm8.learners import GaussianProcessLearner, RandomForestLearner, XGBoostLearner, EnsembleLearner

learner = GaussianProcessLearner(alpha=1e-6, random_state=42)

# Create ensemble with multiple learner instances
ensemble = EnsembleLearner([
    RandomForestLearner(n_estimators=100, random_state=42),
    GaussianProcessLearner(alpha=1e-6, random_state=42),
    XGBoostLearner(n_estimators=100, random_state=42)
])

# Oracles
from learnm8.oracles import CSVOracle, PythonOracle
oracle = CSVOracle('ground_truth.csv')
custom_oracle = PythonOracle('my_module.py', 'scoring_function')
```

### Acquisition Strategy Usage

```python
from learnm8.acquisition import get_acquisition_function

# Get acquisition function by name
acquisition_fn = get_acquisition_function('ucb')
acquisition = acquisition_fn(beta=2.0)

# Available acquisition functions
from learnm8.acquisition import list_acquisition_functions
print(list_acquisition_functions())
```

### Cycle Numbering Convention

LearnM8 uses a **zero-indexed cycle numbering system** where:

- **Cycle 0**: Initialization phase (random selection of initial training set)
- **Cycles 1-N**: Active learning cycles using specified acquisition strategy

**Important Notes:**
- When `n_cycles=10` is specified, you get exactly 10 cycles total: cycle 0 (initialization) + cycles 1-9 (active learning)
- The `cycle_metrics` list contains metrics for all cycles starting from cycle 0
- CSV exports (`cycle_metrics.csv`) start at cycle 0
- This convention ensures clear distinction between initialization and active learning phases

**Example:**
```python
results = run_active_learning(
    compound_pool='data.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer_type='morgan',
    initial_strategy='random',  # Used for cycle 0
    strategy='ucb',             # Used for cycles 1-9
    n_cycles=10                 # Total: 10 cycles (0-9)
)

# Check cycle numbers and strategies
cycle_metrics = results['cycle_metrics']
print(len(cycle_metrics))  # Output: 10
print(cycle_metrics[0]['cycle'])     # Output: 0
print(cycle_metrics[0]['strategy'])  # Output: 'random'
print(cycle_metrics[1]['strategy'])  # Output: 'ucb'
print(cycle_metrics[9]['strategy'])  # Output: 'ucb'
```

## 📦 Installation

### Requirements
- Python 3.11.9 (required)
- Scientific computing: pandas, numpy, scikit-learn
- Chemistry: rdkit
- Machine learning: pytorch, xgboost
- Optional: umap-learn, astartes, bitbirch

### Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate learnm8

# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e .[test]

# Install optional dependencies
pip install -e .[diversity]  # For UMAP and HDBSCAN clustering
pip install -e .[cli]         # For full CLI features (tqdm progress bars)
pip install -e .[full]        # All optional dependencies

# Install BitBIRCH separately (cannot be in setup.py due to git source)
pip install git+https://github.com/mqcomplab/bitbirch.git
```

**Note:** pyyaml is now a required dependency for CLI config file support.

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=learnm8 --cov-report=html

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m molecular      # Tests requiring RDKit
pytest -m "not slow"     # Skip slow tests
```

## ⚡ Performance Features

LearnM8 v1.0.0 introduces significant performance improvements:

### Speed Improvements

- **5-10x faster feature extraction** with automatic parallelization (no configuration needed)
- **100x faster repeated extraction** with HDF5 caching
- **10x faster DataFrame operations** with vectorization
- **Early validation** catches errors before cycles start (saves time)

### Performance Tips

1. **Keep cache directory between runs**: HDF5 cache persists across sessions
2. **Validate once, use many times**: Run `learnm8 validate` before experiments
3. **Automatic parallelization**: Feature extraction auto-detects optimal CPU usage
4. **Use config files**: YAML configs enable reproducible, parameter-tracked experiments

### Benchmarks

| Operation | Old (v0.5.0) | New (v1.0.0) | Speedup |
|-----------|--------------|--------------|---------|
| Feature extraction (10k compounds) | 50s | 5s | 10x |
| Repeated extraction (cached) | 50s | 0.5s | 100x |
| DataFrame updates (per cycle) | 2s | 0.2s | 10x |
| Validation (early detection) | N/A | 1s | ∞ (prevents failures) |

## 📊 Examples

### 1. Basic Benchmark Analysis

```bash
# Compare multiple models on benchmark dataset (oracle auto-detected from compound_pool)
learnm8 run ESSENCE_benchmark_input/ADA.csv --target Activity --learner gp --featurizer morgan --n-cycles 15
learnm8 run ESSENCE_benchmark_input/ADA.csv --target Activity --learner ensemble --featurizer morgan --n-cycles 15
```

### 2. Custom Oracle Deployment

```bash
# Use custom scoring function
learnm8 run compound_library.csv scoring_module.py:calculate_affinity --target binding_score --learner mc_dropout --featurizer morgan --n-cycles 20
```

### 3. Diversity-Focused Screening

```bash
# Mixed strategies (use actual acquisition strategy names from 'learnm8 list acquisition')
learnm8 run compounds.csv oracle.py:score --target binding \
  --cycles "random:0.01 greedy:0.005*5 thompson:0.01" \
  --learner ensemble --featurizer morgan
```

### 4. Uncertainty-Guided Active Learning

```bash
# Uncertainty-based acquisition with pruning
learnm8 run large_library.csv oracle.py:score --target binding \
  --cycles "random:0.005 ucb:0.003*8 thompson:0.005*2" \
  --learner gp \
  --featurizer morgan \
  --pruning-strategy uncertainty_threshold \
  --pruning-params '{"threshold": 0.2}'
```

### 5. Production Screening Pipeline

```bash
# High-performance ensemble
learnm8 run virtual_library.csv docking_oracle.py:calculate_score --target binding_affinity \
  --cycles "random:0.005 greedy:0.002*5 ucb:0.003*5 thompson:0.01" \
  --learner mixed_ensemble \
  --featurizer descriptors \
  --output results/
```

## 🏆 Key Features

- **Early Validation**: Catch invalid compounds before running experiments
- **Parallel Processing**: 5-10x faster feature extraction with automatic parallelization
- **HDF5 Caching**: 100x faster repeated operations with persistent caching
- **Vectorized Operations**: 10x faster DataFrame updates with optimized operations
- **Config File Support**: YAML/JSON for reproducible experiments
- **Predefined Schedules**: Quick start templates (quick, standard, intensive, diverse)
- **Pure Functional Architecture**: No complex state management
- **Explicit Cycle Control**: Fine-grained control over active learning workflow
- **Comprehensive Model Suite**: 15+ machine learning models with uncertainty quantification
- **Rich Acquisition Strategies**: 20+ selection strategies from basic to sophisticated
- **Molecular-Specific**: Built for chemical space with RDKit integration
- **Design Space Pruning**: Intelligent compound pool reduction
- **Production Ready**: Robust error handling and comprehensive evaluation

## 📖 Migration Guide

Migrating from v0.5.0? See **[MIGRATION.md](MIGRATION.md)** for detailed instructions.

**Key changes:**
- New API location: Import from `learnm8` or `learnm8.api`
- New CLI syntax with subcommands (`run`, `validate`, `list`)
- CycleConfig for advanced cycle control
- Performance improvements: 5-100x faster

## 📄 License

LearnM8 is developed for molecular screening and active learning research.

## 🤝 Contributing

LearnM8 follows a functional architecture with modular design. When contributing:

1. Maintain pure functional patterns
2. Use `extract_features()` for all feature extraction
3. Follow existing code conventions
4. Add comprehensive tests with real molecular data
5. Update compatibility matrices when adding new components

## 📚 Citation

If you use LearnM8 in your research, please cite our work (citation details to be added).

---

**LearnM8**: Intelligent molecular screening through active learning 🧬🤖
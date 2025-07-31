# LearnM8: Active Learning Framework for Molecular Screening

LearnM8 is a comprehensive active learning framework designed for molecular property prediction and compound screening. It provides a pure functional API with sophisticated machine learning models, acquisition strategies, and design space pruning techniques specifically tailored for chemical space exploration.

## 🚀 Quick Start

```bash
# Install dependencies
conda env create -f environment.yml
conda activate base
pip install -e .

# Basic usage
learnm8 compounds.csv compounds.csv Activity -l gp -c 10

# Advanced cycle specification
learnm8 compounds.csv oracle.py:calculate_score target -l ensemble --cycles-spec "random:0.01 greedy:0.005*5 diverse:0.01"
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

LearnM8 follows a **pure functional architecture** with explicit dependency injection:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Compound      │    │   Oracle         │    │   Learner       │
│   Pool (CSV)    │───▶│   (CSV/Python)   │───▶│   (ML Model)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  DataManager    │    │  Acquisition     │    │   Evaluation    │
│  (Features)     │    │  Strategy        │    │   Metrics       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                        ┌─────────────────┐
                        │ Active Learning │
                        │ Core Engine     │
                        └─────────────────┘
```

### Core Principles

1. **Pure Functional API**: No complex state management - driven by `run_active_learning()` function
2. **Explicit Cycle Control**: Users specify exactly what happens each cycle
3. **Dependency Injection**: Components receive dependencies explicitly (especially DataManager)
4. **Composition over Inheritance**: Simple interfaces with duck typing

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

### Basic Commands

```bash
# Benchmark mode (CSV oracle)
learnm8 data.csv data.csv Activity -l gp -c 10

# Custom oracle mode  
learnm8 compounds.csv oracle.py:calculate_score target -l ensemble

# Ensemble learning
learnm8 data.csv data.csv Activity -l mixed_ensemble --cycles-spec "random:0.01 greedy:0.005*10"
```

### Advanced Configuration

```bash
# Explicit cycle specification
learnm8 compounds.csv oracle.py:score target \
  --cycles-spec "random:0.01 ucb:0.005*5 diverse:0.01 bitbirch:0.005*3" \
  -l ensemble \
  --featurizer morgan \
  --pruning-strategy uncertainty_threshold \
  --pruning-params '{"threshold": 0.1}'

# Predefined schedules
learnm8 data.csv data.csv Activity -l gp --schedule intensive

# Diversity-focused screening
learnm8 compounds.csv oracle.py:score target \
  --schedule diversity \
  -l mc_dropout \
  --featurizer descriptors
```

### CLI Parameters

#### Required Arguments
- `compound_pool`: CSV file with ID, SMILES columns
- `oracle`: Oracle specification (CSV file or Python module:function)  
- `target_column`: Target property column name

#### Model Selection
- `-l/--learner`: Choose model (`rf`, `gp`, `xgb`, `mlp`, `mc_dropout`, `ensemble`, `rf_ensemble`, `lr_ensemble`, `xgb_ensemble`, `dt_ensemble`, `mixed_ensemble`)

#### Cycle Control
- `--cycles-spec`: Explicit cycle specification (e.g., `"random:0.01 greedy:0.005*5"`)
- `--schedule`: Predefined schedules (`quick`, `standard`, `intensive`, `diversity`)
- `-c/--cycles`: Number of cycles (legacy mode)
- `-b/--batch-fraction`: Batch fraction (legacy mode)

#### Configuration
- `--featurizer`: Molecular features (`morgan`, `descriptors`, `maccs`, `ecfp6`)
- `--pruning-strategy`: Pruning method
- `--random-state`: Random seed
- `-o/--output`: Output directory

## 🔧 API Reference

### Core Function

```python
from learnm8 import run_active_learning

# Simple usage
results = run_active_learning(
    compound_pool=df,
    oracle=oracle_instance,
    learner=learner_instance,
    target_column='Activity',
    strategy='greedy',
    n_cycles=10,
    batch_fraction=0.01
)

# Advanced cycle specification  
results = run_active_learning(
    compound_pool=df,
    oracle=oracle_instance,
    learner=learner_instance,
    target_column='Activity',
    cycles=[
        ('random', 0.01),     # Initial random sampling
        ('greedy', 0.005),    # Greedy exploitation  
        ('diverse', 0.01)     # Final diverse exploration
    ]
)
```

### Component Creation

```python
# Learners
from learnm8.learners import GaussianProcessLearner, EnsembleLearner
learner = GaussianProcessLearner(kernel='rbf')
ensemble = EnsembleLearner(models=['rf', 'gp', 'xgb'])

# Oracles
from learnm8.oracles import CSVOracle, PythonOracle
oracle = CSVOracle('ground_truth.csv')
custom_oracle = PythonOracle('my_module.py', 'scoring_function')

# Data Management
from learnm8.core.data_manager import DataManager
dm = DataManager(featurizer='morgan', cache_dir='./cache')
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
conda activate base

# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e .[test]

# Install optional dependencies
pip install -e .[diversity]  # For UMAP and HDBSCAN clustering
pip install -e .[full]       # All optional dependencies

# Install BitBIRCH separately (cannot be in setup.py due to git source)
pip install git+https://github.com/mqcomplab/bitbirch.git
```

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

## 📊 Examples

### 1. Basic Benchmark Analysis

```bash
# Compare multiple models on benchmark dataset
learnm8 ESSENCE_benchmark_input/ADA.csv ESSENCE_benchmark_input/ADA.csv Activity -l gp -c 15
learnm8 ESSENCE_benchmark_input/ADA.csv ESSENCE_benchmark_input/ADA.csv Activity -l ensemble -c 15
```

### 2. Custom Oracle Deployment

```bash
# Use custom scoring function
learnm8 compound_library.csv scoring_module.py:calculate_affinity binding_score -l mc_dropout -c 20
```

### 3. Diversity-Focused Screening

```bash
# Mixed diversity strategies  
learnm8 compounds.csv oracle.py:score target \
  --cycles-spec "random:0.01 umap_dbscan:0.005*3 bitbirch:0.005*3 kennard_stone:0.01" \
  -l ensemble --featurizer morgan
```

### 4. Uncertainty-Guided Active Learning

```bash
# Uncertainty-based acquisition with pruning
learnm8 large_library.csv oracle.py:score target \
  --cycles-spec "random:0.005 ucb:0.003*8 thompson:0.005*2" \
  -l gp \
  --pruning-strategy uncertainty_threshold \
  --pruning-params '{"threshold": 0.2}'
```

### 5. Production Screening Pipeline

```bash
# High-performance ensemble with simulated annealing
learnm8 virtual_library.csv docking_oracle.py:calculate_score binding_affinity \
  --cycles-spec "random:0.005 greedy:0.002*5 simulated_annealing:0.003*8 diverse:0.01" \
  -l mixed_ensemble \
  --featurizer descriptors \
  --max-batch-size 500 \
  --export-csv \
  -o results/
```

## 🏆 Key Features

- **Pure Functional Architecture**: No complex state management
- **Explicit Cycle Control**: Fine-grained control over active learning workflow  
- **Comprehensive Model Suite**: 15+ machine learning models with uncertainty quantification
- **Rich Acquisition Strategies**: 20+ selection strategies from basic to sophisticated
- **Molecular-Specific**: Built for chemical space with RDKit integration
- **Design Space Pruning**: Intelligent compound pool reduction
- **Production Ready**: Robust error handling and comprehensive evaluation
- **High Performance**: HDF5 caching, parallel processing, GPU acceleration

## 📄 License

LearnM8 is developed for molecular screening and active learning research.

## 🤝 Contributing

LearnM8 follows a functional architecture with dependency injection. When contributing:

1. Maintain pure functional patterns
2. Use DataManager for all feature extraction
3. Follow existing code conventions  
4. Add comprehensive tests with real molecular data
5. Update compatibility matrices when adding new components

## 📚 Citation

If you use LearnM8 in your research, please cite our work (citation details to be added).

---

**LearnM8**: Intelligent molecular screening through active learning 🧬🤖
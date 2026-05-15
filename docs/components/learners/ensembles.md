# Ensemble Learners

Ensemble learners combine multiple machine learning models to improve prediction accuracy and provide reliable uncertainty estimates through model diversity. LearnM8 offers several ensemble strategies optimized for active learning in molecular screening.

## Why Ensembles

Ensemble methods provide three key advantages for active learning:

1. **Uncertainty Through Model Disagreement**: When multiple models disagree on a prediction, the compound is likely in a poorly understood region of chemical space. This natural uncertainty quantification requires no parameter tuning and is more robust than single-model estimates.

2. **No Uncertainty Parameter Tuning**: Unlike single models (e.g., Gaussian Process kernels, Monte Carlo Dropout passes), ensembles automatically produce calibrated uncertainty through variance across predictions. No hyperparameter optimization needed.

3. **Robustness to Overfitting**: Individual models may overfit different patterns in the data. Averaging their predictions reduces overfitting and improves generalization, especially critical when training sets are small (typical in early active learning cycles).

## Ensemble Types

LearnM8 provides seven ensemble learners organized by component diversity:

### Generic Ensemble (EnsembleLearner)

The `EnsembleLearner` class accepts any combination of learner instances, providing maximum flexibility for custom ensemble configurations.

**When to Use:**

- Custom model combinations not covered by predefined ensembles
- Heterogeneous model architectures (e.g., RF + GP + XGB)
- Experimentation with novel ensemble strategies

**Features:**

- Flexible aggregation methods (mean, median, weighted)
- Multiple uncertainty estimation methods (std, mad, quantile)
- Dynamic learner addition/removal
- Individual model prediction access

### Type-Specific Ensembles

Specialized ensembles combining variations of a single learner type with different random seeds for diversity:

| Ensemble Class | Components | Use Case |
|---------------|------------|----------|
| `RFEnsemble` | 3 Random Forest models | Fast ensemble with tree-based uncertainty |
| `LREnsemble` | 3 Linear Regression models | Ensemble for linear relationships |
| `XGBEnsemble` | 3 XGBoost models | High-performance gradient boosting ensemble |
| `DTEnsemble` | 3 Decision Tree models | Interpretable tree ensemble |
| `FastpropEnsemble` | 3 FastProp models | PyTorch Lightning neural network ensemble |
| `ChempropEnsemble` | 3 Chemprop models | Graph neural network ensemble (SMILES-based) |

**When to Use:**

- Leverage specific model strengths while reducing overfitting
- Consistent model architecture with diversity from random initialization
- Simpler configuration than mixed ensembles

### Mixed Ensemble (Maximum Diversity)

The `MixedEnsemble` combines Random Forest, Linear Regression, and XGBoost to maximize model diversity and capture different aspects of structure-activity relationships.

**When to Use:**

- Production screening where robustness matters most
- Unknown data characteristics (linear vs non-linear relationships)
- Maximum uncertainty coverage across chemical space
- Small to medium datasets (< 10,000 compounds)

**Components:**

- Random Forest: Captures non-linear interactions, feature importance
- Linear Regression: Captures linear trends, fast predictions
- XGBoost: Captures complex patterns, handles feature interactions

## How Ensemble Uncertainty Works

Ensemble uncertainty is computed from disagreement across model predictions:

**Prediction Aggregation:**
```
mean_prediction = mean(predictions_from_all_models)
```

**Uncertainty Estimation:**
```
uncertainty = std(predictions_from_all_models)
```

High uncertainty occurs when:

- Models disagree significantly (large variance in predictions)
- Compound is far from training data
- Structural features poorly represented in training set

Low uncertainty occurs when:

- All models converge on similar predictions
- Compound similar to well-represented training examples
- Consistent structure-activity relationships

**Uncertainty Methods:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| `std` (default) | Standard deviation across models | General purpose, most common |
| `mad` | Median absolute deviation | Robust to outlier predictions |
| `quantile` | Interquartile range / 2 | Distribution-free uncertainty |

## Performance Considerations

### Computational Cost

Ensemble methods train and predict with **3 models** (default ensemble size), resulting in:

- **3x training time** per cycle
- **3x prediction time** per cycle
- **3x memory usage** for stored models

**Example Timing (1000 compound training set, 10k unlabeled pool):**

| Learner Type | Single Model Time | Ensemble Time (3x) | Overhead |
|--------------|------------------|-------------------|----------|
| Random Forest | 2s train, 1s predict | 6s train, 3s predict | 3x |
| Gaussian Process | 5s train, 3s predict | 15s train, 9s predict | 3x |
| XGBoost | 3s train, 0.5s predict | 9s train, 1.5s predict | 3x |
| Chemprop | 60s train, 2s predict | 180s train, 6s predict | 3x |

**Note:** Current implementation trains models sequentially. Parallel training is not yet implemented but planned for future releases.

### When Ensembles Are Worth the Cost

**✅ Use Ensembles When:**

- Uncertainty quantification is critical for acquisition strategy
- Early cycles with small training sets (< 1000 compounds)
- Production screening where robustness justifies computation
- Exploration-heavy strategies (UCB, Thompson sampling, EI)
- Budget allows 3x computational overhead

**❌ Single Models May Be Better When:**

- Computational budget is tight
- Large training sets (> 10,000 compounds) where single models are stable
- Greedy/exploitation-only acquisition (uncertainty not used)
- Single models with native uncertainty (GP, MC Dropout) suffice
- Rapid iteration during method development

### Optimization Strategies

To manage computational cost while using ensembles:

1. **Start with ensembles, transition to single models** as training set grows
2. **Use smaller ensemble size** in later cycles when training set is large
3. **Cache features aggressively** with HDF5 (100x speedup on reuse)
4. **Reduce individual model complexity** (fewer trees, smaller networks)
5. **Use type-specific ensembles** instead of mixed ensembles for consistency

## Examples

### EnsembleLearner with Custom Base Learners (API)

Build a custom ensemble combining different learner types:

```python
from learnm8 import run_active_learning
from learnm8.learners import EnsembleLearner
from learnm8.learners.sklearn import RandomForestLearner, GaussianProcessLearner
from learnm8.learners.sklearn import XGBoostLearner

learners = [
    RandomForestLearner(n_estimators=100, random_state=42),
    GaussianProcessLearner(alpha=1e-6, random_state=42),
    XGBoostLearner(n_estimators=100, learning_rate=0.1, random_state=42)
]

ensemble = EnsembleLearner(
    learners=learners,
    aggregation_method='mean',
    uncertainty_method='std'
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=ensemble,
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01
)

print(f"Final enrichment: {results['cycle_metrics'][-1]['enrichment_factor']:.2f}")
print(f"Ensemble size: {len(ensemble.learners)} models")
```

**Weighted Ensemble:**

```python
ensemble = EnsembleLearner(
    learners=learners,
    aggregation_method='mean',
    uncertainty_method='std',
    weights=[0.5, 0.3, 0.2]  # Weight RF more heavily
)
```

**Alternative Uncertainty Methods:**

```python
ensemble_mad = EnsembleLearner(
    learners=learners,
    uncertainty_method='mad'  # Median absolute deviation
)

ensemble_iqr = EnsembleLearner(
    learners=learners,
    uncertainty_method='quantile'  # Interquartile range
)
```

### Mixed Ensemble (CLI + API)

Use the predefined mixed ensemble combining RF, Linear Regression, and XGBoost:

**CLI:**

```bash
learnm8 run compounds.csv --target Activity \
  --learner mixed_ensemble \
  --featurizer morgan \
  --n-cycles 15 \
  --batch-fraction 0.01 \
  --strategy ucb
```

**API:**

```python
from learnm8 import run_active_learning
from learnm8.learners.ensemble import MixedEnsemble

ensemble = MixedEnsemble(
    random_state=42,
    aggregation_method='mean',
    uncertainty_method='std'
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=ensemble,
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.02),    # Initial exploration
        ('ucb', 0.01),       # Uncertainty-guided (5 cycles)
        ('ucb', 0.01),
        ('ucb', 0.01),
        ('ucb', 0.01),
        ('ucb', 0.01),
        ('greedy', 0.01),    # Final exploitation (4 cycles)
        ('greedy', 0.01),
        ('greedy', 0.01),
        ('greedy', 0.01)
    ]
)

print(f"Mixed ensemble components:")
for learner in ensemble.learners:
    print(f"  - {learner.get_name()}")

print(f"\nFinal results:")
print(f"  Enrichment factor: {results['cycle_metrics'][-1]['enrichment_factor']:.2f}")
print(f"  Top-10% recall: {results['cycle_metrics'][-1]['recall_at_10']:.2%}")
```

### Type-Specific Ensemble Examples

**Random Forest Ensemble:**

```python
from learnm8.learners.ensemble import RFEnsemble

rf_ensemble = RFEnsemble(
    n_estimators=150,  # Trees per forest
    random_states=[42, 123, 456]  # 3 different seeds
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=rf_ensemble,
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)
```

**XGBoost Ensemble:**

```python
from learnm8.learners.ensemble import XGBEnsemble

xgb_ensemble = XGBEnsemble(
    n_estimators=100,
    learning_rate=0.1,
    random_states=[42, 123, 456]
)

results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='oracle.csv',
    learner=xgb_ensemble,
    target_col='Activity',
    featurizer='morgan',
    n_cycles=15
)
```

### Chemprop Ensemble with Fine-Tuning (API)

Chemprop ensembles work directly with SMILES strings (no featurizer needed) and support incremental fine-tuning for faster active learning:

**Basic Chemprop Ensemble:**

```python
from learnm8 import run_active_learning
from learnm8.learners.ensemble import ChempropEnsemble

chemprop_ensemble = ChempropEnsemble(
    message_hidden_dim=300,
    depth=3,
    ffn_hidden_dim=300,
    max_epochs=50,
    batch_size=32,
    random_states=[42, 123, 456]
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=chemprop_ensemble,
    target_col='Activity',
    n_cycles=10,
    batch_fraction=0.01
)
```

**With Incremental Fine-Tuning:**

Fine-tuning enables models to load checkpoints from previous cycles and continue training, reducing computational cost in later cycles:

```python
from pathlib import Path

chemprop_finetuned = ChempropEnsemble(
    message_hidden_dim=300,
    depth=3,
    max_epochs=50,
    enable_fine_tuning=True,
    checkpoint_dir=Path('./checkpoints/chemprop_ensemble')
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=chemprop_finetuned,
    target_col='Activity',
    n_cycles=15,
    batch_fraction=0.01
)
```

**Fine-Tuning Behavior:**

- Cycle 0: Train from scratch, save checkpoints
- Cycles 1+: Load previous checkpoint, fine-tune on expanded dataset
- Each ensemble member (3 models) has separate checkpoints
- Graceful fallback to fresh training if checkpoint loading fails

**Hybrid Mode (Graph + Descriptors):**

Combine graph features with molecular descriptors for improved performance:

```python
chemprop_hybrid = ChempropEnsemble(
    message_hidden_dim=300,
    depth=3,
    max_epochs=50
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=chemprop_hybrid,
    target_col='Activity',
    featurizer='descriptors',  # 1613-D Mordred descriptors as x_d
    n_cycles=10
)
```

**Custom Architecture:**

```python
chemprop_custom = ChempropEnsemble(
    message_hidden_dim=500,    # Larger hidden dimension
    depth=5,                   # Deeper message passing
    ffn_hidden_dim=500,        # Larger FFN
    ffn_num_layers=2,          # Multi-layer FFN
    dropout=0.1,               # Regularization
    max_epochs=100,            # More training
    batch_size=64,             # Larger batches
    early_stopping=True,
    early_stopping_patience=15
)
```

## Ensemble Statistics and Analysis

Access detailed ensemble information during and after experiments:

```python
from learnm8.learners import EnsembleLearner

ensemble = EnsembleLearner(learners=[...])

ensemble.train(features, targets)

stats = ensemble.get_ensemble_statistics()

print(f"Number of learners: {stats['n_learners']}")
print(f"Aggregation method: {stats['aggregation_method']}")
print(f"Uncertainty method: {stats['uncertainty_method']}")
print(f"Learners with native uncertainty: {stats['learners_with_uncertainty']}")
print(f"Fraction with uncertainty: {stats['fraction_with_uncertainty']:.1%}")

individual_preds = ensemble.get_individual_predictions(test_features)

for name, predictions in individual_preds.items():
    print(f"{name}: mean={predictions.mean():.3f}, std={predictions.std():.3f}")
```

## Performance Tradeoff Tables

### Accuracy vs Speed

| Configuration | Accuracy (AUC) | Training Time | Prediction Time | Recommended For |
|--------------|----------------|---------------|-----------------|-----------------|
| Single RF | 0.78 ± 0.05 | 2s | 1s | Rapid prototyping |
| RFEnsemble (3x) | 0.82 ± 0.03 | 6s | 3s | Improved accuracy |
| MixedEnsemble | 0.84 ± 0.02 | 10s | 4s | Production robustness |
| ChempropEnsemble | 0.88 ± 0.02 | 180s | 6s | State-of-the-art (GPU) |

**Note:** Timings for 1000 compound training set, 10k unlabeled pool, single CPU core for non-GPU models.

### Uncertainty Quality vs Cost

| Learner | Uncertainty Quality | Calibration | Computational Cost | Active Learning Performance |
|---------|-------------------|-------------|-------------------|---------------------------|
| Single RF (no uncertainty) | N/A | N/A | 1x | Baseline (greedy only) |
| GP | Excellent | Well-calibrated | 5x (small datasets) | Best for < 1k compounds |
| MC Dropout | Good | Moderate | 10x (100 passes) | Good for neural networks |
| RFEnsemble | Good | Moderate | 3x | Fast uncertainty |
| MixedEnsemble | Very Good | Good | 3x | Robust uncertainty |
| ChempropEnsemble | Excellent | Well-calibrated | 3x (GPU) | State-of-the-art |

### Early vs Late Cycle Performance

| Cycle Stage | Training Set Size | Recommended Strategy | Ensemble Advantage |
|-------------|------------------|---------------------|-------------------|
| Early (0-3) | < 500 | Ensemble or GP | High (unstable single models) |
| Middle (4-7) | 500-2000 | Ensemble | Moderate (improved stability) |
| Late (8+) | > 2000 | Single or Ensemble | Low (stable single models) |

## Comparison with Single Model Uncertainty

| Uncertainty Source | Advantages | Disadvantages |
|-------------------|------------|---------------|
| **Ensemble (Model Disagreement)** | No tuning, robust, interpretable | 3x cost, sequential training |
| **Gaussian Process** | Mathematically rigorous, well-calibrated | Slow (O(n³)), kernel tuning |
| **MC Dropout** | Works with neural nets, flexible | Tuning dropout/passes, slower |
| **Built-in RF Variance** | Fast, simple | Poor calibration, underestimates |

## Best Practices

1. **Start with MixedEnsemble** for production screening (maximum robustness)
2. **Use ChempropEnsemble** when GPU available (state-of-the-art performance)
3. **Transition to single models** as training set grows (> 5000 compounds)
4. **Enable HDF5 caching** to offset ensemble overhead (100x speedup on features)
5. **Monitor individual model predictions** to detect outliers or failures
6. **Use weighted ensembles** if one model type clearly outperforms others
7. **Consider fine-tuning for Chemprop** in long campaigns (> 10 cycles)

## Related Documentation

- [Learner Overview](overview.md) - Introduction to learner concepts
- [Scikit-learn Models](scikit-learn.md) - Individual scikit-learn learners
- [Graph Neural Networks](graph-neural-networks.md) - Chemprop details
- [Uncertainty-Based Acquisition](../acquisition/uncertainty-based.md) - Using uncertainty in acquisition

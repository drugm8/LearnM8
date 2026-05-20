# Learners Overview

Learners are the machine learning models at the core of the LearnM8 active learning framework. They provide predictions for molecular properties and, when applicable, uncertainty estimates that guide intelligent compound selection.

## What are Learners?

Learners in LearnM8 serve two critical functions:

1. **Property Prediction**: Train on labeled compounds and predict properties for unlabeled compounds
2. **Uncertainty Quantification**: Estimate prediction confidence to guide acquisition strategies

Every learner implements a consistent protocol defined in `learnm8.core.interfaces.Learner`, ensuring seamless integration with the active learning pipeline.

## The Learner Protocol

All learners implement four core methods:

```python
from learnm8.core.interfaces import Learner
import numpy as np
from typing import Tuple, Optional, List

class MyLearner(Learner):
    def train(self,
              features: np.ndarray,
              targets: np.ndarray,
              smiles: Optional[List[str]] = None) -> None:
        """Train the model on feature matrix or SMILES strings.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
            smiles: Optional SMILES strings (required by graph-based learners)
        """
        pass

    def predict(self,
                features: np.ndarray,
                smiles: Optional[List[str]] = None,
                *,
                compute_uncertainty: bool = True
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict on feature matrix or SMILES strings.

        Args:
            features: Feature matrix (n_samples, n_features)
            smiles: Optional SMILES strings (required by graph-based learners)
            compute_uncertainty: When False and the learner is skip-eligible,
                the uncertainty compute path is elided and None is returned
                as the second element. Cannot disable uncertainty when the
                active acquisition strategy requires it (e.g. UCB, EI).

        Returns:
            Tuple of (predictions, uncertainties).
            Uncertainties is None if the learner doesn't support them,
            or if compute_uncertainty=False on a skip-eligible learner.
        """
        pass

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        pass

    def supports_uncertainty(self) -> bool:
        """Return True if this learner provides uncertainty estimates."""
        return False

    def requires_smiles(self) -> bool:
        """Return True if this learner needs SMILES strings.

        Most learners work with pre-computed features (morgan, descriptors).
        Graph neural networks (e.g., Chemprop) override to return True.
        """
        return False
```

## Learner Types

LearnM8 provides four categories of learners, organized by framework and methodology:

### Scikit-learn Based Models

Traditional machine learning models built on scikit-learn, offering robust baseline performance with minimal dependencies:

- **RandomForestLearner**: Fast ensemble baseline
- **GaussianProcessLearner**: Gold standard for uncertainty quantification
- **XGBoostLearner**: High-performance gradient boosting
- **DecisionTreeLearner**: Interpretable single trees
- **LinearRegressionLearner**: Simple linear baselines

### PyTorch Neural Networks

Deep learning models for complex non-linear patterns, with optional GPU acceleration:

- **MLPLearner**: Multi-layer perceptron (feedforward neural network)
- **MCDropoutLearner**: Monte Carlo Dropout for uncertainty estimation
- **FastpropLearner**: PyTorch Lightning implementation with optimized training

### Graph Neural Networks

State-of-the-art models that work directly with molecular graphs (SMILES strings):

- **ChempropLearner**: Message Passing Neural Network (single model)
- **ChempropEnsemble**: 3-model ensemble with uncertainty via model disagreement

### Ensemble Learners

Multiple models combined for improved predictions and uncertainty through model disagreement:

- **EnsembleLearner**: Generic ensemble (mix any learner types)
- **RFEnsemble**: Random Forest variants ensemble
- **LREnsemble**: Linear model variants ensemble
- **XGBEnsemble**: XGBoost variants ensemble
- **DTEnsemble**: Decision Tree variants ensemble
- **MixedEnsemble**: Maximum diversity (RF + LR + XGB)
- **FastpropEnsemble**: FastProp neural network ensemble

## Choosing a Learner

Selection depends on dataset characteristics, computational resources, and whether uncertainty quantification is needed.

### Decision Matrix by Dataset Size

| Dataset Size | Uncertainty Needed? | GPU Available? | Recommended Learner | Rationale |
|--------------|---------------------|----------------|---------------------|-----------|
| < 1,000 compounds | No | Any | `rf` | Fast training, robust baseline |
| < 1,000 compounds | Yes | No | `gp` | Best uncertainty for small data |
| < 1,000 compounds | Yes | Yes | `chemprop_ensemble` | State-of-the-art with uncertainty |
| 1,000-10,000 | No | Any | `xgb` | High performance, scalable |
| 1,000-10,000 | Yes | No | `mixed_ensemble` | Model diversity for uncertainty |
| 1,000-10,000 | Yes | Yes | `chemprop_ensemble` | Superior performance at scale |
| > 10,000 | No | No | `xgb` | Memory efficient, fast |
| > 10,000 | No | Yes | `mlp` | Deep learning efficiency |
| > 10,000 | Yes | No | `rf_ensemble` | Scalable ensemble |
| > 10,000 | Yes | Yes | `chemprop_ensemble` | Best overall performance |

### Decision Matrix by Use Case

| Priority | Recommended Learner | Notes |
|----------|---------------------|-------|
| Speed | `rf`, `xgb` | Fastest training times |
| Uncertainty Quality | `gp`, `chemprop_ensemble`, `mc_dropout` | Principled uncertainty quantification |
| Interpretability | `dt`, `lr` | Simple, explainable models |
| State-of-the-art Performance | `chemprop_ensemble` | MPNN ensemble optimized for molecules |
| No Featurizer | `chemprop`, `chemprop_ensemble`, `fastprop`, `fastprop_ensemble` | Work directly with SMILES |
| Robustness | `mixed_ensemble`, `rf_ensemble` | Multiple models reduce overfitting risk |

### Featurizer Compatibility

| Learner Type | Best Featurizer | Alternative | Notes |
|--------------|-----------------|-------------|-------|
| Linear models (`lr`) | `descriptors` | `morgan` | Descriptors capture linear relationships |
| Tree-based (`rf`, `xgb`, `dt`) | `morgan` | `ecfp6`, `maccs` | Fingerprints work well with splits |
| Neural networks (`mlp`, `mc_dropout`) | `descriptors` | `morgan` | Rich features for deep learning |
| Gaussian Process (`gp`) | `descriptors` | `morgan` | Smooth kernels benefit from descriptors |
| Graph neural networks (`chemprop`) | None (SMILES) | `descriptors` (hybrid) | Can optionally use descriptors as x_d |
| Ensembles | Same as base learner | - | Consistency across ensemble members |

## Learner Registry

All available learners with their properties and dependencies:

| Shortcut | Full Name | Type | Uncertainty | GPU Support | Optional Deps | Requires Featurizer |
|----------|-----------|------|-------------|-------------|---------------|---------------------|
| `rf` | RandomForestLearner | Scikit-learn | ✅ | ❌ | - | ✅ |
| `gp` | GaussianProcessLearner | Scikit-learn | ✅ | ❌ | - | ✅ |
| `xgb` | XGBoostLearner | Scikit-learn | ❌ | ❌ | xgboost | ✅ |
| `dt` | DecisionTreeLearner | Scikit-learn | ✅ | ❌ | - | ✅ |
| `lr` | LinearRegressionLearner | Scikit-learn | ✅ | ❌ | - | ✅ |
| `mlp` | MLPLearner | PyTorch | ❌ | ✅ | torch | ✅ |
| `mc_dropout` | MCDropoutLearner | PyTorch | ✅ | ✅ | torch | ✅ |
| `fastprop` | FastpropLearner | PyTorch Lightning | ❌ | ✅ | torch, lightning | ✅ |
| `chemprop` | ChempropLearner | Graph Neural Network | ❌ | ✅ | chemprop, torch | ❌ |
| `chemprop_ensemble` | ChempropEnsemble | Ensemble | ✅ | ✅ | chemprop, torch | ❌ |
| `ensemble` | EnsembleLearner | Ensemble | ✅ | ❌ | - | ✅ |
| `rf_ensemble` | RFEnsemble | Ensemble | ✅ | ❌ | - | ✅ |
| `lr_ensemble` | LREnsemble | Ensemble | ✅ | ❌ | - | ✅ |
| `xgb_ensemble` | XGBEnsemble | Ensemble | ✅ | ❌ | xgboost | ✅ |
| `dt_ensemble` | DTEnsemble | Ensemble | ✅ | ❌ | - | ✅ |
| `mixed_ensemble` | MixedEnsemble | Ensemble | ✅ | ❌ | xgboost | ✅ |
| `fastprop_ensemble` | FastpropEnsemble | Ensemble | ✅ | ✅ | torch, lightning | ✅ |
| `rf_fil` | RfFilLearner | GPU (cuML) | ✅ | ✅ | cuml | ✅ |
| `ridge_cuml` | RidgeCumlLearner | GPU (cuML) | ✅ | ✅ | cuml | ✅ |
| `gpu_gp` | GPyTorchGPLearner | GPU (GPyTorch) | ✅ | ✅ | gpytorch | ✅ |
| `svgp` | SVGPLearner | GPU (GPyTorch) | ✅ | ✅ | gpytorch | ✅ |

**Notes:**

- ✅ = Supported/Required, ❌ = Not supported/Not required
- `rf`, `dt`, `lr` uncertainty: tree std dev, leaf impurity, and leverage-based proxies respectively — suitable for ranking (UCB, EI, PI) but not for absolute calibration analyses
- `chemprop` is a single MPNN model with no uncertainty; use `chemprop_ensemble` for uncertainty via model disagreement
- `ensemble` is a generic ensemble wrapper requiring explicit member specification. Use `mixed_ensemble` (RF + LR + XGB) for a pre-configured ensemble.
- GPU learners (`rf_fil`, `ridge_cuml`, `gpu_gp`, `svgp`) require optional dependencies and are added to the registry only when those imports succeed

## Using Learners

### Python API Usage

Use learner shortcuts or instantiate custom learner instances:

```python
from learnm8 import run_active_learning

# Using shortcut string
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)

# Using custom learner instance
from learnm8.learners import GaussianProcessLearner

custom_learner = GaussianProcessLearner(
    alpha=1e-5,
    random_state=42
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=custom_learner,
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)
```

### CLI Usage

Specify a learner using the `--learner` flag with any shortcut from the registry:

```bash
# Random Forest
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan

# Gaussian Process with uncertainty
learnm8 run compounds.csv --target Activity --learner gp --featurizer descriptors

# Chemprop (featurizer required by CLI but not used by the model)
learnm8 run compounds.csv --target Activity --learner chemprop --featurizer morgan

# Ensemble for robust uncertainty
learnm8 run compounds.csv --target Activity --learner ensemble --featurizer morgan
```

### Listing Available Learners

Discover available learners dynamically:

```python
# Python API
from learnm8.api import list_available_learners
print(list_available_learners())
```

```python
# Python API
from learnm8.api import list_available_learners

available = list_available_learners()
print(available)
# ['rf', 'gp', 'xgb', 'mlp', 'mc_dropout', 'chemprop', 'ensemble', ...]
```

## Uncertainty Quantification Methods

Different learners provide uncertainty through different mechanisms:

| Method | Learners | Mechanism | Computational Cost |
|--------|----------|-----------|-------------------|
| Analytical | `gp` | Gaussian Process posterior variance | Low |
| Monte Carlo Dropout | `mc_dropout` | Multiple forward passes with dropout | Medium (100 passes) |
| Model Disagreement | All ensembles | Variance across ensemble predictions | High (3x training) |
| Graph-based | `chemprop_ensemble` | Ensemble variance across 3 MPNN models | Medium to High |

**Choosing an Uncertainty Method:**

- **Small data, best uncertainty**: `gp` (analytical)
- **Neural networks**: `mc_dropout` (Monte Carlo)
- **Robust uncertainty**: Any ensemble (model disagreement)
- **State-of-the-art**: `chemprop` (graph-based ensemble)

## Performance Considerations

### Training Time

Relative training times for 1000 compounds on CPU (baseline = RF):

| Learner | Relative Time | Parallelization | Notes |
|---------|---------------|-----------------|-------|
| `rf` | 1x | ✅ (n_jobs) | Baseline reference |
| `lr` | 0.5x | ❌ | Fastest training |
| `dt` | 0.5x | ❌ | Very fast |
| `xgb` | 1.5x | ✅ (n_jobs) | Slightly slower than RF |
| `gp` | 3x | ❌ | O(n³) scaling |
| `mlp` | 2x (CPU) | ❌ | Much faster on GPU |
| `mc_dropout` | 2x (CPU) | ❌ | Training same as MLP |
| `chemprop` | 5-10x (CPU) | ❌ | Requires GPU for practicality |
| Ensembles | 3x (base model) | ❌ | 3 models trained sequentially |

**GPU Acceleration Impact:**

- `mlp`, `mc_dropout`, `fastprop`: 5-10x faster on GPU
- `chemprop`: 10-50x faster on GPU (essential for large datasets)

### Prediction Time

Relative prediction times (baseline = RF):

| Learner | Relative Time | Notes |
|---------|---------------|-------|
| `lr` | 0.2x | Fastest predictions |
| `rf`, `xgb`, `dt` | 1x | Fast tree inference |
| `gp` | 1.5x | Depends on training set size |
| `mlp`, `fastprop` | 0.5x | Fast neural network inference |
| `mc_dropout` | 50x | 100 forward passes |
| `chemprop` | 2-5x | Graph computation overhead |
| Ensembles | 3x (base model) | 3 model predictions |

### Memory Requirements

| Learner | Memory Scaling | Large Dataset Viability (>100k) |
|---------|----------------|----------------------------------|
| `lr`, `dt` | O(features) | ✅ Excellent |
| `rf` | O(trees × nodes) | ✅ Good |
| `xgb` | O(trees × nodes) | ✅ Good |
| `gp` | O(n²) | ❌ Poor (limited to ~10k) |
| `mlp`, `mc_dropout`, `fastprop` | O(parameters) | ✅ Good |
| `chemprop` | O(parameters + graphs) | ✅ Good with GPU |
| Ensembles | 3x (base model) | Same as base model |

## Random State and Reproducibility

All learners accept a `random_state` parameter for reproducibility:

```python
# Ensemble learners use derived random states
# Given random_state=42:
# - Model 1: random_state=42
# - Model 2: random_state=123 (42 + 81)
# - Model 3: random_state=356 (42 + 314)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='ensemble',
    target_col='Activity',
    featurizer='morgan',
    random_state=42  # Fully reproducible
)
```

Deterministic behavior across:

- Train/test splits in cross-validation
- Model initialization
- Ensemble member diversity (via offset random states)

## Next Steps

Explore learner categories in detail:

- **[Scikit-learn Models](scikit-learn.md)**: Traditional ML models (RF, GP, XGB, DT, LR)
- **[PyTorch Models](pytorch.md)**: Neural networks (MLP, MC Dropout, FastProp)
- **[Graph Neural Networks](graph-neural-networks.md)**: Chemprop MPNN architecture
- **[Ensembles](ensembles.md)**: Ensemble methods for robust uncertainty

Or learn how to create your own:

- **[Custom Learners](../../customization/custom-learners.md)**: Implement the Learner protocol

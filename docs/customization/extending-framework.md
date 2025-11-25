# Extending the Framework

This guide covers advanced customization of LearnM8 beyond custom learners and acquisition functions, including custom featurizers, pruners, evaluation metrics, and ensemble learners.

## Custom Featurizers

Featurizers convert SMILES strings to numerical representations for machine learning. LearnM8 provides built-in featurizers (morgan, maccs, ecfp6, descriptors), but you can add custom molecular representations.

### Function Signature

Custom featurizers should follow this pattern:

```python
import numpy as np
from typing import List

def custom_featurizer(smiles_list: List[str]) -> np.ndarray:
    """Convert SMILES to custom molecular features.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Numpy array of shape (n_compounds, n_features) with feature vectors
    """
    features = []

    for smiles in smiles_list:
        # Implement your feature extraction logic
        feature_vector = compute_features(smiles)
        features.append(feature_vector)

    return np.array(features)
```

### Example: Simple Descriptor Featurizer

This example creates a featurizer based on basic RDKit descriptors:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import List

def simple_descriptor_featurizer(smiles_list: List[str]) -> np.ndarray:
    """Extract basic molecular descriptors."""
    features = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                # Invalid SMILES - return zero vector
                features.append(np.zeros(10))
                continue

            # Calculate descriptors
            descriptor_vector = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                rdMolDescriptors.CalcNumHeavyAtoms(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol)
            ]

            features.append(descriptor_vector)

        except Exception:
            # On error, use zero vector
            features.append(np.zeros(10))

    return np.array(features, dtype=np.float32)
```

### Integration with extract_features()

Use custom featurizers with LearnM8's caching system:

```python
from learnm8.core.features import extract_features

# Method 1: Direct function call (no caching)
smiles = ['CCO', 'CCC', 'CCCO']
features = simple_descriptor_featurizer(smiles)

# Method 2: With extract_features for caching
from pathlib import Path

# Add your featurizer to the registry
from learnm8.utils import featurizers
featurizers.FEATURIZER_REGISTRY['simple_desc'] = simple_descriptor_featurizer

# Now use with caching
features = extract_features(
    smiles_list=smiles,
    featurizer='simple_desc',
    cache_dir=Path('.cache')
)
```

### Advanced Example: Pharmacophore Features

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Pharmacophore
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
import numpy as np

def pharmacophore_featurizer(smiles_list: List[str], radius=3) -> np.ndarray:
    """Extract pharmacophore fingerprints."""
    factory = Gobbi_Pharm2D.factory

    features = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                features.append(np.zeros(39972))  # Standard 2D pharmacophore length
                continue

            # Generate pharmacophore fingerprint
            fp = Generate.Gen2DFingerprint(mol, factory)

            # Convert to numpy array
            arr = np.zeros(len(fp), dtype=np.uint8)
            for i in range(len(fp)):
                arr[i] = fp[i]

            features.append(arr)

        except Exception:
            features.append(np.zeros(39972))

    return np.array(features)
```

### Parallel Featurizer

For computationally intensive featurization:

```python
from joblib import Parallel, delayed
import numpy as np

def parallel_featurizer(smiles_list: List[str], n_jobs=-1) -> np.ndarray:
    """Compute features in parallel."""

    def compute_single(smiles):
        # Your feature computation logic
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(100)

        # Complex feature calculation
        return compute_complex_features(mol)

    # Parallel computation
    features = Parallel(n_jobs=n_jobs)(
        delayed(compute_single)(smiles) for smiles in smiles_list
    )

    return np.array(features)
```

## Custom Pruners

Pruners reduce the unlabeled compound pool by removing unlikely candidates, improving computational efficiency.

### Pruner Protocol

Custom pruners should implement the `DesignSpacePruner` protocol:

```python
from learnm8.pruning.base import DesignSpacePruner, PruningError
import polars as pl
import numpy as np
from typing import Dict, Any, Optional

class MyCustomPruner(DesignSpacePruner):
    def prune(self,
              compounds: pl.DataFrame,
              predictions: np.ndarray,
              uncertainties: Optional[np.ndarray] = None) -> pl.DataFrame:
        """Prune compounds based on custom criteria."""
        pass

    def get_pruning_stats(self) -> Dict[str, Any]:
        """Return statistics about most recent pruning operation."""
        pass

    def get_name(self) -> str:
        """Return descriptive name for this pruning strategy."""
        pass

    def requires_uncertainty(self) -> bool:
        """Return True if uncertainty estimates are required."""
        return False
```

### Example: Percentile-Based Pruner

```python
from learnm8.pruning.base import DesignSpacePruner
import polars as pl
import numpy as np

class PercentilePruner(DesignSpacePruner):
    """Prune compounds below a prediction percentile."""

    def __init__(self, percentile=25, score_direction='higher'):
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")

        if score_direction not in ['higher', 'lower']:
            raise ValueError("score_direction must be 'higher' or 'lower'")

        self.percentile = percentile
        self.score_direction = score_direction
        self._last_stats = {}

    def prune(self,
              compounds: pl.DataFrame,
              predictions: np.ndarray,
              uncertainties: Optional[np.ndarray] = None) -> pl.DataFrame:

        # Validate inputs
        self.validate_inputs(compounds, predictions, uncertainties)

        n_compounds = len(compounds)

        # Calculate threshold based on percentile
        if self.score_direction == 'higher':
            # Keep compounds above percentile
            threshold = np.percentile(predictions, self.percentile)
            keep_mask = predictions >= threshold
        else:
            # Keep compounds below percentile
            threshold = np.percentile(predictions, 100 - self.percentile)
            keep_mask = predictions <= threshold

        # Prune compounds
        pruned_compounds = self._safe_prune_by_indices(compounds, keep_mask)

        # Store statistics
        self._last_stats = {
            'compounds_before_pruning': n_compounds,
            'compounds_after_pruning': len(pruned_compounds),
            'compounds_pruned': n_compounds - len(pruned_compounds),
            'pruning_fraction': self._calculate_pruning_fraction(n_compounds, len(pruned_compounds)),
            'threshold': float(threshold),
            'percentile': self.percentile,
            'score_direction': self.score_direction
        }

        return pruned_compounds

    def get_pruning_stats(self) -> Dict[str, Any]:
        return self._last_stats.copy()

    def get_name(self) -> str:
        return f"PercentilePruner(p={self.percentile}, dir={self.score_direction})"

    def requires_uncertainty(self) -> bool:
        return False
```

### Example: Uncertainty-Based Pruner

```python
class UncertaintyPercentilePruner(DesignSpacePruner):
    """Prune compounds with high uncertainty (low confidence regions)."""

    def __init__(self, uncertainty_percentile=75):
        if not 0 <= uncertainty_percentile <= 100:
            raise ValueError("uncertainty_percentile must be between 0 and 100")

        self.uncertainty_percentile = uncertainty_percentile
        self._last_stats = {}

    def prune(self,
              compounds: pl.DataFrame,
              predictions: np.ndarray,
              uncertainties: Optional[np.ndarray] = None) -> pl.DataFrame:

        self.validate_inputs(compounds, predictions, uncertainties)

        if uncertainties is None:
            raise PruningError("UncertaintyPercentilePruner requires uncertainty estimates")

        n_compounds = len(compounds)

        # Calculate uncertainty threshold
        threshold = np.percentile(uncertainties, self.uncertainty_percentile)

        # Keep compounds with uncertainty below threshold (more confident)
        keep_mask = uncertainties <= threshold

        pruned_compounds = self._safe_prune_by_indices(compounds, keep_mask)

        self._last_stats = {
            'compounds_before_pruning': n_compounds,
            'compounds_after_pruning': len(pruned_compounds),
            'compounds_pruned': n_compounds - len(pruned_compounds),
            'pruning_fraction': self._calculate_pruning_fraction(n_compounds, len(pruned_compounds)),
            'uncertainty_threshold': float(threshold),
            'uncertainty_percentile': self.uncertainty_percentile
        }

        return pruned_compounds

    def get_pruning_stats(self) -> Dict[str, Any]:
        return self._last_stats.copy()

    def get_name(self) -> str:
        return f"UncertaintyPercentilePruner(p={self.uncertainty_percentile})"

    def requires_uncertainty(self) -> bool:
        return True
```

### Integration with Active Learning

Use custom pruners in `run_active_learning()`:

```python
from learnm8 import run_active_learning, CycleConfig

# Create custom pruner instance
pruner = PercentilePruner(percentile=30, score_direction='higher')

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig(
            strategy='random',
            batch_fraction=0.01,
            pruner=pruner  # Use custom pruner for this cycle
        )
    ]
)
```

## Custom Evaluation Metrics

Add custom metrics to evaluate active learning performance.

### Metric Function Signature

```python
from typing import Dict, Any
import numpy as np

def custom_metric(predictions: np.ndarray,
                  ground_truth: np.ndarray,
                  **kwargs) -> Dict[str, Any]:
    """Calculate custom evaluation metric.

    Args:
        predictions: Model predictions for compounds
        ground_truth: True values for compounds
        **kwargs: Additional parameters (e.g., threshold, top_k)

    Returns:
        Dictionary with metric name as key and value
    """
    metric_value = compute_metric(predictions, ground_truth)

    return {'custom_metric': metric_value}
```

### Example: Precision@K Metric

```python
import numpy as np
from typing import Dict, Any

def precision_at_k(predictions: np.ndarray,
                   ground_truth: np.ndarray,
                   k: int = 100,
                   threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate precision in top-k predicted compounds.

    Args:
        predictions: Model predictions
        ground_truth: True activity values
        k: Number of top compounds to consider
        threshold: Activity threshold for defining actives

    Returns:
        Dictionary with precision@k metric
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have same length")

    k = min(k, len(predictions))  # Handle k > dataset size

    # Get indices of top-k predictions
    top_k_indices = np.argsort(predictions)[-k:]

    # Count true actives in top-k
    actives_in_top_k = np.sum(ground_truth[top_k_indices] >= threshold)

    # Calculate precision
    precision = actives_in_top_k / k

    return {
        f'precision@{k}': precision,
        f'actives_in_top_{k}': int(actives_in_top_k)
    }
```

### Example: Receiver Operating Characteristic AUC

```python
from sklearn.metrics import roc_auc_score, roc_curve

def roc_metrics(predictions: np.ndarray,
                ground_truth: np.ndarray,
                threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate ROC AUC and related metrics.

    Args:
        predictions: Model predictions (continuous)
        ground_truth: True activity values (continuous)
        threshold: Threshold for binarizing ground truth

    Returns:
        Dictionary with ROC-related metrics
    """
    # Binarize ground truth
    binary_labels = (ground_truth >= threshold).astype(int)

    # Calculate ROC AUC
    auc = roc_auc_score(binary_labels, predictions)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(binary_labels, predictions)

    # Find optimal threshold (max Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    return {
        'roc_auc': auc,
        'optimal_threshold': float(optimal_threshold),
        'optimal_tpr': float(tpr[optimal_idx]),
        'optimal_fpr': float(fpr[optimal_idx])
    }
```

### Integration with Evaluation Module

Add custom metrics to LearnM8's evaluation pipeline:

```python
from learnm8.evaluation import metrics

# Register custom metric
metrics.METRIC_REGISTRY['precision_at_k'] = precision_at_k
metrics.METRIC_REGISTRY['roc_metrics'] = roc_metrics

# Now use in evaluation
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    custom_metrics=['precision_at_k', 'roc_metrics']
)

# Access custom metrics in results
for cycle_result in results['cycle_metrics']:
    print(f"Cycle {cycle_result['cycle']}: "
          f"Precision@100 = {cycle_result['precision@100']:.3f}, "
          f"ROC AUC = {cycle_result['roc_auc']:.3f}")
```

## Extending Ensemble Learners

Create custom ensemble combinations for improved uncertainty quantification.

### Using EnsembleLearner with Custom Models

```python
from learnm8.learners.ensemble import EnsembleLearner
from learnm8.learners import get_learner

# Create ensemble with specific model mix
base_learners = [
    get_learner('rf', n_estimators=100),
    get_learner('gp', kernel='rbf'),
    get_learner('xgb', n_estimators=100),
    get_learner('mlp', hidden_dims=[512, 256])
]

ensemble = EnsembleLearner(base_learners=base_learners)

# Use in active learning
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=ensemble,  # Pass ensemble instance
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)
```

### Custom Ensemble Aggregation

Create an ensemble with custom prediction aggregation:

```python
from learnm8.core.interfaces import Learner
import numpy as np
from typing import Tuple, Optional, List

class WeightedEnsembleLearner(Learner):
    """Ensemble with weighted model contributions."""

    def __init__(self, base_learners: List[Learner], weights: Optional[List[float]] = None):
        self.base_learners = base_learners
        self.n_models = len(base_learners)

        if weights is None:
            # Equal weights by default
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            if len(weights) != self.n_models:
                raise ValueError("Number of weights must match number of models")

            # Normalize weights
            self.weights = np.array(weights) / np.sum(weights)

    def train(self, features: np.ndarray, targets: np.ndarray, smiles: Optional[List[str]] = None):
        """Train all base learners."""
        for learner in self.base_learners:
            learner.train(features, targets, smiles)

    def predict(self, features: np.ndarray, smiles: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Weighted ensemble prediction with uncertainty."""
        all_predictions = []
        all_uncertainties = []

        # Get predictions from all models
        for learner in self.base_learners:
            pred, unc = learner.predict(features, smiles)
            all_predictions.append(pred)
            if unc is not None:
                all_uncertainties.append(unc)

        all_predictions = np.array(all_predictions)

        # Weighted mean prediction
        weighted_pred = np.average(all_predictions, axis=0, weights=self.weights)

        # Uncertainty combining prediction uncertainty and model disagreement
        if all_uncertainties:
            all_uncertainties = np.array(all_uncertainties)

            # Weighted prediction variance
            pred_variance = np.average(all_uncertainties ** 2, axis=0, weights=self.weights)

            # Model disagreement
            disagreement = np.average(
                (all_predictions - weighted_pred) ** 2,
                axis=0,
                weights=self.weights
            )

            # Total uncertainty
            uncertainty = np.sqrt(pred_variance + disagreement)
        else:
            # Use model disagreement only
            uncertainty = np.sqrt(
                np.average((all_predictions - weighted_pred) ** 2, axis=0, weights=self.weights)
            )

        return weighted_pred, uncertainty

    def get_name(self) -> str:
        model_names = [learner.get_name() for learner in self.base_learners]
        return f"WeightedEnsemble({', '.join(model_names)})"

    def supports_uncertainty(self) -> bool:
        return True  # Ensemble always provides uncertainty
```

### Adaptive Ensemble Weighting

Dynamically adjust model weights based on performance:

```python
class AdaptiveEnsemble(Learner):
    """Ensemble with performance-based weight adaptation."""

    def __init__(self, base_learners: List[Learner], adaptation_rate=0.1):
        self.base_learners = base_learners
        self.n_models = len(base_learners)
        self.weights = np.ones(self.n_models) / self.n_models
        self.adaptation_rate = adaptation_rate
        self.performance_history = []

    def train(self, features: np.ndarray, targets: np.ndarray, smiles: Optional[List[str]] = None):
        """Train all models and update weights based on validation performance."""
        # Split for validation
        n_val = max(1, int(len(features) * 0.2))
        val_indices = np.random.choice(len(features), n_val, replace=False)
        train_indices = np.setdiff1d(np.arange(len(features)), val_indices)

        X_train, y_train = features[train_indices], targets[train_indices]
        X_val, y_val = features[val_indices], targets[val_indices]

        smiles_train = [smiles[i] for i in train_indices] if smiles else None
        smiles_val = [smiles[i] for i in val_indices] if smiles else None

        # Train all models
        model_errors = []
        for learner in self.base_learners:
            learner.train(X_train, y_train, smiles_train)

            # Evaluate on validation set
            val_pred, _ = learner.predict(X_val, smiles_val)
            mse = np.mean((val_pred - y_val) ** 2)
            model_errors.append(mse)

        # Update weights (inverse of error)
        inverse_errors = 1.0 / (np.array(model_errors) + 1e-8)
        new_weights = inverse_errors / inverse_errors.sum()

        # Smooth weight update
        self.weights = (
            (1 - self.adaptation_rate) * self.weights +
            self.adaptation_rate * new_weights
        )

        self.performance_history.append({
            'model_errors': model_errors,
            'weights': self.weights.copy()
        })

    def predict(self, features: np.ndarray, smiles: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Adaptive weighted prediction."""
        all_predictions = []

        for learner in self.base_learners:
            pred, _ = learner.predict(features, smiles)
            all_predictions.append(pred)

        all_predictions = np.array(all_predictions)

        # Weighted prediction
        weighted_pred = np.average(all_predictions, axis=0, weights=self.weights)

        # Uncertainty from weighted disagreement
        disagreement = np.average(
            (all_predictions - weighted_pred) ** 2,
            axis=0,
            weights=self.weights
        )
        uncertainty = np.sqrt(disagreement)

        return weighted_pred, uncertainty

    def get_name(self) -> str:
        return "AdaptiveEnsemble"

    def supports_uncertainty(self) -> bool:
        return True
```

## Contributing Extensions to LearnM8

If you've developed a useful extension, consider contributing it to LearnM8.

### Testing Requirements

All contributions must include comprehensive tests:

```python
# tests/test_custom_featurizer.py
import pytest
import numpy as np

def test_custom_featurizer_basic():
    """Test basic featurization functionality."""
    smiles = ['CCO', 'CCC', 'CCCO']
    features = simple_descriptor_featurizer(smiles)

    assert features.shape[0] == len(smiles)
    assert features.shape[1] == 10  # Expected feature dimension
    assert features.dtype == np.float32

def test_custom_featurizer_invalid_smiles():
    """Test handling of invalid SMILES."""
    smiles = ['CCO', 'INVALID', 'CCC']
    features = simple_descriptor_featurizer(smiles)

    # Invalid SMILES should produce zero vector
    assert np.all(features[1] == 0)

    # Valid SMILES should have non-zero features
    assert np.any(features[0] != 0)
    assert np.any(features[2] != 0)
```

### Documentation Requirements

Provide clear documentation for your extension:

```python
def custom_featurizer(smiles_list: List[str]) -> np.ndarray:
    """Extract custom molecular features from SMILES.

    This featurizer computes X, Y, and Z features that are useful
    for predicting molecular property ABC.

    Args:
        smiles_list: List of SMILES strings to featurize

    Returns:
        Numpy array of shape (n_compounds, n_features) with feature vectors.
        Invalid SMILES are represented as zero vectors.

    Example:
        >>> smiles = ['CCO', 'CCC']
        >>> features = custom_featurizer(smiles)
        >>> features.shape
        (2, 10)

    Note:
        This featurizer requires RDKit version >= 2023.09.1
    """
    pass
```

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Implement your extension** following LearnM8 coding standards
3. **Add comprehensive tests** with >80% code coverage
4. **Update documentation** including docstrings and user guide
5. **Run the test suite**: `pytest tests/`
6. **Submit a pull request** with clear description of changes

### Code Quality Standards

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Provide comprehensive docstrings (Google or NumPy style)
- Handle errors gracefully with informative messages
- Optimize for performance where relevant
- Maintain compatibility with existing API

## Summary

LearnM8's modular architecture supports extensive customization:

- **Custom Featurizers**: Implement domain-specific molecular representations
- **Custom Pruners**: Create tailored design space reduction strategies
- **Custom Metrics**: Add specialized performance evaluation
- **Custom Ensembles**: Build sophisticated model combinations

All extensions follow consistent patterns with clear interfaces, comprehensive testing, and integration with LearnM8's caching and optimization systems.

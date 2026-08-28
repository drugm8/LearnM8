# Extending the Framework

This guide covers advanced customization of LearnM8 beyond custom learners and acquisition functions: custom featurizers and ensemble learners.

Pruning and evaluation metrics are not extensible at runtime. Pruning resolves `pruning_strategy` against a fixed table whose only entry is `'score'`, and there is no metric registry — adding either means changing the package, not registering from user code.

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

### Calling a custom featurizer

Call the function directly on your SMILES:

```python
smiles = ['CCO', 'CCC', 'CCCO']
features = simple_descriptor_featurizer(smiles)
```

`FEATURIZER_REGISTRY` is a frozenset derived from `_FEATURIZER_CONFIG`, not a mutable mapping, so a featurizer cannot be registered at runtime and custom featurizers do not go through the HDF5 cache. To make one cacheable and usable by name, add an entry to `_FEATURIZER_CONFIG` in `learnm8/features/__init__.py` — see [Contributing Extensions](#contributing-extensions-to-learnm8) below.

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

## Extending Ensemble Learners

Create custom ensemble combinations for improved uncertainty quantification.

### Using EnsembleLearner with Custom Models

```python
from learnm8.learners.ensemble import EnsembleLearner
from learnm8.learners.sklearn import (
    GaussianProcessLearner,
    RandomForestLearner,
    XGBoostLearner,
)
from learnm8.learners.torch import MLPLearner

# Create ensemble with specific model mix
learners = [
    RandomForestLearner(n_estimators=100),
    GaussianProcessLearner(),
    XGBoostLearner(n_estimators=100),
    MLPLearner(hidden_sizes=[512, 256])
]

ensemble = EnsembleLearner(learners=learners)

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

    def predict(self, features: np.ndarray, smiles: Optional[List[str]] = None,
                *, compute_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

    def predict(self, features: np.ndarray, smiles: Optional[List[str]] = None,
                *, compute_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

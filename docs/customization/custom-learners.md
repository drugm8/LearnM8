# Custom Learners

LearnM8's learner system is designed for extensibility through a clean protocol-based interface. You can implement custom machine learning models by following the `Learner` protocol without modifying the framework source code.

## Learner Protocol

All learners in LearnM8 implement the `Learner` protocol defined in `learnm8.core.interfaces`. This protocol specifies the contract that any learner must fulfill:

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import numpy as np

class Learner(ABC):
    """Base class for all machine learning models."""

    @abstractmethod
    def train(self,
              features: np.ndarray,
              targets: np.ndarray,
              smiles: Optional[List[str]] = None) -> None:
        """Train the model on feature matrix or SMILES.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
            smiles: Optional SMILES strings (required by some learners)

        Raises:
            ValueError: If input shapes invalid
            RuntimeError: If training fails
        """
        pass

    @abstractmethod
    def predict(self,
                features: np.ndarray,
                smiles: Optional[List[str]] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict on feature matrix or SMILES.

        Args:
            features: Feature matrix (n_samples, n_features)
            smiles: Optional SMILES strings (required by some learners)

        Returns:
            Tuple of (predictions, uncertainties).
            uncertainties can be None if model doesn't provide uncertainty.

        Raises:
            RuntimeError: If model is not trained or prediction fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a descriptive name for this learner.

        Returns:
            String identifier for the learner type and configuration
        """
        pass

    def supports_uncertainty(self) -> bool:
        """Return True if this learner can provide uncertainty estimates.

        Returns:
            Boolean indicating uncertainty support
        """
        return False

    def requires_smiles(self) -> bool:
        """Return True if this learner needs SMILES strings.

        Returns:
            False by default (feature-based learners)
        """
        return False
```

## Method Signatures

### train()

The `train()` method trains the model on labeled data.

**Parameters:**

- `features` (np.ndarray): Feature matrix with shape `(n_samples, n_features)`. For feature-based learners, these are pre-computed molecular features (fingerprints or descriptors). For graph neural networks, this may be ignored.
- `targets` (np.ndarray): Target values with shape `(n_samples,)`. This is a 1D array of regression values or classification labels.
- `smiles` (`Optional[List[str]]`): SMILES strings for each compound. Required by learners that work directly with molecular structures (e.g., Chemprop). Default is None for feature-based learners.

**Returns:** None (modifies learner state)

**Raises:**

- `ValueError`: If input shapes are incompatible or data is invalid
- `RuntimeError`: If training fails due to model errors

**Example shapes:**

```python
features.shape  # (100, 2048) for 100 compounds with Morgan fingerprints
targets.shape   # (100,) for 100 activity values
len(smiles)     # 100 SMILES strings (if required)
```

### predict()

The `predict()` method generates predictions and optional uncertainty estimates on new data.

**Parameters:**

- `features` (np.ndarray): Feature matrix with shape `(n_samples, n_features)`. Must have the same number of features as the training data.
- `smiles` (`Optional[List[str]]`): SMILES strings for prediction. Required only if `requires_smiles()` returns True.

**Returns:** `Tuple[np.ndarray, Optional[np.ndarray]]`

- First element: Predictions array with shape `(n_samples,)`
- Second element: Uncertainty estimates with shape `(n_samples,)` or None if not available

**Raises:**

- `RuntimeError`: If model is not trained or prediction fails

**Example return values:**

```python
predictions, uncertainties = learner.predict(features)
predictions.shape     # (50,) for 50 compounds
uncertainties.shape   # (50,) or None if not supported
```

### get_name()

The `get_name()` method returns a human-readable identifier for the learner.

**Returns:** str - Descriptive name including model type and key hyperparameters

**Example names:**

```python
"RandomForest(n_estimators=100,unlimited_depth)"
"GaussianProcess(RBF,α=1e-10)"
"TorchMLPLearner"
```

### supports_uncertainty()

The `supports_uncertainty()` method indicates whether the learner provides uncertainty estimates.

**Returns:** bool - True if `predict()` returns non-None uncertainties

**Implementation:**

```python
def supports_uncertainty(self) -> bool:
    return True  # Override to True if providing uncertainties
```

### requires_smiles()

The `requires_smiles()` method indicates whether the learner needs SMILES strings.

**Returns:** bool - True if learner works directly with molecular structures

**When to override:**

- Graph neural networks (Chemprop): Return True
- Feature-based learners (RF, GP, XGB): Return False (default)

## Implementing a Custom Learner

Here's a complete step-by-step guide to implementing a custom learner.

### Step 1: Choose Your Base Class

LearnM8 provides two base classes for common scenarios:

**SklearnLearner** - For scikit-learn compatible models:

```python
from learnm8.learners.base import SklearnLearner
```

**TorchLearner** - For PyTorch neural networks:

```python
from learnm8.learners.base import TorchLearner
```

**From scratch** - Implement `Learner` protocol directly for maximum control.

### Step 2: Implement Required Methods

#### Example: Custom Support Vector Regression Learner

```python
from learnm8.core.interfaces import Learner
from typing import Tuple, Optional, List
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class CustomSVRLearner(Learner):
    """Support Vector Regression learner with uncertainty estimation."""

    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 gamma: str = 'scale',
                 random_state: int = 42):
        """Initialize SVR learner.

        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
            epsilon: Epsilon in epsilon-SVR model
            gamma: Kernel coefficient
            random_state: Random seed for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.random_state = random_state

        self.model = None
        self.scaler = None
        self.is_trained = False

    def train(self,
              features: np.ndarray,
              targets: np.ndarray,
              smiles: Optional[List[str]] = None) -> None:
        """Train SVR model on feature matrix.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
            smiles: Unused for SVR (feature-based learner)

        Raises:
            ValueError: If input shapes invalid
            RuntimeError: If training fails
        """
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Features and targets must have same length: "
                f"{features.shape[0]} vs {targets.shape[0]}"
            )

        if features.shape[0] == 0:
            raise ValueError("Cannot train on empty dataset")

        logger.info(f"Training {self.get_name()} on {len(features)} samples")

        try:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features)

            self.model = SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon,
                gamma=self.gamma
            )

            self.model.fit(X_scaled, targets)
            self.is_trained = True

            logger.info(f"Successfully trained {self.get_name()}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"SVR training failed: {e}") from e

    def predict(self,
                features: np.ndarray,
                smiles: Optional[List[str]] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict on feature matrix with distance-based uncertainty.

        Args:
            features: Feature matrix (n_samples, n_features)
            smiles: Unused for SVR (feature-based learner)

        Returns:
            Tuple of (predictions, uncertainties).
            Uncertainty estimated from distance to support vectors.

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        try:
            X_scaled = self.scaler.transform(features)
            predictions = self.model.predict(X_scaled)

            uncertainties = self._estimate_uncertainty(X_scaled)

            logger.debug(f"Predicted {len(predictions)} samples")

            return predictions, uncertainties

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"SVR prediction failed: {e}") from e

    def _estimate_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Estimate uncertainty from distance to support vectors.

        Args:
            X: Scaled feature matrix

        Returns:
            Uncertainty estimates based on support vector distances
        """
        support_vectors = self.model.support_vectors_

        min_distances = np.zeros(len(X))
        for i, x in enumerate(X):
            distances = np.linalg.norm(support_vectors - x, axis=1)
            min_distances[i] = np.min(distances)

        uncertainties = min_distances / (np.max(min_distances) + 1e-10)

        return uncertainties

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f"SVR(kernel={self.kernel},C={self.C},ε={self.epsilon})"

    def supports_uncertainty(self) -> bool:
        """Return True since we provide distance-based uncertainty."""
        return True

    def requires_smiles(self) -> bool:
        """Return False since SVR uses pre-computed features."""
        return False
```

### Step 3: Add Input Validation

Robust input validation prevents downstream errors:

```python
def train(self, features: np.ndarray, targets: np.ndarray,
          smiles: Optional[List[str]] = None) -> None:

    if not isinstance(features, np.ndarray):
        raise TypeError(f"features must be np.ndarray, got {type(features)}")

    if not isinstance(targets, np.ndarray):
        raise TypeError(f"targets must be np.ndarray, got {type(targets)}")

    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")

    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D, got shape {targets.shape}")

    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Sample count mismatch: features={features.shape[0]}, "
            f"targets={targets.shape[0]}"
        )

    if np.any(np.isnan(features)):
        raise ValueError("features contains NaN values")

    if np.any(np.isnan(targets)):
        raise ValueError("targets contains NaN values")
```

### Step 4: Implement Uncertainty Estimation

Uncertainty estimation varies by model type:

#### Random Forest - Tree Variance

```python
def predict(self, features: np.ndarray, smiles: Optional[List[str]] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    X_scaled = self.scaler.transform(features)

    tree_predictions = np.array([
        tree.predict(X_scaled) for tree in self.model.estimators_
    ])

    predictions = tree_predictions.mean(axis=0)
    uncertainties = tree_predictions.std(axis=0)

    return predictions, uncertainties
```

#### Gaussian Process - Native Uncertainty

```python
def predict(self, features: np.ndarray, smiles: Optional[List[str]] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    X_scaled = self.scaler.transform(features)

    predictions, std = self.model.predict(X_scaled, return_std=True)

    return predictions, std
```

#### Neural Network - Monte Carlo Dropout

```python
def predict(self, features: np.ndarray, smiles: Optional[List[str]] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    X_tensor = torch.FloatTensor(self.scaler.transform(features))

    self.model.train()

    predictions_list = []
    for _ in range(100):
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy().squeeze()
            predictions_list.append(pred)

    predictions_array = np.array(predictions_list)
    predictions = predictions_array.mean(axis=0)
    uncertainties = predictions_array.std(axis=0)

    return predictions, uncertainties
```

## Integrating Your Learner

Once implemented, your custom learner integrates directly into LearnM8 through dependency injection.

### Using Custom Learner in API

```python
from learnm8 import run_active_learning
from my_module import CustomSVRLearner
import polars as pl

compounds = pl.read_csv('compounds.csv')

custom_learner = CustomSVRLearner(
    kernel='rbf',
    C=10.0,
    epsilon=0.05,
    random_state=42
)

results = run_active_learning(
    compound_pool=compounds,
    oracle='oracle.csv',
    learner=custom_learner,
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.01),
        ('ucb', 0.005),
        ('ucb', 0.005),
        ('greedy', 0.005)
    ]
)
```

### No Registration Required

LearnM8 uses dependency injection rather than a global registry:

```python
learner_instance = CustomSVRLearner()

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=learner_instance,
    target_col='Activity',
    featurizer='descriptors'
)
```

The framework calls your learner's methods directly without any registration step.

### Example Workflow

Complete workflow with custom learner:

```python
import polars as pl
from learnm8 import run_active_learning
from my_learners import CustomSVRLearner

compounds = pl.DataFrame({
    'ID': ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5'],
    'SMILES': ['CCO', 'CCC', 'CCCO', 'CCCC', 'CCCCO'],
    'Activity': [0.8, 0.6, 0.9, 0.7, 0.85]
})

learner = CustomSVRLearner(
    kernel='rbf',
    C=5.0,
    epsilon=0.1
)

results = run_active_learning(
    compound_pool=compounds,
    oracle=compounds,
    learner=learner,
    target_col='Activity',
    featurizer='morgan',
    n_cycles=5,
    batch_fraction=0.2
)

print(f"Final enrichment: {results['aggregate_metrics']['final_enrichment']:.2f}")
```

## Best Practices

### Input Validation

Always validate inputs in `train()` and `predict()`:

```python
def train(self, features, targets, smiles=None):
    if features.shape[0] != targets.shape[0]:
        raise ValueError("Sample count mismatch")

    if features.shape[0] < 2:
        raise ValueError("Need at least 2 samples for training")

    if np.any(np.isinf(features)):
        raise ValueError("features contains inf values")
```

### Error Handling

Wrap model operations in try-except blocks:

```python
def train(self, features, targets, smiles=None):
    try:
        X_scaled = self.scaler.fit_transform(features)
        self.model.fit(X_scaled, targets)
        self.is_trained = True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Model training failed: {e}") from e
```

### State Management

Track training state to prevent prediction before training:

```python
def __init__(self):
    self.is_trained = False
    self.model = None

def train(self, features, targets, smiles=None):
    self.model.fit(features, targets)
    self.is_trained = True

def predict(self, features, smiles=None):
    if not self.is_trained:
        raise RuntimeError("Model must be trained before prediction")
    return self.model.predict(features), None
```

### Performance Considerations

For large-scale applications:

```python
class EfficientLearner(Learner):
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(n_jobs=n_jobs)

    def train(self, features, targets, smiles=None):
        if len(features) > 10000:
            logger.warning("Large dataset, training may take time")

        self.model.fit(features, targets)
        self.is_trained = True
```

### Testing Your Learner

Comprehensive testing ensures correctness:

```python
import pytest
import numpy as np

def test_custom_learner_training():
    learner = CustomSVRLearner()

    features = np.random.rand(100, 50)
    targets = np.random.rand(100)

    learner.train(features, targets)

    assert learner.is_trained
    assert learner.model is not None

def test_custom_learner_prediction():
    learner = CustomSVRLearner()

    features = np.random.rand(100, 50)
    targets = np.random.rand(100)
    learner.train(features, targets)

    test_features = np.random.rand(20, 50)
    predictions, uncertainties = learner.predict(test_features)

    assert predictions.shape == (20,)
    assert uncertainties.shape == (20,)
    assert np.all(uncertainties >= 0)

def test_custom_learner_untrained_error():
    learner = CustomSVRLearner()

    features = np.random.rand(20, 50)

    with pytest.raises(RuntimeError, match="must be trained"):
        learner.predict(features)

def test_custom_learner_shape_mismatch():
    learner = CustomSVRLearner()

    features = np.random.rand(100, 50)
    targets = np.random.rand(90)

    with pytest.raises(ValueError, match="same length"):
        learner.train(features, targets)

def test_custom_learner_supports_uncertainty():
    learner = CustomSVRLearner()
    assert learner.supports_uncertainty() is True

def test_custom_learner_requires_smiles():
    learner = CustomSVRLearner()
    assert learner.requires_smiles() is False
```

Run tests with:

```bash
pytest test_custom_learner.py -v
```

### Logging

Use Python logging for debugging and monitoring:

```python
import logging

logger = logging.getLogger(__name__)

class CustomLearner(Learner):
    def train(self, features, targets, smiles=None):
        logger.info(f"Training {self.get_name()} on {len(features)} samples")

        self.model.fit(features, targets)

        logger.debug(f"Model parameters: {self.model.get_params()}")
        logger.info(f"Training complete")
```

### Documentation

Document your learner with comprehensive docstrings:

````python
class CustomSVRLearner(Learner):
    """Support Vector Regression learner with distance-based uncertainty.

    This learner uses scikit-learn's SVR with uncertainty estimates
    derived from distances to support vectors. Suitable for small to
    medium datasets where kernel methods are effective.

    Attributes:
        kernel: Kernel type (rbf, linear, poly)
        C: Regularization parameter
        epsilon: Epsilon in epsilon-SVR model
        model: Trained SVR model instance
        scaler: StandardScaler for feature normalization
        is_trained: Whether model has been trained

    Example:
        ```python
        learner = CustomSVRLearner(kernel='rbf', C=10.0)
        learner.train(features, targets)
        predictions, uncertainties = learner.predict(test_features)
        ```
    """
````

By following these guidelines, you can create custom learners that integrate seamlessly with LearnM8's active learning framework while maintaining robustness and performance.

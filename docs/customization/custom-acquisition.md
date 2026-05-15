# Custom Acquisition Functions

This guide explains how to implement custom acquisition strategies for LearnM8's active learning framework. Acquisition functions determine which compounds are selected for measurement in each cycle, making them central to active learning performance.

## AcquisitionFunction Protocol

All acquisition functions must implement the `AcquisitionFunction` protocol defined in `learnm8.core.interfaces`:

```python
from abc import ABC, abstractmethod
import polars as pl

class AcquisitionFunction(ABC):
    @abstractmethod
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select compounds for labeling."""
        pass

    def requires_uncertainty(self) -> bool:
        """Return True if uncertainty estimates are required."""
        return False

    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        return self.__class__.__name__
```

## Method Signatures

### select() Method

The core method that performs compound selection:

**Input:**

- `compounds`: Polars DataFrame with columns:
  - `ID` (str): Unique compound identifier
  - `SMILES` (str): Molecular structure
  - `prediction` (float): Model predictions
  - `uncertainty` (float, optional): Model uncertainty estimates

**Output:**

- Polars DataFrame subset with selected compounds (same schema as input)

**Validation Requirements:**

- Check that required columns exist
- Validate `n_select` is positive and within bounds
- Ensure predictions/uncertainties are valid (no NaN/infinite values)

### requires_uncertainty() Method

Optional method indicating if uncertainty estimates are needed:

```python
def requires_uncertainty(self) -> bool:
    return True  # Override if your strategy needs uncertainties
```

If `True`, the framework ensures that only learners providing uncertainties are used with this acquisition function.

## Implementing a Custom Acquisition Function

### Step 1: Basic Structure

Create a class extending `AcquisitionFunction` from `learnm8.acquisition.base`:

```python
from learnm8.acquisition.base import AcquisitionFunction
import polars as pl
import numpy as np

class MyCustomAcquisition(AcquisitionFunction):
    def __init__(self, my_parameter=1.0, score_direction='higher', **kwargs):
        super().__init__(score_direction=score_direction, **kwargs)
        self.my_parameter = my_parameter
```

**Important:** Always call `super().__init__()` to initialize base class properties like `score_direction` and `maximize`.

### Step 2: Implement select() Method

```python
def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
    # Step 1: Validate input
    self.validate_input(compounds, n_select)

    # Step 2: Extract predictions (and uncertainties if needed)
    predictions = compounds.get_column('prediction').to_numpy()

    # Step 3: Calculate acquisition scores
    scores = self._calculate_acquisition_scores(predictions)

    # Step 4: Select top compounds
    selected = self._safe_select_top_k(
        compounds,
        scores,
        n_select,
        ascending=False  # True for lowest scores, False for highest
    )

    return selected

def _calculate_acquisition_scores(self, predictions):
    # Implement your scoring logic
    return predictions * self.my_parameter
```

### Step 3: Implement Helper Methods

Override optional methods as needed:

```python
def requires_uncertainty(self) -> bool:
    return False  # Change to True if uncertainty is needed

def get_name(self) -> str:
    return f"MyCustomAcquisition(param={self.my_parameter})"
```

## Complete Example: Distance-Weighted Acquisition

This example implements an acquisition function that selects compounds based on a weighted combination of predictions and distance from already-labeled compounds:

```python
from learnm8.acquisition.base import AcquisitionFunction
import polars as pl
import numpy as np
from typing import Optional

class DistanceWeightedAcquisition(AcquisitionFunction):
    """Select compounds based on prediction quality and distance from labeled set."""

    def __init__(self,
                 distance_weight=0.5,
                 score_direction='higher',
                 labeled_smiles: Optional[list] = None,
                 **kwargs):
        super().__init__(score_direction=score_direction, **kwargs)

        if not 0 <= distance_weight <= 1:
            raise ValueError("distance_weight must be between 0 and 1")

        self.distance_weight = distance_weight
        self.labeled_smiles = labeled_smiles or []

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        # Validate input
        self.validate_input(compounds, n_select)

        # Extract predictions
        predictions = compounds.get_column('prediction').to_numpy()
        smiles_list = compounds.get_column('SMILES').to_list()

        # Calculate diversity scores
        diversity_scores = self._calculate_diversity(smiles_list)

        # Normalize both scores to [0, 1]
        norm_predictions = self._normalize_scores(predictions)
        norm_diversity = self._normalize_scores(diversity_scores)

        # Combine prediction quality and diversity
        combined_scores = (
            (1 - self.distance_weight) * norm_predictions +
            self.distance_weight * norm_diversity
        )

        # Adjust for score direction
        if not self.maximize:
            combined_scores = 1 - combined_scores

        # Select top compounds
        selected = self._safe_select_top_k(
            compounds,
            combined_scores,
            n_select,
            ascending=False
        )

        return selected

    def _calculate_diversity(self, smiles_list):
        """Calculate diversity as minimum Tanimoto distance to labeled compounds."""
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        if not self.labeled_smiles:
            # No labeled compounds yet, all equally diverse
            return np.ones(len(smiles_list))

        diversity_scores = []

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                diversity_scores.append(0.0)
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

            # Calculate minimum similarity to any labeled compound
            min_similarity = 1.0
            for labeled_smiles in self.labeled_smiles:
                labeled_mol = Chem.MolFromSmiles(labeled_smiles)
                if labeled_mol is not None:
                    labeled_fp = AllChem.GetMorganFingerprintAsBitVect(
                        labeled_mol, radius=2, nBits=2048
                    )
                    similarity = DataStructs.TanimotoSimilarity(fp, labeled_fp)
                    min_similarity = min(min_similarity, similarity)

            # Convert similarity to diversity (distance)
            diversity = 1 - min_similarity
            diversity_scores.append(diversity)

        return np.array(diversity_scores)

    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] range."""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def update_labeled_smiles(self, new_smiles):
        """Update the set of labeled SMILES after each cycle."""
        self.labeled_smiles.extend(new_smiles)

    def get_name(self) -> str:
        return f"DistanceWeighted(w={self.distance_weight})"

    def requires_uncertainty(self) -> bool:
        return False
```

## Handling Uncertainty

For acquisition functions that use uncertainty estimates:

```python
from learnm8.acquisition.base import validate_uncertainty_inputs

class UncertaintyBasedAcquisition(AcquisitionFunction):
    def __init__(self, exploration_factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.exploration_factor = exploration_factor

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        self.validate_input(compounds, n_select)

        # Extract and validate predictions and uncertainties
        predictions, uncertainties = validate_uncertainty_inputs(compounds)

        # Calculate acquisition scores (e.g., UCB-style)
        if self.maximize:
            scores = predictions + self.exploration_factor * uncertainties
        else:
            scores = predictions - self.exploration_factor * uncertainties

        selected = self._safe_select_top_k(
            compounds, scores, n_select, ascending=not self.maximize
        )

        return selected

    def requires_uncertainty(self) -> bool:
        return True  # This strategy requires uncertainties
```

The `validate_uncertainty_inputs()` helper automatically:

- Checks that `uncertainty` column exists
- Validates no NaN or negative values
- Returns numpy arrays ready for computation

## Validation Helpers

The base class provides validation utilities:

### validate_input()

Checks basic DataFrame structure and column requirements:

```python
def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
    # Automatically validates:
    # - DataFrame is not empty
    # - Required columns exist (ID, SMILES, prediction)
    # - Uncertainty column exists if requires_uncertainty() is True
    # - n_select is positive
    # - No NaN values in predictions/uncertainties
    # - No duplicate IDs
    self.validate_input(compounds, n_select)
```

### _safe_select_top_k()

Safely selects top-k compounds with automatic handling of edge cases:

```python
selected = self._safe_select_top_k(
    compounds,      # Input DataFrame
    scores,         # Numpy array of acquisition scores
    n_select,       # Number to select
    ascending=False # False for highest scores, True for lowest
)
```

This method:

- Handles infinite/NaN scores gracefully
- Ensures score array length matches DataFrame
- Adds `acquisition_score` column to output
- Sorts output by acquisition score

## Integration with LearnM8

### Using Custom Acquisition in CLI

Custom acquisition functions cannot be used directly via CLI (only registered strategies). For custom strategies, use the Python API.

### Using Custom Acquisition in Python API

Pass a custom acquisition instance directly:

```python
from learnm8 import run_active_learning
from my_module import MyCustomAcquisition

# Create custom acquisition instance
custom_acquisition = MyCustomAcquisition(my_parameter=2.5)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    strategy=custom_acquisition  # Pass instance directly
)
```

### Passing Parameters via acquisition_params

For registered strategies or per-cycle customization:

```python
from learnm8 import run_active_learning, CycleConfig

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
            acquisition_params={'random_state': 42}
        ),
        CycleConfig(
            strategy='ucb',
            batch_fraction=0.005,
            acquisition_params={'beta': 3.0}  # Custom beta value
        )
    ]
)
```

## Best Practices

### Input Validation

Always validate inputs at the start of `select()`:

```python
def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
    # Required validation
    self.validate_input(compounds, n_select)

    # Additional custom validation
    if 'custom_column' in self.required_columns:
        if 'custom_column' not in compounds.columns:
            raise ValueError("Missing required custom_column")
```

### Polars vs Pandas

LearnM8 uses **Polars internally** for 10-50x performance gains:

```python
# Extract columns as numpy arrays for computation
predictions = compounds.get_column('prediction').to_numpy()
smiles_list = compounds.get_column('SMILES').to_list()

# Filter DataFrame (Polars syntax)
filtered = compounds.filter(pl.col('prediction') > threshold)

# Add new columns
result = compounds.with_columns(
    pl.Series('acquisition_score', scores)
)

# Use standard Python indexing for row selection
selected_compounds = compounds[selected_indices]
```

**Don't convert to Pandas** unless absolutely necessary (e.g., interfacing with libraries that only accept Pandas).

### Score Direction Handling

Respect the `score_direction` parameter:

```python
def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
    # Use self.maximize (set by score_direction parameter)
    if self.maximize:
        scores = predictions + bonus
    else:
        scores = predictions - bonus

    # _safe_select_top_k handles sorting correctly
    selected = self._safe_select_top_k(
        compounds, scores, n_select, ascending=not self.maximize
    )
```

### Error Handling

Provide informative error messages:

```python
def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
    try:
        self.validate_input(compounds, n_select)
        scores = self._calculate_scores(compounds)
        return self._safe_select_top_k(compounds, scores, n_select, ascending=False)
    except KeyError as e:
        raise ValueError(f"Missing required column in compounds DataFrame: {e}")
    except Exception as e:
        raise RuntimeError(f"Acquisition selection failed: {e}") from e
```

### Testing Your Acquisition Function

Comprehensive testing ensures reliability:

```python
import pytest
import polars as pl
import numpy as np

def test_custom_acquisition_basic():
    """Test basic selection functionality."""
    compounds = pl.DataFrame({
        'ID': ['comp1', 'comp2', 'comp3', 'comp4'],
        'SMILES': ['CCO', 'CCC', 'CCCO', 'CCCC'],
        'prediction': [0.9, 0.3, 0.7, 0.1]
    })

    acquisition = MyCustomAcquisition(my_parameter=2.0)
    selected = acquisition.select(compounds, n_select=2)

    assert len(selected) == 2
    assert 'acquisition_score' in selected.columns
    assert set(selected['ID']) == {'comp1', 'comp3'}

def test_custom_acquisition_with_uncertainty():
    """Test with uncertainty estimates."""
    compounds = pl.DataFrame({
        'ID': ['comp1', 'comp2', 'comp3'],
        'SMILES': ['CCO', 'CCC', 'CCCO'],
        'prediction': [0.8, 0.6, 0.9],
        'uncertainty': [0.1, 0.3, 0.05]
    })

    acquisition = UncertaintyBasedAcquisition(exploration_factor=2.0)
    selected = acquisition.select(compounds, n_select=2)

    assert len(selected) == 2
    assert acquisition.requires_uncertainty()

def test_custom_acquisition_edge_cases():
    """Test edge cases and error handling."""
    compounds = pl.DataFrame({
        'ID': ['comp1'],
        'SMILES': ['CCO'],
        'prediction': [0.5]
    })

    acquisition = MyCustomAcquisition()

    # Request more than available
    selected = acquisition.select(compounds, n_select=10)
    assert len(selected) == 1

    # Empty DataFrame should raise error
    empty_df = pl.DataFrame({'ID': [], 'SMILES': [], 'prediction': []})
    with pytest.raises(ValueError):
        acquisition.select(empty_df, n_select=1)

def test_custom_acquisition_score_direction():
    """Test score direction handling."""
    compounds = pl.DataFrame({
        'ID': ['comp1', 'comp2', 'comp3'],
        'SMILES': ['CCO', 'CCC', 'CCCO'],
        'prediction': [0.9, 0.3, 0.7]
    })

    # Test 'higher' direction
    acq_higher = MyCustomAcquisition(score_direction='higher')
    selected_higher = acq_higher.select(compounds, n_select=1)
    assert selected_higher['ID'][0] == 'comp1'  # Highest prediction

    # Test 'lower' direction
    acq_lower = MyCustomAcquisition(score_direction='lower')
    selected_lower = acq_lower.select(compounds, n_select=1)
    assert selected_lower['ID'][0] == 'comp2'  # Lowest prediction
```

## Advanced Patterns

### Stateful Acquisition

Acquisition functions that adapt based on previous cycles:

```python
class AdaptiveAcquisition(AcquisitionFunction):
    def __init__(self, initial_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.weight = initial_weight
        self.cycle_count = 0

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        self.validate_input(compounds, n_select)

        predictions = compounds.get_column('prediction').to_numpy()

        # Adapt weight based on cycle count
        adaptive_weight = self.weight * (0.95 ** self.cycle_count)
        scores = predictions * adaptive_weight

        self.cycle_count += 1

        return self._safe_select_top_k(
            compounds, scores, n_select, ascending=False
        )
```

### Multi-Objective Acquisition

Combining multiple criteria:

```python
class MultiObjectiveAcquisition(AcquisitionFunction):
    def __init__(self,
                 prediction_weight=0.5,
                 uncertainty_weight=0.3,
                 diversity_weight=0.2,
                 **kwargs):
        super().__init__(**kwargs)

        total = prediction_weight + uncertainty_weight + diversity_weight
        self.pred_w = prediction_weight / total
        self.unc_w = uncertainty_weight / total
        self.div_w = diversity_weight / total

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        self.validate_input(compounds, n_select)

        predictions, uncertainties = validate_uncertainty_inputs(compounds)

        # Normalize all components
        norm_pred = (predictions - predictions.min()) / (predictions.max() - predictions.min())
        norm_unc = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())

        # Calculate diversity (simplified)
        diversity = self._calculate_diversity(compounds)
        norm_div = (diversity - diversity.min()) / (diversity.max() - diversity.min())

        # Weighted combination
        scores = (
            self.pred_w * norm_pred +
            self.unc_w * norm_unc +
            self.div_w * norm_div
        )

        return self._safe_select_top_k(
            compounds, scores, n_select, ascending=False
        )

    def requires_uncertainty(self) -> bool:
        return self.unc_w > 0
```

Custom acquisition functions provide powerful flexibility for optimizing active learning strategies to specific molecular screening challenges.

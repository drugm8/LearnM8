# LearnM8: Complete Architecture Documentation

## Table of Contents

1. [Package Overview](#package-overview)
2. [Architectural Philosophy](#architectural-philosophy)
3. [System Architecture](#system-architecture)
4. [Core Interfaces](#core-interfaces)
5. [Model System](#model-system)
6. [Data Management Architecture](#data-management-architecture)
7. [Active Learning Orchestration](#active-learning-orchestration)
8. [Acquisition Functions](#acquisition-functions)
9. [Design Space Pruning](#design-space-pruning)
10. [Command-Line Interface](#command-line-interface)
11. [Functional API](#functional-api)
12. [Evaluation System](#evaluation-system)
13. [Extension Guidelines](#extension-guidelines)
14. [Technical Specifications](#technical-specifications)

---

## Package Overview

LearnM8 is a comprehensive active learning framework for molecular screening that combines state-of-the-art machine learning with sophisticated uncertainty quantification and production-ready performance optimization. The framework adopts a **hybrid functional-object-oriented architecture** that prioritizes simplicity for new users while maintaining extensibility for advanced applications.

---

## Architectural Philosophy

### 1. **Functional-First Design**
The core API is built around pure functions rather than complex object hierarchies:
- **Pure functional orchestration**: Active learning cycles as immutable transformations
- **No complex state management**: Simple variables and DataFrames instead of state objects
- **Functional cycle execution**: `execute_single_cycle()` as a pure function
- **Strategy dispatch**: Simple string-based strategy selection without class hierarchies

### 2. **Dependency Injection Over Tight Coupling**
Components receive their dependencies explicitly rather than creating them internally:
- Learners receive DataManager instances for feature extraction
- Clear testing boundaries and mockable interfaces
- Eliminates hidden dependencies and circular imports
- Promotes clean separation of concerns

### 3. **Duck Typing Over Complex Inheritance**
Capabilities are determined by behavior, not inheritance hierarchies:
- Uncertainty support: check if `predict()` returns non-None uncertainty
- GPU support: check for device attributes
- No multiple inheritance confusion or diamond problems
- Simplified testing and component swapping

### 4. **Composition Over Inheritance**
Complex behaviors built by combining simple components:
- Ensemble uncertainty via multiple learners composition
- Functional strategy composition in acquisition functions
- Reduces inheritance depth and coupling
- Enables flexible runtime component assembly

### 5. **Centralized Data Management**
All data operations flow through a single DataManager:
- Unified HDF5-based feature extraction and caching
- Memory optimization and compression
- Eliminates data duplication across components
- Clear performance bottleneck identification

### 6. **Hybrid Architecture**
Balances functional simplicity with object-oriented extensibility:
- **Functional core**: Pure functional API for most users
- **Object-oriented extension**: Rich interfaces for advanced customization
- **Legacy compatibility**: Extensive backward compatibility support
- **Progressive disclosure**: Simple defaults with advanced options available

---

## System Architecture

### High-Level Component Organization

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Interface                           │
│           (learnm8.cli.cli - Functional)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Functional API                              │
│              (learnm8.learnm8)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│            Active Learning Engine                           │
│            (learnm8.py - Pure Functions)                   │
└─────┬─────────┬─────────┬─────────┬─────────┬───────────────┘
      │         │         │         │         │
      ▼         ▼         ▼         ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐
│ Models  │ │Acquisition│ │Uncertain│ │ Pruning │ │ Data Flow   │
│ System  │ │Functions  │ │ty System│ │ System  │ │ Management  │
│         │ │           │ │ (Duck   │ │         │ │ (HDF5)      │
└─────────┘ └─────────┘ └─Typing)──┘ └─────────┘ └─────────────┘
      │         │         │         │         │
      ▼         ▼         ▼         ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                Data Management Layer                        │
│           (learnm8.core.data_manager - HDF5)               │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│              Molecular Featurization                        │
│              (learnm8.utils.featurizers)                   │
└─────────────────────────────────────────────────────────────┘
```

### Package Structure (Actual Implementation)

```
learnm8/
├── __init__.py                        # Functional API exports (v0.5.0)
├── learnm8.py                         # Main functional implementation
├── core/                              # Core system components
│   ├── interfaces.py                  # Abstract base classes
│   ├── data_manager.py                # HDF5-based data management
│   ├── active_learning_loops.py       # Loop strategy interfaces (unused)
│   ├── experiment_state.py            # State management interfaces (unused)
│   ├── evaluation_strategy.py         # Evaluation interfaces (unused)
│   └── simple_data_manager.py         # Simplified data operations
├── learners/                          # Model implementations
│   ├── base.py                        # SklearnLearner and TorchLearner bases
│   ├── sklearn/                       # Scikit-learn models
│   │   ├── random_forest.py           # RandomForestLearner
│   │   ├── gaussian_process.py        # GaussianProcessLearner
│   │   └── xgboost_learner.py         # XGBoostLearner
│   ├── torch/                         # PyTorch models
│   │   ├── mlp.py                     # MLPLearner
│   │   └── mc_dropout.py              # MCDropoutLearner
│   └── ensemble/                      # Meta-learners
│       └── ensemble.py                # EnsembleLearner composition
├── acquisition/                       # Selection strategies
│   ├── base.py                        # AcquisitionFunction ABC
│   ├── basic.py                       # Greedy, Random, TopK
│   ├── uncertainty_based.py           # UCB, EI, PI, Thompson, Entropy
│   └── bitbirch.py                    # BitBIRCH molecular clustering
├── oracles/                           # Property measurement
│   ├── csv_oracle.py                  # CSV lookup oracle for benchmarks
│   └── python_oracle.py               # Custom Python function oracle
├── pruning/                           # Design space reduction
│   ├── base.py                        # Base pruning interface
│   ├── score_based.py                 # Score-based pruning implementation
│   └── utils.py                       # Pruning utilities
├── evaluation/                        # Performance metrics
│   ├── __init__.py                    # Evaluation API
│   └── metrics.py                     # Core metric calculations
├── cli/                               # Command-line interface
│   ├── cli.py                         # Simple functional CLI
│   ├── __main__.py                    # Module entry point
│   └── common.py                      # Shared CLI utilities
├── utils/                             # Shared utilities
│   ├── featurizers.py                 # Molecular representations
│   ├── data_loaders.py                # Data loading utilities
│   ├── cycle_utils.py                 # Cycle specification parsing
│   └── logging.py                     # Logging configuration
└── visualization/                     # Data visualization and plotting
    └── plotting.py                    # Scientific plotting utilities
```

---

## Core Interfaces

### 1. Learner Interface (Dependency Injection)

The fundamental abstraction for all machine learning models with dependency injection:

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd
import numpy as np

class Learner(ABC):
    """Abstract base class for all machine learning models."""
    
    @abstractmethod
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: 'DataManager') -> None:
        """
        Train the model on labeled compound data.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', and target columns
            target_column: Name of the target property column  
            data_manager: Central data manager for feature extraction and caching
        """
        pass
    
    @abstractmethod 
    def predict(self, compounds: pd.DataFrame, data_manager: 'DataManager') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict scores for compounds.
        
        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            data_manager: Central data manager for feature extraction
            
        Returns:
            Tuple of (predictions, uncertainties). 
            uncertainties can be None if model doesn't provide uncertainty estimates.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        pass
    
    def supports_uncertainty(self) -> bool:
        """Return True if this learner can provide uncertainty estimates."""
        # Duck typing approach - can be overridden for efficiency
        return False  # Conservative default
```

### 2. Oracle Interface

Provides compound measurement capabilities:

```python
class Oracle(ABC):
    """Abstract interface for compound measurement/scoring."""
    
    @abstractmethod
    def measure(self, compounds: pd.DataFrame, properties: List[str]) -> pd.DataFrame:
        """
        Measure properties for given compounds.
        
        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            properties: List of property names to measure
            
        Returns:
            DataFrame with 'ID' column and requested property columns
        """
        pass
```

### 3. AcquisitionFunction Interface

Simple interface for compound selection strategies:

```python
class AcquisitionFunction(ABC):
    """Base class for compound selection strategies."""
    
    @abstractmethod
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """
        Select compounds for labeling.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
                      May also contain 'uncertainty' column if available
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds
        """
        pass
    
    def requires_uncertainty(self) -> bool:
        """Return True if this acquisition function requires uncertainty estimates."""
        return False
    
    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return self.__class__.__name__
```

### 4. DataManager Interface (Centralized Data Operations)

Central hub for all data operations with HDF5 caching:

```python
class DataManager:
    """Simplified data manager with HDF5 caching for molecular features."""
    
    def __init__(self, 
                 cache_dir: str = 'learnm8_cache',
                 enable_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.featurizers = {}  # Lazy-loaded featurizers
        
    def get_features(self,
                    compound_ids: List[str],
                    smiles_list: Optional[List[str]],
                    featurizer_type: str) -> Tuple[np.ndarray, List[str]]:
        """
        Get molecular features with automatic HDF5 caching.
        This is the single entry point for all feature extraction.

        Returns:
            Tuple of (features_array, valid_compound_ids)
        """
        pass

    def prepare_training_data(self,
                             compounds: pd.DataFrame,
                             target_column: str,
                             featurizer_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare optimized training data with feature extraction.

        Returns:
            Tuple of (X, y) arrays
        """
        pass

    def prepare_prediction_data(self,
                               compounds: pd.DataFrame,
                               featurizer_type: str) -> Tuple[np.ndarray, List[str]]:
        """Prepare prediction data with feature extraction.

        Returns:
            Tuple of (features_array, valid_compound_ids)
        """
        pass
```

---

## Model System

LearnM8 provides 6 essential model classes organized by framework and purpose. The simplified hierarchy promotes composition over inheritance and eliminates redundancy.

### Model Hierarchy

```
Learner (ABC)
├── SklearnLearner               # Base for sklearn models
│   ├── RandomForestLearner      # Ensemble method
│   ├── GaussianProcessLearner   # GP with uncertainty
│   └── XGBoostLearner          # Gradient boosting
├── TorchLearner                # Base for PyTorch models  
│   ├── MLPLearner              # Neural network
│   └── MCDropoutLearner        # Dropout uncertainty
└── EnsembleLearner             # Meta-learner for ensembles
```

### 1. SklearnLearner Base Class

Base class for scikit-learn compatible models with dependency injection:

```python
from sklearn.base import BaseEstimator

class SklearnLearner(Learner):
    """Base class for scikit-learn compatible models."""
    
    def __init__(self, 
                 model: BaseEstimator,
                 featurizer_type: str = 'morgan',
                 random_state: int = 42):
        self.model = model
        self.featurizer_type = featurizer_type
        self.random_state = random_state
        self.is_trained = False
    
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: DataManager) -> None:
        """Train sklearn model using DataManager for features."""
        X, y = data_manager.prepare_training_data(compounds, target_column, self.featurizer_type)
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, compounds: pd.DataFrame, data_manager: DataManager) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict using sklearn model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X, valid_compound_ids = data_manager.prepare_prediction_data(compounds, self.featurizer_type)
        predictions = self.model.predict(X)

        return predictions, None  # Base sklearn models don't provide uncertainty
```

#### RandomForestLearner
```python
class RandomForestLearner(SklearnLearner):
    """Random Forest with optimized hyperparameters for molecular data."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 random_state: int = 42,
                 **kwargs):
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        super().__init__(model, **kwargs)
    
    def get_name(self) -> str:
        return f"RandomForest(n_estimators={self.model.n_estimators})"
```

#### GaussianProcessLearner
```python
class GaussianProcessLearner(SklearnLearner):
    """Gaussian Process learner with native uncertainty support."""
    
    def __init__(self, kernel=None, alpha: float = 1e-10, **kwargs):
        if kernel is None:
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
            kernel = C(1.0, (1e-4, 1e7)) * RBF(1.0, (1e-4, 1e7))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=kwargs.get('random_state', 42)
        )
        super().__init__(model, **kwargs)
    
    def predict(self, compounds: pd.DataFrame, data_manager: DataManager) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with native GP uncertainty."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X, valid_compound_ids = data_manager.prepare_prediction_data(compounds, self.featurizer_type)
        predictions, std = self.model.predict(X, return_std=True)

        return predictions, std  # GP naturally provides uncertainty
    
    def supports_uncertainty(self) -> bool:
        return True
    
    def get_name(self) -> str:
        return "GaussianProcess"
```

### 2. TorchLearner Base Class

Base class for PyTorch models with GPU support and dependency injection:

```python
import torch
import torch.nn as nn

class TorchLearner(Learner):
    """Base class for PyTorch models with GPU support."""
    
    def __init__(self,
                 device: str = 'auto',
                 batch_size: int = 1024,
                 max_epochs: int = 100,
                 learning_rate: float = 0.001,
                 early_stopping_patience: int = 10,
                 featurizer_type: str = 'morgan',
                 random_state: int = 42):
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Training configuration
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.featurizer_type = featurizer_type
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.is_trained = False
        
        # Set random seeds
        torch.manual_seed(random_state)
    
    @abstractmethod
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the PyTorch model architecture."""
        pass
    
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: DataManager) -> None:
        """Train PyTorch model using DataManager for features."""
        X, y = data_manager.prepare_training_data(compounds, target_column, self.featurizer_type)
        
        # Initialize model if needed
        if self.model is None:
            self.model = self._create_model(X.shape[1]).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Feature normalization
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Training loop implementation
        self._train_model(X_scaled, y)
        self.is_trained = True
```

#### MCDropoutLearner
```python
class MCDropoutLearner(TorchLearner):
    """MLP with Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, 
                 hidden_sizes: Tuple[int, ...] = (256, 128),
                 dropout_rate: float = 0.2,
                 n_dropout_samples: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.n_dropout_samples = n_dropout_samples
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create MLP with dropout layers."""
        layers = []
        prev_size = input_size
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*layers)
    
    def predict(self, compounds: pd.DataFrame, data_manager: DataManager) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with Monte Carlo Dropout uncertainty."""
        X, valid_compound_ids = data_manager.prepare_prediction_data(compounds, self.featurizer_type)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        predictions_list = []
        for _ in range(self.n_dropout_samples):
            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy().squeeze()
                predictions_list.append(pred)
        
        predictions = np.array(predictions_list)
        mean_predictions = np.mean(predictions, axis=0)
        uncertainties = np.std(predictions, axis=0)
        
        return mean_predictions, uncertainties
    
    def supports_uncertainty(self) -> bool:
        return True
    
    def get_name(self) -> str:
        return f"MCDropout(samples={self.n_dropout_samples})"
```

### 3. EnsembleLearner (Composition-Based)

Meta-learner that combines multiple models for uncertainty estimation:

```python
class EnsembleLearner(Learner):
    """Meta-learner that combines multiple models through composition."""
    
    def __init__(self, learners: List[Learner], aggregation_method: str = 'mean'):
        self.learners = learners
        self.aggregation_method = aggregation_method
        self.is_trained = False
    
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: DataManager) -> None:
        """Train all ensemble learners."""
        for learner in self.learners:
            learner.train(compounds, target_column, data_manager)
        self.is_trained = True
    
    def predict(self, compounds: pd.DataFrame, data_manager: DataManager) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble uncertainty."""
        predictions_list = []
        for learner in self.learners:
            pred, _ = learner.predict(compounds, data_manager)
            predictions_list.append(pred)
        
        predictions_array = np.array(predictions_list)
        
        # Aggregate predictions
        if self.aggregation_method == 'mean':
            ensemble_predictions = np.mean(predictions_array, axis=0)
        elif self.aggregation_method == 'median':
            ensemble_predictions = np.median(predictions_array, axis=0)
        else:
            ensemble_predictions = np.mean(predictions_array, axis=0)
        
        # Uncertainty as standard deviation across models
        uncertainties = np.std(predictions_array, axis=0)
        
        return ensemble_predictions, uncertainties
    
    def supports_uncertainty(self) -> bool:
        return True
    
    def get_name(self) -> str:
        learner_names = [learner.get_name() for learner in self.learners]
        return f"Ensemble({'+'.join(learner_names[:3])}{'...' if len(learner_names) > 3 else ''})"
```

---

## Data Management Architecture

### HDF5-Based DataManager

The `DataManager` provides a streamlined data management system with HDF5 caching for efficient molecular feature handling:

```python
class DataManager:
    """Simplified data manager with HDF5 caching for molecular features."""
    
    def __init__(self, 
                 cache_dir: str = 'learnm8_cache',
                 enable_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_cache = enable_cache
        self.featurizers = {}  # Lazy-loaded featurizers
        
        # HDF5 cache files for each featurizer type
        self.cache_files = {
            'morgan': self.cache_dir / 'morgan_features.h5',
            'maccs': self.cache_dir / 'maccs_features.h5',
            'ecfp6': self.cache_dir / 'ecfp6_features.h5',
            'descriptors': self.cache_dir / 'descriptors_features.h5'
        }
    
    def get_features(self,
                    compound_ids: List[str],
                    smiles_list: Optional[List[str]],
                    featurizer_type: str) -> Tuple[np.ndarray, List[str]]:
        """
        Get molecular features with automatic HDF5 caching.

        Returns:
            Tuple of (features_array, valid_compound_ids)
        """
        if smiles_list is None:
            smiles_list = compound_ids

        if featurizer_type not in self.featurizers:
            raise ValueError(f"Unknown featurizer type: {featurizer_type}")

        cache_file = self._get_cache_file(featurizer_type)
        features = []
        valid_compound_ids = []

        try:
            with h5py.File(cache_file, 'a') as f:
                # Process each SMILES, skipping invalid ones
                for compound_id, smiles in zip(compound_ids, smiles_list):
                    try:
                        feat = self._get_or_compute_feature(f, smiles, featurizer_type)
                        if feat is not None:
                            features.append(feat)
                            valid_compound_ids.append(compound_id)
                    except Exception as e:
                        logger.warning(f"Skipping invalid SMILES {smiles}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"HDF5 cache error: {e}. Computing without caching.")
            # Fallback without caching
            for compound_id, smiles in zip(compound_ids, smiles_list):
                try:
                    feat = self.featurizers[featurizer_type](smiles)
                    features.append(feat)
                    valid_compound_ids.append(compound_id)
                except Exception:
                    continue

        if features:
            return np.stack(features), valid_compound_ids
        else:
            return np.array([]).reshape(0, -1), []
```

**Key Features:**
- **HDF5 structure**: One file per featurizer type with `/features/{compound_hash}` datasets
- **Compression**: gzip compression reduces storage requirements for large libraries
- **Partial loading**: Load individual compounds without reading entire cache
- **Persistent caching**: Survives across experiment runs
- **Enhanced error handling**: Skips invalid SMILES entirely instead of using placeholder values
- **Data integrity**: Returns valid compound IDs to maintain data consistency
- **Graceful fallbacks**: Handles corrupted cache with automatic fallback to direct computation
- **Large-scale support**: Efficient handling of 1M+ molecule libraries

### Molecular Featurization

The featurization system supports multiple molecular representations:

```python
# Morgan Fingerprints (default) - returns features and valid compound IDs
features, valid_ids = data_manager.get_features(compound_ids, smiles_list, 'morgan')

# MACCS Keys
features, valid_ids = data_manager.get_features(compound_ids, smiles_list, 'maccs')

# Extended Connectivity Fingerprints (ECFP6)
features, valid_ids = data_manager.get_features(compound_ids, smiles_list, 'ecfp6')

# Mordred Descriptors
features, valid_ids = data_manager.get_features(compound_ids, smiles_list, 'descriptors')
```

---

## Active Learning Orchestration

### Functional Orchestration Pattern

LearnM8 uses a **pure functional approach** for active learning orchestration, avoiding complex state management:

```python
def run_active_learning(
    compound_pool: Union[str, pd.DataFrame],
    oracle: Union[str, Oracle],
    target_column: str,
    learner: Union[str, Learner] = 'rf',
    acquisition: Union[str, AcquisitionFunction] = 'greedy',
    n_cycles: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Pure functional active learning orchestration.
    
    Returns simple dictionary with results - no complex state objects.
    """
    
    # Initialize components
    data_manager = DataManager(**kwargs.get('data_manager_params', {}))
    learner_instance = _resolve_learner(learner)
    oracle_instance = _resolve_oracle(oracle)
    acquisition_instance = _resolve_acquisition(acquisition)
    
    # Load compound pool
    if isinstance(compound_pool, str):
        compound_pool = pd.read_csv(compound_pool)
    
    # Initialize with random selection
    initial_size = max(10, int(len(compound_pool) * 0.01))
    initial_compounds = compound_pool.sample(n=initial_size, random_state=42)
    
    # Measure initial compounds
    labeled_data = oracle_instance.measure(initial_compounds, [target_column])
    unlabeled_pool = compound_pool[~compound_pool['ID'].isin(labeled_data['ID'])]
    
    # Run active learning cycles
    cycle_results = []
    for cycle in range(n_cycles):
        if unlabeled_pool.empty:
            break
            
        # Execute single cycle functionally
        cycle_result = execute_single_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            learner=learner_instance,
            oracle=oracle_instance,
            acquisition_fn=acquisition_instance,
            data_manager=data_manager,
            target_column=target_column,
            cycle=cycle,
            **kwargs
        )
        
        # Update data (immutable transformations)
        labeled_data = pd.concat([labeled_data, cycle_result['measured_compounds']])
        unlabeled_pool = cycle_result['updated_pool']
        cycle_results.append(cycle_result['metrics'])
    
    return {
        'labeled_data': labeled_data,
        'cycle_results': cycle_results,
        'total_cycles': len(cycle_results),
        'oracle_budget_used': len(labeled_data)
    }

def execute_single_cycle(
    labeled_data: pd.DataFrame,
    unlabeled_pool: pd.DataFrame,
    learner: Learner,
    oracle: Oracle,
    acquisition_fn: AcquisitionFunction,
    data_manager: DataManager,
    target_column: str,
    cycle: int,
    **config
) -> Dict[str, Any]:
    """
    Pure function for executing a single active learning cycle.
    No side effects - returns new state.
    """
    
    # 1. Train model
    if not labeled_data.empty:
        learner.train(labeled_data, target_column, data_manager)
    
    # 2. Make predictions
    predictions, uncertainties = learner.predict(unlabeled_pool, data_manager)
    
    # 3. Select next batch
    batch_size = min(
        int(len(unlabeled_pool) * config.get('batch_fraction', 0.1)),
        config.get('max_batch_size', 1000)
    )
    
    # Prepare acquisition input
    acquisition_input = unlabeled_pool.copy()
    acquisition_input['prediction'] = predictions
    if uncertainties is not None:
        acquisition_input['uncertainty'] = uncertainties
    
    selected_batch = acquisition_fn.select(acquisition_input, batch_size)
    
    # 4. Measure selected compounds
    measured_compounds = oracle.measure(selected_batch, [target_column])
    
    # 5. Update pool (immutable)
    updated_pool = unlabeled_pool[~unlabeled_pool['ID'].isin(measured_compounds['ID'])]
    
    # 6. Calculate metrics
    metrics = _calculate_cycle_metrics(cycle, predictions, uncertainties, measured_compounds)
    
    return {
        'measured_compounds': measured_compounds,
        'updated_pool': updated_pool,
        'metrics': metrics,
        'predictions': predictions,
        'uncertainties': uncertainties
    }
```

### Strategy Dispatch

Rather than complex strategy classes, LearnM8 uses simple string-based strategy dispatch:

```python
def _resolve_acquisition(acquisition: Union[str, AcquisitionFunction]) -> AcquisitionFunction:
    """Simple strategy dispatch without class hierarchies."""
    
    if isinstance(acquisition, AcquisitionFunction):
        return acquisition
    
    # String-based component selection
    acquisition_registry = {
        'greedy': lambda: GreedyAcquisition(),
        'random': lambda: RandomAcquisition(),
        'topk': lambda: TopKAcquisition(),
        'ucb': lambda: UCBAcquisition(),
        'ei': lambda: ExpectedImprovementAcquisition(),
        'pi': lambda: ProbabilityImprovementAcquisition(),
        'thompson': lambda: ThompsonSamplingAcquisition(),
        'entropy': lambda: EntropyAcquisition(),
        'bitbirch': lambda: BitBIRCHAcquisition()
    }
    
    if acquisition in acquisition_registry:
        return acquisition_registry[acquisition]()
    else:
        raise ValueError(f"Unknown acquisition strategy: {acquisition}")
```

---

## Acquisition Functions

### Simplified Acquisition System

The acquisition function system provides essential selection strategies without unnecessary complexity:

### 1. Basic Strategies

```python
class GreedyAcquisition(AcquisitionFunction):
    """Select compounds with highest predicted values."""
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select based on highest predictions."""
        top_indices = compounds['prediction'].nlargest(n_select).index
        return compounds.loc[top_indices].copy()
    
    def get_name(self) -> str:
        return "Greedy"

class RandomAcquisition(AcquisitionFunction):
    """Random selection for baseline comparison."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Random selection."""
        return compounds.sample(n=n_select, random_state=self.random_state)
    
    def get_name(self) -> str:
        return "Random"
```

### 2. Uncertainty-Based Strategies

```python
class UCBAcquisition(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""
    
    def __init__(self, beta: float = 2.0):
        self.beta = beta
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select using Upper Confidence Bound."""
        if 'uncertainty' not in compounds.columns:
            raise ValueError("UCB requires uncertainty estimates")
        
        # Calculate UCB scores
        ucb_scores = compounds['prediction'] + self.beta * compounds['uncertainty']
        
        # Select top compounds
        top_indices = ucb_scores.nlargest(n_select).index
        return compounds.loc[top_indices].copy()
    
    def requires_uncertainty(self) -> bool:
        return True
    
    def get_name(self) -> str:
        return f"UCB(β={self.beta})"

class ExpectedImprovementAcquisition(AcquisitionFunction):
    """Expected Improvement acquisition function."""
    
    def __init__(self, xi: float = 0.01):
        self.xi = xi  # Exploration parameter
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select using Expected Improvement."""
        if 'uncertainty' not in compounds.columns:
            raise ValueError("EI requires uncertainty estimates")
        
        from scipy.stats import norm
        
        # Current best value (from training data, would need to be passed)
        # For simplicity, using max prediction as proxy
        best_value = compounds['prediction'].max()
        
        # Calculate EI
        improvement = compounds['prediction'] - best_value - self.xi
        Z = improvement / compounds['uncertainty']
        ei_scores = improvement * norm.cdf(Z) + compounds['uncertainty'] * norm.pdf(Z)
        
        # Select top compounds
        top_indices = ei_scores.nlargest(n_select).index
        return compounds.loc[top_indices].copy()
    
    def requires_uncertainty(self) -> bool:
        return True
    
    def get_name(self) -> str:
        return f"EI(ξ={self.xi})"

class ThompsonSamplingAcquisition(AcquisitionFunction):
    """Thompson Sampling acquisition function."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select using Thompson Sampling."""
        if 'uncertainty' not in compounds.columns:
            raise ValueError("Thompson Sampling requires uncertainty estimates")
        
        # Sample from posterior distributions
        samples = self.rng.normal(
            compounds['prediction'].values,
            compounds['uncertainty'].values
        )
        
        # Select top samples
        top_indices = pd.Series(samples, index=compounds.index).nlargest(n_select).index
        return compounds.loc[top_indices].copy()
    
    def requires_uncertainty(self) -> bool:
        return True
    
    def get_name(self) -> str:
        return "ThompsonSampling"
```

### 3. Molecular Diversity Strategies

```python
class BitBIRCHAcquisition(AcquisitionFunction):
    """Molecular clustering using BitBIRCH algorithm for diversity-aware selection."""

    def __init__(self,
                 n_clusters: int = 10,
                 threshold: float = 0.5,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.random_state = random_state

        # Import BitBIRCH with fallback
        try:
            from bitbirch import BitBIRCH
            self._clustering_available = True
        except ImportError:
            self._clustering_available = False
            import warnings
            warnings.warn("BitBIRCH not available. Install with: pip install git+https://github.com/mqcomplab/bitbirch.git")

    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select diverse compounds using BitBIRCH clustering."""
        if not self._clustering_available:
            # Fallback to random selection
            return compounds.sample(n=n_select, random_state=self.random_state)

        from bitbirch import BitBIRCH
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        import numpy as np

        # Generate molecular fingerprints
        fingerprints = []
        valid_indices = []

        for idx, smiles in zip(compounds.index, compounds['SMILES']):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints.append(np.array(fp))
                valid_indices.append(idx)

        if not fingerprints:
            return compounds.sample(n=n_select, random_state=self.random_state)

        # Perform BitBIRCH clustering
        X = np.array(fingerprints)
        n_clusters = min(self.n_clusters, len(fingerprints), n_select)

        clustering = BitBIRCH(n_clusters=n_clusters, threshold=self.threshold)
        cluster_labels = clustering.fit_predict(X)

        # Select one compound from each cluster, prioritizing high predictions
        selected_indices = []
        valid_compounds = compounds.loc[valid_indices].copy()
        valid_compounds['cluster'] = cluster_labels

        for cluster_id in range(n_clusters):
            cluster_compounds = valid_compounds[valid_compounds['cluster'] == cluster_id]
            if not cluster_compounds.empty:
                # Select compound with highest prediction in cluster
                best_in_cluster = cluster_compounds.loc[cluster_compounds['prediction'].idxmax()]
                selected_indices.append(best_in_cluster.name)

        # Fill remaining slots with highest scoring compounds
        while len(selected_indices) < n_select:
            remaining = valid_compounds[~valid_compounds.index.isin(selected_indices)]
            if remaining.empty:
                break
            best_remaining = remaining.loc[remaining['prediction'].idxmax()]
            selected_indices.append(best_remaining.name)

        return compounds.loc[selected_indices[:n_select]].copy()

    def requires_uncertainty(self) -> bool:
        return False

    def get_name(self) -> str:
        return f"BitBIRCH(clusters={self.n_clusters})"
```

### Acquisition Function Implementation Status

**✅ Fully Implemented and Available:**

| Function | Registry Key | Requirements | Use Case |
|----------|-------------|-------------|----------|
| **GreedyAcquisition** | `'greedy'` | None | Exploitation, highest predicted values |
| **RandomAcquisition** | `'random'` | None | Baseline comparison, unbiased sampling |
| **TopKAcquisition** | `'topk'` | None | Percentile-based selection with randomization |
| **UCBAcquisition** | `'ucb'` | Uncertainty estimates | Exploration-exploitation balance |
| **ExpectedImprovementAcquisition** | `'ei'` | Uncertainty + scipy | Bayesian optimization |
| **ProbabilityImprovementAcquisition** | `'pi'` | Uncertainty + scipy | Complementary to EI |
| **ThompsonSamplingAcquisition** | `'thompson'` | Uncertainty estimates | Stochastic posterior sampling |
| **EntropyAcquisition** | `'entropy'` | Uncertainty estimates | Maximum information gain |

**🚧 Conditionally Available:**

| Function | Requirements | Status |
|----------|-------------|--------|
| **BitBIRCHAcquisition** | `'bitbirch'` | BitBIRCH package | Molecular clustering diversity |

---

## Design Space Pruning

### Score-Based Pruning System

Design space pruning reduces the search space by removing compounds with poor predicted scores:

```python
class ScoreBasedPruner:
    """Score-based design space pruning."""

    def __init__(self,
                 pruning_fraction: float = 0.3,
                 score_direction: str = 'higher'):
        self.pruning_fraction = pruning_fraction
        self.score_direction = score_direction

        if not 0.0 <= pruning_fraction <= 0.9:
            raise ValueError("pruning_fraction must be between 0.0 and 0.9")

        if score_direction not in ['higher', 'lower']:
            raise ValueError("score_direction must be 'higher' or 'lower'")

    def prune(self,
              compounds: pd.DataFrame,
              predictions: np.ndarray) -> pd.DataFrame:
        """
        Prune compounds based on predicted scores.

        Args:
            compounds: Compound pool DataFrame
            predictions: Model predictions

        Returns:
            Pruned compound pool with worst-scoring compounds removed
        """
        if len(compounds) == 0:
            return compounds.copy()

        # Calculate number of compounds to remove
        n_remove = int(len(compounds) * self.pruning_fraction)
        n_keep = len(compounds) - n_remove

        if n_keep <= 0:
            n_keep = 1  # Always keep at least one compound

        # Create compound DataFrame with predictions for sorting
        compound_scores = compounds.copy()
        compound_scores['prediction'] = predictions

        # Sort by predictions and keep top compounds
        if self.score_direction == 'higher':
            # Keep compounds with highest predictions (remove lowest)
            pruned_compounds = compound_scores.nlargest(n_keep, 'prediction')
        else:
            # Keep compounds with lowest predictions (remove highest)
            pruned_compounds = compound_scores.nsmallest(n_keep, 'prediction')

        # Remove prediction column and return
        pruned_compounds = pruned_compounds.drop(columns=['prediction'])

        return pruned_compounds

    def get_pruning_info(self, compounds: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
        """Get detailed pruning statistics."""
        n_total = len(compounds)
        n_remove = int(n_total * self.pruning_fraction)
        n_keep = n_total - n_remove

        # Calculate score thresholds
        if self.score_direction == 'higher':
            threshold = np.partition(predictions, n_remove)[n_remove] if n_remove > 0 else predictions.min()
            kept_scores = predictions[predictions >= threshold]
        else:
            threshold = np.partition(predictions, -n_remove-1)[-n_remove-1] if n_remove > 0 else predictions.max()
            kept_scores = predictions[predictions <= threshold]

        return {
            'total_compounds': n_total,
            'compounds_removed': min(n_remove, n_total),
            'compounds_kept': min(n_keep, n_total),
            'pruning_fraction': self.pruning_fraction,
            'score_direction': self.score_direction,
            'score_threshold': threshold,
            'kept_score_range': (kept_scores.min(), kept_scores.max()) if len(kept_scores) > 0 else (0, 0)
        }

# Integration with functional orchestration
def apply_pruning_strategy(
    pool: pd.DataFrame,
    predictions: np.ndarray,
    uncertainties: Optional[np.ndarray],
    strategy: str = 'score_based',
    params: Dict[str, Any] = None,
    score_direction: str = 'higher'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply pruning strategy to compound pool."""

    if strategy == 'score_based':
        pruner = ScoreBasedPruner(
            pruning_fraction=params.get('pruning_fraction', 0.3),
            score_direction=score_direction
        )
        pruned_pool = pruner.prune(pool, predictions)
        pruning_info = pruner.get_pruning_info(pool, predictions)

        return pruned_pool, pruning_info
    else:
        raise ValueError(f"Unknown pruning strategy: {strategy}")
```

---

## Command-Line Interface

### Simplified Functional CLI

The CLI provides a simple, functional interface without complex configuration management:

```python
def main():
    """Simple functional CLI entry point."""
    parser = argparse.ArgumentParser(description='LearnM8 Active Learning')
    
    # Required arguments
    parser.add_argument('compound_pool', help='Path to compound pool CSV')
    parser.add_argument('oracle', help='Oracle (CSV file or Python module:function)')
    parser.add_argument('target_column', help='Target property column name')
    
    # Learning configuration
    parser.add_argument('-l', '--learner', default='rf',
                       choices=['rf', 'gp', 'xgb', 'mlp', 'mc_dropout', 'ensemble',
                               'rf_ensemble', 'lr_ensemble', 'xgb_ensemble', 'dt_ensemble', 'mixed_ensemble'],
                       help='Machine learning model')
    parser.add_argument('-a', '--acquisition', default='greedy',
                       choices=['greedy', 'random', 'topk', 'ucb', 'ei', 'pi', 'thompson', 'entropy', 'bitbirch'],
                       help='Acquisition function')
    
    # Active learning parameters
    parser.add_argument('-c', '--cycles', type=int, default=10,
                       help='Number of active learning cycles')
    parser.add_argument('-b', '--batch-fraction', type=float, default=0.1,
                       help='Fraction of compounds per cycle')
    parser.add_argument('--max-batch-size', type=int, default=1000,
                       help='Maximum compounds per batch')
    parser.add_argument('--initial-size', type=int, default=None,
                       help='Initial training set size')
    
    # Data configuration
    parser.add_argument('--featurizer', default='morgan',
                       choices=['morgan', 'maccs', 'ecfp6', 'descriptors'],
                       help='Molecular featurizer')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers')
    
    # Output
    parser.add_argument('-o', '--output', default='learnm8_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Call functional API
    try:
        results = run_active_learning(
            compound_pool=args.compound_pool,
            oracle=args.oracle,
            target_column=args.target_column,
            learner=args.learner,
            acquisition=args.acquisition,
            n_cycles=args.cycles,
            batch_fraction=args.batch_fraction,
            max_batch_size=args.max_batch_size,
            initial_size=args.initial_size,
            featurizer_type=args.featurizer,
            random_state=args.random_state,
            n_workers=args.n_workers,
            results_dir=args.output
        )
        
        # Simple output
        print(f"Active learning completed!")
        print(f"Total cycles: {results['total_cycles']}")
        print(f"Final training set size: {len(results['labeled_data'])}")
        print(f"Oracle budget used: {results['oracle_budget_used']}")
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        results['labeled_data'].to_csv(output_dir / 'final_labeled_data.csv', index=False)
        
        # Save cycle metrics
        import json
        with open(output_dir / 'cycle_metrics.json', 'w') as f:
            json.dump(results['cycle_results'], f, indent=2)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

# Example usage:
# learnm8 compounds.csv data.csv Activity -l gp -a ucb -c 15
# learnm8 compounds.csv oracle.py:calculate_score binding_affinity -l ensemble -a thompson
```

---

## Functional API

### Main Functional Interface

The primary interface is the `run_active_learning` function:

```python
from learnm8 import run_active_learning

# Simple string-based usage
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='data.csv',  # CSV oracle for benchmark
    target_column='Activity',
    learner='gp',
    acquisition='ucb',
    n_cycles=15
)

# Direct component instantiation  
from learnm8.learners import GaussianProcessLearner
from learnm8.acquisition import UCBAcquisition

results = run_active_learning(
    compound_pool=my_dataframe,
    oracle=custom_oracle,
    target_column='binding_affinity',
    learner=GaussianProcessLearner(length_scale=1.0),
    acquisition=UCBAcquisition(beta=2.0)
)

# With pruning enabled
results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='data.csv',
    target_column='activity',
    learner='ensemble',
    acquisition='ei',
    enable_pruning=True,
    pruning_params={'threshold': 0.1, 'min_pool_fraction': 0.1}
)
```

### Convenience Functions

```python
from learnm8 import quick_benchmark, quick_run

# Quick benchmark mode
results = quick_benchmark('data.csv', 'Activity', learner='ensemble', cycles=10)

# Quick run mode with Python oracle
results = quick_run('compounds.csv', 'oracle.py', 'calculate_score', 'target', learner='gp')
```

---

## Evaluation System

### Unified Evaluation API

The evaluation system provides comprehensive metrics without complex strategy patterns:

```python
from learnm8.evaluation import evaluate_cycle

def evaluate_cycle(
    cycle: int,
    predictions: np.ndarray,
    ground_truth: Optional[pd.DataFrame],
    labeled_data: pd.DataFrame,
    selected_compounds: pd.DataFrame,
    target_column: str
) -> Dict[str, Any]:
    """
    Evaluate active learning cycle performance.
    
    Returns comprehensive metrics dictionary.
    """
    metrics = {
        'cycle': cycle,
        'labeled_count': len(labeled_data),
        'selected_count': len(selected_compounds)
    }
    
    # Model performance metrics (if ground truth available)
    if ground_truth is not None:
        labeled_ids = set(labeled_data['ID'])
        labeled_ground_truth = ground_truth[ground_truth['ID'].isin(labeled_ids)]
        
        if not labeled_ground_truth.empty:
            from sklearn.metrics import r2_score, mean_squared_error
            
            true_values = labeled_ground_truth[target_column]
            predicted_values = predictions[:len(true_values)]  # Match lengths
            
            metrics.update({
                'r2_score': r2_score(true_values, predicted_values),
                'rmse': np.sqrt(mean_squared_error(true_values, predicted_values)),
                'mae': np.mean(np.abs(true_values - predicted_values))
            })
    
    # Active learning specific metrics
    if 'prediction' in selected_compounds.columns:
        metrics.update({
            'mean_prediction': selected_compounds['prediction'].mean(),
            'std_prediction': selected_compounds['prediction'].std()
        })
    
    if 'uncertainty' in selected_compounds.columns:
        metrics.update({
            'mean_uncertainty': selected_compounds['uncertainty'].mean(),
            'std_uncertainty': selected_compounds['uncertainty'].std()
        })
    
    # Enrichment metrics (top-k analysis)
    if ground_truth is not None:
        top_k_enrichment = _calculate_enrichment(selected_compounds, ground_truth, target_column)
        metrics['top_k_enrichment'] = top_k_enrichment
    
    return metrics

def _calculate_enrichment(selected: pd.DataFrame, ground_truth: pd.DataFrame, target_column: str) -> float:
    """Calculate enrichment factor for selected compounds."""
    # Merge selected compounds with ground truth
    merged = selected.merge(ground_truth[['ID', target_column]], on='ID', how='left')
    
    if merged.empty or merged[target_column].isna().all():
        return 0.0
    
    # Calculate enrichment (simplified)
    selected_mean = merged[target_column].mean()
    population_mean = ground_truth[target_column].mean()
    
    return selected_mean / population_mean if population_mean != 0 else 1.0
```

### Progress Monitoring

```python
def format_progress_output(cycle: int, metrics: Dict[str, Any]) -> str:
    """Format cycle metrics for console output."""
    output = f"\n=== Cycle {cycle} Results ==="
    
    # Core metrics
    output += f"\nLabeled compounds: {metrics.get('labeled_count', 'N/A')}"
    output += f"\nSelected compounds: {metrics.get('selected_count', 'N/A')}"
    
    # Model performance
    if 'r2_score' in metrics:
        output += f"\nModel R²: {metrics['r2_score']:.3f}"
        output += f"\nRMSE: {metrics['rmse']:.3f}"
    
    # Uncertainty metrics
    if 'mean_uncertainty' in metrics:
        output += f"\nMean uncertainty: {metrics['mean_uncertainty']:.3f}"
    
    # Enrichment
    if 'top_k_enrichment' in metrics:
        output += f"\nEnrichment factor: {metrics['top_k_enrichment']:.2f}x"
    
    return output
```

---

## Extension Guidelines

### Creating Custom Learners

```python
from learnm8.learners.base import SklearnLearner
from sklearn.ensemble import ExtraTreesRegressor

class ExtraTreesLearner(SklearnLearner):
    """Custom Extra Trees learner."""
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
        super().__init__(model, **kwargs)
    
    def get_name(self) -> str:
        return f"ExtraTrees(n_estimators={self.model.n_estimators})"

# Register with functional API
def _resolve_learner(learner):
    """Extended learner resolution."""
    if learner == 'extra_trees':
        return ExtraTreesLearner()
    # ... existing resolvers
```

### Creating Custom Acquisition Functions

```python
from learnm8.acquisition.base import AcquisitionFunction

class CustomAcquisition(AcquisitionFunction):
    """Custom acquisition strategy."""
    
    def __init__(self, parameter: float = 1.0):
        self.parameter = parameter
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Custom selection logic."""
        # Implement your custom selection strategy
        scores = compounds['prediction'] * self.parameter
        
        if 'uncertainty' in compounds.columns:
            scores += compounds['uncertainty']
        
        top_indices = scores.nlargest(n_select).index
        return compounds.loc[top_indices].copy()
    
    def requires_uncertainty(self) -> bool:
        return False  # Uncertainty is optional
    
    def get_name(self) -> str:
        return f"Custom(param={self.parameter})"
```

### Creating Custom Oracles

```python
from learnm8.oracles.base import Oracle

class DatabaseOracle(Oracle):
    """Oracle that queries external database."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection = None  # Initialize database connection
    
    def measure(self, compounds: pd.DataFrame, properties: List[str]) -> pd.DataFrame:
        """Query database for compound properties."""
        results = []
        
        for _, compound in compounds.iterrows():
            # Query database for compound properties
            compound_data = {'ID': compound['ID']}
            
            for prop in properties:
                # Database query implementation
                value = self._query_database(compound['SMILES'], prop)
                compound_data[prop] = value
            
            results.append(compound_data)
        
        return pd.DataFrame(results)
    
    def _query_database(self, smiles: str, property_name: str) -> float:
        """Execute database query."""
        # Implement database query logic
        pass
```

---

## Technical Specifications

### Model-Uncertainty-Acquisition Compatibility

| Model Type | Native Uncertainty | Ensemble Uncertainty | UCB/EI/PI | Thompson | Notes |
|------------|-------------------|---------------------|-----------|----------|-------|
| **RandomForestLearner** | ❌ | ✅ | 🔶¹ | 🔶¹ | Requires ensemble wrapper |
| **GaussianProcessLearner** | ✅ | ✅ | ✅ | ✅ | Gold standard uncertainty |
| **XGBoostLearner** | ❌ | ✅ | 🔶¹ | 🔶¹ | Ensemble recommended |
| **MLPLearner** | ❌ | ✅ | 🔶¹ | 🔶¹ | Base PyTorch MLP |
| **MCDropoutLearner** | ✅ | ✅ | ✅ | ✅ | Built-in uncertainty |
| **EnsembleLearner** | ✅ | ✅ | ✅ | ✅ | Variance across models |

**Legend:**
- ✅ Full compatibility and recommended
- 🔶¹ Requires ensemble uncertainty wrapper

### Recommended Combinations

**High-Performance Scenarios:**
```python
model = EnsembleLearner([GaussianProcessLearner(), MCDropoutLearner()])
acquisition = UCBAcquisition(beta=2.0)
```

**Fast Prototyping:**
```python
model = MCDropoutLearner(n_dropout_samples=50)
acquisition = ThompsonSamplingAcquisition()
```

**Large Datasets:**
```python
model = XGBoostLearner()
# Use ensemble for uncertainty
ensemble_model = EnsembleLearner([model] * 3)
acquisition = BitBIRCHAcquisition(n_clusters=10)
```




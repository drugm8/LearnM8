# learnm8/__init__.py
"""LearnM8: Active Learning for Molecular Screening - Modern Architecture

This package provides active learning for molecular screening with early validation,
performance optimizations, and flexible configuration. The modular architecture
keeps a clean, functional API.

Key features:
- Early compound validation before running experiments
- HDF5-based feature caching for fast repeated operations
- Automatic parallel processing for feature extraction
- Flexible cycle configuration with CycleConfig
- Comprehensive CSV export for analysis

"""

import importlib.metadata

__version__ = importlib.metadata.version('learnm8')

import contextlib

# Acquisition strategies
from .acquisition import (
    EntropyAcquisition,
    ExpectedImprovementAcquisition,
    GreedyAcquisition,
    ProbabilityImprovementAcquisition,
    RandomAcquisition,
    ThompsonSamplingAcquisition,
    TopKAcquisition,
    UCBAcquisition,
)
from .api import run_active_learning
from .core.config import CycleConfig

# Core interfaces (maintained for component creation)
from .core.interfaces import Learner, Oracle

# New core APIs
from .core.validation import ValidationResult, validate_compound_pool

# Exceptions and warnings
from .exceptions import (
    AcquisitionError,
    ConfigurationError,
    ConvergenceWarning,
    FeatureExtractionError,
    LearnerError,
    LearnM8Error,
    LearnM8Warning,
    OracleError,
    PersistenceError,
    PruningError,
    ValidationError,
)
from .features import extract_features

# Ensemble learners
from .learners.ensemble import (
    ChempropEnsemble,
    DTEnsemble,
    EnsembleLearner,
    FastpropEnsemble,
    LREnsemble,
    MixedEnsemble,
    XGBEnsemble,
)

# GPU learners (cuML imported lazily at train time)
from .learners.gpu import RfFilLearner, RidgeCumlLearner

# Sklearn learners
from .learners.sklearn import (
    DecisionTreeLearner,
    GaussianProcessLearner,
    LinearRegressionLearner,
    RandomForestLearner,
    XGBoostLearner,
)

# Torch learners
from .learners.torch import (
    ChempropLearner,
    FastpropLearner,
    MCDropoutLearner,
    MLPLearner,
)

# Oracles
from .oracles.csv_oracle import CSVOracle
from .oracles.python_oracle import PythonOracle

# Pruning strategies
from .pruning import DesignSpacePruner, ScoreBasedPruner
from .suggest import suggest_next_batch

# GPyTorch learners (optional dependency)
_OPTIONAL_EXPORTS: list[str] = []
with contextlib.suppress(ImportError):
    from .learners.gpytorch import GPyTorchGPLearner  # noqa: F401

    _OPTIONAL_EXPORTS.append('GPyTorchGPLearner')
with contextlib.suppress(ImportError):
    from .learners.gpytorch import SVGPLearner  # noqa: F401

    _OPTIONAL_EXPORTS.append('SVGPLearner')

# Utility functions

__all__ = [
    'AcquisitionError',
    'CSVOracle',
    'ChempropEnsemble',
    'ChempropLearner',
    'ConfigurationError',
    'ConvergenceWarning',
    'CycleConfig',
    'DTEnsemble',
    'DecisionTreeLearner',
    'DesignSpacePruner',
    'EnsembleLearner',
    'EntropyAcquisition',
    'ExpectedImprovementAcquisition',
    'FastpropEnsemble',
    'FastpropLearner',
    'FeatureExtractionError',
    'GaussianProcessLearner',
    'GreedyAcquisition',
    'LREnsemble',
    'LearnM8Error',
    'LearnM8Warning',
    'Learner',
    'LearnerError',
    'LinearRegressionLearner',
    'MCDropoutLearner',
    'MLPLearner',
    'MixedEnsemble',
    'Oracle',
    'OracleError',
    'PersistenceError',
    'ProbabilityImprovementAcquisition',
    'PruningError',
    'PythonOracle',
    'RandomAcquisition',
    'RandomForestLearner',
    'RfFilLearner',
    'RidgeCumlLearner',
    'ScoreBasedPruner',
    'ThompsonSamplingAcquisition',
    'TopKAcquisition',
    'UCBAcquisition',
    'ValidationError',
    'ValidationResult',
    'XGBEnsemble',
    'XGBoostLearner',
    'extract_features',
    'run_active_learning',
    'suggest_next_batch',
    'validate_compound_pool',
    *_OPTIONAL_EXPORTS,
]

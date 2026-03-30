# learnm8/__init__.py
"""LearnM8: Active Learning for Molecular Screening - Modern Architecture

This package provides active learning for molecular screening with early validation,
performance optimizations, and flexible configuration. The new modular architecture
enables 5-100x performance improvements while maintaining a clean, functional API.

Key features:
- Early compound validation before running experiments
- HDF5-based feature caching for 100x faster repeated operations
- Automatic parallel processing for 5-10x faster feature extraction
- Flexible cycle configuration with CycleConfig
- Comprehensive CSV export for analysis

"""

__version__ = "1.0.0"

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
    DataConversionWarning,
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
    DTEnsemble,
    EnsembleLearner,
    LREnsemble,
    MixedEnsemble,
    RFEnsemble,
    XGBEnsemble,
)

# Sklearn learners
from .learners.sklearn import (
    GaussianProcessLearner,
    RandomForestLearner,
    XGBoostLearner,
)

# Torch learners
from .learners.torch import MCDropoutLearner, MLPLearner

# GPyTorch learners (optional dependency)
with contextlib.suppress(ImportError):
    from .learners.gpytorch import GPyTorchGPLearner  # noqa: F401
with contextlib.suppress(ImportError):
    from .learners.gpytorch import SVGPLearner  # noqa: F401

# Oracles
from .oracles.csv_oracle import CSVOracle
from .oracles.python_oracle import PythonOracle

# Pruning strategies
from .pruning import DesignSpacePruner, ScoreBasedPruner

# Utility functions
from .utils.logging import setup_logging

__all__ = [
    'AcquisitionError',
    'CSVOracle',
    'ConfigurationError',
    'ConvergenceWarning',
    'CycleConfig',
    'DTEnsemble',
    'DataConversionWarning',
    'DesignSpacePruner',
    'EnsembleLearner',
    'EntropyAcquisition',
    'ExpectedImprovementAcquisition',
    'FeatureExtractionError',
    'GaussianProcessLearner',
    'GreedyAcquisition',
    'LREnsemble',
    'LearnM8Error',
    'LearnM8Warning',
    'Learner',
    'LearnerError',
    'MCDropoutLearner',
    'MLPLearner',
    'MixedEnsemble',
    'Oracle',
    'OracleError',
    'PersistenceError',
    'ProbabilityImprovementAcquisition',
    'PruningError',
    'PythonOracle',
    'RFEnsemble',
    'RandomAcquisition',
    'RandomForestLearner',
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
    'setup_logging',
    'validate_compound_pool',
]

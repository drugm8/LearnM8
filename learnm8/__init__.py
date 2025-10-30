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

from .api import run_active_learning

# Core interfaces (maintained for component creation)
from .core.interfaces import Learner, Oracle

# New core APIs
from .core.validation import validate_compound_pool, ValidationResult
from .features import extract_features
from .core.config import CycleConfig

# Utility functions
from .utils.logging import setup_logging


# Optional components (with graceful import failures)
try:
    from .learners.sklearn import RandomForestLearner, GaussianProcessLearner, XGBoostLearner
    from .learners.ensemble import (
        EnsembleLearner, RFEnsemble, LREnsemble, XGBEnsemble, DTEnsemble, MixedEnsemble
    )
except ImportError:
    pass

try:
    from .learners.torch import MLPLearner, MCDropoutLearner
except ImportError:
    pass

try:
    from .acquisition.basic import GreedyAcquisition, RandomAcquisition, TopKAcquisition
    from .acquisition.uncertainty_based import (
        UCBAcquisition, ExpectedImprovementAcquisition, 
        ProbabilityImprovementAcquisition, ThompsonSamplingAcquisition, EntropyAcquisition
    )
except ImportError:
    pass

try:
    from .pruning.probabilistic import ProbabilisticPruner, PredictionThresholdPruner
    from .pruning.adaptive import AdaptivePruner
except ImportError:
    pass

try:
    from .oracles.csv_oracle import CSVOracle
    from .oracles.python_oracle import PythonOracle
except ImportError:
    pass


# Build __all__ dynamically based on successful imports
__all__ = [
    # Main functional API
    'run_active_learning',

    # New core APIs
    'validate_compound_pool', 'ValidationResult',
    'extract_features',
    'CycleConfig',

    # Utility functions
    'setup_logging',

    # Core interfaces (for advanced users)
    'Learner', 'Oracle',
]

# Add optional components only if imports succeeded
if 'RandomForestLearner' in dir():
    __all__.extend(['RandomForestLearner', 'GaussianProcessLearner', 'XGBoostLearner'])
if 'EnsembleLearner' in dir():
    __all__.extend(['EnsembleLearner', 'RFEnsemble', 'LREnsemble', 'XGBEnsemble', 'DTEnsemble', 'MixedEnsemble'])
if 'MLPLearner' in dir():
    __all__.extend(['MLPLearner', 'MCDropoutLearner'])

if 'GreedyAcquisition' in dir():
    __all__.extend(['GreedyAcquisition', 'RandomAcquisition', 'TopKAcquisition'])
if 'UCBAcquisition' in dir():
    __all__.extend([
        'UCBAcquisition', 'ExpectedImprovementAcquisition',
        'ProbabilityImprovementAcquisition', 'ThompsonSamplingAcquisition', 'EntropyAcquisition'
    ])

if 'ProbabilisticPruner' in dir():
    __all__.extend(['ProbabilisticPruner', 'PredictionThresholdPruner'])
if 'AdaptivePruner' in dir():
    __all__.append('AdaptivePruner')

if 'CSVOracle' in dir():
    __all__.extend(['CSVOracle', 'PythonOracle'])
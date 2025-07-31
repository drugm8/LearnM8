# learnm8/__init__.py
"""LearnM8: Active Learning for Molecular Screening - PAPI (Alternative 1)

This package provides a approach to active learning that eliminates
complex state management in favor of explicit cycle control. Users specify exactly
what happens each cycle through simple lists of (strategy, batch_fraction) tuples.

"""

__version__ = "0.5.0"  

from .learnm8 import (
    run_active_learning,
    create_learner_from_string
)

# Core interfaces (maintained for component creation)
from .core.interfaces import Learner, Oracle
from .core.data_manager import DataManager

# NOTE: ExperimentState and ActiveLearningLoop are deprecated in Alternative 1
# They are no longer exported in the main API

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


__all__ = [

    'run_active_learning',     # Main functional API with explicit cycle control
    'create_learner_from_string',  # Factory function for creating learners from strings
    
    # Core interfaces (for advanced users)
    'Learner', 'Oracle', 'DataManager',
    
    # Common learners (if available)
    'RandomForestLearner', 'GaussianProcessLearner', 'EnsembleLearner',
    'RFEnsemble', 'LREnsemble', 'XGBEnsemble', 'DTEnsemble', 'MixedEnsemble',
    'MLPLearner', 'MCDropoutLearner',
    
    # Common oracles (if available)
    'CSVOracle', 'PythonOracle',
]
"""Core active learning functionality - Architecture (v0.5.0).

Architecture (Alternative 1):
- Functional approach with explicit cycle control
- Simple functions instead of complex class hierarchies  
- No state management objects (ExperimentState deprecated)
- No active learning loop classes (StandardActiveLoop deprecated)
- Centralized data management for feature caching only
"""

# Core interfaces (maintained for component creation)
from .interfaces import Oracle, Learner

# Core data management (maintained for feature caching)
try:
    from .data_manager import DataManager
except ImportError:
    pass

# DEPRECATED: These are no longer part of the core functional API
# - ExperimentState: Replaced by simple variables in functional approach
# - ActiveLearningLoop: Replaced by simple functions
# - StandardActiveLoop: Replaced by execute_single_cycle function

__all__ = [
    # Core interfaces (maintained)
    'Oracle', 'Learner',
    
    # Data management (maintained for feature caching)
    'DataManager',
    
    # NOTE: ExperimentState and ActiveLearningLoop classes are deprecated
    # in the approach (Alternative 1)
]
"""Design space pruning for the LearnM8 active learning framework.

This package provides various strategies for reducing the size of the unlabeled
compound pool by removing compounds that are unlikely to be valuable, based on
model predictions and uncertainty estimates.

New Architecture (v0.3.0):
- Object-oriented pruning strategies with consistent interfaces
- Statistical analysis and validation
- Adaptive and performance-based pruning
- Comprehensive utility functions
"""

from .base import DesignSpacePruner, PruningError
from .probabilistic import (
    ProbabilisticPruner,
    UncertaintyThresholdPruner,
    PredictionThresholdPruner,
    ConfidenceIntervalPruner
)
from .adaptive import (
    CycleBudgetPruner,
    PerformanceBasedPruner
)
from .utils import (
    validate_pruning_parameters,
    create_pruning_strategy,
)

__all__ = [
    # Base classes
    'DesignSpacePruner',
    'PruningError',
    
    # Probabilistic pruning strategies
    'ProbabilisticPruner',
    'UncertaintyThresholdPruner', 
    'PredictionThresholdPruner',
    'ConfidenceIntervalPruner',
    
    # Adaptive pruning strategies
    'CycleBudgetPruner',
    'PerformanceBasedPruner',
    
    # Utility functions
    'analyze_pruning_effectiveness',
    'estimate_pruning_impact',
    'recommend_pruning_strategy',
    'validate_pruning_parameters',
    'create_pruning_strategy',
    'compare_pruning_strategies'
]
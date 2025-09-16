"""Simple score-based pruning for the LearnM8 active learning framework.

This package provides a single, straightforward pruning strategy that removes
compounds based on their predicted scores, making it easy to focus the search
on the most promising regions of chemical space.

Simplified Architecture (v0.6.0):
- Single score-based pruning strategy
- No uncertainty dependencies
- Simple fraction-based removal
- Score direction awareness
"""

from .base import DesignSpacePruner, PruningError
from .score_based import ScoreBasedPruner
from .utils import (
    validate_pruning_parameters,
    create_pruning_strategy,
)

__all__ = [
    # Base classes
    'DesignSpacePruner',
    'PruningError',
    
    # Score-based pruning strategy
    'ScoreBasedPruner',
    
    # Utility functions
    'validate_pruning_parameters',
    'create_pruning_strategy',
]
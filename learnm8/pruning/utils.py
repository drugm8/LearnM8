"""Utility functions for simple score-based pruning in the LearnM8 framework.

This module provides helper functions for the simplified score-based
pruning strategy.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_pruning_parameters(strategy_name: str,
                               parameters: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate parameters for the score-based pruning strategy.

    Args:
        strategy_name: Name of the pruning strategy (should be 'score')
        parameters: Dictionary of strategy parameters

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Only validate score-based pruner parameters
    if strategy_name == 'score':
        # Validate pruning_fraction
        if 'pruning_fraction' in parameters:
            fraction = parameters['pruning_fraction']
            if not isinstance(fraction, (int, float)):
                errors.append("pruning_fraction must be a number")
            elif not 0.0 <= fraction <= 0.9:
                errors.append("pruning_fraction must be between 0.0 and 0.9")
        
        # Validate score_direction
        if 'score_direction' in parameters:
            direction = parameters['score_direction']
            if not isinstance(direction, str):
                errors.append("score_direction must be a string")
            elif direction not in ['higher', 'lower']:
                errors.append("score_direction must be 'higher' or 'lower'")
        
        # Check for unexpected parameters
        valid_params = {'pruning_fraction', 'score_direction'}
        unexpected_params = set(parameters.keys()) - valid_params
        if unexpected_params:
            errors.append(f"Unexpected parameters: {', '.join(unexpected_params)}")
    
    else:
        errors.append(
            f"Unknown pruning strategy '{strategy_name}'. "
            f"Only 'score' is currently supported. "
            f"Use pruning_strategy='score' with pruning_params={{'pruning_fraction': 0.3}}."
        )

    return len(errors) == 0, errors


def create_pruning_strategy(strategy_name: str,
                           parameters: dict[str, Any]):
    """Factory function to create the score-based pruning strategy.

    Args:
        strategy_name: Name of the pruning strategy (should be 'score')
        parameters: Dictionary of strategy parameters

    Returns:
        Initialized ScoreBasedPruner instance

    Raises:
        ValueError: If strategy name is invalid or parameters are invalid
    """
    # Validate parameters first
    is_valid, errors = validate_pruning_parameters(strategy_name, parameters)
    if not is_valid:
        raise ValueError(
            f"Invalid parameters for pruning strategy '{strategy_name}': {'; '.join(errors)}. "
            f"Valid parameters for 'score': pruning_fraction (0.0-0.9), score_direction ('higher'/'lower')."
        )

    # Import strategy class
    from .score_based import ScoreBasedPruner

    strategy_registry = {
        'score': ScoreBasedPruner
    }

    if strategy_name not in strategy_registry:
        raise ValueError(
            f"Unknown pruning strategy '{strategy_name}'. Only 'score' is currently supported. "
            f"Use pruning_strategy='score' with pruning_params={{'pruning_fraction': 0.3}}."
        )

    strategy_class = strategy_registry[strategy_name]

    try:
        return strategy_class(**parameters)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to create pruning strategy '{strategy_name}' with parameters {parameters}: {e}. "
            f"Valid parameters: pruning_fraction (0.0-0.9), score_direction ('higher'/'lower')."
        ) from e

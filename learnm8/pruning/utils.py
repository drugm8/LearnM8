"""Utility functions for design space pruning in the LearnM8 framework.

This module provides helper functions and utilities for implementing
and analyzing design space pruning strategies.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_pruning_parameters(strategy_name: str, 
                               parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate parameters for a pruning strategy.
    
    Args:
        strategy_name: Name of the pruning strategy
        parameters: Dictionary of strategy parameters
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Common validations
    if 'retention_fraction' in parameters:
        retention = parameters['retention_fraction']
        if not 0 < retention <= 1:
            errors.append("retention_fraction must be between 0 and 1")
    
    if 'probability_threshold' in parameters:
        prob_thresh = parameters['probability_threshold']
        if not 0 <= prob_thresh <= 1:
            errors.append("probability_threshold must be between 0 and 1")
    
    # Strategy-specific validations
    if strategy_name == 'ProbabilisticPruner':
        required_params = ['value_threshold']
        for param in required_params:
            if param not in parameters:
                errors.append(f"ProbabilisticPruner requires parameter: {param}")
    
    elif strategy_name == 'ConfidenceIntervalPruner':
        required_params = ['target_min', 'target_max']
        for param in required_params:
            if param not in parameters:
                errors.append(f"ConfidenceIntervalPruner requires parameter: {param}")
        
        if 'target_min' in parameters and 'target_max' in parameters:
            if parameters['target_min'] >= parameters['target_max']:
                errors.append("target_min must be less than target_max")
        
        if 'confidence_level' in parameters:
            conf_level = parameters['confidence_level']
            if not 0 < conf_level < 1:
                errors.append("confidence_level must be between 0 and 1")
    
    elif strategy_name == 'CycleBudgetPruner':
        required_params = ['total_cycles']
        for param in required_params:
            if param not in parameters:
                errors.append(f"CycleBudgetPruner requires parameter: {param}")
        
        if 'total_cycles' in parameters and parameters['total_cycles'] <= 0:
            errors.append("total_cycles must be positive")
        
        if ('initial_retention_fraction' in parameters and 
            'final_retention_fraction' in parameters):
            initial = parameters['initial_retention_fraction']
            final = parameters['final_retention_fraction']
            if initial < final:
                errors.append("initial_retention_fraction should be >= final_retention_fraction for typical pruning")
    
    elif strategy_name == 'PerformanceBasedPruner':
        if 'performance_window' in parameters:
            window = parameters['performance_window']
            if window <= 0:
                errors.append("performance_window must be positive")
        
        if 'improvement_threshold' in parameters:
            threshold = parameters['improvement_threshold']
            if threshold < 0:
                errors.append("improvement_threshold must be non-negative")
    
    return len(errors) == 0, errors


def create_pruning_strategy(strategy_name: str, 
                           parameters: Dict[str, Any]):
    """Factory function to create pruning strategies.
    
    Args:
        strategy_name: Name of the pruning strategy class
        parameters: Dictionary of strategy parameters
        
    Returns:
        Initialized pruning strategy instance
        
    Raises:
        ValueError: If strategy name is invalid or parameters are invalid
    """
    # Validate parameters first
    is_valid, errors = validate_pruning_parameters(strategy_name, parameters)
    if not is_valid:
        raise ValueError(f"Invalid parameters for {strategy_name}: {'; '.join(errors)}")
    
    # Import strategy classes
    from .probabilistic import (
        ProbabilisticPruner, UncertaintyThresholdPruner, 
        PredictionThresholdPruner, ConfidenceIntervalPruner
    )
    from .adaptive import CycleBudgetPruner, PerformanceBasedPruner
    
    strategy_registry = {
        'ProbabilisticPruner': ProbabilisticPruner,
        'UncertaintyThresholdPruner': UncertaintyThresholdPruner,
        'PredictionThresholdPruner': PredictionThresholdPruner,
        'ConfidenceIntervalPruner': ConfidenceIntervalPruner,
        'CycleBudgetPruner': CycleBudgetPruner,
        'PerformanceBasedPruner': PerformanceBasedPruner
    }
    
    if strategy_name not in strategy_registry:
        available_strategies = ', '.join(strategy_registry.keys())
        raise ValueError(f"Unknown pruning strategy '{strategy_name}'. Available strategies: {available_strategies}")
    
    strategy_class = strategy_registry[strategy_name]
    
    try:
        return strategy_class(**parameters)
    except Exception as e:
        raise ValueError(f"Failed to create {strategy_name} with parameters {parameters}: {e}") from e

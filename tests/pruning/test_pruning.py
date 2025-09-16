"""Tests for pruning integration and utilities.

Tests the pruning utility functions and integration with the main active learning workflow.
"""

import pytest
import numpy as np
import pandas as pd
import json

from learnm8.pruning.utils import validate_pruning_parameters, create_pruning_strategy
from learnm8.pruning.score_based import ScoreBasedPruner
from learnm8.pruning.base import PruningError


def test_validate_pruning_parameters_valid():
    """Test validation with valid ScoreBasedPruner parameters."""
    # Valid parameters
    is_valid, errors = validate_pruning_parameters('ScoreBasedPruner', {
        'pruning_fraction': 0.3,
        'score_direction': 'higher'
    })
    assert is_valid == True
    assert len(errors) == 0
    
    # Valid with minimal parameters
    is_valid, errors = validate_pruning_parameters('ScoreBasedPruner', {
        'pruning_fraction': 0.0
    })
    assert is_valid == True
    assert len(errors) == 0


def test_validate_pruning_parameters_invalid():
    """Test validation with invalid parameters."""
    # Invalid pruning fraction - too high
    is_valid, errors = validate_pruning_parameters('ScoreBasedPruner', {
        'pruning_fraction': 1.0
    })
    assert is_valid == False
    assert any('pruning_fraction must be between 0.0 and 0.9' in error for error in errors)
    
    # Invalid pruning fraction - negative
    is_valid, errors = validate_pruning_parameters('ScoreBasedPruner', {
        'pruning_fraction': -0.1
    })
    assert is_valid == False
    assert any('pruning_fraction must be between 0.0 and 0.9' in error for error in errors)
    
    # Invalid score direction
    is_valid, errors = validate_pruning_parameters('ScoreBasedPruner', {
        'score_direction': 'invalid'
    })
    assert is_valid == False
    assert any("score_direction must be 'higher' or 'lower'" in error for error in errors)
    
    # Invalid pruning fraction type
    is_valid, errors = validate_pruning_parameters('ScoreBasedPruner', {
        'pruning_fraction': 'invalid'
    })
    assert is_valid == False
    assert any('pruning_fraction must be a number' in error for error in errors)
    
    # Invalid score direction type
    is_valid, errors = validate_pruning_parameters('ScoreBasedPruner', {
        'score_direction': 123
    })
    assert is_valid == False
    assert any('score_direction must be a string' in error for error in errors)
    
    # Unexpected parameters
    is_valid, errors = validate_pruning_parameters('ScoreBasedPruner', {
        'pruning_fraction': 0.2,
        'unexpected_param': 'value'
    })
    assert is_valid == False
    assert any('Unexpected parameters' in error for error in errors)


def test_validate_pruning_parameters_unknown_strategy():
    """Test validation with unknown strategy."""
    is_valid, errors = validate_pruning_parameters('UnknownStrategy', {})
    assert is_valid == False
    assert any("Unknown pruning strategy 'UnknownStrategy'" in error for error in errors)


def test_create_pruning_strategy_success():
    """Test successful creation of ScoreBasedPruner."""
    # Basic creation
    pruner = create_pruning_strategy('ScoreBasedPruner', {
        'pruning_fraction': 0.2,
        'score_direction': 'higher'
    })
    assert isinstance(pruner, ScoreBasedPruner)
    assert pruner.pruning_fraction == 0.2
    assert pruner.score_direction == 'higher'
    
    # Creation with default parameters
    pruner = create_pruning_strategy('ScoreBasedPruner', {})
    assert isinstance(pruner, ScoreBasedPruner)
    assert pruner.pruning_fraction == 0.1  # Default
    assert pruner.score_direction == 'higher'  # Default


def test_create_pruning_strategy_validation_failure():
    """Test that creation fails with invalid parameters."""
    with pytest.raises(ValueError, match="Invalid parameters for ScoreBasedPruner"):
        create_pruning_strategy('ScoreBasedPruner', {
            'pruning_fraction': 1.5  # Invalid
        })


def test_create_pruning_strategy_unknown_strategy():
    """Test that creation fails with unknown strategy."""
    with pytest.raises(ValueError, match="Unknown pruning strategy 'UnknownStrategy'"):
        create_pruning_strategy('UnknownStrategy', {})


def test_create_pruning_strategy_creation_failure():
    """Test handling of creation failures."""
    # This should trigger a validation error for invalid parameters
    with pytest.raises(ValueError, match="Invalid parameters for ScoreBasedPruner"):
        create_pruning_strategy('ScoreBasedPruner', {
            'score_direction': 'invalid_direction'  # This will cause validation to fail
        })


def test_pruning_integration_with_main_workflow(small_real_compounds):
    """Test that pruning integrates correctly with the main workflow."""
    from learnm8.learnm8 import apply_pruning_strategy
    
    # Create test data
    predictions = pd.DataFrame({'prediction': np.random.rand(len(small_real_compounds))})
    
    # Test score-based pruning
    pruned_pool, pruning_info = apply_pruning_strategy(
        pool=small_real_compounds,
        predictions=predictions,
        uncertainties=None,
        strategy='score_based',
        params={'pruning_fraction': 0.3},
        score_direction='higher'
    )
    
    # Verify results
    assert len(pruned_pool) < len(small_real_compounds)
    assert 'pruned_count' in pruning_info
    assert 'original_pool_size' in pruning_info
    assert 'pruned_pool_size' in pruning_info
    assert 'pruning_fraction' in pruning_info
    
    assert pruning_info['original_pool_size'] == len(small_real_compounds)
    assert pruning_info['pruned_pool_size'] == len(pruned_pool)
    assert pruning_info['pruned_count'] == len(small_real_compounds) - len(pruned_pool)


def test_pruning_integration_with_score_direction_injection(small_real_compounds):
    """Test that score_direction is correctly injected into pruning parameters."""
    from learnm8.learnm8 import apply_pruning_strategy
    
    predictions = pd.DataFrame({'prediction': np.random.rand(len(small_real_compounds))})
    
    # Test without score_direction in params - should be injected
    pruned_pool, pruning_info = apply_pruning_strategy(
        pool=small_real_compounds,
        predictions=predictions,
        uncertainties=None,
        strategy='score_based',
        params={'pruning_fraction': 0.3},
        score_direction='lower'  # Should be injected into params
    )
    
    # Should work correctly
    assert len(pruned_pool) < len(small_real_compounds)


def test_pruning_integration_error_handling(small_real_compounds):
    """Test error handling in pruning integration."""
    from learnm8.learnm8 import apply_pruning_strategy
    
    predictions = pd.DataFrame({'prediction': np.random.rand(len(small_real_compounds))})
    
    # Test unknown strategy
    with pytest.raises(ValueError, match="Failed to create pruning strategy"):
        apply_pruning_strategy(
            pool=small_real_compounds,
            predictions=predictions,
            uncertainties=None,
            strategy='unknown_strategy',
            params={},
            score_direction='higher'
        )
    
    # Test invalid parameters
    with pytest.raises(ValueError, match="Failed to create pruning strategy"):
        apply_pruning_strategy(
            pool=small_real_compounds,
            predictions=predictions,
            uncertainties=None,
            strategy='score_based',
            params={'pruning_fraction': 1.5},  # Invalid
            score_direction='higher'
        )


def test_cli_parameter_parsing_simulation():
    """Test CLI-style parameter parsing for pruning."""
    # Simulate CLI parameter parsing
    cli_params = '{"pruning_fraction": 0.25, "score_direction": "lower"}'
    params = json.loads(cli_params)
    
    # Should be able to create strategy from CLI params
    pruner = create_pruning_strategy('ScoreBasedPruner', params)
    assert isinstance(pruner, ScoreBasedPruner)
    assert pruner.pruning_fraction == 0.25
    assert pruner.score_direction == 'lower'


def test_pruning_backward_compatibility_removed():
    """Test that old pruning strategies are no longer available."""
    # These should all fail as old strategies are removed
    old_strategies = [
        'ProbabilisticPruner',
        'UncertaintyThresholdPruner', 
        'PredictionThresholdPruner',
        'ConfidenceIntervalPruner',
        'CycleBudgetPruner',
        'PerformanceBasedPruner'
    ]
    
    for strategy in old_strategies:
        with pytest.raises(ValueError, match=f"Unknown pruning strategy '{strategy}'"):
            create_pruning_strategy(strategy, {})
    
    # CLI-friendly names should also fail
    old_cli_names = [
        'probabilistic',
        'uncertainty_threshold',
        'prediction_threshold', 
        'confidence_interval',
        'cycle_budget',
        'performance_based'
    ]
    
    from learnm8.learnm8 import apply_pruning_strategy
    predictions = pd.DataFrame({'prediction': [0.5]})
    test_compounds = pd.DataFrame({'ID': ['test'], 'SMILES': ['CCO']})
    
    for strategy in old_cli_names:
        with pytest.raises(ValueError, match="Failed to create pruning strategy"):
            apply_pruning_strategy(
                pool=test_compounds,
                predictions=predictions,
                uncertainties=None,
                strategy=strategy,
                params={},
                score_direction='higher'
            )
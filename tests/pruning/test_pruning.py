"""Tests for pruning integration via public factory API.

Tests the pruning factory function and integration with the main active learning workflow.
Uses only public API (create_pruning_strategy) rather than internal validators.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
import json

from learnm8.pruning import create_pruning_strategy
from learnm8.pruning.score_based import ScoreBasedPruner

pytestmark = pytest.mark.unit


def test_create_pruning_strategy_valid_params():
    """Test factory with valid ScoreBasedPruner parameters."""
    pruner = create_pruning_strategy('score', {
        'pruning_fraction': 0.3,
        'score_direction': 'higher'
    })
    assert isinstance(pruner, ScoreBasedPruner)
    assert pruner.pruning_fraction == 0.3
    assert pruner.score_direction == 'higher'

    pruner = create_pruning_strategy('score', {
        'pruning_fraction': 0.0
    })
    assert isinstance(pruner, ScoreBasedPruner)
    assert pruner.pruning_fraction == 0.0


def test_create_pruning_strategy_invalid_fraction_too_high():
    """Test factory rejects invalid pruning fraction (too high)."""
    with pytest.raises(ValueError):
        create_pruning_strategy('score', {
            'pruning_fraction': 1.0
        })


def test_create_pruning_strategy_invalid_fraction_negative():
    """Test factory rejects invalid pruning fraction (negative)."""
    with pytest.raises(ValueError):
        create_pruning_strategy('score', {
            'pruning_fraction': -0.1
        })


def test_create_pruning_strategy_invalid_score_direction():
    """Test factory rejects invalid score_direction."""
    with pytest.raises(ValueError):
        create_pruning_strategy('score', {
            'score_direction': 'invalid'
        })


def test_create_pruning_strategy_invalid_fraction_type():
    """Test factory rejects invalid pruning_fraction type."""
    with pytest.raises(ValueError):
        create_pruning_strategy('score', {
            'pruning_fraction': 'invalid'
        })


def test_create_pruning_strategy_invalid_direction_type():
    """Test factory rejects invalid score_direction type."""
    with pytest.raises(ValueError):
        create_pruning_strategy('score', {
            'score_direction': 123
        })


def test_create_pruning_strategy_unexpected_params():
    """Test factory rejects unexpected parameters."""
    with pytest.raises(ValueError):
        create_pruning_strategy('score', {
            'pruning_fraction': 0.2,
            'unexpected_param': 'value'
        })


def test_create_pruning_strategy_unknown_strategy():
    """Test factory rejects unknown strategy."""
    with pytest.raises(ValueError, match="Unknown pruning strategy 'UnknownStrategy'"):
        create_pruning_strategy('UnknownStrategy', {})


def test_create_pruning_strategy_success():
    """Test successful creation of ScoreBasedPruner."""
    # Basic creation
    pruner = create_pruning_strategy('score', {
        'pruning_fraction': 0.2,
        'score_direction': 'higher'
    })
    assert isinstance(pruner, ScoreBasedPruner)
    assert pruner.pruning_fraction == 0.2
    assert pruner.score_direction == 'higher'

    # Creation with default parameters
    pruner = create_pruning_strategy('score', {})
    assert isinstance(pruner, ScoreBasedPruner)
    assert pruner.pruning_fraction == 0.1  # Default
    assert pruner.score_direction == 'higher'  # Default


def test_create_pruning_strategy_validation_failure():
    """Test that creation fails with invalid parameters."""
    with pytest.raises(ValueError, match="Invalid parameters for score"):
        create_pruning_strategy('score', {
            'pruning_fraction': 1.5  # Invalid
        })


def test_create_pruning_strategy_unknown_strategy():
    """Test that creation fails with unknown strategy."""
    with pytest.raises(ValueError, match="Unknown pruning strategy 'UnknownStrategy'"):
        create_pruning_strategy('UnknownStrategy', {})


def test_create_pruning_strategy_creation_failure():
    """Test handling of creation failures."""
    # This should trigger a validation error for invalid parameters
    with pytest.raises(ValueError, match="Invalid parameters for score"):
        create_pruning_strategy('score', {
            'score_direction': 'invalid_direction'  # This will cause validation to fail
        })


def test_pruning_integration_with_main_workflow(small_real_compounds):
    """Test that pruning integrates correctly with the main workflow."""
    from learnm8.pruning import create_pruning_strategy

    # Create test data
    predictions = np.random.rand(len(small_real_compounds))

    # Test score-based pruning
    pruner = create_pruning_strategy(
        'score',
        {'pruning_fraction': 0.3, 'score_direction': 'higher'}
    )
    pruned_pool = pruner.prune(small_real_compounds, predictions, None)

    # Verify results
    assert len(pruned_pool) < len(small_real_compounds)
    assert len(pruned_pool) <= len(small_real_compounds)


def test_pruning_integration_with_score_direction_injection(small_real_compounds):
    """Test that score_direction is correctly injected into pruning parameters."""
    from learnm8.pruning import create_pruning_strategy

    predictions = np.random.rand(len(small_real_compounds))

    # Test with score_direction in params
    pruner = create_pruning_strategy(
        'score',
        {'pruning_fraction': 0.3, 'score_direction': 'lower'}
    )
    pruned_pool = pruner.prune(small_real_compounds, predictions, None)

    # Should work correctly
    assert len(pruned_pool) < len(small_real_compounds)


def test_pruning_integration_error_handling(small_real_compounds):
    """Test error handling in pruning integration."""
    from learnm8.pruning import create_pruning_strategy

    predictions = np.random.rand(len(small_real_compounds))

    # Test unknown strategy
    with pytest.raises(ValueError, match="Unknown pruning strategy"):
        create_pruning_strategy('unknown_strategy', {})

    # Test invalid parameters
    with pytest.raises(ValueError, match="Invalid parameters"):
        create_pruning_strategy(
            'score',
            {'pruning_fraction': 1.5}  # Invalid
        )


def test_cli_parameter_parsing_simulation():
    """Test CLI-style parameter parsing for pruning."""
    # Simulate CLI parameter parsing
    cli_params = '{"pruning_fraction": 0.25, "score_direction": "lower"}'
    params = json.loads(cli_params)

    # Should be able to create strategy from CLI params
    pruner = create_pruning_strategy('score', params)
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

    for strategy in old_cli_names:
        with pytest.raises(ValueError, match="Unknown pruning strategy"):
            create_pruning_strategy(strategy, {})
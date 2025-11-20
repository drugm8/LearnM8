"""Performance sequence fixtures for LearnM8 tests.

Provides performance metric sequences for testing pruning and adaptation logic.

Complex sequences (lists of dicts):
- performance_data_sequence: Improving performance over 5 cycles
- declining_performance_sequence: Declining pattern over 4 cycles
- rapid_change_performance: Volatile pattern

Simple sequences (lists of floats):
- stagnant_performance_sequence: 7 flat values
- improving_performance_sequence: 7 improving values
- declining_performance_values: 7 declining values

Configuration:
- performance_metric_names: Valid metric name list
- adaptation_scenarios: Adaptation config dict
"""

from typing import List, Dict, Any
import pytest


@pytest.fixture
def performance_data_sequence() -> List[Dict[str, Any]]:
    """Create sequence of performance data for adaptation testing."""
    return [
        {
            'improvement_rate': 0.05,
            'selection_diversity': 0.7,
            'oracle_efficiency': 0.8,
            'cycle': 1
        },
        {
            'improvement_rate': 0.08,
            'selection_diversity': 0.75,
            'oracle_efficiency': 0.85,
            'cycle': 2
        },
        {
            'improvement_rate': 0.03,
            'selection_diversity': 0.6,
            'oracle_efficiency': 0.7,
            'cycle': 3
        },
        {
            'improvement_rate': -0.02,
            'selection_diversity': 0.5,
            'oracle_efficiency': 0.6,
            'cycle': 4
        },
        {
            'improvement_rate': 0.12,
            'selection_diversity': 0.9,
            'oracle_efficiency': 0.95,
            'cycle': 5
        }
    ]


@pytest.fixture
def declining_performance_sequence() -> List[Dict[str, Any]]:
    """Create declining performance sequence for edge case testing."""
    return [
        {
            'improvement_rate': 0.1,
            'selection_diversity': 0.8,
            'oracle_efficiency': 0.9,
            'cycle': 1
        },
        {
            'improvement_rate': 0.05,
            'selection_diversity': 0.7,
            'oracle_efficiency': 0.8,
            'cycle': 2
        },
        {
            'improvement_rate': -0.05,
            'selection_diversity': 0.5,
            'oracle_efficiency': 0.6,
            'cycle': 3
        },
        {
            'improvement_rate': -0.15,
            'selection_diversity': 0.3,
            'oracle_efficiency': 0.4,
            'cycle': 4
        }
    ]


@pytest.fixture
def rapid_change_performance() -> List[Dict[str, Any]]:
    """Create rapid performance changes for edge case testing."""
    return [
        {'improvement_rate': 0.1, 'selection_diversity': 0.8, 'oracle_efficiency': 0.9},
        {'improvement_rate': -0.2, 'selection_diversity': 0.2, 'oracle_efficiency': 0.3},
        {'improvement_rate': 0.3, 'selection_diversity': 0.9, 'oracle_efficiency': 0.95},
        {'improvement_rate': -0.1, 'selection_diversity': 0.4, 'oracle_efficiency': 0.5},
        {'improvement_rate': 0.25, 'selection_diversity': 0.85, 'oracle_efficiency': 0.9}
    ]


@pytest.fixture
def stagnant_performance_sequence() -> List[float]:
    """Create stagnant performance sequence for PerformanceBasedPruner."""
    return [0.5, 0.51, 0.49, 0.5, 0.52, 0.48, 0.5]


@pytest.fixture
def improving_performance_sequence() -> List[float]:
    """Create improving performance sequence for PerformanceBasedPruner."""
    return [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]


@pytest.fixture
def declining_performance_values() -> List[float]:
    """Create declining performance values for PerformanceBasedPruner."""
    return [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25]


@pytest.fixture
def performance_metric_names() -> List[str]:
    """List of valid performance metric names."""
    return ['improvement_rate', 'diversity', 'efficiency']


@pytest.fixture
def adaptation_scenarios() -> Dict[str, Dict[str, Any]]:
    """Different adaptation scenarios for comprehensive testing."""
    return {
        'conservative': {
            'adaptation_rate': 0.05,
            'min_retention_fraction': 0.5,
            'max_retention_fraction': 0.9
        },
        'aggressive': {
            'adaptation_rate': 0.2,
            'min_retention_fraction': 0.1,
            'max_retention_fraction': 0.8
        },
        'balanced': {
            'adaptation_rate': 0.1,
            'min_retention_fraction': 0.3,
            'max_retention_fraction': 0.8
        }
    }

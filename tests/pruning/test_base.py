"""Tests for base pruning functionality.

Tests the base DesignSpacePruner interface and utility functions using real molecular data.
"""

import pytest
import numpy as np
import polars as pl
from typing import Dict, Any, Optional

from learnm8.pruning.base import (
    DesignSpacePruner,
    PruningError,
)

pytestmark = pytest.mark.unit


class MockPruner(DesignSpacePruner):
    """Mock pruner for testing base functionality."""
    
    def __init__(self, requires_uncertainty=False, name="MockPruner"):
        self.requires_uncertainty_flag = requires_uncertainty
        self.name = name
        self.last_pruning_stats = {}
    
    def requires_uncertainty(self) -> bool:
        return self.requires_uncertainty_flag
    
    def get_name(self) -> str:
        return self.name
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        return self.last_pruning_stats.copy()
    
    def prune(self, compounds: pl.DataFrame, predictions: np.ndarray,
              uncertainties: Optional[np.ndarray] = None) -> pl.DataFrame:
        # Validate inputs
        self.validate_inputs(compounds, predictions, uncertainties)

        # Simple mock: keep first half of compounds
        n_keep = len(compounds) // 2
        pruned = compounds.head(n_keep)

        self.last_pruning_stats = {
            'compounds_before_pruning': len(compounds),
            'compounds_after_pruning': len(pruned),
            'compounds_pruned': len(compounds) - len(pruned),
            'pruning_fraction': self._calculate_pruning_fraction(len(compounds), len(pruned))
        }

        return pruned


def test_mock_pruner_basic_functionality(sample_compounds):
    """Test that MockPruner works correctly."""
    predictions = np.random.rand(len(sample_compounds))

    pruner = MockPruner()
    pruned = pruner.prune(sample_compounds, predictions)

    # Should keep approximately half
    assert len(pruned) == len(sample_compounds) // 2
    assert 'ID' in pruned.columns
    assert 'SMILES' in pruned.columns

    # Check stats
    stats = pruner.get_pruning_stats()
    assert stats['compounds_before_pruning'] == len(sample_compounds)
    assert stats['compounds_after_pruning'] == len(pruned)
    assert stats['compounds_pruned'] > 0


def test_pruner_interface_validation(sample_compounds):
    """Test input validation in base pruner class."""
    predictions = np.random.rand(len(sample_compounds))
    pruner = MockPruner()

    # Test empty DataFrame
    empty_df = sample_compounds.head(0)
    with pytest.raises(PruningError, match="compounds DataFrame is empty"):
        pruner.prune(empty_df, np.array([]))

    # Test missing columns
    bad_df = sample_compounds.drop(['SMILES'])
    with pytest.raises(PruningError, match="Missing required columns"):
        pruner.prune(bad_df, predictions)

    # Test mismatched prediction length
    with pytest.raises(PruningError, match="Predictions length"):
        pruner.prune(sample_compounds, predictions[:5])

    # Test NaN predictions
    bad_predictions = predictions.copy()
    bad_predictions[0] = np.nan
    with pytest.raises(PruningError, match="Predictions contain NaN values"):
        pruner.prune(sample_compounds, bad_predictions)


def test_uncertainty_validation(sample_compounds):
    """Test uncertainty validation."""
    predictions = np.random.rand(len(sample_compounds))
    uncertainties = np.random.rand(len(sample_compounds))

    # Test uncertainty-requiring pruner without uncertainty
    uncertainty_pruner = MockPruner(requires_uncertainty=True)
    with pytest.raises(PruningError, match="requires uncertainty estimates"):
        uncertainty_pruner.prune(sample_compounds, predictions)

    # Test mismatched uncertainty length
    with pytest.raises(PruningError, match="Uncertainties length"):
        uncertainty_pruner.prune(sample_compounds, predictions, uncertainties[:5])

    # Test negative uncertainties
    bad_uncertainties = uncertainties.copy()
    bad_uncertainties[0] = -1.0
    with pytest.raises(PruningError, match="Uncertainties must be non-negative"):
        uncertainty_pruner.prune(sample_compounds, predictions, bad_uncertainties)

    # Test NaN uncertainties
    bad_uncertainties = uncertainties.copy()
    bad_uncertainties[0] = np.nan
    with pytest.raises(PruningError, match="Uncertainties contain NaN values"):
        uncertainty_pruner.prune(sample_compounds, predictions, bad_uncertainties)


def test_safe_prune_by_indices(sample_compounds):
    """Test the _safe_prune_by_indices utility method."""
    predictions = np.random.rand(len(sample_compounds))
    pruner = MockPruner()

    # Test boolean indexing
    keep_mask = np.random.choice([True, False], size=len(sample_compounds))
    pruned = pruner._safe_prune_by_indices(sample_compounds, keep_mask)
    expected_count = np.sum(keep_mask)
    assert len(pruned) == expected_count

    # Test integer indexing
    keep_indices = np.array([0, 2, 4])
    pruned = pruner._safe_prune_by_indices(sample_compounds, keep_indices)
    assert len(pruned) == 3

    # Test empty selection
    empty_indices = np.array([], dtype=int)
    pruned = pruner._safe_prune_by_indices(sample_compounds, empty_indices)
    assert len(pruned) == 0
    assert list(pruned.columns) == list(sample_compounds.columns)


def test_pruning_fraction_calculation():
    """Test the _calculate_pruning_fraction utility method."""
    pruner = MockPruner()
    
    # Normal case
    assert pruner._calculate_pruning_fraction(100, 80) == 0.2
    
    # Edge cases
    assert pruner._calculate_pruning_fraction(0, 0) == 0.0
    assert pruner._calculate_pruning_fraction(10, 0) == 1.0
    assert pruner._calculate_pruning_fraction(10, 10) == 0.0


def test_requires_uncertainty_default():
    """Test that requires_uncertainty defaults to False."""
    pruner = MockPruner()
    assert pruner.requires_uncertainty() == False


def test_pruner_name_and_stats():
    """Test pruner name and stats functionality."""
    pruner = MockPruner(name="TestPruner")
    assert pruner.get_name() == "TestPruner"
    
    # Stats should be empty initially
    stats = pruner.get_pruning_stats()
    assert isinstance(stats, dict)
    assert len(stats) == 0
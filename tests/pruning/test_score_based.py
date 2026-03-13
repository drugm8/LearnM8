"""Tests for score-based pruning functionality.

Tests the ScoreBasedPruner implementation using real molecular data.
"""

import pytest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from learnm8.pruning.score_based import ScoreBasedPruner
from learnm8.pruning.base import PruningError

pytestmark = pytest.mark.unit


def test_score_based_pruner_initialization():
    """Test ScoreBasedPruner initialization and parameter validation."""
    # Valid initialization
    pruner = ScoreBasedPruner(pruning_fraction=0.2, score_direction='higher')
    assert pruner.pruning_fraction == 0.2
    assert pruner.score_direction == 'higher'
    
    # Default parameters
    default_pruner = ScoreBasedPruner()
    assert default_pruner.pruning_fraction == 0.1
    assert default_pruner.score_direction == 'higher'
    
    # Invalid pruning fraction
    with pytest.raises(PruningError, match="pruning_fraction must be between 0.0 and 0.9"):
        ScoreBasedPruner(pruning_fraction=1.0)
    
    with pytest.raises(PruningError, match="pruning_fraction must be between 0.0 and 0.9"):
        ScoreBasedPruner(pruning_fraction=-0.1)
    
    # Invalid score direction
    with pytest.raises(PruningError, match="score_direction must be 'higher' or 'lower'"):
        ScoreBasedPruner(score_direction='invalid')


def test_score_based_pruner_basic_functionality(sample_compounds):
    """Test basic pruning functionality with score-based approach."""
    n_compounds = len(sample_compounds)
    predictions = np.random.rand(n_compounds)
    
    # Test with 30% pruning
    pruner = ScoreBasedPruner(pruning_fraction=0.3, score_direction='higher')
    pruned = pruner.prune(sample_compounds, predictions)
    
    # Should keep approximately 70% of compounds
    expected_kept = n_compounds - int(n_compounds * 0.3)
    assert len(pruned) == expected_kept
    assert 'ID' in pruned.columns
    assert 'SMILES' in pruned.columns

    # Check statistics
    stats = pruner.get_pruning_stats()
    assert stats['compounds_before_pruning'] == n_compounds
    assert stats['compounds_after_pruning'] == expected_kept
    assert stats['compounds_pruned'] == n_compounds - expected_kept
    assert abs(stats['pruning_fraction_actual'] - 0.3) <= 0.1
    assert stats['score_direction'] == 'higher'


def test_score_based_pruner_higher_direction(sample_compounds):
    """Test pruning with 'higher' score direction."""
    predictions = np.array([0.1, 0.9, 0.3, 0.7, 0.5])  # Known values
    test_compounds = sample_compounds.head(5)
    
    # Remove 40% (2 compounds), keep 60% (3 compounds) with highest scores
    pruner = ScoreBasedPruner(pruning_fraction=0.4, score_direction='higher')
    pruned = pruner.prune(test_compounds, predictions)
    
    assert len(pruned) == 3
    
    # Check pruning statistics - for 'higher' direction, should keep compounds with highest scores
    stats = pruner.get_pruning_stats()
    assert stats['score_threshold'] == 0.5  # Minimum score among kept compounds (0.5, 0.7, 0.9)
    assert stats['score_range_kept']['min'] == 0.5
    assert stats['score_range_kept']['max'] == 0.9
    assert stats['compounds_pruned'] == 2
    assert stats['score_direction'] == 'higher'


def test_score_based_pruner_lower_direction(sample_compounds):
    """Test pruning with 'lower' score direction."""
    predictions = np.array([0.1, 0.9, 0.3, 0.7, 0.5])  # Known values
    test_compounds = sample_compounds.head(5)
    
    # Remove 40% (2 compounds), keep 60% (3 compounds) with lowest scores
    pruner = ScoreBasedPruner(pruning_fraction=0.4, score_direction='lower')
    pruned = pruner.prune(test_compounds, predictions)
    
    assert len(pruned) == 3
    
    # Check pruning statistics - for 'lower' direction, should keep compounds with lowest scores
    stats = pruner.get_pruning_stats()
    assert stats['score_threshold'] == 0.5  # Maximum score among kept compounds (0.1, 0.3, 0.5)
    assert stats['score_range_kept']['min'] == 0.1
    assert stats['score_range_kept']['max'] == 0.5
    assert stats['compounds_pruned'] == 2
    assert stats['score_direction'] == 'lower'


def test_score_based_pruner_no_pruning(sample_compounds):
    """Test behavior when pruning fraction is 0.0."""
    predictions = np.random.rand(len(sample_compounds))
    
    pruner = ScoreBasedPruner(pruning_fraction=0.0)
    pruned = pruner.prune(sample_compounds, predictions)
    
    # Should keep all compounds
    assert len(pruned) == len(sample_compounds)
    assert_frame_equal(pruned, sample_compounds)
    
    stats = pruner.get_pruning_stats()
    assert stats['compounds_pruned'] == 0
    assert stats['pruning_fraction_actual'] == 0.0


def test_score_based_pruner_edge_cases(sample_compounds):
    """Test edge cases for score-based pruning."""
    # Single compound
    single_compound = sample_compounds.head(1)
    predictions = np.array([0.5])
    
    pruner = ScoreBasedPruner(pruning_fraction=0.5)
    pruned = pruner.prune(single_compound, predictions)
    
    # Should keep the single compound (never prune to zero)
    assert len(pruned) == 1
    
    # Very high pruning fraction should still keep at least one compound
    predictions = np.random.rand(len(sample_compounds))
    high_pruning_pruner = ScoreBasedPruner(pruning_fraction=0.9)
    pruned = high_pruning_pruner.prune(sample_compounds, predictions)
    
    assert len(pruned) >= 1  # Should keep at least one compound


def test_score_based_pruner_statistics(sample_compounds):
    """Test comprehensive statistics collection."""
    predictions = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    test_compounds = sample_compounds.head(5)
    
    pruner = ScoreBasedPruner(pruning_fraction=0.4, score_direction='higher')
    pruned = pruner.prune(test_compounds, predictions)
    
    stats = pruner.get_pruning_stats()
    
    # Check all required statistics
    assert 'compounds_before_pruning' in stats
    assert 'compounds_after_pruning' in stats
    assert 'compounds_pruned' in stats
    assert 'pruning_fraction_actual' in stats
    assert 'pruning_fraction_requested' in stats
    assert 'score_direction' in stats
    assert 'score_threshold' in stats
    assert 'score_range_kept' in stats
    
    # Check score range statistics
    score_range = stats['score_range_kept']
    assert 'min' in score_range
    assert 'max' in score_range
    assert 'mean' in score_range
    
    assert score_range['min'] <= score_range['max']
    assert score_range['min'] <= score_range['mean'] <= score_range['max']


def test_score_based_pruner_doesnt_require_uncertainty():
    """Test that ScoreBasedPruner doesn't require uncertainty."""
    pruner = ScoreBasedPruner()
    assert pruner.requires_uncertainty() == False


def test_score_based_pruner_name():
    """Test pruner name formatting."""
    pruner = ScoreBasedPruner(pruning_fraction=0.2, score_direction='lower')
    name = pruner.get_name()
    
    assert 'ScoreBasedPruner' in name
    assert '0.2' in name
    assert 'lower' in name


def test_score_based_pruner_with_uncertainties(sample_compounds):
    """Test that ScoreBasedPruner ignores uncertainties when provided."""
    predictions = np.random.rand(len(sample_compounds))
    uncertainties = np.random.rand(len(sample_compounds))
    
    pruner = ScoreBasedPruner(pruning_fraction=0.3)
    
    # Should work fine even with uncertainties provided
    pruned_with_unc = pruner.prune(sample_compounds, predictions, uncertainties)
    pruned_without_unc = pruner.prune(sample_compounds, predictions, None)
    
    # Results should be identical (uncertainties are ignored)
    assert_frame_equal(pruned_with_unc, pruned_without_unc)


def test_score_based_pruner_input_validation(sample_compounds):
    """Test input validation inherits from base class."""
    pruner = ScoreBasedPruner()
    predictions = np.random.rand(len(sample_compounds))
    
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
        pruner.prune(sample_compounds, predictions[:3])


def test_score_based_pruner_deterministic_behavior(sample_compounds):
    """Test that pruning behavior is deterministic given same inputs."""
    predictions = np.random.rand(len(sample_compounds))
    
    pruner1 = ScoreBasedPruner(pruning_fraction=0.3, score_direction='higher')
    pruner2 = ScoreBasedPruner(pruning_fraction=0.3, score_direction='higher')
    
    pruned1 = pruner1.prune(sample_compounds, predictions)
    pruned2 = pruner2.prune(sample_compounds, predictions)

    # Results should be identical
    assert_frame_equal(pruned1, pruned2)


def test_score_based_pruner_preserves_dataframe_structure(sample_compounds):
    """Test that pruning preserves DataFrame structure and types."""
    predictions = np.random.rand(len(sample_compounds))
    
    pruner = ScoreBasedPruner(pruning_fraction=0.2)
    pruned = pruner.prune(sample_compounds, predictions)
    
    # Should preserve all columns
    assert list(pruned.columns) == list(sample_compounds.columns)

    # Should preserve data types
    for col in pruned.columns:
        assert pruned[col].dtype == sample_compounds[col].dtype
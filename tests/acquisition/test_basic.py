"""
Basic acquisition function tests: Greedy and Random acquisition.

Tests core functionality for the most commonly used acquisition methods.
"""

import pytest
import numpy as np
import polars as pl
from learnm8.acquisition import GreedyAcquisition, RandomAcquisition, TopKAcquisition


class TestGreedyAcquisition:
    """Test GreedyAcquisition functionality."""
    
    def test_greedy_basic_functionality(self, small_real_compounds):
        """Test basic greedy acquisition with real molecular data."""
        compounds = small_real_compounds.clone()
        # Add synthetic predictions
        np.random.seed(42)
        compounds = compounds.with_columns(
            pl.Series('prediction', np.random.uniform(0, 1, len(compounds)))
        )

        acq = GreedyAcquisition()
        selected = acq.select(compounds, n_select=5)

        # Should select compounds with highest predictions
        assert len(selected) == 5
        assert 'acquisition_score' in selected.columns

        # Verify greedy selection (highest scores first)
        selected_scores = selected.get_column('prediction').to_numpy()
        assert np.all(selected_scores[:-1] >= selected_scores[1:])
    
    def test_greedy_with_activity_scores(self, small_real_compounds):
        """Test greedy selection using actual activity scores."""
        compounds = small_real_compounds.clone()
        # Use Activity as prediction (meaningful docking scores)
        compounds = compounds.with_columns(
            pl.col('Activity').alias('prediction')
        )
        
        acq = GreedyAcquisition(score_direction='higher')  # Higher is better
        selected = acq.select(compounds, n_select=10)
        
        assert len(selected) == 10
        # Should select compounds with highest activities (less negative for docking scores)
        selected_activities = selected.get_column('Activity').to_numpy()
        assert np.all(selected_activities[:-1] >= selected_activities[1:])
    
    def test_greedy_deterministic_selection(self, small_real_compounds):
        """Test that greedy selection is deterministic."""
        compounds = small_real_compounds.clone()
        compounds = compounds.with_columns(
            pl.col('Activity').alias('prediction')
        )

        acq = GreedyAcquisition()
        selected1 = acq.select(compounds, n_select=5)
        selected2 = acq.select(compounds, n_select=5)

        # Should select identical compounds
        assert selected1.get_column('ID').to_list() == selected2.get_column('ID').to_list()
    
    def test_greedy_score_direction(self, small_real_compounds):
        """Test greedy selection with different score directions."""
        compounds = small_real_compounds.head(10).clone()
        # Use Activity scores (docking scores, lower is better)
        compounds = compounds.with_columns(
            pl.col('Activity').alias('prediction')
        )

        # Test 'lower' direction (typical for docking scores)
        acq_lower = GreedyAcquisition(score_direction='lower')
        selected_lower = acq_lower.select(compounds, n_select=3)

        # Test 'higher' direction
        acq_higher = GreedyAcquisition(score_direction='higher')
        selected_higher = acq_higher.select(compounds, n_select=3)

        # Should select different compounds
        lower_ids = set(selected_lower.get_column('ID').to_list())
        higher_ids = set(selected_higher.get_column('ID').to_list())
        assert lower_ids != higher_ids


class TestRandomAcquisition:
    """Test RandomAcquisition functionality."""
    
    def test_random_basic_functionality(self, small_real_compounds):
        """Test basic random acquisition."""
        compounds = small_real_compounds.clone()
        acq = RandomAcquisition(random_state=42)
        selected = acq.select(compounds, n_select=8)
        
        assert len(selected) == 8
        assert 'acquisition_score' in selected.columns

        # All compounds should have been valid selections
        compound_ids = compounds.get_column('ID').to_list()
        selected_ids = selected.get_column('ID').to_list()
        assert all(id in compound_ids for id in selected_ids)
    
    def test_random_reproducibility(self, small_real_compounds):
        """Test random acquisition reproducibility with seed."""
        compounds = small_real_compounds.clone()
        acq1 = RandomAcquisition(random_state=42)
        acq2 = RandomAcquisition(random_state=42)
        
        selected1 = acq1.select(compounds, n_select=10)
        selected2 = acq2.select(compounds, n_select=10)

        # Should select identical compounds with same seed
        assert selected1.get_column('ID').to_list() == selected2.get_column('ID').to_list()
    
    def test_random_different_seeds(self, small_real_compounds):
        """Test that different seeds produce different selections."""
        compounds = small_real_compounds.clone()
        acq1 = RandomAcquisition(random_state=42)
        acq2 = RandomAcquisition(random_state=123)
        
        selected1 = acq1.select(compounds, n_select=15)
        selected2 = acq2.select(compounds, n_select=15)

        # Should select different compounds with different seeds
        assert selected1.get_column('ID').to_list() != selected2.get_column('ID').to_list()
    
    def test_random_selection_coverage(self, medium_real_compounds):
        """Test random selection covers dataset reasonably."""
        compounds = medium_real_compounds.clone()

        if len(compounds) < 50:
            pytest.skip("Insufficient compounds for coverage test")

        # Use different random states for different selections
        acq1 = RandomAcquisition(random_state=42)
        acq2 = RandomAcquisition(random_state=123)

        # Multiple selections should cover different parts of dataset
        selected1 = acq1.select(compounds, n_select=20)
        selected2 = acq2.select(compounds, n_select=20)

        # Should have some overlap but not complete overlap
        ids1 = set(selected1.get_column('ID').to_list())
        ids2 = set(selected2.get_column('ID').to_list())
        overlap = len(ids1 & ids2)
        assert 0 < overlap < 20  # Some but not complete overlap


class TestTopKAcquisition:
    """Test TopKAcquisition functionality."""
    
    def test_topk_basic_functionality(self, small_real_compounds):
        """Test TopK acquisition basic functionality."""
        compounds = small_real_compounds.clone()
        # Add predictions based on activity
        compounds = compounds.with_columns(
            (pl.col('Activity') + pl.Series('noise', np.random.normal(0, 0.1, len(compounds)))).alias('prediction')
        )
        
        acq = TopKAcquisition(k_fraction=0.5)
        selected = acq.select(compounds, n_select=5)
        
        assert len(selected) == 5
        assert 'acquisition_score' in selected.columns
    
    def test_topk_fraction_behavior(self, small_real_compounds):
        """Test TopK behavior with different fractions."""
        compounds = small_real_compounds.clone()
        compounds = compounds.with_columns(
            pl.col('Activity').alias('prediction')
        )

        # Test different k_fractions
        acq_small = TopKAcquisition(k_fraction=0.2)
        acq_large = TopKAcquisition(k_fraction=0.8)

        selected_small = acq_small.select(compounds, n_select=5)
        selected_large = acq_large.select(compounds, n_select=5)

        assert len(selected_small) == 5
        assert len(selected_large) == 5

        # Both should select valid compounds
        compound_ids = compounds.get_column('ID').to_list()
        assert all(id in compound_ids for id in selected_small.get_column('ID').to_list())
        assert all(id in compound_ids for id in selected_large.get_column('ID').to_list())


class TestBasicAcquisitionIntegration:
    """Integration tests for basic acquisition functions."""
    
    def test_acquisition_with_real_molecular_workflow(self, medium_real_compounds):
        """Test acquisition functions in realistic molecular workflow."""
        compounds = medium_real_compounds.clone()
        
        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for workflow test")
        
        # Simulate predictions from different models
        np.random.seed(42)
        compounds = compounds.with_columns(
            (pl.col('Activity') + pl.Series('noise', np.random.normal(0, 2, len(compounds)))).alias('prediction')
        )

        # Test multiple acquisition strategies
        greedy_acq = GreedyAcquisition()
        random_acq = RandomAcquisition(random_state=42)
        topk_acq = TopKAcquisition(k_fraction=0.3)

        n_select = 10
        greedy_selected = greedy_acq.select(compounds, n_select=n_select)
        random_selected = random_acq.select(compounds, n_select=n_select)
        topk_selected = topk_acq.select(compounds, n_select=n_select)

        # All should select correct number of compounds
        assert len(greedy_selected) == n_select
        assert len(random_selected) == n_select
        assert len(topk_selected) == n_select

        # All should be valid selections
        all_ids = set(compounds.get_column('ID').to_list())
        assert set(greedy_selected.get_column('ID').to_list()).issubset(all_ids)
        assert set(random_selected.get_column('ID').to_list()).issubset(all_ids)
        assert set(topk_selected.get_column('ID').to_list()).issubset(all_ids)
        
        # Greedy and random should generally select different compounds
        greedy_ids = set(greedy_selected.get_column('ID').to_list())
        random_ids = set(random_selected.get_column('ID').to_list())
        overlap = len(greedy_ids & random_ids)
        assert overlap < n_select  # Should have some diversity
    
    def test_acquisition_with_diverse_molecular_targets(self, diverse_real_compounds):
        """Test acquisition with multi-target molecular data."""
        compounds = diverse_real_compounds.clone()
        
        if len(compounds) == 0:
            pytest.skip("No diverse molecular data available")
        
        # Add target-aware predictions
        compounds = compounds.with_columns(
            (pl.col('Activity') + pl.Series('noise', np.random.normal(0, 1, len(compounds)))).alias('prediction')
        )

        acq = GreedyAcquisition()
        selected = acq.select(compounds, n_select=min(15, len(compounds)))

        assert len(selected) <= len(compounds)

        # Should work with multi-target data
        if 'Target' in compounds.columns:
            # Check that selection covers multiple targets (if present)
            selected_targets = set(selected.get_column('Target').to_list()) if 'Target' in selected.columns else set()
            compound_targets = set(compounds.get_column('Target').to_list())
            
            # Should select from multiple targets (diversity is good)
            assert len(selected_targets) > 0
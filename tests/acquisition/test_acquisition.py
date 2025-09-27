"""Tests for basic LearnM8 acquisition functions.

Focused tests for core acquisition methods using real molecular data.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from numpy.testing import assert_allclose

from learnm8.core.data_manager import DataManager
from learnm8.acquisition import get_acquisition_function
from learnm8.acquisition.basic import GreedyAcquisition, RandomAcquisition, TopKAcquisition
from learnm8.acquisition.uncertainty_based import (
    UCBAcquisition, ExpectedImprovementAcquisition, 
    ProbabilityImprovementAcquisition, ThompsonSamplingAcquisition
)


class TestBasicAcquisition:
    """Test basic acquisition methods with real molecular data."""
    
    def test_greedy_acquisition_functionality(self, small_real_compounds):
        """Test greedy acquisition selects highest predictions."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add realistic predictions
        np.random.seed(42)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        acq = GreedyAcquisition()
        selected = acq.select(compounds, n_select=5)
        
        # Verify selection
        assert len(selected) == 5
        assert isinstance(selected, pd.DataFrame)
        assert all(col in selected.columns for col in ['ID', 'SMILES'])
        
        # Verify highest predictions were selected
        selected_predictions = selected['prediction'].values
        all_predictions = compounds['prediction'].values
        sorted_all = np.sort(all_predictions)[::-1]  # Descending order
        
        # Selected predictions should be among the top values
        assert np.all(selected_predictions >= sorted_all[4])  # At least top 5
    
    def test_greedy_acquisition_deterministic(self, small_real_compounds):
        """Test that greedy acquisition is deterministic."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add fixed predictions
        compounds['prediction'] = np.linspace(0, 1, len(compounds))
        
        acq = GreedyAcquisition()
        selected1 = acq.select(compounds, n_select=3)
        selected2 = acq.select(compounds, n_select=3)
        
        # Should be identical
        assert set(selected1['ID']) == set(selected2['ID'])
    
    def test_random_acquisition_functionality(self, small_real_compounds):
        """Test random acquisition with real molecular data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add predictions (not used by random, but needed for interface)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        acq = RandomAcquisition(random_state=42)
        selected = acq.select(compounds, n_select=5)
        
        # Verify selection
        assert len(selected) == 5
        assert isinstance(selected, pd.DataFrame)
        assert all(col in selected.columns for col in ['ID', 'SMILES'])
        
        # All selected compounds should be from original set
        assert all(id_val in compounds['ID'].values for id_val in selected['ID'])
    
    def test_random_acquisition_reproducible(self, small_real_compounds):
        """Test random acquisition reproducibility."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        # Same seed should give same results
        acq1 = RandomAcquisition(random_state=42)
        acq2 = RandomAcquisition(random_state=42)
        
        selected1 = acq1.select(compounds, n_select=3)
        selected2 = acq2.select(compounds, n_select=3)
        
        assert set(selected1['ID']) == set(selected2['ID'])
        
        # Different seed should give different results (with high probability)
        acq3 = RandomAcquisition(random_state=123)
        selected3 = acq3.select(compounds, n_select=3)
        
        assert set(selected1['ID']) != set(selected3['ID'])
    
    def test_topk_acquisition_functionality(self, small_real_compounds):
        """Test TopK acquisition functionality."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add predictions
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        acq = TopKAcquisition(k_fraction=0.5)
        selected = acq.select(compounds, n_select=3)
        
        # Should select exactly n_select compounds
        assert len(selected) == 3
        
        # Test that it selects the requested number of compounds
        selected_large = acq.select(compounds, n_select=10)
        assert len(selected_large) == 10
    
    def test_topk_equivalent_to_greedy(self, small_real_compounds):
        """Test that TopK with small k_fraction works correctly."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use fixed seed for reproducible test
        np.random.seed(42)
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        greedy_acq = GreedyAcquisition()
        topk_acq = TopKAcquisition(k_fraction=0.2)  # Consider top 20%
        
        greedy_selected = greedy_acq.select(compounds, n_select=5)
        topk_selected = topk_acq.select(compounds, n_select=5)
        
        # Verify both methods select the correct number
        assert len(greedy_selected) == 5
        assert len(topk_selected) == 5
        
        # Greedy should select the absolute top 5
        all_predictions = compounds['prediction'].values
        top_5_threshold = np.sort(all_predictions)[-5]
        
        greedy_scores = greedy_selected['prediction'].values
        assert np.all(greedy_scores >= top_5_threshold)
        
        # TopK should select from top 20% of compounds
        top_20_percent_threshold = np.percentile(all_predictions, 80)
        topk_scores = topk_selected['prediction'].values
        assert np.all(topk_scores >= top_20_percent_threshold)


class TestUncertaintyBasedAcquisition:
    """Test uncertainty-based acquisition methods."""
    
    def test_ucb_acquisition_functionality(self, compounds_with_uncertainty):
        """Test UCB acquisition with real compounds."""
        compounds = compounds_with_uncertainty.copy()

        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")

        # Test default behavior (score_direction='higher')
        acq = UCBAcquisition(beta=1.0)
        selected = acq.select(compounds, n_select=5)

        # Verify selection
        assert len(selected) == 5
        assert isinstance(selected, pd.DataFrame)
        assert all(col in selected.columns for col in ['ID', 'SMILES'])

        # UCB should favor high prediction + uncertainty for maximization
        selected_ucb_scores = (
            selected['prediction'].values +
            1.0 * selected['uncertainty'].values
        )

        # Calculate UCB for all compounds
        all_ucb_scores = (
            compounds['prediction'].values +
            1.0 * compounds['uncertainty'].values
        )

        # Selected compounds should have high UCB scores
        sorted_all_ucb = np.sort(all_ucb_scores)[::-1]
        min_selected_ucb = np.min(selected_ucb_scores)

        # Should be among top UCB scores
        assert min_selected_ucb >= sorted_all_ucb[4]  # At least top 5

    def test_ucb_score_direction_lower(self, compounds_with_uncertainty):
        """Test UCB with score_direction='lower' (LCB behavior)."""
        compounds = compounds_with_uncertainty.copy()

        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")

        # Test LCB behavior for minimization
        acq = UCBAcquisition(beta=1.0, score_direction='lower')
        selected = acq.select(compounds, n_select=5)

        # Verify selection
        assert len(selected) == 5
        assert isinstance(selected, pd.DataFrame)

        # For LCB, we want LOWEST (prediction - beta * uncertainty) scores
        selected_lcb_scores = (
            selected['prediction'].values -
            1.0 * selected['uncertainty'].values
        )

        # Calculate LCB for all compounds
        all_lcb_scores = (
            compounds['prediction'].values -
            1.0 * compounds['uncertainty'].values
        )

        # Selected compounds should have the lowest LCB scores
        sorted_all_lcb = np.sort(all_lcb_scores)  # Ascending order (lowest first)
        max_selected_lcb = np.max(selected_lcb_scores)

        # Should be among lowest LCB scores
        assert max_selected_lcb <= sorted_all_lcb[4]  # At most the 5th lowest

    def test_ucb_beta_parameter_effect(self, compounds_with_uncertainty):
        """Test effect of beta parameter in UCB."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        # Conservative (low beta) vs exploratory (high beta)
        acq_conservative = UCBAcquisition(beta=0.1)
        acq_exploratory = UCBAcquisition(beta=3.0)
        
        selected_conservative = acq_conservative.select(compounds, n_select=5)
        selected_exploratory = acq_exploratory.select(compounds, n_select=5)
        
        # Both should select 5 compounds
        assert len(selected_conservative) == 5
        assert len(selected_exploratory) == 5
        
        # Conservative should favor higher predictions
        conservative_pred_mean = selected_conservative['prediction'].mean()
        exploratory_pred_mean = selected_exploratory['prediction'].mean()
        
        # This is probabilistic, but conservative should generally have higher pred mean
        # (with some tolerance for randomness)
        assert conservative_pred_mean >= exploratory_pred_mean - 0.2
    
    def test_expected_improvement_functionality(self, compounds_with_uncertainty):
        """Test Expected Improvement acquisition."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        acq = ExpectedImprovementAcquisition(xi=0.01)
        selected = acq.select(compounds, n_select=5)
        
        # Verify basic functionality
        assert len(selected) == 5
        assert isinstance(selected, pd.DataFrame)
        assert all(col in selected.columns for col in ['ID', 'SMILES'])
        
        # All selected should be from original set
        assert all(id_val in compounds['ID'].values for id_val in selected['ID'])
    
    def test_probability_improvement_functionality(self, compounds_with_uncertainty):
        """Test Probability of Improvement acquisition."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        acq = ProbabilityImprovementAcquisition(xi=0.01)
        selected = acq.select(compounds, n_select=3)
        
        # Verify basic functionality
        assert len(selected) == 3
        assert isinstance(selected, pd.DataFrame)
        assert all(col in selected.columns for col in ['ID', 'SMILES'])
    
    def test_thompson_sampling_functionality(self, compounds_with_uncertainty):
        """Test Thompson Sampling acquisition."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        acq = ThompsonSamplingAcquisition(random_state=42)
        selected = acq.select(compounds, n_select=4)
        
        # Verify basic functionality
        assert len(selected) == 4
        assert isinstance(selected, pd.DataFrame)
        assert all(col in selected.columns for col in ['ID', 'SMILES'])
    
    def test_thompson_sampling_reproducible(self, compounds_with_uncertainty):
        """Test Thompson Sampling reproducibility."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        # Same seed should give same results
        acq1 = ThompsonSamplingAcquisition(random_state=42)
        acq2 = ThompsonSamplingAcquisition(random_state=42)
        
        selected1 = acq1.select(compounds, n_select=3)
        selected2 = acq2.select(compounds, n_select=3)
        
        assert set(selected1['ID']) == set(selected2['ID'])
    
    def test_uncertainty_requirement_validation(self, small_real_compounds):
        """Test that uncertainty-based methods require uncertainty column."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add predictions but no uncertainty
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        # Should fail without uncertainty
        acq = UCBAcquisition()
        with pytest.raises(ValueError, match="requires.*uncertainty"):
            acq.select(compounds, n_select=3)
        
        # Should work with uncertainty
        compounds['uncertainty'] = np.random.uniform(0.1, 0.5, len(compounds))
        selected = acq.select(compounds, n_select=3)
        assert len(selected) == 3


class TestAcquisitionIntegration:
    """Test acquisition function integration and workflows."""
    
    def test_acquisition_function_registry(self):
        """Test acquisition function registration and retrieval."""
        # Test basic functions
        basic_functions = ['greedy', 'random', 'topk', 'ucb', 'ei', 'pi', 'thompson']
        
        for func_name in basic_functions:
            acq_cls = get_acquisition_function(func_name)
            assert callable(acq_cls)
            
            # Test instantiation (basic functions don't need DataManager)
            if func_name not in ['pca_dbscan', 'umap_dbscan', 'bitbirch']:
                acq = acq_cls()
                assert hasattr(acq, 'select')
                assert hasattr(acq, 'requires_uncertainty')
                assert hasattr(acq, 'get_name')
    
    def test_acquisition_error_handling(self, small_real_compounds):
        """Test acquisition function error handling."""
        if len(small_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        compounds = small_real_compounds.copy()
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        
        acq = GreedyAcquisition()
        
        # Test empty DataFrame
        empty_df = pd.DataFrame(columns=['ID', 'SMILES', 'prediction'])
        with pytest.raises(ValueError, match="empty"):
            acq.select(empty_df, n_select=1)
        
        # Test missing required columns
        invalid_df = compounds.drop(columns=['prediction'])
        with pytest.raises(ValueError, match="Missing required columns"):
            acq.select(invalid_df, n_select=1)
        
        # Test invalid n_select
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(compounds, n_select=0)
        
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(compounds, n_select=-1)
    
    def test_acquisition_with_minimal_data(self):
        """Test acquisition functions with minimal datasets."""
        # Single compound
        single_compound = pd.DataFrame({
            'ID': ['mol_1'],
            'SMILES': ['CCO'],
            'prediction': [0.5],
            'uncertainty': [0.2]
        })
        
        # Test basic acquisition
        acq = GreedyAcquisition()
        selected = acq.select(single_compound, n_select=1)
        assert len(selected) == 1
        assert selected['ID'].iloc[0] == 'mol_1'
        
        # Test uncertainty-based acquisition
        acq_ucb = UCBAcquisition()
        selected_ucb = acq_ucb.select(single_compound, n_select=1)
        assert len(selected_ucb) == 1
        
        # Test requesting more than available
        selected_too_many = acq.select(single_compound, n_select=5)
        assert len(selected_too_many) == 1  # Should return what's available
    
    def test_acquisition_with_real_workflow(self, medium_real_compounds, tmp_path):
        """Test acquisition in realistic active learning workflow."""
        if len(medium_real_compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Use subset for faster testing
        compounds = medium_real_compounds.head(50).copy()
        
        # Simulate active learning cycle
        labeled_compounds = compounds.head(30).copy()
        
        # Add realistic predictions and uncertainties
        np.random.seed(42)
        labeled_compounds['prediction'] = (
            labeled_compounds['Activity'].values + 
            np.random.normal(0, 0.1, len(labeled_compounds))
        )
        labeled_compounds['uncertainty'] = 0.1 + 0.3 * np.abs(
            labeled_compounds['prediction'] - 0.5
        )
        
        # Test multiple acquisition strategies
        strategies = [
            ('greedy', GreedyAcquisition()),
            ('random', RandomAcquisition(random_state=42)),
            ('ucb', UCBAcquisition(beta=1.0))
        ]
        
        selections = {}
        for name, acq in strategies:
            selected = acq.select(labeled_compounds, n_select=5)
            selections[name] = selected
            
            # Verify each selection
            assert len(selected) == 5
            assert isinstance(selected, pd.DataFrame)
            assert all(col in selected.columns for col in ['ID', 'SMILES'])
        
        # Different strategies should make different selections
        greedy_ids = set(selections['greedy']['ID'])
        random_ids = set(selections['random']['ID'])
        ucb_ids = set(selections['ucb']['ID'])
        
        # At least some difference expected (probabilistic test)
        assert len(greedy_ids.union(random_ids).union(ucb_ids)) > 5


class TestAcquisitionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_acquisition_with_nan_predictions(self, small_real_compounds):
        """Test acquisition handling NaN predictions."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add predictions with NaN values
        compounds['prediction'] = np.random.uniform(0, 1, len(compounds))
        compounds.loc[0, 'prediction'] = np.nan
        compounds.loc[1, 'prediction'] = np.nan
        
        acq = GreedyAcquisition()
        
        # Should handle NaN values appropriately
        try:
            selected = acq.select(compounds, n_select=3)
            # If it succeeds, should not select NaN compounds
            assert not selected['prediction'].isna().any()
        except (ValueError, RuntimeError):
            # This is also acceptable behavior
            pass
    
    def test_acquisition_with_identical_predictions(self, small_real_compounds):
        """Test acquisition with identical predictions."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # All predictions are identical
        compounds['prediction'] = 0.5
        compounds['uncertainty'] = np.random.uniform(0.1, 0.5, len(compounds))
        
        # Greedy should still work (arbitrary selection among equal values)
        acq_greedy = GreedyAcquisition()
        selected = acq_greedy.select(compounds, n_select=3)
        assert len(selected) == 3
        
        # UCB should differentiate based on uncertainty
        acq_ucb = UCBAcquisition(beta=1.0)
        selected_ucb = acq_ucb.select(compounds, n_select=3)
        assert len(selected_ucb) == 3
        
        # UCB should select higher uncertainty compounds
        selected_uncertainty = selected_ucb['uncertainty'].mean()
        all_uncertainty = compounds['uncertainty'].mean()
        assert selected_uncertainty >= all_uncertainty - 0.1
    
    def test_acquisition_with_extreme_values(self, small_real_compounds):
        """Test acquisition with extreme prediction/uncertainty values."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add extreme values
        compounds['prediction'] = [0.0, 0.001, 0.999, 1.0] + [0.5] * (len(compounds) - 4)
        compounds['uncertainty'] = [0.001, 0.999, 0.001, 0.999] + [0.1] * (len(compounds) - 4)
        
        # Should handle extreme values gracefully
        acq_greedy = GreedyAcquisition()
        selected_greedy = acq_greedy.select(compounds, n_select=2)
        assert len(selected_greedy) == 2
        
        acq_ucb = UCBAcquisition(beta=2.0)
        selected_ucb = acq_ucb.select(compounds, n_select=2)
        assert len(selected_ucb) == 2
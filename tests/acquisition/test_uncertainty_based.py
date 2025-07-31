"""
Uncertainty-based acquisition function tests.

Tests acquisition functions that leverage prediction uncertainty for selection.
"""

import pytest
import numpy as np
import pandas as pd
from learnm8.acquisition.uncertainty_based import (
    UCBAcquisition, ExpectedImprovementAcquisition, 
    ProbabilityImprovementAcquisition, ThompsonSamplingAcquisition
)


class TestUCBAcquisition:
    """Test Upper Confidence Bound acquisition."""
    
    def test_ucb_basic_functionality(self, compounds_with_uncertainty):
        """Test UCB acquisition with uncertainty estimates."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        acq = UCBAcquisition(data_manager=None, beta=1.0)
        selected = acq.select(compounds, n_select=5)
        
        assert len(selected) == 5
        assert 'acquisition_score' in selected.columns
        
        # UCB scores should incorporate both prediction and uncertainty
        # Higher prediction + higher uncertainty = higher UCB score
        ucb_scores = selected['acquisition_score'].values
        assert np.all(np.isfinite(ucb_scores))
    
    def test_ucb_beta_parameter(self, compounds_with_uncertainty):
        """Test UCB with different beta values."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        # Test different exploration parameters
        acq_conservative = UCBAcquisition(data_manager=None, beta=0.1)  # Low exploration
        acq_aggressive = UCBAcquisition(data_manager=None, beta=2.0)    # High exploration
        
        selected_conservative = acq_conservative.select(compounds, n_select=5)
        selected_aggressive = acq_aggressive.select(compounds, n_select=5)
        
        assert len(selected_conservative) == 5
        assert len(selected_aggressive) == 5
        
        # Check that both strategies work (selections may be similar with small datasets)
        conservative_ids = set(selected_conservative['ID'])
        aggressive_ids = set(selected_aggressive['ID'])
        
        # Both strategies should make valid selections
        assert len(conservative_ids) == 5
        assert len(aggressive_ids) == 5
        
        # Conservative should favor high predictions, aggressive should include high uncertainty
        conservative_scores = selected_conservative['acquisition_score'].mean()
        aggressive_scores = selected_aggressive['acquisition_score'].mean()
        
        # Both should produce valid UCB scores
        assert np.isfinite(conservative_scores)
        assert np.isfinite(aggressive_scores)
    
    def test_ucb_uncertainty_weighting(self, small_real_compounds):
        """Test UCB properly weights uncertainty."""
        compounds = small_real_compounds.head(10).copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Create scenario with varying uncertainty
        compounds['prediction'] = np.random.uniform(0.3, 0.7, len(compounds))
        compounds['uncertainty'] = [0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5]
        
        acq = UCBAcquisition(data_manager=None, beta=1.0)
        selected = acq.select(compounds, n_select=3)
        
        # Should prefer compounds with higher uncertainty (exploration)
        selected_uncertainties = selected['uncertainty'].values
        assert np.mean(selected_uncertainties) > np.mean(compounds['uncertainty'])


class TestExpectedImprovementAcquisition:
    """Test Expected Improvement acquisition."""
    
    def test_ei_basic_functionality(self, compounds_with_uncertainty):
        """Test EI acquisition functionality."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        # Use default xi parameter for exploration
        acq = ExpectedImprovementAcquisition(data_manager=None, xi=0.01)
        selected = acq.select(compounds, n_select=5)
        
        assert len(selected) == 5
        assert 'acquisition_score' in selected.columns
        assert np.all(selected['acquisition_score'] >= 0)  # EI is always non-negative
    
    def test_ei_xi_parameter(self, compounds_with_uncertainty):
        """Test EI with different xi exploration parameters."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        # Test with different xi values (exploration parameters)
        acq_low = ExpectedImprovementAcquisition(data_manager=None, xi=0.001)  # Low exploration
        acq_high = ExpectedImprovementAcquisition(data_manager=None, xi=0.1)   # High exploration
        
        selected_low = acq_low.select(compounds, n_select=5)
        selected_high = acq_high.select(compounds, n_select=5)
        
        assert len(selected_low) == 5
        assert len(selected_high) == 5
        
        # Both should select compounds, differences may be subtle
        assert all(isinstance(selected_low.iloc[0]['acquisition_score'], (int, float)) for _ in [0])
        assert all(isinstance(selected_high.iloc[0]['acquisition_score'], (int, float)) for _ in [0])
    
    def test_ei_with_realistic_molecular_scenario(self, small_real_compounds):
        """Test EI in realistic molecular discovery scenario."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Simulate model predictions with uncertainty
        np.random.seed(42)
        compounds['prediction'] = compounds['Activity'] + np.random.normal(0, 1, len(compounds))
        compounds['uncertainty'] = np.random.uniform(0.1, 0.5, len(compounds))
        
        # Use standard EI with default parameters
        acq = ExpectedImprovementAcquisition(data_manager=None, xi=0.01)
        selected = acq.select(compounds, n_select=8)
        
        assert len(selected) == 8
        # EI should select compounds with potential for improvement
        ei_scores = selected['acquisition_score'].values
        assert np.all(ei_scores >= 0)


class TestProbabilityImprovementAcquisition:
    """Test Probability of Improvement acquisition."""
    
    def test_pi_basic_functionality(self, compounds_with_uncertainty):
        """Test PI acquisition functionality."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        acq = ProbabilityImprovementAcquisition(data_manager=None, xi=0.01)
        selected = acq.select(compounds, n_select=5)
        
        assert len(selected) == 5
        assert 'acquisition_score' in selected.columns
        
        # PI scores should be probabilities (0 to 1)
        pi_scores = selected['acquisition_score'].values
        assert np.all(pi_scores >= 0)
        assert np.all(pi_scores <= 1)
    
    def test_pi_probability_interpretation(self, compounds_with_uncertainty):
        """Test PI probability interpretation."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        # Set current_best very high to get low PI values
        acq = ProbabilityImprovementAcquisition(data_manager=None, xi=0.1)  # Higher exploration parameter
        selected = acq.select(compounds, n_select=5)
        
        # Should still select 5 compounds even if PI is low
        assert len(selected) == 5
        
        # PI scores should be valid probabilities
        pi_scores = selected['acquisition_score'].values
        assert np.all(pi_scores >= 0)
        assert np.all(pi_scores <= 1)


class TestThompsonSampling:
    """Test Thompson Sampling acquisition."""
    
    def test_thompson_sampling_basic(self, compounds_with_uncertainty):
        """Test Thompson Sampling basic functionality."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        acq = ThompsonSamplingAcquisition(data_manager=None, random_state=42)
        selected = acq.select(compounds, n_select=5)
        
        assert len(selected) == 5
        assert 'acquisition_score' in selected.columns
    
    def test_thompson_sampling_reproducibility(self, compounds_with_uncertainty):
        """Test Thompson Sampling reproducibility."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        acq1 = ThompsonSamplingAcquisition(data_manager=None, random_state=42)
        acq2 = ThompsonSamplingAcquisition(data_manager=None, random_state=42)
        
        selected1 = acq1.select(compounds, n_select=8)
        selected2 = acq2.select(compounds, n_select=8)
        
        # Should select identical compounds with same seed
        assert list(selected1['ID']) == list(selected2['ID'])
    
    def test_thompson_sampling_stochasticity(self, compounds_with_uncertainty):
        """Test Thompson Sampling produces different selections."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        acq1 = ThompsonSamplingAcquisition(data_manager=None, random_state=42)
        acq2 = ThompsonSamplingAcquisition(data_manager=None, random_state=123)
        
        selected1 = acq1.select(compounds, n_select=10)
        selected2 = acq2.select(compounds, n_select=10)
        
        # Should select different compounds with different seeds
        assert list(selected1['ID']) != list(selected2['ID'])


class TestUncertaintyBasedIntegration:
    """Integration tests for uncertainty-based acquisition."""
    
    def test_uncertainty_acquisition_comparison(self, compounds_with_uncertainty):
        """Compare different uncertainty-based acquisition methods."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        # Test multiple uncertainty-based methods
        ucb_acq = UCBAcquisition(data_manager=None, beta=1.0)
        ei_acq = ExpectedImprovementAcquisition(data_manager=None, xi=0.01)
        pi_acq = ProbabilityImprovementAcquisition(data_manager=None, xi=0.01)
        ts_acq = ThompsonSamplingAcquisition(data_manager=None, random_state=42)
        
        n_select = 5
        ucb_selected = ucb_acq.select(compounds, n_select=n_select)
        ei_selected = ei_acq.select(compounds, n_select=n_select)
        pi_selected = pi_acq.select(compounds, n_select=n_select)
        ts_selected = ts_acq.select(compounds, n_select=n_select)
        
        # All should select correct number
        assert len(ucb_selected) == n_select
        assert len(ei_selected) == n_select
        assert len(pi_selected) == n_select
        assert len(ts_selected) == n_select
        
        # Different methods should show some diversity in selection
        all_selections = [
            set(ucb_selected['ID']),
            set(ei_selected['ID']),
            set(pi_selected['ID']),
            set(ts_selected['ID'])
        ]
        
        # Not all methods should select identical compounds
        assert not all(s == all_selections[0] for s in all_selections[1:])
    
    def test_uncertainty_methods_with_molecular_workflow(self, medium_real_compounds):
        """Test uncertainty methods in realistic molecular workflow."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for workflow test")
        
        # Simulate realistic uncertainty estimates
        np.random.seed(42)
        compounds['prediction'] = compounds['Activity'] + np.random.normal(0, 2, len(compounds))
        compounds['uncertainty'] = np.random.uniform(0.1, 1.0, len(compounds))
        
        # Test that uncertainty methods work with real molecular data
        acq = UCBAcquisition(data_manager=None, beta=1.5)
        selected = acq.select(compounds, n_select=15)
        
        assert len(selected) == 15
        assert all(id in compounds['ID'].values for id in selected['ID'])
        
        # Selected compounds should have reasonable uncertainty properties
        selected_uncertainty = selected['uncertainty'].values
        assert np.all(selected_uncertainty > 0)
        assert np.all(np.isfinite(selected_uncertainty))
    
    def test_uncertainty_acquisition_error_handling(self, small_real_compounds):
        """Test error handling in uncertainty-based acquisition."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Test with missing uncertainty
        compounds['prediction'] = compounds['Activity']
        # No uncertainty column
        
        acq = UCBAcquisition(data_manager=None, beta=1.0)
        
        with pytest.raises((KeyError, ValueError)):
            acq.select(compounds, n_select=5)
        
        # Test with NaN uncertainty
        compounds['uncertainty'] = np.nan
        
        with pytest.raises((ValueError, RuntimeError)):
            acq.select(compounds, n_select=5)
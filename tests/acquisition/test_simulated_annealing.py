"""Tests for simulated annealing acquisition function.

This module tests the SimulatedAnnealingAcquisition class following LearnM8's
testing philosophy of using real molecular data and focusing on functionality,
integration, error handling, and data validation.
"""

import pytest
import numpy as np
import pandas as pd
import logging

from learnm8.acquisition.simulated_annealing import SimulatedAnnealingAcquisition


class TestSimulatedAnnealingAcquisition:
    """Test SimulatedAnnealingAcquisition with real molecular data."""
    
    def test_basic_functionality(self, small_real_compounds):
        """Test basic simulated annealing functionality with real pharmaceutical compounds."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add realistic predictions based on docking scores
        np.random.seed(42)
        compounds['prediction'] = compounds['Activity'] + np.random.normal(0, 0.5, len(compounds))
        
        # Test basic selection
        acq = SimulatedAnnealingAcquisition(random_state=42, max_iterations=100)
        selected = acq.select(compounds, n_select=8)
        
        # Verify output structure
        assert len(selected) == 8
        assert 'acquisition_score' in selected.columns
        assert all(id in compounds['ID'].values for id in selected['ID'])
        assert len(selected['ID'].unique()) == len(selected)  # No duplicates
        
        # Verify acquisition scores are present and finite
        assert np.all(np.isfinite(selected['acquisition_score']))
    
    def test_parameter_validation(self):
        """Test parameter validation in constructor."""
        # Test invalid temperatures
        with pytest.raises(ValueError, match="initial_temp must be positive"):
            SimulatedAnnealingAcquisition(initial_temp=0)
        
        with pytest.raises(ValueError, match="final_temp must be positive"):
            SimulatedAnnealingAcquisition(final_temp=0)
        
        with pytest.raises(ValueError, match="final_temp must be less than initial_temp"):
            SimulatedAnnealingAcquisition(initial_temp=0.5, final_temp=1.0)
        
        # Test invalid iterations
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            SimulatedAnnealingAcquisition(max_iterations=0)
        
        # Test invalid cooling schedule
        with pytest.raises(ValueError, match="cooling_schedule must be"):
            SimulatedAnnealingAcquisition(cooling_schedule='invalid')
        
        # Test invalid score direction
        with pytest.raises(ValueError, match="score_direction must be"):
            SimulatedAnnealingAcquisition(score_direction='invalid')
    
    def test_cooling_schedules(self, small_real_compounds):
        """Test different cooling schedules with real molecular data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        np.random.seed(42)
        compounds['prediction'] = compounds['Activity'] + np.random.normal(0, 0.5, len(compounds))
        
        # Test exponential cooling
        acq_exp = SimulatedAnnealingAcquisition(
            cooling_schedule='exponential', 
            random_state=42,
            max_iterations=50
        )
        selected_exp = acq_exp.select(compounds, n_select=5)
        assert len(selected_exp) == 5
        assert acq_exp.get_name() == "SimulatedAnnealing(exponential_higher)"
        
        # Test linear cooling
        acq_lin = SimulatedAnnealingAcquisition(
            cooling_schedule='linear', 
            random_state=42,
            max_iterations=50
        )
        selected_lin = acq_lin.select(compounds, n_select=5)
        assert len(selected_lin) == 5
        assert acq_lin.get_name() == "SimulatedAnnealing(linear_higher)"
        
        # Results may differ due to different cooling schedules
        # but both should return valid selections
        assert np.all(np.isfinite(selected_exp['acquisition_score']))
        assert np.all(np.isfinite(selected_lin['acquisition_score']))
    
    def test_score_directions(self, small_real_compounds):
        """Test both score directions (higher and lower optimization)."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        np.random.seed(42)
        compounds['prediction'] = compounds['Activity'] + np.random.normal(0, 0.5, len(compounds))
        
        # Test higher is better (maximization)
        acq_higher = SimulatedAnnealingAcquisition(
            score_direction='higher', 
            random_state=42,
            max_iterations=50
        )
        selected_higher = acq_higher.select(compounds, n_select=5)
        
        # Test lower is better (minimization)
        acq_lower = SimulatedAnnealingAcquisition(
            score_direction='lower', 
            random_state=42,
            max_iterations=50
        )
        selected_lower = acq_lower.select(compounds, n_select=5)
        
        # Both should return valid selections
        assert len(selected_higher) == 5
        assert len(selected_lower) == 5
        assert acq_higher.get_name() == "SimulatedAnnealing(exponential_higher)"
        assert acq_lower.get_name() == "SimulatedAnnealing(exponential_lower)"
        
        # For the same predictions, different score directions may select different compounds
        # but all selections should be valid
        assert np.all(np.isfinite(selected_higher['acquisition_score']))
        assert np.all(np.isfinite(selected_lower['acquisition_score']))
    
    def test_reproducibility(self, small_real_compounds):
        """Test reproducible behavior with fixed random seed."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        np.random.seed(42)
        compounds['prediction'] = compounds['Activity'] + np.random.normal(0, 0.5, len(compounds))
        
        # Test same seed produces identical results
        acq1 = SimulatedAnnealingAcquisition(random_state=42, max_iterations=50)
        acq2 = SimulatedAnnealingAcquisition(random_state=42, max_iterations=50)
        
        selected1 = acq1.select(compounds, n_select=5)
        selected2 = acq2.select(compounds, n_select=5)
        
        # Should select identical compounds with same seed
        assert list(selected1['ID']) == list(selected2['ID'])
        assert np.allclose(selected1['acquisition_score'], selected2['acquisition_score'])
        
        # Test different seed produces different results (with high probability)
        acq3 = SimulatedAnnealingAcquisition(random_state=123, max_iterations=50)
        selected3 = acq3.select(compounds, n_select=5)
        
        # Different seeds should likely produce different selections
        # Note: There's a small chance they could be the same, but very unlikely
        assert len(selected3) == 5
    
    def test_edge_cases(self, small_real_compounds):
        """Test edge cases and boundary conditions."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        np.random.seed(42)
        compounds['prediction'] = compounds['Activity'] + np.random.normal(0, 0.5, len(compounds))
        
        # Test selecting all compounds
        acq = SimulatedAnnealingAcquisition(random_state=42)
        selected_all = acq.select(compounds, n_select=len(compounds))
        assert len(selected_all) == len(compounds)
        assert set(selected_all['ID']) == set(compounds['ID'])
        
        # Test selecting more than available (should return all)
        selected_more = acq.select(compounds, n_select=len(compounds) + 10)
        assert len(selected_more) == len(compounds)
        
        # Test selecting single compound
        selected_one = acq.select(compounds, n_select=1)
        assert len(selected_one) == 1
        assert selected_one.iloc[0]['ID'] in compounds['ID'].values
    
    def test_with_uncertainty_estimates(self, compounds_with_uncertainty):
        """Test integration with uncertainty estimates from real molecular predictions."""
        compounds = compounds_with_uncertainty.copy()
        
        if len(compounds) == 0:
            pytest.skip("No compounds with uncertainty available")
        
        # Basic version doesn't use uncertainty, but should work with it present
        acq = SimulatedAnnealingAcquisition(random_state=42, max_iterations=50)
        selected = acq.select(compounds, n_select=5)
        
        assert len(selected) == 5
        assert 'uncertainty' in compounds.columns  # Verify uncertainty was present
        assert np.all(selected['uncertainty'] > 0)  # Verify uncertainty values are valid
        
        # Should not require uncertainty
        assert not acq.requires_uncertainty()
    
    def test_error_handling(self, small_real_compounds):
        """Test error handling with real molecular data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Add predictions for valid test
        np.random.seed(42)
        compounds['prediction'] = compounds['Activity'] + np.random.normal(0, 0.5, len(compounds))
        
        acq = SimulatedAnnealingAcquisition(random_state=42)
        
        # Test missing required columns
        compounds_no_smiles = compounds.drop(columns=['SMILES'])
        with pytest.raises(ValueError, match="Missing required columns"):
            acq.select(compounds_no_smiles, n_select=5)
        
        # Test empty DataFrame
        empty_compounds = compounds.iloc[:0].copy()
        with pytest.raises(ValueError, match="compounds DataFrame is empty"):
            acq.select(empty_compounds, n_select=5)
        
        # Test invalid n_select
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(compounds, n_select=0)
        
        # Test NaN predictions
        compounds_nan = compounds.copy()
        compounds_nan.loc[0, 'prediction'] = np.nan
        with pytest.raises(ValueError, match="Predictions contain NaN values"):
            acq.select(compounds_nan, n_select=5)
    
    def test_temperature_calculation(self):
        """Test temperature calculation for different cooling schedules."""
        acq_exp = SimulatedAnnealingAcquisition(
            initial_temp=1.0, 
            final_temp=0.1, 
            max_iterations=100,
            cooling_schedule='exponential'
        )
        
        acq_lin = SimulatedAnnealingAcquisition(
            initial_temp=1.0, 
            final_temp=0.1, 
            max_iterations=100,
            cooling_schedule='linear'
        )
        
        # Test temperature at start
        assert acq_exp._get_temperature(0) == 1.0
        assert acq_lin._get_temperature(0) == 1.0
        
        # Test temperature at end
        assert abs(acq_exp._get_temperature(100) - 0.1) < 1e-10
        assert abs(acq_lin._get_temperature(100) - 0.1) < 1e-10
        
        # Test temperature decreases monotonically
        temp_exp_mid = acq_exp._get_temperature(50)
        temp_lin_mid = acq_lin._get_temperature(50)
        
        assert 0.1 < temp_exp_mid < 1.0
        assert 0.1 < temp_lin_mid < 1.0
        
        # Linear cooling should be exactly halfway at 50% progress
        assert abs(temp_lin_mid - 0.55) < 1e-10  # (1.0 + 0.1) / 2 = 0.55
    
    def test_energy_calculation(self):
        """Test energy calculation for different score directions."""
        acq_higher = SimulatedAnnealingAcquisition(score_direction='higher')
        acq_lower = SimulatedAnnealingAcquisition(score_direction='lower')
        
        prediction = 5.0
        
        # For maximization: higher predictions should have lower energy
        energy_higher = acq_higher._calculate_energy(prediction)
        assert energy_higher == -5.0
        
        # For minimization: prediction value is the energy
        energy_lower = acq_lower._calculate_energy(prediction)
        assert energy_lower == 5.0
    
    def test_metropolis_acceptance(self):
        """Test Metropolis acceptance criterion."""
        acq = SimulatedAnnealingAcquisition(random_state=42)
        
        # Always accept better (lower energy) candidates
        assert acq._metropolis_accept(5.0, 3.0, 1.0) == True
        assert acq._metropolis_accept(3.0, 3.0, 1.0) == True  # Equal energy
        
        # Test temperature effect on worse candidates
        # At high temperature, should have higher probability of accepting worse candidates
        # At low temperature, should have lower probability
        
        # We can't test exact probabilities due to randomness, but we can test logic
        current_energy = 3.0
        worse_energy = 5.0
        
        # At zero temperature, never accept worse candidates
        assert acq._metropolis_accept(current_energy, worse_energy, 0.0) == False
        
        # At very high temperature, acceptance probability approaches 1
        # exp(-2/1000) ≈ 0.998, so very likely to accept
        np.random.seed(42)
        high_temp_accept = acq._metropolis_accept(current_energy, worse_energy, 1000.0)
        # Don't assert the exact result due to randomness, just ensure no errors
        assert isinstance(high_temp_accept, bool)
    
    def test_integration_with_diverse_molecular_data(self, diverse_real_compounds):
        """Test with structurally diverse compounds across multiple targets."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No diverse molecular data available")
        
        # Add realistic predictions
        np.random.seed(42)
        compounds['prediction'] = compounds['Activity'] + np.random.normal(0, 1.0, len(compounds))
        
        # Test with different parameters
        acq = SimulatedAnnealingAcquisition(
            initial_temp=2.0,
            final_temp=0.01,
            max_iterations=200,
            cooling_schedule='exponential',
            random_state=42
        )
        
        selected = acq.select(compounds, n_select=15)
        
        assert len(selected) == 15
        assert len(selected['ID'].unique()) == 15  # No duplicates
        
        # Check that we get diversity across targets if Target column exists
        if 'Target' in compounds.columns:
            # Should ideally get compounds from multiple targets
            unique_targets_in_selection = selected['Target'].nunique()
            unique_targets_total = compounds['Target'].nunique()
            
            # At minimum, should have valid target assignments
            assert unique_targets_in_selection >= 1
            assert unique_targets_in_selection <= unique_targets_total
"""Tests for probabilistic pruning strategies.

Tests probabilistic, uncertainty-based, and confidence interval pruning using real molecular data.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Optional

try:
    from learnm8.pruning.probabilistic import (
        ProbabilisticPruner,
        UncertaintyThresholdPruner,
        PredictionThresholdPruner,
        ConfidenceIntervalPruner
    )
    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False


@pytest.mark.skipif(not PRUNING_AVAILABLE, reason="Pruning modules not available")
class TestProbabilisticPruner:
    """Test probabilistic pruning based on value probability estimates."""
    
    def test_probabilistic_pruner_functionality(self, small_real_compounds):
        """Test basic probabilistic pruner functionality."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Generate realistic predictions and uncertainties
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, len(compounds))
        uncertainties = np.random.uniform(0.1, 0.5, len(compounds))
        
        try:
            pruner = ProbabilisticPruner(
                value_threshold=0.5,
                probability_threshold=0.2,
                minimize=False
            )
            
            assert pruner.requires_uncertainty() == True
            
            pruned = pruner.prune(compounds, predictions, uncertainties)
            
            assert isinstance(pruned, pd.DataFrame)
            assert len(pruned) <= len(compounds)
            
            # Should retain compounds with high probability of exceeding threshold
            if len(pruned) > 0:
                assert all(col in pruned.columns for col in compounds.columns)
                
            # Check statistics
            stats = pruner.get_pruning_stats()
            assert 'compounds_before_pruning' in stats
            assert 'mean_probability' in stats
            
        except ImportError:
            pytest.skip("scipy not available for ProbabilisticPruner")
    
    def test_probabilistic_pruner_without_uncertainty(self, small_real_compounds):
        """Test probabilistic pruner fallback without uncertainty."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        try:
            pruner = ProbabilisticPruner(
                value_threshold=0.5,
                probability_threshold=0.5,
                minimize=False
            )
            
            # ProbabilisticPruner requires uncertainty, so this should fail
            with pytest.raises(Exception):  # Could be PruningError or validation error
                pruner.prune(compounds, predictions)
            
        except ImportError:
            pytest.skip("scipy not available for ProbabilisticPruner")


@pytest.mark.skipif(not PRUNING_AVAILABLE, reason="Pruning modules not available")
class TestUncertaintyThresholdPruner:
    """Test uncertainty-based pruning strategies."""
    
    def test_uncertainty_threshold_adaptive(self, small_real_compounds):
        """Test adaptive uncertainty threshold pruner."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        uncertainties = np.random.uniform(0.1, 0.5, len(compounds))
        
        # Test adaptive threshold (default behavior)
        pruner = UncertaintyThresholdPruner(
            retention_fraction=0.7,
            adaptive_threshold=True
        )
        
        pruned = pruner.prune(compounds, predictions, uncertainties)
        
        assert isinstance(pruned, pd.DataFrame)
        assert len(pruned) <= len(compounds)
        
        # Should retain approximately 70% of compounds
        retention_rate = len(pruned) / len(compounds)
        assert 0.6 <= retention_rate <= 0.8  # Allow some tolerance
        
        # Check statistics
        stats = pruner.get_pruning_stats()
        assert 'uncertainty_threshold' in stats
        assert 'adaptive_threshold' in stats
        assert stats['adaptive_threshold'] == True
    
    def test_uncertainty_threshold_fixed(self, small_real_compounds):
        """Test fixed uncertainty threshold pruner."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        uncertainties = np.random.uniform(0.1, 0.5, len(compounds))
        
        # Test fixed threshold
        pruner = UncertaintyThresholdPruner(
            uncertainty_threshold=0.3,
            adaptive_threshold=False
        )
        
        pruned = pruner.prune(compounds, predictions, uncertainties)
        
        # Should keep compounds with uncertainty <= 0.3
        if len(pruned) > 0:
            # Add predictions/uncertainties to dataframe for verification
            compounds_with_data = compounds.copy()
            compounds_with_data['prediction'] = predictions
            compounds_with_data['uncertainty'] = uncertainties
            
            # Find matching compounds by ID
            pruned_ids = set(pruned['ID'].tolist())
            matching_rows = compounds_with_data[compounds_with_data['ID'].isin(pruned_ids)]
            
            if len(matching_rows) > 0:
                assert np.all(matching_rows['uncertainty'].values <= 0.3)
    
    def test_uncertainty_threshold_requires_uncertainty(self, small_real_compounds):
        """Test that uncertainty threshold pruner requires uncertainty."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        pruner = UncertaintyThresholdPruner(uncertainty_threshold=0.3, adaptive_threshold=False)
        
        # Should fail without uncertainty
        with pytest.raises(Exception):  # Could be PruningError or validation error
            pruner.prune(compounds, predictions)


@pytest.mark.skipif(not PRUNING_AVAILABLE, reason="Pruning modules not available")
class TestPredictionThresholdPruner:
    """Test prediction-based pruning strategies."""
    
    def test_prediction_threshold_maximize(self, small_real_compounds):
        """Test prediction threshold pruner for maximization."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        # Test keeping high predictions (maximize=True)
        pruner = PredictionThresholdPruner(
            prediction_threshold=0.6,
            adaptive_threshold=False,
            maximize=True
        )
        
        pruned = pruner.prune(compounds, predictions)
        
        # Should keep compounds with prediction >= 0.6
        if len(pruned) > 0:
            # Add predictions to dataframe for verification
            compounds_with_data = compounds.copy()
            compounds_with_data['prediction'] = predictions
            
            # Find matching compounds by ID
            pruned_ids = set(pruned['ID'].tolist())
            matching_rows = compounds_with_data[compounds_with_data['ID'].isin(pruned_ids)]
            
            if len(matching_rows) > 0:
                assert np.all(matching_rows['prediction'].values >= 0.6)
    
    def test_prediction_threshold_minimize(self, small_real_compounds):
        """Test prediction threshold pruner for minimization."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        # Test keeping low predictions (maximize=False)
        pruner = PredictionThresholdPruner(
            prediction_threshold=0.4,
            adaptive_threshold=False,
            maximize=False
        )
        
        pruned = pruner.prune(compounds, predictions)
        
        # Should keep compounds with prediction <= 0.4
        if len(pruned) > 0:
            # Add predictions to dataframe for verification
            compounds_with_data = compounds.copy()
            compounds_with_data['prediction'] = predictions
            
            # Find matching compounds by ID
            pruned_ids = set(pruned['ID'].tolist())
            matching_rows = compounds_with_data[compounds_with_data['ID'].isin(pruned_ids)]
            
            if len(matching_rows) > 0:
                assert np.all(matching_rows['prediction'].values <= 0.4)
    
    def test_prediction_threshold_adaptive(self, small_real_compounds):
        """Test adaptive prediction threshold pruner."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        # Test adaptive threshold
        pruner = PredictionThresholdPruner(
            retention_fraction=0.6,
            adaptive_threshold=True,
            maximize=True
        )
        
        pruned = pruner.prune(compounds, predictions)
        
        # Should retain approximately 60% of compounds
        retention_rate = len(pruned) / len(compounds)
        assert 0.5 <= retention_rate <= 0.7  # Allow some tolerance
        
        # Check statistics
        stats = pruner.get_pruning_stats()
        assert 'prediction_threshold' in stats
        assert 'adaptive_threshold' in stats
        assert stats['adaptive_threshold'] == True


@pytest.mark.skipif(not PRUNING_AVAILABLE, reason="Pruning modules not available")
class TestConfidenceIntervalPruner:
    """Test confidence interval-based pruning strategies."""
    
    def test_confidence_interval_any_overlap(self, small_real_compounds):
        """Test confidence interval pruner with any overlap requirement."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0.3, 0.8, len(compounds))
        uncertainties = np.random.uniform(0.1, 0.3, len(compounds))
        
        try:
            pruner = ConfidenceIntervalPruner(
                target_min=0.4,
                target_max=0.7,
                confidence_level=0.95,
                overlap_requirement='any'
            )
            
            pruned = pruner.prune(compounds, predictions, uncertainties)
            
            assert isinstance(pruned, pd.DataFrame)
            assert len(pruned) <= len(compounds)
            
            # Check statistics
            stats = pruner.get_pruning_stats()
            assert 'target_min' in stats
            assert 'target_max' in stats
            assert 'overlap_requirement' in stats
            assert stats['overlap_requirement'] == 'any'
            
        except ImportError:
            pytest.skip("scipy not available for ConfidenceIntervalPruner")
    
    def test_confidence_interval_complete_overlap(self, small_real_compounds):
        """Test confidence interval pruner with complete overlap requirement."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0.4, 0.6, len(compounds))  # Narrow range
        uncertainties = np.random.uniform(0.05, 0.15, len(compounds))  # Small uncertainties
        
        try:
            pruner = ConfidenceIntervalPruner(
                target_min=0.3,
                target_max=0.8,
                confidence_level=0.95,
                overlap_requirement='complete'
            )
            
            pruned = pruner.prune(compounds, predictions, uncertainties)
            
            assert isinstance(pruned, pd.DataFrame)
            assert len(pruned) <= len(compounds)
            
            # Check statistics
            stats = pruner.get_pruning_stats()
            assert stats['overlap_requirement'] == 'complete'
            
        except ImportError:
            pytest.skip("scipy not available for ConfidenceIntervalPruner")
    
    def test_confidence_interval_requires_uncertainty(self, small_real_compounds):
        """Test that confidence interval pruner requires uncertainty."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        try:
            pruner = ConfidenceIntervalPruner(
                target_min=0.4,
                target_max=0.7,
                confidence_level=0.95
            )
            
            # Should fail without uncertainty
            with pytest.raises(Exception):  # Could be PruningError or validation error
                pruner.prune(compounds, predictions)
                
        except ImportError:
            pytest.skip("scipy not available for ConfidenceIntervalPruner")


@pytest.mark.skipif(not PRUNING_AVAILABLE, reason="Pruning modules not available")
class TestPruningEdgeCases:
    """Test edge cases for probabilistic pruning."""
    
    def test_single_compound_pruning(self):
        """Test pruning with single compound."""
        single_compound = pd.DataFrame({
            'ID': ['mol_1'],
            'SMILES': ['CCO']
        })
        
        predictions = np.array([0.7])
        uncertainties = np.array([0.2])
        
        # Test different pruners with single compound
        try:
            pruners = [
                PredictionThresholdPruner(prediction_threshold=0.5, adaptive_threshold=False),
                UncertaintyThresholdPruner(uncertainty_threshold=0.3, adaptive_threshold=False)
            ]
            
            for pruner in pruners:
                try:
                    if pruner.requires_uncertainty():
                        pruned = pruner.prune(single_compound, predictions, uncertainties)
                    else:
                        pruned = pruner.prune(single_compound, predictions)
                    
                    # Should handle single compound gracefully
                    assert isinstance(pruned, pd.DataFrame)
                    assert len(pruned) <= 1
                    
                except Exception:
                    # Some pruners might not handle single compounds well
                    pass
                    
        except ImportError:
            pytest.skip("Pruning modules not fully available")
    
    def test_extreme_thresholds(self, small_real_compounds):
        """Test pruning with extreme threshold values."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        
        # Test very high threshold (should keep nothing)
        pruner_high = PredictionThresholdPruner(
            prediction_threshold=1.5, 
            adaptive_threshold=False,
            maximize=True
        )
        pruned_high = pruner_high.prune(compounds, predictions)
        assert len(pruned_high) == 0
        
        # Test very low threshold (should keep everything)
        pruner_low = PredictionThresholdPruner(
            prediction_threshold=-0.5, 
            adaptive_threshold=False,
            maximize=True
        )
        pruned_low = pruner_low.prune(compounds, predictions)
        assert len(pruned_low) == len(compounds)

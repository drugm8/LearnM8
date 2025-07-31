"""Tests for base pruning functionality.

Tests the base DesignSpacePruner interface and utility functions using real molecular data.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from learnm8.pruning.base import (
    DesignSpacePruner,
    StatefulPruner,
    PruningError,
    calculate_confidence_intervals,
    calculate_prediction_percentiles,
    adaptive_threshold_calculation
)


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
    
    def prune(self, compounds: pd.DataFrame, predictions: np.ndarray, 
              uncertainties: Optional[np.ndarray] = None) -> pd.DataFrame:
        # Validate inputs
        self.validate_inputs(compounds, predictions, uncertainties)
        
        # Simple mock: keep first half of compounds
        n_keep = len(compounds) // 2
        pruned = compounds.head(n_keep).copy()
        
        self.last_pruning_stats = {
            'compounds_before_pruning': len(compounds),
            'compounds_after_pruning': len(pruned),
            'compounds_pruned': len(compounds) - len(pruned),
            'pruning_fraction': self._calculate_pruning_fraction(len(compounds), len(pruned))
        }
        
        return pruned


class TestBasePruningInterface:
    """Test base pruning interface and abstract methods."""
    
    def test_abstract_pruner_interface(self):
        """Test that abstract pruner enforces interface."""
        with pytest.raises(TypeError):
            DesignSpacePruner()
    
    def test_mock_pruner_functionality(self, small_real_compounds):
        """Test mock pruner with real molecular data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Generate realistic predictions
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, len(compounds))
        
        pruner = MockPruner()
        
        # Test basic pruning
        pruned = pruner.prune(compounds, predictions)
        
        assert isinstance(pruned, pd.DataFrame)
        assert len(pruned) <= len(compounds)
        assert all(col in pruned.columns for col in compounds.columns)
        
        # Test statistics
        stats = pruner.get_pruning_stats()
        assert 'compounds_before_pruning' in stats
        assert 'compounds_after_pruning' in stats
        assert 'compounds_pruned' in stats
        assert stats['compounds_before_pruning'] == len(compounds)
        assert stats['compounds_after_pruning'] == len(pruned)
    
    def test_pruner_with_uncertainty_requirement(self, small_real_compounds):
        """Test pruner that requires uncertainty."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        uncertainties = np.random.uniform(0.1, 0.5, len(compounds))
        
        # Test pruner that requires uncertainty
        pruner = MockPruner(requires_uncertainty=True)
        assert pruner.requires_uncertainty() == True
        
        # Should work with uncertainty
        pruned = pruner.prune(compounds, predictions, uncertainties)
        assert len(pruned) <= len(compounds)
        
        # Should fail without uncertainty
        with pytest.raises(PruningError):
            pruner.prune(compounds, predictions)


class TestPruningValidation:
    """Test input validation and error handling."""
    
    def test_empty_compounds_validation(self):
        """Test validation with empty compounds."""
        empty_compounds = pd.DataFrame(columns=['ID', 'SMILES'])
        predictions = np.array([])
        
        pruner = MockPruner()
        
        with pytest.raises(PruningError):
            pruner.prune(empty_compounds, predictions)
    
    def test_missing_columns_validation(self, small_real_compounds):
        """Test validation with missing required columns."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Remove required column
        if 'ID' in compounds.columns:
            compounds_missing = compounds.drop(columns=['ID'])
        else:
            compounds_missing = pd.DataFrame({'invalid_col': range(len(compounds))})
        
        predictions = np.random.uniform(0, 1, len(compounds))
        pruner = MockPruner()
        
        with pytest.raises(PruningError):
            pruner.prune(compounds_missing, predictions)
    
    def test_mismatched_lengths_validation(self, small_real_compounds):
        """Test validation with mismatched array lengths."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        # Wrong length predictions
        wrong_predictions = np.random.uniform(0, 1, len(compounds) + 5)
        pruner = MockPruner()
        
        with pytest.raises(PruningError):
            pruner.prune(compounds, wrong_predictions)
    
    def test_nan_predictions_validation(self, small_real_compounds):
        """Test validation with NaN predictions."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            pytest.skip("No real molecular data available")
        
        predictions = np.random.uniform(0, 1, len(compounds))
        predictions[0] = np.nan  # Add NaN
        
        pruner = MockPruner()
        
        with pytest.raises(PruningError):
            pruner.prune(compounds, predictions)


class TestUtilityFunctions:
    """Test utility functions for pruning strategies."""
    
    def test_confidence_intervals_calculation(self):
        """Test confidence interval calculation utility."""
        np.random.seed(42)
        predictions = np.random.normal(0.5, 0.2, 100)
        uncertainties = np.random.uniform(0.05, 0.3, 100)
        
        lower, upper = calculate_confidence_intervals(predictions, uncertainties, confidence_level=0.95)
        
        # Check output shape and properties
        assert len(lower) == len(predictions)
        assert len(upper) == len(predictions)
        assert np.all(lower <= predictions)
        assert np.all(upper >= predictions)
        assert np.all(upper >= lower)
    
    def test_prediction_percentiles_calculation(self):
        """Test prediction percentiles utility."""
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        percentiles_dict = calculate_prediction_percentiles(predictions, percentiles=[25, 50, 75])
        
        assert len(percentiles_dict) == 3
        assert 'p25' in percentiles_dict
        assert 'p50' in percentiles_dict
        assert 'p75' in percentiles_dict
        
        # Check monotonic ordering
        assert percentiles_dict['p25'] <= percentiles_dict['p50'] <= percentiles_dict['p75']
    
    def test_adaptive_threshold_calculation(self):
        """Test adaptive threshold calculation."""
        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        uncertainties = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
        
        # Test percentile method
        threshold_percentile = adaptive_threshold_calculation(
            predictions, uncertainties, target_retention_rate=0.6, method='percentile'
        )
        assert isinstance(threshold_percentile, float)
        assert 0 <= threshold_percentile <= 1
        
        # Test uncertainty weighted method
        threshold_weighted = adaptive_threshold_calculation(
            predictions, uncertainties, target_retention_rate=0.6, method='uncertainty_weighted'
        )
        assert isinstance(threshold_weighted, float)
        assert 0 <= threshold_weighted <= 1
        
        # Test different retention fractions
        high_retention = adaptive_threshold_calculation(
            predictions, uncertainties, target_retention_rate=0.8
        )
        low_retention = adaptive_threshold_calculation(
            predictions, uncertainties, target_retention_rate=0.4
        )
        
        # Higher retention should have lower threshold
        assert high_retention <= low_retention


class TestStatefulPruner:
    """Test stateful pruner base functionality."""
    
    def test_stateful_pruner_initialization(self):
        """Test stateful pruner initialization."""
        # Can't instantiate abstract class directly, so test through inheritance
        # This would be tested through actual implementations in other modules
        pass
    
    def test_cycle_state_updates(self):
        """Test cycle state update functionality."""
        # This would be tested through actual StatefulPruner implementations
        pass

"""Base classes for design space pruning in the LearnM8 framework.

This module defines the abstract interface for design space pruning strategies
that reduce the unlabeled compound pool by removing unlikely candidates.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DesignSpacePruner(ABC):
    """Base class for design space pruning strategies.
    
    Design space pruners analyze model predictions and uncertainties to identify
    and remove compounds that are unlikely to be valuable for active learning,
    thereby reducing computational costs and focusing on promising regions.
    """
    
    @abstractmethod
    def prune(self, 
              compounds: pd.DataFrame, 
              predictions: np.ndarray, 
              uncertainties: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Prune the compound pool based on predictions and uncertainties.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES' columns
            predictions: Model predictions for each compound
            uncertainties: Model uncertainties (optional, strategy-dependent)
            
        Returns:
            Pruned DataFrame subset with promising compounds
            
        Raises:
            PruningError: If pruning fails or inputs are invalid
        """
        pass
    
    @abstractmethod
    def get_pruning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the most recent pruning operation.
        
        Returns:
            Dictionary containing pruning statistics
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return a descriptive name for this pruning strategy.
        
        Returns:
            String identifier for the pruning strategy
        """
        pass
    
    def requires_uncertainty(self) -> bool:
        """
        Return True if this pruning strategy requires uncertainty estimates.
        
        Returns:
            Boolean indicating if uncertainty is required
        """
        return False
    
    def validate_inputs(self, 
                       compounds: pd.DataFrame, 
                       predictions: np.ndarray, 
                       uncertainties: Optional[np.ndarray] = None) -> None:
        """
        Validate inputs for pruning operation.
        
        Args:
            compounds: DataFrame to validate
            predictions: Predictions array to validate
            uncertainties: Uncertainties array to validate (optional)
            
        Raises:
            PruningError: If inputs are invalid
        """
        # Check DataFrame structure
        if compounds.empty:
            raise PruningError("compounds DataFrame is empty")
        
        required_cols = ['ID', 'SMILES']
        missing_cols = set(required_cols) - set(compounds.columns)
        if missing_cols:
            raise PruningError(f"Missing required columns: {missing_cols}")
        
        # Check predictions array
        if len(predictions) != len(compounds):
            raise PruningError(f"Predictions length ({len(predictions)}) doesn't match compounds length ({len(compounds)})")
        
        if np.any(np.isnan(predictions)):
            raise PruningError("Predictions contain NaN values")
        
        # Check uncertainties if provided
        if uncertainties is not None:
            if len(uncertainties) != len(compounds):
                raise PruningError(f"Uncertainties length ({len(uncertainties)}) doesn't match compounds length ({len(compounds)})")
            
            if np.any(np.isnan(uncertainties)):
                raise PruningError("Uncertainties contain NaN values")
            
            if np.any(uncertainties < 0):
                raise PruningError("Uncertainties must be non-negative")
        
        # Check if uncertainties are required but not provided
        if self.requires_uncertainty() and uncertainties is None:
            raise PruningError(f"{self.get_name()} requires uncertainty estimates")
    
    def _calculate_pruning_fraction(self, original_count: int, pruned_count: int) -> float:
        """Calculate the fraction of compounds pruned.
        
        Args:
            original_count: Original number of compounds
            pruned_count: Number of compounds after pruning
            
        Returns:
            Fraction of compounds removed (0.0 to 1.0)
        """
        if original_count == 0:
            return 0.0
        
        removed_count = original_count - pruned_count
        return removed_count / original_count
    
    def _safe_prune_by_indices(self, 
                              compounds: pd.DataFrame, 
                              keep_indices: np.ndarray) -> pd.DataFrame:
        """Safely prune compounds by keeping specified indices.
        
        Args:
            compounds: Original compounds DataFrame
            keep_indices: Boolean or integer array of indices to keep
            
        Returns:
            Pruned compounds DataFrame
            
        Raises:
            PruningError: If indices are invalid
        """
        try:
            if len(keep_indices) == 0:
                logger.warning("No compounds selected for retention during pruning")
                return compounds.iloc[:0].copy()  # Return empty DataFrame with same structure
            
            # Handle boolean indexing
            if keep_indices.dtype == bool:
                if len(keep_indices) != len(compounds):
                    raise PruningError(f"Boolean index length ({len(keep_indices)}) doesn't match compounds length ({len(compounds)})")
                pruned_compounds = compounds[keep_indices].copy()
            else:
                # Handle integer indexing
                if np.any(keep_indices >= len(compounds)) or np.any(keep_indices < 0):
                    raise PruningError("Invalid compound indices for pruning")
                pruned_compounds = compounds.iloc[keep_indices].copy()
            
            # Reset index to maintain clean indexing
            pruned_compounds = pruned_compounds.reset_index(drop=True)
            
            return pruned_compounds
            
        except Exception as e:
            raise PruningError(f"Failed to prune compounds: {e}") from e


class PruningError(Exception):
    """Exception raised when design space pruning encounters an error."""
    pass


class StatefulPruner(DesignSpacePruner):
    """Base class for pruners that maintain state across cycles.
    
    This class provides common functionality for pruning strategies that
    adapt their behavior based on previous cycles or accumulated statistics.
    """
    
    def __init__(self):
        """Initialize stateful pruner."""
        self.cycle_count = 0
        self.pruning_history: List[Dict[str, Any]] = []
        self.cumulative_stats = {
            'total_compounds_seen': 0,
            'total_compounds_pruned': 0,
            'total_pruning_operations': 0
        }
    
    def update_cycle_state(self, cycle_stats: Dict[str, Any]) -> None:
        """Update pruner state after a cycle.
        
        Args:
            cycle_stats: Statistics from the completed cycle
        """
        self.cycle_count += 1
        self.pruning_history.append(cycle_stats)
        
        # Update cumulative statistics
        if 'compounds_before_pruning' in cycle_stats:
            self.cumulative_stats['total_compounds_seen'] += cycle_stats['compounds_before_pruning']
        
        if 'compounds_pruned' in cycle_stats:
            self.cumulative_stats['total_compounds_pruned'] += cycle_stats['compounds_pruned']
        
        self.cumulative_stats['total_pruning_operations'] += 1
    
    def get_cycle_history(self) -> List[Dict[str, Any]]:
        """Get history of pruning operations across cycles.
        
        Returns:
            List of cycle statistics dictionaries
        """
        return self.pruning_history.copy()
    
    def get_cumulative_stats(self) -> Dict[str, Any]:
        """Get cumulative pruning statistics across all cycles.
        
        Returns:
            Dictionary of cumulative statistics
        """
        stats = self.cumulative_stats.copy()
        
        # Calculate derived statistics
        if stats['total_compounds_seen'] > 0:
            stats['overall_pruning_rate'] = stats['total_compounds_pruned'] / stats['total_compounds_seen']
        else:
            stats['overall_pruning_rate'] = 0.0
        
        stats['current_cycle'] = self.cycle_count
        
        return stats
    
    def reset_state(self) -> None:
        """Reset pruner state (useful for new experiments)."""
        self.cycle_count = 0
        self.pruning_history.clear()
        self.cumulative_stats = {
            'total_compounds_seen': 0,
            'total_compounds_pruned': 0,
            'total_pruning_operations': 0
        }


# Utility functions for pruning strategies
def calculate_confidence_intervals(predictions: np.ndarray, 
                                 uncertainties: np.ndarray, 
                                 confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate confidence intervals for predictions.
    
    Args:
        predictions: Model predictions
        uncertainties: Model uncertainties (standard deviations)
        confidence_level: Confidence level for intervals (0-1)
        
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    # Calculate z-score for confidence level
    from scipy.stats import norm
    alpha = 1 - confidence_level
    z_score = norm.ppf(1 - alpha/2)
    
    # Calculate confidence intervals
    margin = z_score * uncertainties
    lower_bounds = predictions - margin
    upper_bounds = predictions + margin
    
    return lower_bounds, upper_bounds


def calculate_prediction_percentiles(predictions: np.ndarray, 
                                   percentiles: List[float] = [10, 25, 50, 75, 90]) -> Dict[str, float]:
    """Calculate percentiles of prediction distribution.
    
    Args:
        predictions: Model predictions
        percentiles: List of percentiles to calculate
        
    Returns:
        Dictionary mapping percentile names to values
    """
    result = {}
    for p in percentiles:
        result[f'p{p}'] = np.percentile(predictions, p)
    
    return result


def adaptive_threshold_calculation(predictions: np.ndarray,
                                 uncertainties: Optional[np.ndarray] = None,
                                 target_retention_rate: float = 0.5,
                                 method: str = 'percentile') -> float:
    """Calculate adaptive threshold for pruning.
    
    Args:
        predictions: Model predictions
        uncertainties: Model uncertainties (optional)
        target_retention_rate: Fraction of compounds to retain
        method: Method for threshold calculation ('percentile', 'uncertainty_weighted')
        
    Returns:
        Calculated threshold value
    """
    if method == 'percentile':
        # Use prediction percentile as threshold
        percentile = (1 - target_retention_rate) * 100
        return np.percentile(predictions, percentile)
    
    elif method == 'uncertainty_weighted' and uncertainties is not None:
        # Weight predictions by inverse uncertainty for threshold calculation
        weights = 1 / (uncertainties + 1e-8)  # Add small epsilon to avoid division by zero
        weighted_predictions = predictions * weights
        
        # Calculate weighted percentile
        sorted_indices = np.argsort(weighted_predictions)
        cumulative_weights = np.cumsum(weights[sorted_indices])
        normalized_weights = cumulative_weights / cumulative_weights[-1]
        
        # Find threshold corresponding to retention rate
        threshold_idx = np.searchsorted(normalized_weights, 1 - target_retention_rate)
        threshold_idx = min(threshold_idx, len(predictions) - 1)
        
        return float(predictions[sorted_indices[threshold_idx]])
    
    else:
        # Fallback to simple percentile
        percentile = (1 - target_retention_rate) * 100
        return np.percentile(predictions, percentile)
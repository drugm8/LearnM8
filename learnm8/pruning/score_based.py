"""Simple score-based pruning for the LearnM8 framework.

This module provides a single, straightforward pruning strategy that removes
compounds based on their predicted scores, making it easy to focus the search
on the most promising regions of chemical space.
"""

import logging
from typing import Any
import polars as pl
import numpy as np

from .base import DesignSpacePruner, PruningError

logger = logging.getLogger(__name__)


class ScoreBasedPruner(DesignSpacePruner):
    """Simple score-based pruning strategy.
    
    This pruner removes a specified fraction of compounds with the worst
    predicted scores, based on the optimization direction. For 'higher' 
    direction, it removes compounds with the lowest predicted scores.
    For 'lower' direction, it removes compounds with the highest scores.
    
    Args:
        pruning_fraction: Fraction of compounds to remove (0.0 to 0.9)
        score_direction: Optimization direction ('higher' or 'lower')
    """
    
    def __init__(self, pruning_fraction: float = 0.1, score_direction: str = 'higher'):
        """Initialize the score-based pruner.
        
        Args:
            pruning_fraction: Fraction of compounds to remove (0.0-0.9)
            score_direction: Score optimization direction ('higher' or 'lower')
            
        Raises:
            PruningError: If parameters are invalid
        """
        if not 0.0 <= pruning_fraction <= 0.9:
            raise PruningError(
                f"pruning_fraction must be between 0.0 and 0.9, got {pruning_fraction}. "
                f"Values above 0.9 risk removing too many candidates. "
                f"Use 0.0 to disable pruning or set pruning_strategy=None."
            )

        if score_direction not in ['higher', 'lower']:
            raise PruningError(
                f"score_direction must be 'higher' or 'lower', got '{score_direction}'. "
                f"Use 'higher' to maximize the target property or 'lower' to minimize it."
            )
        
        self.pruning_fraction = pruning_fraction
        self.score_direction = score_direction
        self._last_stats: dict[str, Any] = {}
    
    def prune(self,
              compounds: pl.DataFrame,
              predictions: np.ndarray,
              uncertainties: np.ndarray | None = None) -> pl.DataFrame:
        """Prune compounds based on predicted scores.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES' columns
            predictions: Model predictions for each compound
            uncertainties: Not used for score-based pruning
            
        Returns:
            DataFrame with worst-scoring compounds removed
            
        Raises:
            PruningError: If pruning fails or inputs are invalid
        """
        self.validate_inputs(compounds, predictions, uncertainties)
        
        n_compounds = len(compounds)
        
        # Skip pruning if fraction is 0 or if we have too few compounds
        if self.pruning_fraction == 0.0 or n_compounds <= 1:
            self._last_stats = {
                'compounds_before_pruning': n_compounds,
                'compounds_after_pruning': n_compounds,
                'compounds_pruned': 0,
                'pruning_fraction_actual': 0.0,
                'score_direction': self.score_direction,
                'pruning_fraction_requested': self.pruning_fraction
            }
            return compounds.clone()
        
        # Calculate how many compounds to remove
        n_to_remove = int(n_compounds * self.pruning_fraction)
        n_to_keep = n_compounds - n_to_remove
        
        # Ensure we keep at least one compound
        if n_to_keep <= 0:
            n_to_keep = 1
            n_to_remove = n_compounds - 1
        
        # Sort compounds by score based on optimization direction
        if self.score_direction == 'higher':
            # Keep compounds with highest scores (remove lowest)
            keep_indices = np.argsort(predictions)[-n_to_keep:]
        else:
            # Keep compounds with lowest scores (remove highest)
            keep_indices = np.argsort(predictions)[:n_to_keep]
        
        # Create pruned dataset
        pruned_compounds = self._safe_prune_by_indices(compounds, keep_indices)
        
        # Calculate actual pruning statistics
        actual_pruning_fraction = self._calculate_pruning_fraction(n_compounds, len(pruned_compounds))
        
        # Store statistics
        self._last_stats = {
            'compounds_before_pruning': n_compounds,
            'compounds_after_pruning': len(pruned_compounds),
            'compounds_pruned': n_compounds - len(pruned_compounds),
            'pruning_fraction_actual': actual_pruning_fraction,
            'pruning_fraction_requested': self.pruning_fraction,
            'score_direction': self.score_direction,
            'score_threshold': float(predictions[keep_indices].min() if self.score_direction == 'higher' 
                                   else predictions[keep_indices].max()),
            'score_range_kept': {
                'min': float(predictions[keep_indices].min()),
                'max': float(predictions[keep_indices].max()),
                'mean': float(predictions[keep_indices].mean())
            }
        }
        
        logger.info(f"ScoreBasedPruner: removed {self._last_stats['compounds_pruned']} compounds "
                   f"({actual_pruning_fraction:.1%}) with worst scores for '{self.score_direction}' optimization")
        
        return pruned_compounds
    
    def get_pruning_stats(self) -> dict[str, Any]:
        """Get statistics from the most recent pruning operation.
        
        Returns:
            Dictionary containing pruning statistics
        """
        return self._last_stats.copy()
    
    def get_name(self) -> str:
        """Return the name of this pruning strategy.
        
        Returns:
            Strategy name string
        """
        return f"ScoreBasedPruner(fraction={self.pruning_fraction}, direction={self.score_direction})"
    
    def requires_uncertainty(self) -> bool:
        """Return False as this strategy only needs predictions.
        
        Returns:
            False - uncertainty is not required
        """
        return False
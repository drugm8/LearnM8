"""Simplified diversity-based acquisition function for the LearnM8 framework.

This module provides a simple diversity-aware acquisition strategy without complex
diversity weighting parameters.
"""

import logging
import numpy as np
import pandas as pd

from .base import AcquisitionFunction

logger = logging.getLogger(__name__)


class DiverseAcquisition(AcquisitionFunction):
    """Simple diversity-aware acquisition function using random selection.
    
    This simplified strategy provides basic diversity without complex parameters
    or utility-diversity weighting. It uses the base acquisition function to
    rank compounds, then applies simple random selection among top candidates.
    """
    
    def __init__(self, base_acquisition: AcquisitionFunction, top_fraction: float = 0.5):
        """Initialize simple diverse acquisition function.
        
        Args:
            base_acquisition: Base acquisition function for initial ranking
            top_fraction: Fraction of top-ranked compounds to randomly sample from
        """
        self.base_acquisition = base_acquisition
        self.top_fraction = max(0.1, min(1.0, top_fraction))  # Clamp between 0.1 and 1.0
    
    def requires_uncertainty(self) -> bool:
        """Check if this acquisition function requires uncertainty estimates."""
        return self.base_acquisition.requires_uncertainty()
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select compounds using simplified diversity approach.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES' columns and predictions
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds
        """
        # Validate input
        self.validate_input(compounds, n_select)
        
        if n_select >= len(compounds):
            return compounds.copy()
        
        # Get utility scores (use prediction column or random)
        if 'prediction' in compounds.columns:
            scores = compounds['prediction'].values
        else:
            logger.warning("No prediction column found, using random scores")
            scores = np.random.random(len(compounds))
        
        # Select top fraction based on utility scores
        n_top = max(n_select, int(len(compounds) * self.top_fraction))
        top_indices = np.argsort(-scores)[:n_top]  # Sort descending
        
        # Randomly sample from top candidates
        if n_select >= len(top_indices):
            selected_indices = top_indices
        else:
            selected_indices = np.random.choice(top_indices, n_select, replace=False)
        
        logger.info(f"Selected {len(selected_indices)} compounds from top {len(top_indices)} candidates")
        
        return compounds.iloc[selected_indices].copy()
    
    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        base_name = getattr(self.base_acquisition, 'get_name', lambda: 'Unknown')()
        return f"Diverse({base_name})"
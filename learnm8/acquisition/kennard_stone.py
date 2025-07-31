"""Kennard-Stone acquisition for optimal molecular diversity sampling.

This module implements the Kennard-Stone algorithm for selecting diverse compounds
based on maximum distance (minimum similarity) criterion. The implementation is
extracted and optimized from the astartes library for integration with LearnM8.
"""

import logging
from typing import Optional, TYPE_CHECKING
import pandas as pd
import numpy as np

from learnm8.acquisition.base import AcquisitionFunction
from learnm8.acquisition.astartes_utils import (
    compute_tanimoto_distance_matrix,
    get_molecular_features,
    validate_acquisition_input,
    fast_kennard_stone
)

if TYPE_CHECKING:
    from learnm8.core.data_manager import DataManager

logger = logging.getLogger(__name__)

# O(n²) complexity protection constant
MAX_COMPOUNDS = 10000


class KennardStoneAcquisition(AcquisitionFunction):
    """Kennard-Stone acquisition for systematic molecular diversity.
    
    The Kennard-Stone algorithm selects compounds to maximize diversity by:
    1. Starting with the two most distant compounds
    2. Iteratively adding compounds with maximum minimum distance to selected set
    3. Using the maximin criterion for optimal space coverage
    
    This implementation uses cached molecular fingerprints from DataManager
    and vectorized numpy operations for high performance.
    
    Args:
        data_manager: DataManager instance for feature extraction and caching
        featurizer_type: Type of molecular features ('morgan', 'maccs', 'ecfp6')
        random_state: Random seed for reproducibility (though algorithm is deterministic)
    """
    
    def __init__(self, 
                 data_manager: Optional['DataManager'] = None,
                 featurizer_type: str = 'morgan',
                 random_state: Optional[int] = None,
                 **kwargs):
        """Initialize Kennard-Stone acquisition function.
        
        Args:
            data_manager: DataManager instance for feature extraction and caching
            featurizer_type: Type of molecular features ('morgan', 'maccs', 'ecfp6')
            random_state: Random seed for reproducibility (though algorithm is deterministic)
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(data_manager=data_manager, **kwargs)
        
        if data_manager is None:
            raise ValueError("KennardStoneAcquisition requires a DataManager for feature extraction")
        
        self.featurizer_type = featurizer_type
        self.random_state = random_state
        
        # Set numpy random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
    
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """Select diverse compounds using Kennard-Stone algorithm.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
            n_select: Number of compounds to select
            
        Returns:
            DataFrame with selected compounds ordered by selection priority
            
        Raises:
            ValueError: If inputs are invalid or feature extraction fails
            RuntimeError: If algorithm fails to converge
        """
        # Validate inputs using base class method
        self.validate_input(compounds, n_select)
        
        # Additional validation for our specific needs
        validate_acquisition_input(compounds, n_select)
        
        # Check dataset size for O(n²) complexity protection
        if len(compounds) > MAX_COMPOUNDS:
            raise ValueError(f"Too many compounds ({len(compounds)}) for {self.__class__.__name__}. "
                           f"Maximum allowed: {MAX_COMPOUNDS}. "
                           f"Consider using a different acquisition method or reducing dataset size.")
        
        logger.info(f"Starting Kennard-Stone selection of {n_select} compounds "
                   f"from {len(compounds)} candidates using {self.featurizer_type} features")
        
        # Handle edge cases
        if n_select >= len(compounds):
            logger.info("Selecting all available compounds")
            return compounds.copy()
        
        if n_select == 1:
            # For single selection, just pick the first compound
            return compounds.iloc[:1].copy()
        
        try:
            # Extract molecular features using cached fingerprints
            features = get_molecular_features(
                compounds=compounds,
                data_manager=self.data_manager,
                featurizer_type=self.featurizer_type
            )
            
            logger.info(f"Extracted {features.shape[1]} molecular features "
                       f"for {features.shape[0]} compounds")
            
            # Compute Tanimoto distance matrix
            logger.info("Computing pairwise distance matrix...")
            distance_matrix = compute_tanimoto_distance_matrix(features)
            
            # Apply Kennard-Stone algorithm
            logger.info("Running Kennard-Stone selection algorithm...")
            selected_indices = fast_kennard_stone(distance_matrix)
            
            # Select top n_select compounds in order of priority
            top_indices = selected_indices[:n_select]
            selected_compounds = compounds.iloc[top_indices].copy()
            
            # Add acquisition metadata
            selected_compounds['acquisition_score'] = np.arange(n_select, 0, -1)  # Priority order
            selected_compounds['selection_order'] = np.arange(n_select)
            
            logger.info(f"Successfully selected {len(selected_compounds)} diverse compounds")
            
            return selected_compounds
            
        except Exception as e:
            logger.error(f"Kennard-Stone selection failed: {str(e)}")
            raise RuntimeError(f"Kennard-Stone acquisition failed: {str(e)}") from e
    
    def requires_uncertainty(self) -> bool:
        """Kennard-Stone doesn't require uncertainty estimates."""
        return False
    
    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        return f"Kennard-Stone({self.featurizer_type})"


def create_kennard_stone_acquisition(data_manager: 'DataManager',
                                     featurizer_type: str = 'morgan',
                                     random_state: Optional[int] = None) -> KennardStoneAcquisition:
    """Factory function for creating KennardStoneAcquisition instances.
    
    Args:
        data_manager: DataManager instance for feature extraction and caching
        featurizer_type: Type of molecular features ('morgan', 'maccs', 'ecfp6') 
        random_state: Random seed for reproducibility
        
    Returns:
        Configured KennardStoneAcquisition instance
    """
    return KennardStoneAcquisition(
        data_manager=data_manager,
        featurizer_type=featurizer_type,
        random_state=random_state
    )
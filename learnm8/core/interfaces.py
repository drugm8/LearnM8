"""Core interfaces for the LearnM8 active learning framework.

This module defines the abstract base classes that establish the contracts
for all major components in the LearnM8 system.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any, TYPE_CHECKING
from pathlib import Path
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .data_manager import DataManager


class Oracle(ABC):
    """Interface for measuring molecular properties.
    
    Oracles are responsible for evaluating compounds and returning their
    measured properties. This could be experimental measurements, 
    computational simulations, or lookup from databases.
    """
    
    @abstractmethod
    def measure(self, compounds: pd.DataFrame, properties: List[str]) -> pd.DataFrame:
        """
        Measure properties for given compounds.
        
        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            properties: List of property names to measure
            
        Returns:
            DataFrame with 'ID' column and measured property columns
            
        Raises:
            ValueError: If compounds DataFrame is malformed
            RuntimeError: If measurement fails
        """
        pass


class Learner(ABC):
    """Simplified base class for all machine learning models.
    
    Learners are responsible for training models on labeled data and making
    predictions on unlabeled compounds. They use dependency injection to
    receive a DataManager for all feature extraction needs.
    """
    
    @abstractmethod
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: 'DataManager') -> None:
        """
        Train the model on labeled compound data.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', and target columns
            target_column: Name of the target property column  
            data_manager: Central data manager for feature extraction and caching
            
        Raises:
            ValueError: If compounds DataFrame is malformed or target_column missing
            RuntimeError: If training fails
        """
        pass
    
    @abstractmethod
    def predict(self, compounds: pd.DataFrame, data_manager: 'DataManager') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict scores for compounds.

        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            data_manager: Central data manager for feature extraction

        Returns:
            Tuple of (predictions, uncertainties).
            uncertainties can be None if model doesn't provide uncertainty estimates.
            The predictions and uncertainties align with the valid compounds returned
            by data_manager.prepare_prediction_data().

        Raises:
            ValueError: If compounds DataFrame is malformed
            RuntimeError: If model is not trained or prediction fails
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return a descriptive name for this learner.
        
        Returns:
            String identifier for the learner type and configuration
        """
        pass
    
    def supports_uncertainty(self) -> bool:
        """Return True if this learner can provide uncertainty estimates.
        
        This method can be overridden by subclasses for efficiency.
        The default implementation attempts a test prediction to check
        if uncertainty is returned.
        
        Returns:
            Boolean indicating uncertainty support
        """
        # Default implementation - can be overridden for efficiency
        try:
            # This would need proper implementation with actual DataManager
            # For now, subclasses should override this method
            return False
        except Exception:
            return False


class AcquisitionFunction(ABC):
    """Base class for compound selection strategies.
    
    Acquisition functions determine which compounds to select for labeling
    in each active learning cycle based on model predictions and optionally
    uncertainty estimates.
    """
    
    @abstractmethod
    def select(self, compounds: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """
        Select compounds for labeling.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
                      May also contain 'uncertainty' column if available
            n_select: Number of compounds to select
            
        Returns:
            DataFrame subset with selected compounds
            
        Raises:
            ValueError: If required columns are missing or n_select is invalid
            RuntimeError: If selection fails
        """
        pass
    
    def requires_uncertainty(self) -> bool:
        """Return True if this acquisition function requires uncertainty estimates.
        
        Returns:
            Boolean indicating if uncertainty column is required
        """
        return False
    
    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function.
        
        Returns:
            String identifier for the acquisition function
        """
        return self.__class__.__name__
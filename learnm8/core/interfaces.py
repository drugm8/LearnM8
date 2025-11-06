"""Core interfaces for the LearnM8 active learning framework.

This module defines the abstract base classes that establish the contracts
for all major components in the LearnM8 system.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np


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
    """Base class for all machine learning models.

    Learners work with feature matrices (numpy arrays) and are agnostic to
    the molecular domain. Feature extraction happens at the API/cycle level.

    Some learners (e.g., Chemprop) work directly with SMILES strings instead
    of pre-computed features. These learners override requires_smiles() to
    return True and use the smiles parameter in train/predict.
    """

    @abstractmethod
    def train(self,
              features: np.ndarray,
              targets: np.ndarray,
              smiles: Optional[List[str]] = None) -> None:
        """
        Train the model on feature matrix or SMILES.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
            smiles: Optional SMILES strings (required by some learners)

        Raises:
            ValueError: If input shapes invalid
            RuntimeError: If training fails
        """
        pass

    @abstractmethod
    def predict(self,
                features: np.ndarray,
                smiles: Optional[List[str]] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict on feature matrix or SMILES.

        Args:
            features: Feature matrix (n_samples, n_features)
            smiles: Optional SMILES strings (required by some learners)

        Returns:
            Tuple of (predictions, uncertainties).
            uncertainties can be None if model doesn't provide uncertainty estimates.

        Raises:
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
        # Subclasses should override this method with actual logic
        return False

    def requires_smiles(self) -> bool:
        """Return True if this learner needs SMILES strings.

        Override in subclasses that work directly with molecular structures
        instead of pre-computed features (e.g., graph neural networks).

        Returns:
            False by default (feature-based learners)
        """
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
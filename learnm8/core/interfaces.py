"""Abstract interfaces for LearnM8 components."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd
import numpy as np


class Oracle(ABC):
    """Abstract interface for compound measurement/scoring."""
    
    @abstractmethod
    def measure(self, compounds: pd.DataFrame, properties: list[str]) -> pd.DataFrame:
        """
        Measure properties for given compounds.
        
        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            properties: List of property names to measure
            
        Returns:
            DataFrame with 'ID' column and requested property columns
        """
        pass


class Learner(ABC):
    """Abstract base class for active learning models."""
    
    @abstractmethod
    def train(self, compounds: pd.DataFrame, target_column: str) -> None:
        """Train the model on labeled compounds."""
        pass
    
    @abstractmethod
    def predict(self, compounds: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict scores for unlabeled compounds.
        
        Returns:
            Tuple of (predictions, uncertainty) where:
            - predictions: Array of predicted scores
            - uncertainty: Array of uncertainty estimates or None if not supported
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the model name."""
        pass
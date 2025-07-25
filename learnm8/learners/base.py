"""Base learner class for sklearn-compatible models."""

from abc import abstractmethod
import pandas as pd
import numpy as np
from core.interfaces import Learner
from utils.chemistry import smiles_to_fingerprints


class SklearnLearner(Learner):
    """Base class for scikit-learn based learners."""
    
    def __init__(self, model):
        """
        Initialize with a scikit-learn compatible model.
        
        Args:
            model: Scikit-learn model instance
        """
        self.model = model
        self.is_trained = False
        self.training_data = pd.DataFrame()
        self.target_column = None
    
    def train(self, compounds: pd.DataFrame, target_column: str) -> None:
        """
        Train the model on labeled compounds.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', and target columns
            target_column: Name of column containing target values
        """
        if target_column not in compounds.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Accumulate training data
        if self.training_data.empty:
            self.training_data = compounds.copy()
        else:
            self.training_data = pd.concat([self.training_data, compounds], ignore_index=True)
        
        self.target_column = target_column
        
        # Convert SMILES to fingerprints
        X = smiles_to_fingerprints(self.training_data['SMILES'].tolist())
        y = self.training_data[target_column].values
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        print(f"Trained {self.get_name()} on {len(self.training_data)} compounds")
    
    def predict(self, compounds: pd.DataFrame) -> np.ndarray:
        """
        Predict scores for compounds.
        
        Args:
            compounds: DataFrame with 'SMILES' column
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Convert SMILES to fingerprints
        X = smiles_to_fingerprints(compounds['SMILES'].tolist())
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the model name."""
        pass
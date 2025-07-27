"""Base learner class for sklearn-compatible models."""

from abc import abstractmethod
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from learnm8.core.interfaces import Learner
from learnm8.utils.featurizers import get_fingerprints, get_descriptors


class SklearnLearner(Learner):
    """Base class for scikit-learn based learners."""
    
    def __init__(self, model, results_dir: Path = None, featurizer_type: str = 'morgan'):
        """
        Initialize with a scikit-learn compatible model.
        
        Args:
            model: Scikit-learn model instance
            results_dir: Directory for storing representations
            featurizer_type: Type of molecular featurizer ('morgan', 'maccs', 'ecfp6', or 'descriptors')
        """
        self.model = model
        self.results_dir = results_dir
        self.featurizer_type = featurizer_type
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
        
        # Get molecular representations based on featurizer type
        if self.results_dir is None:
            raise RuntimeError("results_dir must be set to use molecular representations")
        
        if self.featurizer_type in ['morgan', 'maccs', 'ecfp6']:
            X = get_fingerprints(self.training_data['ID'].tolist(), self.results_dir, self.featurizer_type)
        elif self.featurizer_type == 'descriptors':
            X = get_descriptors(self.training_data['ID'].tolist(), self.results_dir)
        else:
            raise ValueError(f"Unknown featurizer type: {self.featurizer_type}")
        
        y = self.training_data[target_column].values
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        from learnm8.utils.logging import get_logger
        logger = get_logger()
        logger.debug(f"Trained [cyan]{self.get_name()}[/cyan] on [bold]{len(self.training_data)}[/bold] compounds")
    
    def predict(self, compounds: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict scores for compounds.
        
        Args:
            compounds: DataFrame with 'SMILES' column
            
        Returns:
            Tuple of (predictions, uncertainty) where uncertainty is None for base sklearn models
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Get molecular representations based on featurizer type
        if self.results_dir is None:
            raise RuntimeError("results_dir must be set to use molecular representations")
        
        if self.featurizer_type in ['morgan', 'maccs', 'ecfp6']:
            X = get_fingerprints(compounds['ID'].tolist(), self.results_dir, self.featurizer_type)
        elif self.featurizer_type == 'descriptors':
            X = get_descriptors(compounds['ID'].tolist(), self.results_dir)
        else:
            raise ValueError(f"Unknown featurizer type: {self.featurizer_type}")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        from learnm8.utils.logging import get_logger
        logger = get_logger()
        logger.debug(f"Predicted [bold]{len(predictions)}[/bold] compounds with [cyan]{self.get_name()}[/cyan]")
        
        return predictions, None
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the model name."""
        pass
"""
Scikit-learn based learner implementation for LearnM8 active learning.

This module provides a base class for scikit-learn compatible models
that work with molecular fingerprints derived from SMILES strings.
"""

from abc import ABC, abstractmethod
from learners.learner_abc import learner
import pandas as pd
import numpy as np
from helpers.helpers import convert_list_of_smiles_to_morgan_fingerprints


class sklearn_learner(learner):
    """
    Base class for scikit-learn compatible active learning models.
    
    This class handles the conversion of SMILES strings to molecular fingerprints
    and provides a standard interface for sklearn-based regression models.
    
    Attributes:
        model: The underlying scikit-learn model instance
        model_config: Configuration dictionary for hyperparameter tuning
    """
    
    def __init__(self, sklearn_model):
        """
        Initialize sklearn-based learner with a scikit-learn model.
        
        Args:
            sklearn_model: Initialized scikit-learn model (e.g., RandomForestRegressor)
        """
        self.model = sklearn_model
        self.name = sklearn_model.__class__.__name__
        self.model_config = None

    def teach(self, labeled_compounds):
        """
        Add new labeled compounds to training set and retrain the model.
        
        Args:
            labeled_compounds (pd.DataFrame): New training data with columns:
                - 'ID': Compound identifier
                - 'SMILES': Molecular structure
                - target_column: Target values to learn
        """
        # Add data to internal training set
        self.add_training_data(labeled_compounds)
        
        # Retrain model with updated dataset
        self._train_model()

    def estimate(self, smiles_list):
        """
        Make predictions for a list of SMILES compounds.
        
        Args:
            smiles_list (list or array): SMILES strings to predict
            
        Returns:
            array: Predicted values for the input compounds
        """
        # Convert SMILES to molecular fingerprints
        molecular_fingerprints = convert_list_of_smiles_to_morgan_fingerprints(smiles_list)
        
        # Make predictions using trained model
        predictions = self.model.predict(molecular_fingerprints)
        
        return predictions

    def _train_model(self):
        """
        Train the sklearn model using current training data.
        
        Converts SMILES to molecular fingerprints and fits the model.
        """
        if self.compound_features is None or self.target_values is None:
            raise ValueError("No training data available. Call teach() first.")
        
        # Convert SMILES to molecular fingerprints for sklearn model
        molecular_fingerprints = convert_list_of_smiles_to_morgan_fingerprints(
            self.compound_features
        )
        
        # Debug output for development
        print(f"Training model with {len(molecular_fingerprints)} compounds")
        print(f"Fingerprint shape: {molecular_fingerprints.shape}")
        print(f"Target values shape: {self.target_values.shape}")
        
        # Fit the sklearn model
        self.model.fit(molecular_fingerprints, self.target_values)

    def get_model_info(self):
        """
        Get information about the current model state.
        
        Returns:
            dict: Model information including type, parameters, and training status
        """
        info = {
            'model_type': self.name,
            'n_training_samples': len(self.training_data) if self.training_data is not None else 0,
            'target_column': self.target_column,
            'is_trained': hasattr(self.model, 'feature_importances_') if hasattr(self.model, 'feature_importances_') else True
        }
        
        # Add model-specific parameters if available
        if hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()
            
        return info

    def debug_training_data(self):
        """Print debugging information about current training data."""
        if self.training_data is not None:
            print(f"Training data shape: {self.training_data.shape}")
            print(f"Columns: {list(self.training_data.columns)}")
            print(f"Target column: {self.target_column}")
            print(f"Compound features shape: {self.compound_features.shape if self.compound_features is not None else 'None'}")
            print(f"Target values shape: {self.target_values.shape if self.target_values is not None else 'None'}")
        else:
            print("No training data available")
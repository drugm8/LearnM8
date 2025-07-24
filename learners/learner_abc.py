"""
Abstract Base Class for Active Learning Models in LearnM8

This module defines the interface that all active learning models must implement
for compatibility with the LearnM8 framework. The simplified interface focuses
on single target column learning without consensus scoring complexity.
"""

from abc import ABC, abstractmethod
import os
import pandas as pd
from sklearn.metrics import mean_squared_error


class learner(ABC):
    """
    Abstract base class for active learning models in molecular screening.
    
    All learner implementations must inherit from this class and implement
    the required abstract methods for training, querying, and prediction.
    
    Attributes:
        training_data (pd.DataFrame): Accumulated training dataset
        target_column (str): Name of the column to learn/predict
        compound_features (array): Processed molecular features (SMILES → fingerprints)
        target_values (array): Target values for supervised learning
        seed (int): Random seed for reproducibility
        query_function: Function for selecting next compounds to query
        batch_size (int): Number of compounds to select per active learning iteration
        output_path (str): Directory path for saving results and cache files
    """
    
    # Class attributes with clearer names
    training_data = None
    target_column = None
    compound_features = None  # Renamed from dataset_x
    target_values = None      # Renamed from dataset_y
    seed = None
    query_function = None
    batch_size = None
    output_path = None        # Renamed from path
    score_direction = None    # 'higher' or 'lower' for scoring direction

    @abstractmethod
    def __init__(self):
        """
        Initialize the learner with model-specific configuration.
        
        Note: Removed query_function, initial_x, initial_y parameters from signature
        as they are set through setter methods for cleaner interface.
        """
        pass

    @abstractmethod
    def teach(self, labeled_compounds):
        """
        Add new labeled data to the training set and retrain the model.
        
        Args:
            labeled_compounds (pd.DataFrame): New labeled data with columns:
                - 'ID': Compound identifier
                - 'SMILES': Molecular structure representation
                - target_column: The target values to learn
        """
        pass
    
    @abstractmethod
    def query(self, available_compounds, compound_pool_csv_path, target_column):
        """
        Select the next batch of compounds for labeling using the query strategy.
        
        Args:
            available_compounds (pd.DataFrame): Pool of unlabeled compounds with 'ID' and 'SMILES'
            compound_pool_csv_path (str): Path to the full compound pool CSV file
            target_column (str): Name of the target column being learned
            
        Returns:
            pd.DataFrame: Selected compounds for the next iteration
        """
        pass

    @abstractmethod
    def estimate(self, smiles_list):
        """
        Make predictions for a list of SMILES compounds.
        
        Args:
            smiles_list (list or array): List of SMILES strings to predict
            
        Returns:
            array: Predicted values for the input compounds
        """
        pass
    
    def get_name(self):
        """Get the human-readable name of this learner."""
        return getattr(self, 'name', 'Unknown Learner')
    
    def set_query_function(self, query_func):
        """Set the query strategy function for compound selection."""
        self.query_function = query_func
    
    def set_score_direction(self, score_direction):
        """Set the scoring direction ('higher' or 'lower')."""
        if score_direction not in ['higher', 'lower']:
            raise ValueError("Score direction must be 'higher' or 'lower'")
        self.score_direction = score_direction

    def set_batch_size(self, batch_size):
        """Set the number of compounds to select per active learning iteration."""
        self.batch_size = batch_size
        
    def set_output_path(self, path):
        """Set the directory path for saving results and cache files."""
        self.output_path = path
        
    def set_seed(self, seed):
        """Set random seed for reproducible results."""
        self.seed = seed
        
    def get_seed(self):
        """Get the current random seed."""
        return self.seed
        
    def set_target_column(self, target_column):
        """Set the name of the target column to learn."""
        self.target_column = target_column

    def save_predictions(self, predictions_df):
        """
        Save model predictions to cache file for analysis and evaluation.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with 'ID' and 'prediction' columns
        """
        if not self.output_path:
            return
            
        cache_file_path = os.path.join(self.output_path, "cache.csv")
        
        # Rename prediction column to cycle-specific name
        predictions_copy = predictions_df.copy()
        
        if not os.path.exists(cache_file_path):
            # First cycle - create new cache file
            predictions_copy.rename(columns={"prediction": "cycle_0"}, inplace=True)
            predictions_copy.to_csv(cache_file_path, index=False)
        else:
            # Subsequent cycles - append to existing cache
            existing_cache = pd.read_csv(cache_file_path)
            cycle_number = len(existing_cache.columns) - 1  # Subtract 1 for ID column
            
            predictions_copy.rename(columns={"prediction": f"cycle_{cycle_number}"}, inplace=True)
            
            # Merge with existing cache
            updated_cache = pd.merge(existing_cache, predictions_copy, on='ID', how='outer')
            updated_cache.to_csv(cache_file_path, index=False)

    def prepare_training_data(self):
        """
        Prepare training data by extracting features and target values.
        
        This simplified version works directly with single target columns,
        bypassing the complex consensus scoring and normalization logic.
        """
        if self.training_data is None or self.training_data.empty:
            return
            
        # Extract SMILES as features (to be converted to molecular fingerprints by subclass)
        self.compound_features = self.training_data['SMILES'].values
        
        # Extract target values
        if self.target_column and self.target_column in self.training_data.columns:
            self.target_values = self.training_data[self.target_column].values
        else:
            raise ValueError(f"Target column '{self.target_column}' not found in training data")

    def add_training_data(self, new_labeled_data):
        """
        Add new labeled compounds to the training dataset.
        
        Args:
            new_labeled_data (pd.DataFrame): New data to add with ID, SMILES, and target columns
        """
        if self.training_data is None:
            # First batch of training data
            self.training_data = new_labeled_data.copy()
        else:
            # Append to existing training data
            self.training_data = pd.concat([self.training_data, new_labeled_data], 
                                         axis=0, ignore_index=True)
        
        # Prepare features and targets for model training
        self.prepare_training_data()

    def query(self, available_compounds, compound_pool_csv_path, target_column):
        """
        Default implementation of query method using the configured query function.
        
        This method:
        1. Makes predictions on the full compound pool
        2. Calculates MSE against ground truth (if available)
        3. Uses the query function to select next batch
        4. Saves predictions to cache file
        
        Args:
            available_compounds (pd.DataFrame): Pool of available compounds
            compound_pool_csv_path (str): Path to full compound pool CSV
            target_column (str): Target column name
            
        Returns:
            pd.DataFrame: Selected compounds for next iteration
        """
        try:
            # Load full compound pool for prediction
            full_compound_pool = pd.read_csv(compound_pool_csv_path)
        except Exception as e:
            raise IOError(f"Failed to read compound pool file: {e}")
        
        # Make predictions on all compounds
        all_smiles = full_compound_pool['SMILES'].values
        predictions = self.estimate(all_smiles)
        
        # Create predictions DataFrame
        full_compound_pool['prediction'] = predictions
        
        # Calculate MSE if ground truth is available
        if target_column in full_compound_pool.columns:
            mse = mean_squared_error(full_compound_pool['prediction'], 
                                   full_compound_pool[target_column])
            print(f"Current MSE: {mse:.4f}")
        
        # Save predictions to cache
        predictions_for_cache = full_compound_pool[['ID', 'prediction']].copy()
        self.save_predictions(predictions_for_cache)
        
        # Merge with available compounds to get subset for querying
        queryable_compounds = pd.merge(available_compounds, full_compound_pool, 
                                     on=['ID', 'SMILES'], how='inner')
        
        # Rename 'prediction' to 'estimation' for query function compatibility
        queryable_compounds = queryable_compounds.rename(columns={'prediction': 'estimation'})
        
        # Use query function to select next batch
        if self.query_function is None:
            raise ValueError("Query function not set. Use set_query_function() first.")
        
        # Pass score direction if the query function supports it
        try:
            selected_compounds = self.query_function(queryable_compounds, 
                                                   batch_size=self.batch_size, 
                                                   seed=self.seed,
                                                   score_direction=self.score_direction or 'higher')
        except TypeError:
            # Fallback for query functions that don't support score_direction parameter
            selected_compounds = self.query_function(queryable_compounds, 
                                                   batch_size=self.batch_size, 
                                                   seed=self.seed)
        
        return selected_compounds

    # Legacy method names for backward compatibility (can be removed later)
    def set_int_batch_size(self, batch_size):
        """Legacy method name - use set_batch_size() instead."""
        self.set_batch_size(batch_size)
        
    def set_column_to_learn(self, target_column):
        """Legacy method name - use set_target_column() instead."""
        self.set_target_column(target_column)
        
    def set_path(self, path):
        """Legacy method name - use set_output_path() instead."""
        self.set_output_path(path)
        
    def set_scoring_functions(self, scoring_functions):
        """
        Legacy method for setting scoring functions.
        
        In the simplified interface, this is replaced by set_target_column()
        which handles single target column learning.
        """
        if isinstance(scoring_functions, list) and len(scoring_functions) == 1:
            self.set_target_column(scoring_functions[0])
        else:
            print("Warning: Multiple scoring functions not supported in simplified interface. "
                  "Using single target column learning.")
            
    def set_do_scoring_function_list_prediction(self, value):
        """
        Legacy method for enabling scoring function list prediction.
        
        This feature has been removed in the simplified interface.
        All learning now focuses on single target columns.
        """
        if value:
            print("Warning: Scoring function list prediction not supported in simplified interface. "
                  "Using single target column learning.")
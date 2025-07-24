"""
Random Forest Learner for LearnM8 Active Learning

This module implements a Random Forest-based active learning model that uses
molecular fingerprints for molecular property prediction in drug discovery.
"""

import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from learners.sklearn_learner import sklearn_learner
from helpers.helpers import convert_list_of_smiles_to_morgan_fingerprints


class rf_learner(sklearn_learner):
    """
    Random Forest based active learner for molecular property prediction.
    
    Uses Morgan molecular fingerprints as features and Random Forest regression
    for learning molecular property relationships. Supports hyperparameter
    optimization through randomized search.
    
    Attributes:
        model (RandomForestRegressor): The underlying Random Forest model
        hyperparameter_config (dict): Best hyperparameters from optimization
        max_jobs (int): Maximum number of parallel jobs for training
    """
    
    def __init__(self, max_out_system=True):
        """
        Initialize Random Forest learner with performance optimization.
        
        Args:
            max_out_system (bool): If True, use all available CPU cores (up to 32)
                                  If False, use single core for compatibility
        """
        # Configure parallel processing
        if max_out_system:
            n_jobs = min(os.cpu_count(), 32)  # Limit to 32 cores max for stability
        else:
            n_jobs = 1
        
        # Initialize Random Forest model with reasonable defaults
        rf_model = RandomForestRegressor(
            n_estimators=100,
            n_jobs=n_jobs,
            random_state=42  # For reproducible results
        )
        
        # Initialize parent sklearn_learner
        super().__init__(rf_model)
        
        self.name = "Random Forest Regressor"
        self.max_jobs = n_jobs
        self.hyperparameter_config = None

    def optimize_hyperparameters(self):
        """
        Optimize Random Forest hyperparameters using randomized search.
        
        Performs hyperparameter tuning on the current training data using
        cross-validation to find the best parameter combination.
        
        Note: This method requires training data to be available.
        """
        if self.compound_features is None or self.target_values is None:
            raise ValueError("No training data available for hyperparameter optimization")
        
        # Define hyperparameter search space
        hyperparameter_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 10, 20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'criterion': ['squared_error', 'absolute_error'],
            'max_leaf_nodes': [None, 30, 50, 70, 90],
            'min_impurity_decrease': [0.0, 0.01, 0.1, 0.2]
        }
        
        # Convert SMILES to molecular fingerprints for optimization
        molecular_fingerprints = convert_list_of_smiles_to_morgan_fingerprints(
            self.compound_features
        )
        
        print(f"Starting hyperparameter optimization with {len(molecular_fingerprints)} training compounds")
        
        # Perform randomized search with cross-validation
        randomized_search = RandomizedSearchCV(
            estimator=RandomForestRegressor(n_jobs=self.max_jobs),
            param_distributions=hyperparameter_grid,
            n_iter=100,  # Number of parameter combinations to try
            cv=5,        # 5-fold cross-validation
            verbose=1,   # Print progress
            random_state=42,
            n_jobs=1     # Use single job for the search itself
        )
        
        # Fit the search
        randomized_search.fit(molecular_fingerprints, self.target_values)
        
        # Update model with best parameters
        self.hyperparameter_config = randomized_search.best_params_
        self.model = randomized_search.best_estimator_
        
        print(f"Hyperparameter optimization completed")
        print(f"Best parameters: {self.hyperparameter_config}")
        print(f"Best cross-validation score: {randomized_search.best_score_:.4f}")

    def get_feature_importance(self, top_n=10):
        """
        Get feature importance scores from the trained Random Forest.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            list: Top N feature importance scores with indices
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet. Call teach() first.")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Get top N features
        top_indices = importances.argsort()[-top_n:][::-1]
        top_importances = importances[top_indices]
        
        return list(zip(top_indices, top_importances))

    def get_model_info(self):
        """
        Get comprehensive information about the Random Forest model.
        
        Returns:
            dict: Model information including RF-specific details
        """
        # Get base model info from parent class
        info = super().get_model_info()
        
        # Add Random Forest specific information
        info.update({
            'max_jobs': self.max_jobs,
            'hyperparameter_config': self.hyperparameter_config,
            'n_estimators': self.model.n_estimators if hasattr(self.model, 'n_estimators') else None,
            'max_depth': self.model.max_depth if hasattr(self.model, 'max_depth') else None
        })
        
        # Add feature importance info if model is trained
        if hasattr(self.model, 'feature_importances_'):
            info['feature_importance_available'] = True
            info['n_features'] = len(self.model.feature_importances_)
        else:
            info['feature_importance_available'] = False
            
        return info

    def predict_with_uncertainty(self, smiles_list):
        """
        Make predictions with uncertainty estimates using Random Forest ensemble.
        
        Args:
            smiles_list (list): SMILES strings to predict
            
        Returns:
            tuple: (predictions, uncertainties) where uncertainties are standard deviations
        """
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("Model not trained yet. Call teach() first.")
        
        # Convert SMILES to fingerprints
        molecular_fingerprints = convert_list_of_smiles_to_morgan_fingerprints(smiles_list)
        
        # Get predictions from all trees
        tree_predictions = []
        for tree in self.model.estimators_:
            tree_preds = tree.predict(molecular_fingerprints)
            tree_predictions.append(tree_preds)
        
        # Calculate mean and standard deviation
        import numpy as np
        tree_predictions = np.array(tree_predictions)
        mean_predictions = np.mean(tree_predictions, axis=0)
        uncertainty_estimates = np.std(tree_predictions, axis=0)
        
        return mean_predictions, uncertainty_estimates
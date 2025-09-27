"""Gaussian Process learner implementation for the LearnM8 framework.

This module provides Gaussian Process regression with principled uncertainty
quantification for molecular property prediction tasks.
"""

import logging
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd

# Base class import
from ..base import SklearnLearner

# Optional imports with fallbacks
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    SKLEARN_AVAILABLE = True
except ImportError:
    GaussianProcessRegressor = None
    RBF = None
    C = None
    SKLEARN_AVAILABLE = False


logger = logging.getLogger(__name__)


class GaussianProcessLearner(SklearnLearner):
    """Gaussian Process learner with native uncertainty support.
    
    This learner provides principled uncertainty quantification through
    Gaussian Process regression, making it ideal for active learning
    scenarios where uncertainty estimates are crucial.
    """
    
    def __init__(self,
                 kernel=None,
                 alpha: float = 1e-10,
                 n_restarts_optimizer: int = 5,
                 normalize_y: bool = True,
                 random_state: int = 42,
                 featurizer_type: str = None,
                 **kwargs):
        """Initialize Gaussian Process learner.

        Args:
            kernel: GP kernel (None for RBF with learned hyperparameters)
            alpha: Noise regularization parameter
            n_restarts_optimizer: Number of optimizer restarts for hyperparameter optimization
            normalize_y: Whether to normalize target values
            random_state: Random seed for reproducibility
            featurizer_type: Type of molecular features to use
            **kwargs: Additional arguments passed to SklearnLearner
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for GaussianProcessLearner")
        
        # Default to RBF kernel with learned hyperparameters if none provided
        if kernel is None:
            kernel = C(1.0, (1e-4, 1e7)) * RBF(1.0, (1e-4, 1e7))
        
        # Create Gaussian Process model
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_state=random_state
        )
        
        super().__init__(model, featurizer_type=featurizer_type, random_state=random_state, **kwargs)
        
        # Store configuration for name generation
        self.alpha = alpha
        self.kernel_name = str(kernel).split('(')[0] if kernel else "RBF"
    
    def predict(self, compounds: pd.DataFrame, data_manager: 'DataManager') -> Tuple[np.ndarray, np.ndarray]:
        """Predict with native GP uncertainty.

        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            data_manager: Central data manager for feature extraction

        Returns:
            Tuple of (predictions, uncertainties).
            GP naturally provides uncertainty.
            The predictions and uncertainties align with the valid compounds returned
            by data_manager.prepare_prediction_data().

        Raises:
            ValueError: If compounds DataFrame is malformed
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        try:
            # Use DataManager to prepare prediction data (filters invalid compounds)
            valid_compounds, X = data_manager.prepare_prediction_data(compounds, self.featurizer_type)

            if len(valid_compounds) == 0:
                logger.warning("No compounds could generate valid features for prediction")
                return np.array([]), np.array([])

            # Make predictions with uncertainty
            predictions, std = self.model.predict(X, return_std=True)

            logger.debug(f"Predicted {len(predictions)} compounds with {self.get_name()}")

            return predictions, std  # GP naturally provides uncertainty

        except Exception as e:
            logger.error(f"Failed to predict with {self.get_name()}: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    def supports_uncertainty(self) -> bool:
        """Return True since GP naturally provides uncertainty estimates."""
        return True
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f"GaussianProcess({self.kernel_name},α={self.alpha})"
    
    def get_learned_hyperparameters(self) -> Optional[dict]:
        """Get learned kernel hyperparameters from the trained model.
        
        Returns:
            Dictionary of learned hyperparameters, or None if model not trained
        """
        if not self.is_trained:
            return None
        
        kernel = getattr(self.model, 'kernel_', None)
        if kernel is None:
            return None
        
        return {
            'kernel': str(kernel),
            'theta': kernel.theta,
            'log_marginal_likelihood': getattr(self.model, 'log_marginal_likelihood_value_', None)
        }
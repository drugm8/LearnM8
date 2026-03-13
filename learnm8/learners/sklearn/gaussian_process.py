"""Gaussian Process learner implementation for the LearnM8 framework.

This module provides Gaussian Process regression with principled uncertainty
quantification for molecular property prediction tasks.
"""

import logging
from typing import Tuple, Optional, List
import numpy as np

# Base class import
from ..base import SklearnLearner, _preprocess_features
from learnm8.exceptions import LearnerError

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


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
                 **kwargs):
        """Initialize Gaussian Process learner.

        Args:
            kernel: GP kernel (None for RBF with learned hyperparameters)
            alpha: Noise regularization parameter
            n_restarts_optimizer: Number of optimizer restarts for hyperparameter optimization
            normalize_y: Whether to normalize target values
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments passed to SklearnLearner
        """
        if kernel is None:
            kernel = C(1.0, (1e-4, 1e7)) * RBF(1.0, (1e-4, 1e7))

        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_state=random_state
        )

        super().__init__(model, random_state=random_state, **kwargs)

        logger.debug(f"Initialized GaussianProcessLearner with kernel={kernel}, random_state={random_state}")

        # Store configuration for name generation
        self.alpha = alpha
        self.kernel_name = str(kernel).split('(')[0] if kernel else "RBF"
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on feature matrix with uncertainty.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (predictions, uncertainties).
            GP naturally provides uncertainty estimates.

        Raises:
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise LearnerError(
                f"{self.get_name()} must be trained before prediction. "
                f"Call train() with labeled data first."
            )

        try:
            features, _ = _preprocess_features(
                features,
                valid_feature_mask=self._valid_feature_mask,
                remove_zero_variance=self.remove_zero_variance,
                is_training=False
            )

            predictions, std = self.model.predict(features, return_std=True)
            logger.debug(f"Predicted {len(predictions)} samples with {self.get_name()}")
            return predictions, std

        except Exception as e:
            logger.error(f"Failed to predict with {self.get_name()}: {e}")
            raise LearnerError(
                f"Prediction failed for {self.get_name()} on {len(features)} samples: {e}. "
                f"Check that the input features have the same shape as training features "
                f"and that the featurizer is compatible with the model."
            ) from e
    
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
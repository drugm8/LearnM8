"""Linear Regression learner implementation for the LearnM8 framework.

This module provides Linear Regression with optimized settings
for molecular property prediction tasks.
"""

import logging
import numpy as np

from ..base import SklearnLearner

from sklearn.linear_model import LinearRegression, Ridge


logger = logging.getLogger(__name__)


class LinearRegressionLearner(SklearnLearner):
    """Linear/Ridge Regression with optimized settings for molecular data.
    
    This learner provides both standard Linear Regression and Ridge Regression
    with L2 regularization. If alpha parameter is provided, uses Ridge regression;
    otherwise uses standard Linear Regression with parallel processing capabilities.
    """
    
    def __init__(self,
                 alpha: float | None = None,
                 fit_intercept: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 **kwargs):
        """Initialize Linear/Ridge Regression learner.

        Args:
            alpha: Regularization strength. If None, uses LinearRegression.
                  If provided, uses Ridge regression with L2 regularization.
            fit_intercept: Whether to fit the intercept term
            n_jobs: Number of parallel jobs (-1 for all cores, only for LinearRegression)
            random_state: Random seed
            **kwargs: Additional arguments passed to SklearnLearner
        """
        if alpha is None:
            model = LinearRegression(
                fit_intercept=fit_intercept,
                n_jobs=n_jobs
            )
            self.is_ridge = False
        else:
            model = Ridge(
                alpha=alpha,
                fit_intercept=fit_intercept,
                random_state=random_state
            )
            self.is_ridge = True

        super().__init__(model, random_state=random_state, **kwargs)
        
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs if alpha is None else None
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        intercept_str = "with_intercept" if self.fit_intercept else "no_intercept"
        if self.is_ridge:
            return f"Ridge(α={self.alpha:.3f},{intercept_str})"
        else:
            return f"LinearRegression({intercept_str})"
    
    def get_coefficients(self) -> np.ndarray | None:
        """Get model coefficients from the trained model.
        
        Returns:
            Array of model coefficients, or None if model not trained
        """
        if not self.is_trained:
            return None
        
        return self.model.coef_
    
    def get_intercept(self) -> float | None:
        """Get model intercept from the trained model.
        
        Returns:
            Model intercept, or None if model not trained or no intercept fitted
        """
        if not self.is_trained:
            return None
        
        return getattr(self.model, 'intercept_', None)
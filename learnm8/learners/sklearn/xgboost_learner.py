"""XGBoost learner implementation for the LearnM8 framework.

This module provides XGBoost gradient boosting with optimized hyperparameters
for molecular property prediction tasks.
"""

import logging
from typing import Optional
import numpy as np

# Base class import
from ..base import SklearnLearner

import xgboost as xgb


logger = logging.getLogger(__name__)


class XGBoostLearner(SklearnLearner):
    """XGBoost gradient boosting with optimized hyperparameters.
    
    This learner provides high-performance gradient boosting for molecular
    property prediction with efficient CPU parallelization and robust
    handling of different data distributions.
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 6,
                 min_child_weight: int = 1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """Initialize XGBoost learner.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (step size shrinkage)
            max_depth: Maximum depth of trees
            min_child_weight: Minimum sum of instance weight needed in child
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            **kwargs: Additional arguments passed to SklearnLearner
        """
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
            objective='reg:squarederror',
            tree_method='hist'
        )

        super().__init__(model, random_state=random_state, **kwargs)

        logger.debug(f"Initialized XGBoostLearner with n_estimators={n_estimators}, max_depth={max_depth}")

        # Store hyperparameters for name generation
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f"XGBoost(n_estimators={self.n_estimators},lr={self.learning_rate},depth={self.max_depth})"
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores from the trained model.
        
        Returns:
            Array of feature importance scores, or None if model not trained
        """
        if not self.is_trained:
            return None
        
        return self.model.feature_importances_
    
    def get_booster_stats(self) -> Optional[dict]:
        """Get statistics from the trained XGBoost booster.
        
        Returns:
            Dictionary of booster statistics, or None if model not trained
        """
        if not self.is_trained:
            return None
        
        booster = getattr(self.model, 'get_booster', lambda: None)()
        if booster is None:
            return None
        
        try:
            return {
                'num_boosted_rounds': booster.num_boosted_rounds(),
                'num_features': booster.num_features(),
                'objective': self.model.objective
            }
        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to get booster stats: {e}")
            return None
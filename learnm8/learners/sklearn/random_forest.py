"""Random Forest learner implementation for the LearnM8 framework.

This module provides Random Forest regression with hyperparameters optimized
for molecular property prediction tasks.
"""

import logging
from typing import Optional
import numpy as np

# Base class import
from ..base import SklearnLearner

# Optional imports with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestRegressor = None
    SKLEARN_AVAILABLE = False


logger = logging.getLogger(__name__)


class RandomForestLearner(SklearnLearner):
    """Random Forest with optimized hyperparameters for molecular data.
    
    This learner uses Random Forest regression with hyperparameters optimized
    for molecular property prediction tasks. It provides robust performance
    across different molecular datasets.
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """Initialize Random Forest learner.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split internal node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider at each split
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            **kwargs: Additional arguments passed to SklearnLearner
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RandomForestLearner")

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            bootstrap=True,
            oob_score=True
        )

        super().__init__(model, random_state=random_state, **kwargs)
        
        # Store hyperparameters for name generation
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        depth_str = f"depth={self.max_depth}" if self.max_depth else "unlimited_depth"
        return f"RandomForest(n_estimators={self.n_estimators},{depth_str})"
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores from the trained model.
        
        Returns:
            Array of feature importance scores, or None if model not trained
        """
        if not self.is_trained:
            return None
        
        return self.model.feature_importances_
    
    def get_oob_score(self) -> Optional[float]:
        """Get out-of-bag R² score from the trained model.
        
        Returns:
            Out-of-bag score, or None if model not trained
        """
        if not self.is_trained:
            return None
        
        return getattr(self.model, 'oob_score_', None)
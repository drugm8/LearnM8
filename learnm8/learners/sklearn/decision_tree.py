"""Decision Tree learner implementation for the LearnM8 framework.

This module provides Decision Tree regression with hyperparameters optimized
for molecular property prediction tasks.
"""

import logging
import numpy as np

from ..base import SklearnLearner

from sklearn.tree import DecisionTreeRegressor


logger = logging.getLogger(__name__)


class DecisionTreeLearner(SklearnLearner):
    """Decision Tree with hyperparameters optimized for molecular data.
    
    This learner uses Decision Tree regression with hyperparameters designed
    to balance model complexity and generalization for molecular datasets.
    """
    
    def __init__(self,
                 max_depth: int | None = 10,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 max_features: str | None = None,
                 random_state: int = 42,
                 **kwargs):
        """Initialize Decision Tree learner.

        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split internal node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider at each split
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments passed to SklearnLearner
        """
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

        super().__init__(model, random_state=random_state, **kwargs)
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        depth_str = f"depth={self.max_depth}" if self.max_depth else "unlimited_depth"
        return f"DecisionTree({depth_str},min_split={self.min_samples_split})"
    
    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance scores from the trained model.
        
        Returns:
            Array of feature importance scores, or None if model not trained
        """
        if not self.is_trained:
            return None
        
        return self.model.feature_importances_
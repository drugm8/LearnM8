"""XGBoost learner implementation."""

import os
from pathlib import Path

from xgboost import XGBRegressor

from ...base import SklearnLearner


class XGBoostLearner(SklearnLearner):
    """XGBoost based active learner with CPU parallelization."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3,
                 n_jobs: int = -1, random_state: int = 42, results_dir: Path = None,
                 featurizer: str = 'morgan'):
        """
        Initialize XGBoost learner.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of individual regression estimators
            n_jobs: Number of CPU cores to use (-1 for all cores)
            random_state: Random seed for reproducibility
            results_dir: Directory for storing representations
            featurizer: Type of molecular featurizer ('morgan' or 'descriptors')
        """
        # Limit CPU cores to prevent oversubscription
        if n_jobs == -1:
            n_jobs = min(os.cpu_count() or 1, 32)

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_jobs=n_jobs,
            tree_method="hist",  # Optimized histogram-based algorithm for parallel performance
            random_state=random_state,
            objective='reg:squarederror'
        )

        super().__init__(model, results_dir, featurizer)

    def get_name(self) -> str:
        """Return the model name."""
        return "XGBoost"

"""XGBoost learner implementation for the LearnM8 framework.

This module provides XGBoost gradient boosting with optimized hyperparameters
for molecular property prediction tasks.
"""

import gc
import logging

import numpy as np
import xgboost as xgb

from learnm8.exceptions import LearnerError

from ..base import SklearnLearner

logger = logging.getLogger(__name__)


class XGBoostLearner(SklearnLearner):
    """XGBoost gradient boosting with optimized hyperparameters.

    This learner provides high-performance gradient boosting for molecular
    property prediction with efficient CPU parallelization and robust
    handling of different data distributions.
    """

    def __init__(
        self,
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
        device: str = 'cpu',
        enable_aggressive_gc: bool = True,
        **kwargs,
    ):
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
            device: Device for XGBoost computation ('cpu' or 'cuda')
            enable_aggressive_gc: Delete old model and call gc.collect() before refit
            **kwargs: Additional arguments passed to SklearnLearner
        """
        self.device = device
        self.enable_aggressive_gc = enable_aggressive_gc

        # Store hyperparameters needed for model recreation
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._min_child_weight = min_child_weight
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._reg_alpha = reg_alpha
        self._reg_lambda = reg_lambda
        self._random_state = random_state
        self._n_jobs = n_jobs

        model = self._make_xgb_model()

        super().__init__(model, random_state=random_state, **kwargs)

        logger.debug(
            f'Initialized XGBoostLearner with n_estimators={n_estimators}, max_depth={max_depth}, device={device}'
        )

        # Store hyperparameters for name generation
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def _make_xgb_model(self) -> xgb.XGBRegressor:
        """Create a new XGBRegressor with current parameters."""
        return xgb.XGBRegressor(
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            min_child_weight=self._min_child_weight,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            reg_alpha=self._reg_alpha,
            reg_lambda=self._reg_lambda,
            random_state=self._random_state,
            n_jobs=self._n_jobs,
            verbosity=0,
            objective='reg:squarederror',
            tree_method='hist',
            device=self.device,
        )

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train XGBoost model, with optional GC and CUDA error handling.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)

        Raises:
            LearnerError: If training fails, or if CUDA training fails
        """
        if self.enable_aggressive_gc and self.is_trained:
            del self.model
            gc.collect()
            self.model = self._make_xgb_model()

        try:
            super().train(features, targets)
        except LearnerError as e:
            if self.device == 'cuda' and e.__cause__ is not None:
                raise LearnerError(
                    f'XGBoost CUDA training failed: {e.__cause__}. '
                    f'Ensure CUDA drivers are installed and XGBoost was built with GPU support.'
                ) from e.__cause__
            raise
        except Exception as e:
            if self.device == 'cuda':
                raise LearnerError(
                    f'XGBoost CUDA training failed: {e}. '
                    f'Ensure CUDA drivers are installed and XGBoost was built with GPU support.'
                ) from e
            raise

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f'XGBoost(n_estimators={self.n_estimators},lr={self.learning_rate},depth={self.max_depth})'

    def preferred_feature_dtype(self) -> str:
        return 'uint8' if self._feature_type == 'binary' else 'float32'

"""Random Forest learner implementation for the LearnM8 framework.

This module provides Random Forest regression with hyperparameters optimized
for molecular property prediction tasks.
"""

import logging
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from learnm8.exceptions import LearnerError

from ..base import SklearnLearner, _preprocess_features

logger = logging.getLogger(__name__)


class RandomForestLearner(SklearnLearner):
    """Random Forest with optimized hyperparameters for molecular data.

    This learner uses Random Forest regression with hyperparameters optimized
    for molecular property prediction tasks. It provides robust performance
    across different molecular datasets.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
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
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            bootstrap=True,
            oob_score=True,
        )

        super().__init__(model, random_state=random_state, **kwargs)

        logger.debug(
            f'Initialized RandomForestLearner with n_estimators={n_estimators}, random_state={random_state}'
        )

        # Store hyperparameters for name generation
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        depth_str = f'depth={self.max_depth}' if self.max_depth else 'unlimited_depth'
        return f'RandomForest(n_estimators={self.n_estimators},{depth_str})'

    def preferred_feature_dtype(self) -> str:
        return 'uint8' if self._feature_type == 'binary' else 'float32'

    def supports_uncertainty(self) -> bool:
        """Return True — RF provides tree-level uncertainty estimates."""
        return True

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict with tree-level uncertainty estimation.

        Mean predictions use sklearn's optimized ``model.predict()`` path.
        Uncertainty is computed as ``np.std`` (ddof=0) across individual tree
        predictions from ``model.estimators_``.

        Note: uncertainty estimates are **ordinal** (useful for ranking compounds
        by relative confidence) but **not calibrated** prediction intervals.
        Tree-level std overestimates true variance due to Monte Carlo bias and
        tree correlation (Wager et al. 2014).

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (predictions, uncertainties) where uncertainties is the
            population std across individual tree predictions.

        Raises:
            LearnerError: If model not trained or estimators_ unavailable.
        """
        if not self.is_trained:
            raise LearnerError(
                f'{self.get_name()} must be trained before prediction. '
                f'Call train() with labeled data first. If using the active learning API, '
                f'ensure at least one training cycle has completed before predicting.'
            )

        start_time = time.time()

        try:
            preprocessed, _, _ = _preprocess_features(
                features,
                valid_feature_mask=self._valid_feature_mask,
                remove_zero_variance=self.remove_zero_variance,
                is_training=False,
                feature_type=self._feature_type,
                imputer=self._feature_imputer,
            )

            if not hasattr(self.model, 'estimators_'):
                raise LearnerError(
                    f'{self.get_name()} model is missing estimators_ attribute. '
                    f'Cannot compute tree-level uncertainty.'
                )

            predictions = self.model.predict(preprocessed)

            tree_predictions = np.array(
                [tree.predict(preprocessed) for tree in self.model.estimators_]
            )
            logger.debug(
                f'Computing uncertainty from {len(self.model.estimators_)} trees'
            )
            uncertainty = np.std(tree_predictions, axis=0, ddof=0)

            pred_time = time.time() - start_time
            logger.debug(
                f'Predicted {len(predictions)} samples with {self.get_name()} in {pred_time:.2f}s'
            )

            return predictions, uncertainty

        except LearnerError:
            raise
        except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as e:
            logger.error(f'Failed to predict with {self.get_name()}: {e}')
            raise LearnerError(
                f'Prediction failed for {self.get_name()} on {len(features)} samples: {e}. '
                f'Check that the input features have the same shape as training features '
                f'and that the featurizer is compatible with the model.'
            ) from e

    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance scores from the trained model.

        Returns:
            Array of feature importance scores, or None if model not trained
        """
        if not self.is_trained:
            return None

        return self.model.feature_importances_

    def get_oob_score(self) -> float | None:
        """Get out-of-bag R² score from the trained model.

        Returns:
            Out-of-bag score, or None if model not trained
        """
        if not self.is_trained:
            return None

        return getattr(self.model, 'oob_score_', None)

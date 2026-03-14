"""Advanced Random Forest learner implementation for the LearnM8 framework.

This module provides Random Forest regression with highly optimized hyperparameters
designed for enhanced performance on molecular property prediction tasks.
"""

import logging
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from learnm8.exceptions import LearnerError

from ..base import SklearnLearner, _preprocess_features

logger = logging.getLogger(__name__)


class AdvancedRandomForestLearner(SklearnLearner):
    """Advanced Random Forest with highly optimized hyperparameters.

    This learner uses an enhanced Random Forest configuration with optimized
    hyperparameters, regularization, and advanced features for superior
    performance on molecular datasets.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = 'sqrt',
        max_samples: float = 0.8,
        min_impurity_decrease: float = 0.0001,
        ccp_alpha: float = 0.001,
        bootstrap: bool = True,
        oob_score: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        """Initialize Advanced Random Forest learner.

        Args:
            n_estimators: Number of trees in the forest (increased from standard)
            max_depth: Maximum depth of trees (limited to prevent overfitting)
            min_samples_split: Minimum samples required to split internal node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider at each split
            max_samples: Fraction of samples to use for each tree
            min_impurity_decrease: Minimum improvement required to split
            ccp_alpha: Cost complexity pruning parameter
            bootstrap: Enable bootstrap sampling
            oob_score: Enable out-of-bag scoring
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            **kwargs: Additional arguments passed to SklearnLearner
        """
        if n_jobs == -1:
            n_jobs = min(os.cpu_count() or 1, 32)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_samples=max_samples,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        super().__init__(model, random_state=random_state, **kwargs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.ccp_alpha = ccp_alpha

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        depth_str = f'depth={self.max_depth}' if self.max_depth else 'unlimited_depth'
        return f'AdvancedRandomForest(n_estimators={self.n_estimators},{depth_str},pruning={self.ccp_alpha})'

    def supports_uncertainty(self) -> bool:
        """Return True — AdvancedRF provides tree-level uncertainty estimates."""
        return True

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict with tree-level uncertainty estimation.

        Uncertainty is computed as ``np.std`` (ddof=0) across individual tree
        predictions from ``model.estimators_``, matching the RandomForestLearner
        pattern.

        Note: uncertainty estimates are **ordinal** (useful for ranking compounds
        by relative confidence) but **not calibrated** prediction intervals.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (predictions, uncertainties).
        """
        if not self.is_trained:
            raise LearnerError(
                f'{self.get_name()} must be trained before prediction. '
                f'Call train() with labeled data first.'
            )

        start_time = time.time()

        try:
            preprocessed, _ = _preprocess_features(
                features,
                valid_feature_mask=self._valid_feature_mask,
                remove_zero_variance=self.remove_zero_variance,
                is_training=False,
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
                f'Prediction failed for {self.get_name()} on {len(features)} samples: {e}.'
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

    def get_tree_stats(self) -> dict | None:
        """Get statistics about the trained forest.

        Returns:
            Dictionary with forest statistics, or None if model not trained
        """
        if not self.is_trained:
            return None

        try:
            tree_depths = [tree.tree_.max_depth for tree in self.model.estimators_]
            tree_nodes = [tree.tree_.node_count for tree in self.model.estimators_]

            return {
                'n_trees': len(self.model.estimators_),
                'avg_depth': np.mean(tree_depths),
                'max_depth': np.max(tree_depths),
                'avg_nodes': np.mean(tree_nodes),
                'total_nodes': np.sum(tree_nodes),
                'oob_score': self.get_oob_score(),
            }
        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            logger.warning(f'Failed to get tree stats: {e}')
            return None

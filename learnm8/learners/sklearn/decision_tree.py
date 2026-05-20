"""Decision Tree learner implementation for the LearnM8 framework.

This module provides Decision Tree regression with hyperparameters optimized
for molecular property prediction tasks.
"""

import logging
import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from learnm8.exceptions import LearnerError

from ..base import SklearnLearner, _preprocess_features

logger = logging.getLogger(__name__)


class DecisionTreeLearner(SklearnLearner):
    """Decision Tree with hyperparameters optimized for molecular data.

    This learner uses Decision Tree regression with hyperparameters designed
    to balance model complexity and generalization for molecular datasets.
    """

    def __init__(
        self,
        max_depth: int | None = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str | None = None,
        random_state: int = 42,
        **kwargs,
    ):
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
            random_state=random_state,
        )

        super().__init__(model, random_state=random_state, **kwargs)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        depth_str = f'depth={self.max_depth}' if self.max_depth else 'unlimited_depth'
        return f'DecisionTree({depth_str},min_split={self.min_samples_split})'

    def preferred_feature_dtype(self) -> str:
        return 'uint8' if self._feature_type == 'binary' else 'float32'

    def supports_uncertainty(self) -> bool:
        """Return True if uncertainty is available (requires squared_error criterion)."""
        if not self.is_trained:
            return True
        return self.model.criterion == 'squared_error'

    def predict(
        self,
        features: np.ndarray,
        smiles: list[str] | None = None,
        *,
        compute_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict with leaf impurity uncertainty.

        When ``criterion='squared_error'``, uncertainty is computed as
        ``sqrt(tree_.impurity[leaf_ids])`` — the training-set variance at each
        leaf node.

        Note: uncertainty estimates are **ordinal** (useful for ranking compounds
        by relative confidence) but **not calibrated** prediction intervals.
        Leaf impurity measures training data heterogeneity within each leaf.

        Args:
            features: Feature matrix (n_samples, n_features)
            smiles: Ignored; present for interface compatibility.
            compute_uncertainty: Keyword-only (feature 023). When False, the
                ``model.apply()`` leaf-id lookup and impurity sqrt are
                skipped.

        Returns:
            Tuple of (predictions, uncertainties). Uncertainties is None
            if criterion is not 'squared_error' or compute_uncertainty=False.
        """
        del smiles
        if not self.is_trained:
            raise LearnerError(
                f'{self.get_name()} must be trained before prediction. '
                f'Call train() with labeled data first.'
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

            predictions = self.model.predict(preprocessed)

            if compute_uncertainty and self.model.criterion == 'squared_error':
                leaf_ids = self.model.apply(preprocessed)
                impurity = self.model.tree_.impurity[leaf_ids]
                uncertainty = np.sqrt(np.maximum(impurity, 0.0))

                zero_frac = np.mean(uncertainty == 0.0)
                if zero_frac > 0.5:
                    logger.debug(
                        f'>{zero_frac:.0%} of predictions have zero uncertainty '
                        f'(leaf impurity=0). Consider increasing min_samples_leaf '
                        f'or reducing max_depth.'
                    )
            else:
                uncertainty = None

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

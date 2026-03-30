"""Linear Regression learner implementation for the LearnM8 framework.

This module provides Linear Regression with optimized settings
for molecular property prediction tasks.
"""

import logging
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import LinearRegression, Ridge

from learnm8.exceptions import LearnerError

from ..base import SklearnLearner, _preprocess_features

logger = logging.getLogger(__name__)


class LinearRegressionLearner(SklearnLearner):
    """Linear/Ridge Regression with optimized settings for molecular data.

    This learner provides both standard Linear Regression and Ridge Regression
    with L2 regularization. If alpha parameter is provided, uses Ridge regression;
    otherwise uses standard Linear Regression with parallel processing capabilities.
    """

    def __init__(
        self,
        alpha: float | None = 0.1,
        fit_intercept: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
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
            model = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)
            self.is_ridge = False
        else:
            model = Ridge(
                alpha=alpha, fit_intercept=fit_intercept, random_state=random_state
            )
            self.is_ridge = True

        super().__init__(model, random_state=random_state, scale_features=self.is_ridge, **kwargs)

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs if alpha is None else None

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        intercept_str = 'with_intercept' if self.fit_intercept else 'no_intercept'
        if self.is_ridge:
            return f'Ridge(alpha={self.alpha:.3f},{intercept_str})'
        else:
            return f'LinearRegression({intercept_str})'

    def supports_uncertainty(self) -> bool:
        """Return True — LR provides leverage-based uncertainty estimates."""
        return True

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train linear model and compute Gram matrix Cholesky factorization.

        After fitting the model via the base class, computes and stores:
        - ``_gram_chol``: Cholesky factorization of ``X^T X + (alpha + eps) * I``
        - ``_sigma_hat_sq``: Residual variance ``RSS / max(n - p, 1)``

        These are used by ``predict()`` to compute leverage-based uncertainty.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
        """
        super().train(features, targets)

        preprocessed, _, _ = _preprocess_features(
            features,
            valid_feature_mask=self._valid_feature_mask,
            remove_zero_variance=self.remove_zero_variance,
            is_training=False,
            feature_type=self._feature_type,
            imputer=self._feature_imputer,
        )

        if self._feature_scaler is not None:
            preprocessed = self._feature_scaler.transform(preprocessed)

        n, p = preprocessed.shape

        alpha = self.alpha if self.alpha is not None else 0.0
        epsilon = 1e-10 if alpha == 0.0 else 0.0
        gram = preprocessed.T @ preprocessed + (alpha + epsilon) * np.eye(p)

        cond = np.linalg.cond(gram)
        if cond > 1e10:
            logger.warning(
                f'Gram matrix condition number is {cond:.2e} (>1e10). '
                f'Consider increasing alpha for regularization.'
            )

        self._gram_chol = cho_factor(gram)

        residuals = targets - self.model.predict(preprocessed)
        rss = float(np.sum(residuals**2))
        self._sigma_hat_sq = rss / max(n - p, 1)

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict with leverage-based analytical uncertainty.

        Uncertainty is computed as ``sqrt(sigma_hat_sq * max(1 + h_ii, 0))``
        where ``h_ii = x^T (X^T X + alpha*I)^{-1} x`` is the leverage.

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
            preprocessed, _, _ = _preprocess_features(
                features,
                valid_feature_mask=self._valid_feature_mask,
                remove_zero_variance=self.remove_zero_variance,
                is_training=False,
                feature_type=self._feature_type,
                imputer=self._feature_imputer,
            )

            if self._feature_scaler is not None:
                preprocessed = self._feature_scaler.transform(preprocessed)

            predictions = self.model.predict(preprocessed)

            solved = cho_solve(self._gram_chol, preprocessed.T)
            leverages = np.einsum('ij,ji->i', preprocessed, solved)
            variance = self._sigma_hat_sq * np.maximum(1.0 + leverages, 0.0)
            uncertainty = np.sqrt(variance)

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

"""Gaussian Process learner implementation for the LearnM8 framework.

This module provides Gaussian Process regression with principled uncertainty
quantification for molecular property prediction tasks.
"""

import logging

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from learnm8.exceptions import ConfigurationError, LearnerError

from ..base import SklearnLearner, _preprocess_features
from .kernels import TanimotoKernel

logger = logging.getLogger(__name__)

_VALID_KERNEL_STRINGS = ('auto', 'tanimoto', 'rbf')


class GaussianProcessLearner(SklearnLearner):
    """Gaussian Process learner with native uncertainty support.

    Supports kernel='auto' (auto-detects binary vs continuous features),
    kernel='tanimoto', kernel='rbf', or a custom sklearn Kernel instance.
    """

    def __init__(
        self,
        kernel: str | Kernel | None = 'auto',
        alpha: float = 1e-3,
        n_restarts_optimizer: int = 5,
        normalize_y: bool = True,
        random_state: int = 42,
        max_train_size: int = 5000,
        **kwargs,
    ):
        if kernel is None:
            kernel = 'auto'

        if isinstance(kernel, str):
            if kernel not in _VALID_KERNEL_STRINGS:
                raise ConfigurationError(
                    f"kernel must be 'auto', 'tanimoto', 'rbf', None, "
                    f"or a sklearn Kernel instance, got '{kernel}'"
                )
            self._kernel_config = kernel
        elif isinstance(kernel, Kernel):
            self._kernel_config = kernel
        else:
            raise ConfigurationError(
                f"kernel must be 'auto', 'tanimoto', 'rbf', None, "
                f'or a sklearn Kernel instance, got {type(kernel).__name__}'
            )

        if isinstance(self._kernel_config, str) and self._kernel_config == 'tanimoto':
            self._kernel_name = 'Tanimoto'
        elif isinstance(self._kernel_config, str) and self._kernel_config == 'rbf':
            self._kernel_name = 'RBF'
        elif isinstance(self._kernel_config, Kernel):
            self._kernel_name = str(self._kernel_config).split('(')[0]
        else:
            self._kernel_name = 'auto'

        self.alpha = alpha
        self.max_train_size = max_train_size
        self._n_restarts_optimizer = n_restarts_optimizer
        self._normalize_y = normalize_y

        placeholder_kernel = C(1.0, (1e-4, 1e7)) * RBF(1.0, (1e-4, 1e7))
        model = GaussianProcessRegressor(
            kernel=placeholder_kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_state=random_state,
        )

        super().__init__(model, random_state=random_state, **kwargs)

        logger.debug(
            f'Initialized GaussianProcessLearner with kernel={self._kernel_config}, '
            f'alpha={alpha}, random_state={random_state}'
        )

    def _resolve_kernel(self, features: np.ndarray) -> Kernel:
        if isinstance(self._kernel_config, Kernel):
            return self._kernel_config

        if self._kernel_config == 'auto':
            is_binary = np.all((features == 0) | (features == 1))
            if is_binary:
                self._kernel_name = 'Tanimoto'
                return TanimotoKernel()
            else:
                self._kernel_name = 'RBF'
                return C(1.0, (1e-4, 1e7)) * RBF(1.0, (1e-4, 1e7))

        if self._kernel_config == 'tanimoto':
            if np.any(features < 0):
                logger.warning(
                    'Tanimoto kernel is designed for non-negative features. '
                    'Negative values detected — PSD guarantee may not hold.'
                )
            return TanimotoKernel()

        return C(1.0, (1e-4, 1e7)) * RBF(1.0, (1e-4, 1e7))

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._n_train = features.shape[0] if features.ndim == 2 else len(features)
        n_samples = self._n_train

        if n_samples > self.max_train_size:
            raise LearnerError(
                f'Training set size ({n_samples}) exceeds maximum ({self.max_train_size}). '
                f'GP training is O(n^3) and impractical at this scale. '
                f"Consider using 'rf', 'ensemble', or 'xgb' learners instead, "
                f'or increase max_train_size if you have sufficient compute.'
            )

        warn_threshold = int(self.max_train_size * 0.4)
        if n_samples > warn_threshold:
            logger.warning(
                f'Training set size ({n_samples}) is large for GP (O(n^3) scaling). '
                f"Training may be slow. Consider 'rf', 'ensemble', or 'xgb' for large datasets."
            )

        resolved_kernel = self._resolve_kernel(features)
        self.model = GaussianProcessRegressor(
            kernel=resolved_kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self._n_restarts_optimizer,
            normalize_y=self._normalize_y,
            random_state=self.random_state,
        )

        super().train(features, targets)

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise LearnerError(
                f'{self.get_name()} must be trained before prediction. '
                f'Call train() with labeled data first.'
            )

        try:
            features, _ = _preprocess_features(
                features,
                valid_feature_mask=self._valid_feature_mask,
                remove_zero_variance=self.remove_zero_variance,
                is_training=False,
            )

            predictions, std = self.model.predict(features, return_std=True)
            logger.debug(f'Predicted {len(predictions)} samples with {self.get_name()}')
            return predictions, std

        except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as e:
            logger.error(f'Failed to predict with {self.get_name()}: {e}')
            raise LearnerError(
                f'Prediction failed for {self.get_name()} on {len(features)} samples: {e}. '
                f'Check that the input features have the same shape as training features '
                f'and that the featurizer is compatible with the model.'
            ) from e

    def supports_uncertainty(self) -> bool:
        return True

    def get_name(self) -> str:
        return f'GaussianProcess({self._kernel_name},alpha={self.alpha})'

    def memory_profile(self, n_features: int) -> dict[str, int | float]:
        n_train = getattr(self, '_n_train', 0)
        return {
            'bytes_per_sample': n_features * 8 + n_train * 8 * 2,
            'working_multiplier': 1.0,
            'fixed_overhead': 0,
        }

    def get_learned_hyperparameters(self) -> dict | None:
        if not self.is_trained:
            return None

        kernel = getattr(self.model, 'kernel_', None)
        if kernel is None:
            return None

        return {
            'kernel': str(kernel),
            'theta': kernel.theta,
            'log_marginal_likelihood': getattr(
                self.model, 'log_marginal_likelihood_value_', None
            ),
        }

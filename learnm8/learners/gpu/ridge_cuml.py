"""cuML Ridge GPU learner with leverage-based uncertainty.

Uses cuML Ridge for GPU-accelerated training and CuPy for leverage-based
uncertainty computation. Falls back to CPU scipy for leverage when GPU OOM.
"""

import logging
import time
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from learnm8.core.interfaces import Learner
from learnm8.exceptions import LearnerError
from learnm8.learners.base import (
    _preprocess_features,
    _validate_predict_inputs,
    _validate_train_inputs,
)

logger = logging.getLogger(__name__)

_LEVERAGE_CHUNK_SIZE = 10_000


class RidgeCumlLearner(Learner):
    """cuML Ridge regression with leverage-based uncertainty estimation.

    Uses cuML's GPU-accelerated Ridge regression for training and prediction.
    Uncertainty is computed analytically via leverage scores:
    ``uncertainty = sqrt(sigma_hat_sq * max(1 + h_ii, 0))``
    where ``h_ii = x^T (X^T X + alpha * I)^{-1} x`` is the leverage.

    Falls back to CPU scipy Cholesky for leverage computation if GPU OOM.

    Note:
        Requires cuml (RAPIDS >= 25.04) to be installed. If cuml is not
        available, construction succeeds but train() raises LearnerError.
        Uncertainty is ordinal (proportional to predictive variance) and is
        not calibrated — it reflects relative confidence rather than a
        statistical prediction interval.
    """

    def __init__(
        self,
        alpha: float | None = 0.1,
        fit_intercept: bool = True,
        solver: str = 'eig',
        random_state: int = 42,
        remove_zero_variance: bool = True,
        **kwargs,
    ) -> None:
        """Initialize RidgeCumlLearner.

        Args:
            alpha: Regularization strength (must not be None for GPU Ridge).
            fit_intercept: Whether to fit the intercept term.
            solver: cuML Ridge solver ('eig', 'svd', 'cd').
            random_state: Random seed for reproducibility.
            remove_zero_variance: Remove zero-variance features during training.
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            LearnerError: If alpha is None.
        """
        if alpha is None:
            raise LearnerError(
                'RidgeCumlLearner requires a non-None alpha for L2 regularization. '
                'Provide a positive float (e.g., alpha=0.1). '
                'Use LinearRegressionLearner with alpha=None for unregularized regression.'
            )

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.random_state = random_state
        self.remove_zero_variance = remove_zero_variance

        self._model = None
        self._feature_scaler = None
        self._gram_chol_cpu = None
        self._gram_chol_gpu = None
        self._sigma_hat_sq = None
        self.is_trained = False
        self._valid_feature_mask = None
        self._feature_imputer = None
        self._feature_type = 'binary'

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        smiles: list[str] | None = None,
    ) -> None:
        """Train cuML Ridge on GPU and compute Gram matrix Cholesky factorization.

        After fitting the cuML Ridge model, computes and stores:
        - ``_gram_chol_cpu``: Cholesky factorization of ``X^T X + alpha * I``
        - ``_sigma_hat_sq``: Residual variance ``RSS / max(n - p, 1)``

        These are used by ``predict()`` to compute leverage-based uncertainty.

        Args:
            features: Feature matrix (n_samples, n_features).
            targets: Target values (n_samples,).
            smiles: Unused; present for interface compatibility.

        Raises:
            LearnerError: If cuml is not installed, inputs are invalid, or training fails.
        """
        _validate_train_inputs(features, targets, self.get_name())

        t0 = time.perf_counter_ns()
        features, self._valid_feature_mask, self._feature_imputer = _preprocess_features(
            features,
            remove_zero_variance=self.remove_zero_variance,
            is_training=True,
            feature_type=self._feature_type,
        )
        t1 = time.perf_counter_ns()
        logger.debug('preprocess: %.3f ms', (t1 - t0) / 1e6)

        from sklearn.preprocessing import StandardScaler
        self._feature_scaler = StandardScaler()
        features = self._feature_scaler.fit_transform(features)

        try:
            from cuml.linear_model import Ridge as CumlRidge
        except ImportError as exc:
            raise LearnerError(
                'cuml is required for RidgeCumlLearner but is not installed. '
                'Install RAPIDS cuML >= 25.04 (e.g., conda install -c rapidsai cuml). '
                'Use LinearRegressionLearner for a CPU-only Ridge alternative.'
            ) from exc

        try:
            import cuml
            _ver = getattr(cuml, '__version__', '')
            _parts = tuple(int(x) for x in _ver.split('.')[:2])
            if _parts < (25, 4):
                raise LearnerError(
                    f'RAPIDS cuML >= 25.04 is required for ridge_cuml (found {_ver}). '
                    'Update via conda: conda install -c rapidsai cuml>=25.04'
                )
        except (ValueError, TypeError, AttributeError):
            pass

        try:
            t2 = time.perf_counter_ns()
            self._model = CumlRidge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                solver=self.solver,
            )
            self._model.fit(features, targets)
            t3 = time.perf_counter_ns()
            logger.debug('cuml ridge fit: %.3f ms', (t3 - t2) / 1e6)
        except Exception as exc:
            raise LearnerError(
                f'cuML Ridge training failed for {self.get_name()} '
                f'with {features.shape[0]} samples: {exc}. '
                f'Check that the training data is valid and GPU memory is sufficient.'
            ) from exc

        n, p = features.shape
        X = features.astype(np.float64)

        t4 = time.perf_counter_ns()
        gram = X.T @ X + self.alpha * np.eye(p, dtype=np.float64)
        t5 = time.perf_counter_ns()
        logger.debug('gram matrix: %.3f ms', (t5 - t4) / 1e6)

        cond = np.linalg.cond(gram)
        if cond > 1e10:
            warnings.warn(
                f'Gram matrix condition number is {cond:.2e} (>1e10) for {self.get_name()}. '
                f'Consider increasing alpha for better numerical stability.',
                stacklevel=2,
            )

        t6 = time.perf_counter_ns()
        self._gram_chol_cpu = cho_factor(gram)
        t7 = time.perf_counter_ns()
        logger.debug('CPU cholesky: %.3f ms', (t7 - t6) / 1e6)

        try:
            import cupy as cp
            gram_gpu = cp.asarray(gram)
            self._gram_chol_gpu = cp.linalg.cholesky(gram_gpu)
            t8 = time.perf_counter_ns()
            logger.debug('GPU cholesky: %.3f ms', (t8 - t7) / 1e6)
        except (ImportError, Exception) as exc:
            self._gram_chol_gpu = None
            logger.debug('GPU cholesky unavailable: %s', exc)

        try:
            preds_train = np.asarray(self._model.predict(features)).ravel()
        except Exception:
            preds_train = np.zeros(n, dtype=np.float64)

        residuals = targets.astype(np.float64) - preds_train.astype(np.float64)
        rss = float(np.sum(residuals ** 2))
        self._sigma_hat_sq = rss / max(n - p, 1)

        self.is_trained = True
        logger.debug(
            'Trained %s on %d samples (p=%d), sigma_hat_sq=%.6f',
            self.get_name(), n, p, self._sigma_hat_sq,
        )

    def predict(
        self,
        features: np.ndarray,
        smiles: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict with cuML Ridge and compute leverage-based uncertainty.

        Uncertainty is computed as ``sqrt(sigma_hat_sq * max(1 + h_ii, 0))``
        where ``h_ii = x^T (X^T X + alpha * I)^{-1} x`` is the leverage.

        Args:
            features: Feature matrix (n_samples, n_features).
            smiles: Unused; present for interface compatibility.

        Returns:
            Tuple of (predictions, uncertainties) as numpy float64 arrays.

        Raises:
            LearnerError: If the learner has not been trained or prediction fails.
        """
        _validate_predict_inputs(self.is_trained, self.get_name())

        t0 = time.perf_counter_ns()
        features, _, _ = _preprocess_features(
            features,
            valid_feature_mask=self._valid_feature_mask,
            remove_zero_variance=self.remove_zero_variance,
            is_training=False,
            feature_type=self._feature_type,
            imputer=self._feature_imputer,
        )
        t1 = time.perf_counter_ns()
        logger.debug('predict preprocess: %.3f ms', (t1 - t0) / 1e6)

        if self._feature_scaler is not None:
            features = self._feature_scaler.transform(features)

        try:
            t2 = time.perf_counter_ns()
            raw_preds = np.asarray(self._model.predict(features)).ravel()
            t3 = time.perf_counter_ns()
            logger.debug('cuml ridge predict: %.3f ms', (t3 - t2) / 1e6)
        except Exception as exc:
            raise LearnerError(
                f'cuML Ridge prediction failed for {self.get_name()} '
                f'on {features.shape[0]} samples: {exc}.'
            ) from exc

        predictions = raw_preds.astype(np.float64)

        t4 = time.perf_counter_ns()
        X = features.astype(np.float64)
        leverages = self._compute_leverages(X)

        variance = self._sigma_hat_sq * np.maximum(1.0 + leverages, 0.0)
        uncertainty = np.sqrt(variance)
        t5 = time.perf_counter_ns()
        logger.debug('leverage uncertainty: %.3f ms', (t5 - t4) / 1e6)

        return predictions, uncertainty

    def _compute_leverages(self, X: np.ndarray) -> np.ndarray:
        """Compute leverage scores h_ii, GPU-first with CPU fallback.

        Tries CuPy GPU cho_solve when ``_gram_chol_gpu`` is available.
        Falls back to CPU scipy cho_solve on GPU OOM, with chunked
        DTRSM for n > ``_LEVERAGE_CHUNK_SIZE`` rows.
        """
        if self._gram_chol_gpu is not None:
            try:
                import cupy as cp
                from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular

                t0 = time.perf_counter_ns()
                L = self._gram_chol_gpu
                X_gpu = cp.asarray(X)
                tmp = cp_solve_triangular(L, X_gpu.T, lower=True)
                leverages = cp.asnumpy(cp.einsum('ij,ij->j', tmp, tmp))
                t1 = time.perf_counter_ns()
                logger.debug('GPU leverage: %.3f ms', (t1 - t0) / 1e6)
                return leverages
            except (MemoryError, Exception) as exc:
                if isinstance(exc, MemoryError) or 'out of memory' in str(exc).lower():
                    logger.warning(
                        'GPU OOM during leverage for %s; falling back to CPU.',
                        self.get_name(),
                    )
                else:
                    raise LearnerError(
                        f'GPU leverage computation failed for {self.get_name()}: {exc}.'
                    ) from exc

        try:
            t0 = time.perf_counter_ns()
            n = X.shape[0]
            if n > _LEVERAGE_CHUNK_SIZE:
                leverages = self._compute_leverages_chunked(X)
            else:
                solved = cho_solve(self._gram_chol_cpu, X.T)
                leverages = np.einsum('ij,ji->i', X, solved)
            t1 = time.perf_counter_ns()
            logger.debug('CPU leverage: %.3f ms', (t1 - t0) / 1e6)
            return leverages
        except Exception as exc:
            raise LearnerError(
                f'Leverage computation failed for {self.get_name()}: {exc}.'
            ) from exc

    def _compute_leverages_chunked(self, X: np.ndarray) -> np.ndarray:
        """Compute leverages in chunks for large sample counts."""
        n = X.shape[0]
        leverages = np.empty(n, dtype=np.float64)
        for start in range(0, n, _LEVERAGE_CHUNK_SIZE):
            end = min(start + _LEVERAGE_CHUNK_SIZE, n)
            chunk = X[start:end]
            solved = cho_solve(self._gram_chol_cpu, chunk.T)
            leverages[start:end] = np.einsum('ij,ji->i', chunk, solved)
        return leverages

    def get_name(self) -> str:
        """Return the learner identifier.

        Returns:
            'ridge_cuml'
        """
        return 'ridge_cuml'

    def supports_uncertainty(self) -> bool:
        """Return True — RidgeCumlLearner provides leverage-based uncertainty.

        Returns:
            True
        """
        return True

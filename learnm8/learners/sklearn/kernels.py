"""Custom kernels for Gaussian Process learners.

Provides TanimotoKernel (Jaccard similarity) for binary molecular fingerprints.
"""

import numpy as np
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel


class TanimotoKernel(Kernel):
    """Tanimoto (Jaccard) similarity kernel for binary feature vectors.

    Computes k(x, z) = signal_variance * <x, z> / (||x||^2 + ||z||^2 - <x, z>)

    Designed for binary molecular fingerprints (e.g., Morgan, ECFP, MACCS).
    Positive semi-definite for non-negative inputs. Single trainable
    hyperparameter (signal_variance) optimized via L-BFGS-B in GPR.
    """

    def __init__(
        self,
        signal_variance: float = 1.0,
        signal_variance_bounds: tuple[float, float] | str = (1e-5, 1e5),
    ):
        self.signal_variance = signal_variance
        self.signal_variance_bounds = signal_variance_bounds

    @property
    def hyperparameter_signal_variance(self) -> Hyperparameter:
        return Hyperparameter('signal_variance', 'numeric', self.signal_variance_bounds)

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if eval_gradient and Y is not None:
            raise ValueError('eval_gradient is not supported when Y is not None.')

        if Y is None:
            Y = X

        XY = X @ Y.T
        norm_X = np.sum(X**2, axis=1)
        norm_Y = np.sum(Y**2, axis=1)
        denominator = norm_X[:, None] + norm_Y[None, :] - XY
        T = XY / np.maximum(denominator, 1e-10)
        K = self.signal_variance * T

        if not eval_gradient:
            return K

        if not self.hyperparameter_signal_variance.fixed:
            return K, K[:, :, np.newaxis]
        else:
            return K, np.empty((X.shape[0], X.shape[0], 0))

    def diag(self, X: np.ndarray) -> np.ndarray:
        norm_sq = np.sum(X**2, axis=1)
        result = self.signal_variance * norm_sq / np.maximum(norm_sq, 1e-10)
        result[norm_sq == 0] = 0.0
        return result

    def is_stationary(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f'TanimotoKernel(signal_variance={self.signal_variance:.3g})'

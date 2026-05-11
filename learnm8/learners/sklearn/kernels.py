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
        # Feature 019 FR-014: zero-zero pairs (denominator == 0) get T = 1
        # (canonical k(x,x)=1 identity, removable singularity from |x|→0+).
        # `denom_safe` swaps the zero entries to 1.0 BEFORE the divide so the
        # divide path never sees a zero — avoids the np.where double-eval
        # that would materialise a full divide-result over the entire matrix.
        denom_safe = np.where(denominator > 0, denominator, 1.0)
        T = np.where(denominator > 0, XY / denom_safe, 1.0)
        K = self.signal_variance * T

        if not eval_gradient:
            return K

        if not self.hyperparameter_signal_variance.fixed:
            return K, K[:, :, np.newaxis]
        else:
            return K, np.empty((X.shape[0], X.shape[0], 0))

    def diag(self, X: np.ndarray) -> np.ndarray:
        # Feature 019 FR-014: canonical k(x,x)=1 identity, scaled by signal_variance.
        # Returns signal_variance for every row including zero vectors (removable
        # singularity); the previous ``zero-row → 0`` branch caused LinAlgError
        # when sklearn computed predictive variance on zero-fingerprint rows.
        return np.full(X.shape[0], self.signal_variance, dtype=np.float64)

    def is_stationary(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f'TanimotoKernel(signal_variance={self.signal_variance:.3g})'

import gc
import logging
import warnings

import numpy as np
import torch

from learnm8.core.interfaces import Learner
from learnm8.exceptions import ConfigurationError, ConvergenceWarning, LearnerError
from learnm8.learners.base import _preprocess_features

logger = logging.getLogger(__name__)


class GPyTorchGPLearner(Learner):
    """Gaussian Process learner using GPyTorch with optional GPU acceleration.

    Supports Tanimoto kernel (for binary/fingerprint features) and RBF kernel,
    with automatic selection based on feature type. Uses LOVE approximation for
    fast predictions on large pools.

    Args:
        kernel: Kernel type: 'auto', 'tanimoto', 'rbf', or a gpytorch.kernels.Kernel instance.
            'auto' selects tanimoto for binary features and rbf otherwise.
        alpha: Initial noise value for GaussianLikelihood.
        max_train_size: Maximum allowed training set size.
        n_iterations: Training iterations override. None = auto (100 if n<1000, else 200).
        device: Compute device: 'auto', 'cpu', 'cuda', or 'cuda:N'.
        random_state: Random seed for reproducibility.
        remove_zero_variance: Remove zero-variance features during preprocessing.
        enable_aggressive_gc: Run gc.collect() and empty_cache() after CUDA operations.
        predict_chunk_size: Chunk size for batched prediction to bound GPU memory.
    """

    def __init__(
        self,
        kernel: str = "auto",
        alpha: float = 0.1,
        max_train_size: int = 10000,
        n_iterations: int | None = None,
        device: str = "auto",
        random_state: int = 42,
        remove_zero_variance: bool = True,
        enable_aggressive_gc: bool = True,
        predict_chunk_size: int = 10000,
    ) -> None:
        try:
            import gpytorch  # noqa: F401
        except ImportError as err:
            raise ConfigurationError(
                "GPyTorch required for 'gpu_gp' learner. Install: pip install gpytorch"
            ) from err

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.kernel = kernel
        self.alpha = alpha
        self.max_train_size = max_train_size
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.remove_zero_variance = remove_zero_variance
        self.enable_aggressive_gc = enable_aggressive_gc
        self.predict_chunk_size = predict_chunk_size

        self.is_trained = False
        self._model = None
        self._likelihood = None
        self._valid_feature_mask: np.ndarray | None = None
        self._feature_imputer = None
        self._feature_scaler = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._kernel_name: str = ""
        self._feature_type = 'binary'

        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    def _resolve_kernel(self, features: np.ndarray):
        import gpytorch

        is_binary = self._feature_type == 'binary'
        kernel_spec = self.kernel

        if kernel_spec == "auto":
            kernel_spec = "tanimoto" if is_binary else "rbf"

        if kernel_spec == "tanimoto":
            try:
                from gauche.kernels.fingerprint_kernels.tanimoto_kernel import (
                    TanimotoKernel,
                )
            except ImportError as err:
                raise ConfigurationError(
                    "GAUCHE required for Tanimoto kernel with gpu_gp. Install: pip install gauche"
                ) from err
            return gpytorch.kernels.ScaleKernel(TanimotoKernel()), "tanimoto"

        if kernel_spec == "rbf":
            return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), "rbf"

        if isinstance(kernel_spec, gpytorch.kernels.Kernel):
            if isinstance(kernel_spec, gpytorch.kernels.ScaleKernel):
                return kernel_spec, type(kernel_spec).__name__
            return gpytorch.kernels.ScaleKernel(kernel_spec), type(kernel_spec).__name__

        raise ConfigurationError(f"Unknown kernel: {kernel_spec!r}")

    def _make_model(self, train_x, train_y, likelihood, covar_module):
        import gpytorch

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, covar_module):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = covar_module

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return ExactGPModel(train_x, train_y, likelihood, covar_module)

    def _train_on_device(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        covar_module,
        target_device: torch.device,
    ):
        import gpytorch
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.mlls import ExactMarginalLogLikelihood

        train_x = train_x.to(target_device)
        train_y = train_y.to(target_device)

        noise_constraint = gpytorch.constraints.GreaterThan(1e-6)
        likelihood = GaussianLikelihood(noise_constraint=noise_constraint).double().to(target_device)
        with torch.no_grad():
            likelihood.noise = torch.tensor(self.alpha, dtype=torch.float64, device=target_device)

        covar_module = covar_module.double().to(target_device)
        model = self._make_model(train_x, train_y, likelihood, covar_module)
        model = model.double().to(target_device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()), lr=0.1
        )
        mll = ExactMarginalLogLikelihood(likelihood, model)

        n = train_x.shape[0]
        n_iters = self.n_iterations if self.n_iterations is not None else (100 if n < 1000 else 200)

        initial_loss: float | None = None
        final_loss: float = 0.0

        for i in range(n_iters):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            if torch.isnan(loss):
                raise LearnerError(
                    "GP training produced NaN loss. Try a different kernel or check your features."
                )
            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                logger.debug("GP training iter %d/%d — loss: %.4f", i, n_iters, final_loss)

        if initial_loss is not None and final_loss > initial_loss:
            warnings.warn(
                ConvergenceWarning(
                    f"GP training did not converge: final_loss ({final_loss:.4f}) > "
                    f"initial_loss ({initial_loss:.4f}). Consider more iterations or a different kernel."
                ),
                stacklevel=2,
            )

        model.eval()
        likelihood.eval()
        return model, likelihood

    def train(
        self, features: np.ndarray, targets: np.ndarray, smiles: list[str] | None = None
    ) -> None:
        """Train the GP model on the given features and targets.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            targets: Target values of shape (n_samples,).
            smiles: Unused; present for interface compatibility.

        Raises:
            LearnerError: If training set exceeds max_train_size, features are all zero-variance,
                or training produces NaN loss.
            ConfigurationError: If required dependencies are missing.
        """
        n = features.shape[0]
        self._n_train = n
        if n > self.max_train_size:
            raise LearnerError(
                f"Training set size {n} exceeds max_train_size={self.max_train_size}. "
                "Reduce the training set or increase max_train_size."
            )
        if 5000 < n <= self.max_train_size:
            logger.warning(
                "GP training on %d samples has O(n^2) memory scaling. Consider max_train_size.", n
            )

        covar_module, kernel_name = self._resolve_kernel(features)

        features_proc, mask, self._feature_imputer = _preprocess_features(
            features, remove_zero_variance=self.remove_zero_variance, is_training=True,
            feature_type=self._feature_type,
        )
        if features_proc.shape[1] == 0:
            raise LearnerError("No valid features remain after zero-variance removal.")

        self._valid_feature_mask = mask
        self._kernel_name = kernel_name

        if kernel_name == 'rbf':
            from sklearn.preprocessing import StandardScaler
            self._feature_scaler = StandardScaler()
            features_proc = self._feature_scaler.fit_transform(features_proc)
        else:
            self._feature_scaler = None

        self._target_mean = float(np.mean(targets))
        self._target_std = float(max(np.std(targets), 1e-10))
        targets_std = (targets - self._target_mean) / self._target_std

        train_x = torch.tensor(features_proc, dtype=torch.float64)
        train_y = torch.tensor(targets_std, dtype=torch.float64)

        try:
            model, likelihood = self._train_on_device(
                train_x, train_y, covar_module, self._device
            )
        except RuntimeError as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                "CUDA out of memory" in str(exc)
            )
            if not oom:
                raise
            logger.warning(
                "GPU OOM during GP training — falling back to CPU. Error: %s", exc
            )
            if self._model is not None:
                del self._model
            if self._likelihood is not None:
                del self._likelihood
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            cpu_device = torch.device("cpu")
            covar_module_cpu, _ = self._resolve_kernel(features)
            model, likelihood = self._train_on_device(
                train_x.cpu(), train_y.cpu(), covar_module_cpu, cpu_device
            )
            self._device = cpu_device

        self._model = model
        self._likelihood = likelihood
        self.is_trained = True

        if self.enable_aggressive_gc and self._device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def predict(
        self,
        features: np.ndarray,
        smiles: list[str] | None = None,
        *,
        compute_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict mean and (optionally) standard deviation.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            smiles: Unused; present for interface compatibility.
            compute_uncertainty: Keyword-only (feature 023). When False, only
                ``posterior.mean`` is accessed — ``posterior.variance`` and
                the surrounding ``gpytorch.settings.fast_pred_var()`` context
                are skipped (FR-004 amendment / D3). The Cholesky-backed
                variance solve is the dominant cost.

        Returns:
            Tuple of (means, stds) as float64 arrays of shape (n_samples,).
            ``stds`` is ``None`` if ``compute_uncertainty=False``.

        Raises:
            LearnerError: If model is not trained or GPU OOM occurs during prediction.
        """
        del smiles
        if not self.is_trained:
            raise LearnerError("GPyTorchGPLearner must be trained before predict(). Call train() first.")

        features_proc, _, _ = _preprocess_features(
            features,
            valid_feature_mask=self._valid_feature_mask,
            remove_zero_variance=self.remove_zero_variance,
            is_training=False,
            feature_type=self._feature_type,
            imputer=self._feature_imputer,
        )

        if self._feature_scaler is not None:
            features_proc = self._feature_scaler.transform(features_proc)

        import gpytorch

        all_means = []
        all_stds: list = []
        n = features_proc.shape[0]

        try:
            for start in range(0, n, self.predict_chunk_size):
                chunk = features_proc[start : start + self.predict_chunk_size]
                chunk_tensor = torch.tensor(chunk, dtype=torch.float64, device=self._device)
                if compute_uncertainty:
                    with (
                        torch.no_grad(),
                        gpytorch.settings.fast_pred_var(),
                        gpytorch.settings.max_root_decomposition_size(20),
                    ):
                        preds = self._likelihood(self._model(chunk_tensor))
                        all_means.append(preds.mean.cpu().numpy())
                        all_stds.append(preds.variance.sqrt().cpu().numpy())
                else:
                    # Skip path: posterior.mean only — no variance solve.
                    with torch.no_grad():
                        posterior = self._model(chunk_tensor)
                        all_means.append(posterior.mean.cpu().numpy())
        except RuntimeError as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                "CUDA out of memory" in str(exc)
            )
            if not oom:
                raise
            raise LearnerError(
                "GPU out of memory during prediction. Set device='cpu' or reduce pool size."
            ) from exc

        means = np.concatenate(all_means).astype(np.float64)
        means = means * self._target_std + self._target_mean

        if self.enable_aggressive_gc and self._device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        if not compute_uncertainty:
            return means, None

        stds = np.concatenate(all_stds).astype(np.float64) * self._target_std
        return means, stds

    def get_name(self) -> str:
        """Return a descriptive name for this learner instance."""
        if not self.is_trained or self._likelihood is None:
            return "GPyTorchGP(untrained)"
        noise = self._likelihood.noise.item()
        return f"GPyTorchGP({self._kernel_name},noise={noise:.4f})"

    def supports_uncertainty(self) -> bool:
        """Return True; GP models always provide uncertainty estimates."""
        return True

    def memory_profile(self, n_features: int) -> dict[str, int | float]:
        n_train = getattr(self, '_n_train', 0)
        return {
            'bytes_per_sample': n_features * 4 + n_train * 4 * 2,
            'working_multiplier': 1.0,
            'fixed_overhead': 0,
        }

    def requires_smiles(self) -> bool:
        """Return False; this learner operates on feature arrays."""
        return False

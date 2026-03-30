import gc
import logging
import warnings

import numpy as np
import torch

from learnm8.core.interfaces import Learner
from learnm8.exceptions import ConfigurationError, ConvergenceWarning, LearnerError
from learnm8.learners.base import _preprocess_features

logger = logging.getLogger(__name__)


class SVGPLearner(Learner):
    """Stochastic Variational Gaussian Process learner using GPyTorch.

    Scales GP regression beyond 10K training points via inducing-point
    approximation with minibatch stochastic gradient descent.  Supports the
    same Tanimoto / RBF kernel auto-selection as ``GPyTorchGPLearner``.

    Key differences from ExactGP (``gpu_gp``):
    * Training cost is O(M²B) per minibatch (M = inducing points, B = batch).
    * Memory is O(M²), independent of training set size.
    * Inducing points are fixed at a random training subset (not gradient-optimized)
      to keep binary fingerprints valid for the Tanimoto kernel.
    * Variational posterior systematically underestimates uncertainty.

    Args:
        kernel: 'auto', 'tanimoto', 'rbf', or a gpytorch.kernels.Kernel instance.
        alpha: Initial noise value for GaussianLikelihood.
        n_inducing: Number of inducing points (clamped to n_train if larger).
        batch_size: Minibatch size for variational training.
        n_epochs: Maximum training epochs.
        learning_rate: Adam learning rate for kernel/likelihood parameters.
        early_stopping_patience: Stop after this many epochs with no ELBO improvement.
        device: 'auto', 'cpu', 'cuda', or 'cuda:N'.
        random_state: Random seed.
        remove_zero_variance: Remove zero-variance features during preprocessing.
        enable_aggressive_gc: Run gc.collect() + empty_cache() after CUDA ops.
        predict_chunk_size: Chunk size for batched prediction.
    """

    def __init__(
        self,
        kernel: str = "auto",
        alpha: float = 0.1,
        n_inducing: int = 512,
        batch_size: int = 256,
        n_epochs: int = 50,
        learning_rate: float = 0.01,
        early_stopping_patience: int = 10,
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
                "GPyTorch required for 'svgp' learner. Install: pip install gpytorch"
            ) from err

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.kernel = kernel
        self.alpha = alpha
        self.n_inducing = n_inducing
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.remove_zero_variance = remove_zero_variance
        self.enable_aggressive_gc = enable_aggressive_gc
        self.predict_chunk_size = predict_chunk_size

        self.is_trained = False
        self._model = None
        self._likelihood = None
        self._valid_feature_mask: np.ndarray | None = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._kernel_name: str = ""
        self._effective_m: int = 0

        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    def _resolve_kernel(self, features: np.ndarray):
        import gpytorch

        is_binary = bool(np.all((features == 0) | (features == 1)))
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
                    "GAUCHE required for Tanimoto kernel with svgp. Install: pip install gauche"
                ) from err
            return gpytorch.kernels.ScaleKernel(TanimotoKernel()), "tanimoto"

        if kernel_spec == "rbf":
            return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), "rbf"

        if isinstance(kernel_spec, gpytorch.kernels.Kernel):
            if isinstance(kernel_spec, gpytorch.kernels.ScaleKernel):
                return kernel_spec, type(kernel_spec).__name__
            return gpytorch.kernels.ScaleKernel(kernel_spec), type(kernel_spec).__name__

        raise ConfigurationError(f"Unknown kernel: {kernel_spec!r}")

    def _make_model(self, inducing_points, covar_module):
        import gpytorch

        class SVGPModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points, covar_module):
                variational_distribution = (
                    gpytorch.variational.CholeskyVariationalDistribution(
                        inducing_points.size(0)
                    )
                )
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=False,
                )
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = covar_module

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return SVGPModel(inducing_points, covar_module)

    def _train_on_device(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        covar_module,
        target_device: torch.device,
    ):
        import gpytorch
        from torch.utils.data import DataLoader, TensorDataset

        n_train = train_x.shape[0]
        effective_m = min(self.n_inducing, n_train)
        self._effective_m = effective_m

        if effective_m < self.n_inducing:
            logger.warning(
                "Training size (%d) < n_inducing (%d). Using all training points "
                "as inducing points — SVGP adds overhead without approximation benefit.",
                n_train,
                self.n_inducing,
            )

        rng = torch.Generator()
        rng.manual_seed(self.random_state)
        indices = torch.randperm(n_train, generator=rng)[:effective_m]
        inducing_points = train_x[indices].clone().to(dtype=torch.float64, device=target_device)

        covar_module = covar_module.double().to(target_device)
        model = self._make_model(inducing_points, covar_module)
        model = model.double().to(target_device)

        noise_constraint = gpytorch.constraints.GreaterThan(1e-6)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint
        ).double().to(target_device)
        with torch.no_grad():
            likelihood.noise = torch.tensor(
                self.alpha, dtype=torch.float64, device=target_device
            )

        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
            ],
            lr=self.learning_rate,
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_train)

        train_x_dev = train_x.to(target_device)
        train_y_dev = train_y.to(target_device)
        dataset = TensorDataset(train_x_dev, train_y_dev)
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, n_train),
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )

        model.train()
        likelihood.train()

        ema_loss: float | None = None
        best_ema = float("inf")
        patience_counter = 0
        initial_loss: float | None = None
        final_loss: float = 0.0
        consecutive_nan = 0

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)

                if torch.isnan(loss):
                    consecutive_nan += 1
                    if consecutive_nan >= 3:
                        raise LearnerError(
                            f"SVGP training produced NaN loss for {consecutive_nan} "
                            f"consecutive batches at epoch {epoch}."
                        )
                    continue

                consecutive_nan = 0
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                continue

            avg_loss = epoch_loss / n_batches
            if initial_loss is None:
                initial_loss = avg_loss
            final_loss = avg_loss

            if epoch % 10 == 0:
                logger.debug(
                    "SVGP epoch %d/%d — ELBO: %.4f", epoch, self.n_epochs, avg_loss
                )

            if ema_loss is None:
                ema_loss = avg_loss
            else:
                ema_loss = 0.99 * ema_loss + 0.01 * avg_loss

            if ema_loss < best_ema - 1e-4:
                best_ema = ema_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                logger.debug("SVGP early stopping at epoch %d (patience=%d)", epoch, self.early_stopping_patience)
                break

        if initial_loss is not None and final_loss > initial_loss * 1.1:
            warnings.warn(
                ConvergenceWarning(
                    f"SVGP training loss increased: {initial_loss:.4f} -> {final_loss:.4f}. "
                    "Consider more epochs or a different kernel."
                ),
                stacklevel=2,
            )

        model.eval()
        likelihood.eval()
        return model, likelihood

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        smiles: list[str] | None = None,
    ) -> None:
        if features.ndim != 2:
            raise LearnerError(f"Features must be 2D, got shape {features.shape}")
        if len(features) != len(targets):
            raise LearnerError(
                f"Feature/target length mismatch: {len(features)} vs {len(targets)}"
            )

        covar_module, kernel_name = self._resolve_kernel(features)

        features_proc, mask = _preprocess_features(
            features, remove_zero_variance=self.remove_zero_variance, is_training=True
        )
        if features_proc.shape[1] == 0:
            raise LearnerError("No valid features remain after zero-variance removal.")

        self._valid_feature_mask = mask
        self._kernel_name = kernel_name
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
                "GPU OOM during SVGP training — falling back to CPU. Error: %s", exc
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
        self, features: np.ndarray, smiles: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise LearnerError(
                "SVGPLearner must be trained before predict(). Call train() first."
            )

        features_proc, _ = _preprocess_features(
            features,
            valid_feature_mask=self._valid_feature_mask,
            remove_zero_variance=self.remove_zero_variance,
            is_training=False,
        )

        all_means = []
        all_stds = []
        n = features_proc.shape[0]

        try:
            for start in range(0, n, self.predict_chunk_size):
                chunk = features_proc[start : start + self.predict_chunk_size]
                chunk_tensor = torch.tensor(
                    chunk, dtype=torch.float64, device=self._device
                )
                with torch.no_grad():
                    preds = self._likelihood(self._model(chunk_tensor))
                    all_means.append(preds.mean.cpu().numpy())
                    all_stds.append(preds.variance.sqrt().cpu().numpy())
        except RuntimeError as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                "CUDA out of memory" in str(exc)
            )
            if not oom:
                raise
            raise LearnerError(
                "GPU out of memory during SVGP prediction. "
                "Set device='cpu' or reduce pool size."
            ) from exc

        means = np.concatenate(all_means).astype(np.float64)
        stds = np.concatenate(all_stds).astype(np.float64)

        means = means * self._target_std + self._target_mean
        stds = stds * self._target_std

        if self.enable_aggressive_gc and self._device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        return means, stds

    def get_name(self) -> str:
        if not self.is_trained or self._likelihood is None:
            return "SVGP(untrained)"
        noise = self._likelihood.noise.item()
        return f"SVGP({self._kernel_name},M={self._effective_m},noise={noise:.4f})"

    def supports_uncertainty(self) -> bool:
        return True

    def requires_smiles(self) -> bool:
        return False

"""Fastprop learner for deep learning on molecular features.

This module provides a PyTorch Lightning-based feedforward neural network
learner using the fastprop library. It integrates seamlessly with LearnM8's
featurizer-agnostic architecture, accepting any pre-computed feature matrix.
"""

import logging
import warnings
from typing import Tuple, Optional
import numpy as np

from learnm8.core.interfaces import Learner

try:
    import torch
    from torch.utils.data import TensorDataset
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from fastprop.model import fastprop
    from fastprop.data import fastpropDataLoader, fastpropDataset, standard_scale
    FASTPROP_AVAILABLE = True
except ImportError:
    torch = None
    TensorDataset = None
    Trainer = None
    EarlyStopping = None
    fastprop = None
    fastpropDataLoader = None
    fastpropDataset = None
    standard_scale = None
    FASTPROP_AVAILABLE = False

logger = logging.getLogger(__name__)


class FastpropLearner(Learner):
    """Fastprop feedforward neural network with PyTorch Lightning.

    This learner provides deep learning capabilities using the fastprop library,
    which implements a configurable feedforward neural network with PyTorch Lightning.
    It accepts pre-computed feature matrices and handles all scaling internally.

    Architecture:
        - Accepts ANY numerical feature matrix (morgan, maccs, descriptors, etc.)
        - Uses fastprop's standard_scale() for feature/target normalization
        - Trains via PyTorch Lightning Trainer with early stopping
        - Supports CPU/GPU with automatic device detection

    Key Features:
        - Featurizer-agnostic (works with all LearnM8 featurizers)
        - Automatic feature scaling and denormalization
        - Early stopping prevents overfitting
        - Input clamping (winsorization) for robustness
        - No uncertainty estimates (single model, no ensemble)
    """

    def __init__(self,
                 fnn_layers: int = 2,
                 hidden_size: int = 1800,
                 max_epochs: int = 50,
                 learning_rate: float = 0.0001,
                 batch_size: int = 32,
                 clamp_input: bool = True,
                 early_stopping_patience: int = 5,
                 random_state: int = 42,
                 device: str = 'auto',
                 enable_aggressive_gc: bool = True):
        """Initialize Fastprop learner with conservative defaults.

        Args:
            fnn_layers: Number of hidden layers (0=linear, 2=standard, 3+=deep)
            hidden_size: Hidden layer size (1800 is fastprop's recommended)
            max_epochs: Maximum training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training and prediction
            clamp_input: Apply winsorization to inputs (recommended)
            early_stopping_patience: Patience for early stopping
            random_state: Random seed for reproducibility
            device: Device for computation ('auto', 'cpu', 'cuda')
            enable_aggressive_gc: Enable automatic GPU memory cleanup after
                training and prediction. Recommended for active learning.
                Default: True.

        Raises:
            ImportError: If fastprop or PyTorch Lightning not available
        """
        if not FASTPROP_AVAILABLE:
            raise ImportError(
                "Fastprop requires: pip install fastprop pytorch-lightning\n"
                "These dependencies are optional for LearnM8."
            )

        self.fnn_layers = fnn_layers
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.clamp_input = clamp_input
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.enable_aggressive_gc = enable_aggressive_gc

        self.model = None
        self.trainer = None
        self.is_trained = False

        self.feature_means = None
        self.feature_vars = None
        self.target_means = None
        self.target_vars = None

        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)

        logger.info(f"Initialized FastpropLearner on device: {self.device}")

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train Fastprop model using PyTorch Lightning.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)

        Raises:
            ValueError: If input shapes invalid
            RuntimeError: If training fails
        """
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Features and targets must have same length: "
                f"{features.shape[0]} vs {targets.shape[0]}"
            )

        if features.shape[0] == 0:
            raise ValueError("Cannot train on empty dataset")

        logger.debug(
            f"Training {self.get_name()} on features shape: {features.shape}, "
            f"targets shape: {targets.shape}"
        )

        # Check if root logger is at DEBUG level to enable detailed logging
        root_level = logging.getLogger().level
        if root_level <= logging.DEBUG:
            # Enable detailed logging for Fastprop/Lightning
            logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)
            logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)
            logging.getLogger("fastprop").setLevel(logging.DEBUG)
            logger.info("DEBUG logging enabled for Fastprop and PyTorch Lightning")
        else:
            # Default: suppress Lightning logging
            logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
            logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

        try:
            X = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

            X, self.feature_means, self.feature_vars = standard_scale(X)
            y, self.target_means, self.target_vars = standard_scale(y)

            dataset = fastpropDataset(X, y)
            train_dataloader = fastpropDataLoader(
                dataset,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=0,
                persistent_workers=False
            )

            self.model = fastprop(
                input_size=features.shape[1],
                fnn_layers=self.fnn_layers,
                hidden_size=self.hidden_size,
                clamp_input=self.clamp_input,
                feature_means=self.feature_means,
                feature_vars=self.feature_vars,
                target_means=self.target_means,
                target_vars=self.target_vars,
                learning_rate=self.learning_rate,
            )

            callbacks = [
                EarlyStopping(
                    monitor='train_mse_scaled_loss',
                    patience=self.early_stopping_patience,
                    mode='min',
                    verbose=False
                )
            ]

            # Enable progress bar and model summary if DEBUG logging is active
            enable_verbose = logging.getLogger().level <= logging.DEBUG

            self.trainer = Trainer(
                max_epochs=self.max_epochs,
                enable_progress_bar=enable_verbose,
                enable_model_summary=False,
                callbacks=callbacks,
                accelerator='auto',
                devices=1,
                logger=False
            )

            logger.info(f"Training {self.get_name()} for up to {self.max_epochs} epochs on {len(features)} samples")
            self.trainer.fit(self.model, train_dataloader)
            logger.info(f"{self.get_name()} training completed successfully")

            self.is_trained = True
            logger.info(f"Trained {self.get_name()} on {len(features)} samples")

            self._cleanup_gpu_memory("after training")

        except Exception as e:
            logger.error(f"Failed to train {self.get_name()}: {e}")
            raise RuntimeError(f"Training failed: {e}") from e

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict using trained Fastprop model.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (predictions, uncertainties).
            uncertainties is None for single model.

        Raises:
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        logger.debug(f"Predicting with {self.get_name()} on {len(features)} samples")

        try:
            X = torch.tensor(features, dtype=torch.float32)

            X = standard_scale(X, self.feature_means, self.feature_vars)

            dataset = TensorDataset(X)
            predict_dataloader = fastpropDataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=0,
                persistent_workers=False
            )

            logger.info(f"Starting {self.get_name()} prediction on {len(features)} samples")
            predictions = self.trainer.predict(self.model, predict_dataloader)
            logger.info(f"{self.get_name()} prediction completed, processing results")
            predictions = torch.cat(predictions).cpu().numpy().squeeze()

            if predictions.ndim == 0:
                predictions = np.array([predictions.item()])
            elif np.isscalar(predictions):
                predictions = np.array([predictions])

            self._cleanup_gpu_memory("after prediction")

            logger.info(f"Predicted {len(predictions)} samples with {self.get_name()}")

            return predictions, None

        except Exception as e:
            logger.error(f"Failed to predict with {self.get_name()}: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def get_name(self) -> str:
        """Return descriptive name for this learner."""
        return f"Fastprop(layers={self.fnn_layers},hidden={self.hidden_size})"

    def supports_uncertainty(self) -> bool:
        """Return True if this learner can provide uncertainty estimates.

        Single Fastprop model does not provide uncertainty.
        For uncertainty, use FastpropEnsemble (future implementation).
        """
        return False

    def _cleanup_gpu_memory(self, context: str = "") -> None:
        """Force garbage collection and clear GPU cache if enabled.

        This method performs two cleanup operations:
        1. torch.cuda.empty_cache() - Releases cached GPU memory
        2. gc.collect() - Forces Python garbage collection

        This is particularly important in active learning scenarios where
        models are trained repeatedly over many cycles, which can lead to
        GPU memory accumulation from unreferenced tensors and PyTorch's
        caching allocator.

        The cleanup is a best-effort operation that won't raise exceptions
        if it fails. It only runs if enable_aggressive_gc=True.

        Args:
            context: Optional description of when cleanup is being called,
                    used for debug logging (e.g., "after training")

        Note:
            This is safe to call after predictions have been moved to CPU
            memory via .cpu().numpy(), as it only affects unreferenced
            GPU tensors and Python objects.
        """
        if not self.enable_aggressive_gc:
            return

        try:
            import gc
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if context:
                logger.debug(f"GPU memory cleanup: {context}")

        except Exception as e:
            logger.warning(f"GPU memory cleanup failed ({context}): {e}")

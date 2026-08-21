"""Fastprop learner for deep learning on molecular features.

This module provides a PyTorch Lightning-based feedforward neural network
learner using the fastprop library. It integrates seamlessly with LearnM8's
featurizer-agnostic architecture, accepting any pre-computed feature matrix.
"""

import logging
import time
import warnings

import numpy as np
import torch
from fastprop.data import fastpropDataLoader, fastpropDataset, standard_scale
from fastprop.model import fastprop
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import TensorDataset

from learnm8.core.interfaces import Learner
from learnm8.exceptions import ConfigurationError, LearnerError
from learnm8.learners.base import _cleanup_gpu_memory, _validate_and_resolve_precision

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

    Performance Configuration:
            - batch_size: Training batch size (default: 32)
            - predict_batch_size: Prediction batch size (default: None, uses 4x batch_size)
            - precision: Precision mode - 'auto', '16-mixed', '32-true' (default: 'auto')
            - pin_memory: Enable pinned memory (may not work with fastpropDataLoader, default: False)
    """

    def __init__(
        self,
        fnn_layers: int = 2,
        hidden_size: int = 300,
        max_epochs: int = 50,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        predict_batch_size: int | None = None,
        precision: str = 'auto',
        pin_memory: bool = False,
        clamp_input: bool = True,
        early_stopping_patience: int = 5,
        val_fraction: float = 0.1,
        random_state: int = 42,
        device: str = 'auto',
        enable_aggressive_gc: bool = True,
    ):
        if fnn_layers < 1:
            raise ConfigurationError(
                f'fnn_layers >= 1 required, got {fnn_layers}. '
                f'FastpropLearner requires at least 1 hidden layer.'
            )
        if not (0.0 <= val_fraction < 1.0):
            raise ConfigurationError(
                f'val_fraction must be >= 0.0 and < 1.0, got {val_fraction}'
            )
        self.fnn_layers = fnn_layers
        self.hidden_size = hidden_size
        self.val_fraction = val_fraction
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.predict_batch_size = (
            predict_batch_size if predict_batch_size is not None else (batch_size * 4)
        )
        self.precision = _validate_and_resolve_precision(precision)
        self.pin_memory = pin_memory
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

        logger.debug(f'Initialized FastpropLearner on device: {self.device}')

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
                f'Features and targets must have same length: '
                f'{features.shape[0]} vs {targets.shape[0]}'
            )

        if features.shape[0] == 0:
            raise ValueError('Cannot train on empty dataset')

        logger.debug(
            f'Training {self.get_name()} on features shape: {features.shape}, '
            f'targets shape: {targets.shape}'
        )

        warnings.filterwarnings('ignore', category=UserWarning, module='pytorch_lightning')
        warnings.filterwarnings('ignore', category=UserWarning, module='lightning')

        _pl_loggers = [
            logging.getLogger('pytorch_lightning'),
            logging.getLogger('lightning'),
            logging.getLogger('lightning.pytorch'),
        ]
        _orig_levels = [lg.level for lg in _pl_loggers]
        for lg in _pl_loggers:
            lg.setLevel(logging.WARNING)

        start_time = time.time()
        try:
            from sklearn.model_selection import train_test_split

            X = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

            n_samples = X.shape[0]
            use_val = False
            if self.val_fraction > 0:
                min_samples_for_split = max(5, int(5 / self.val_fraction))
                if n_samples >= min_samples_for_split:
                    use_val = True

            val_dataloader = None
            callbacks = []

            if use_val:
                indices = list(range(n_samples))
                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=self.val_fraction,
                    random_state=self.random_state,
                )
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                X_train, self.feature_means, self.feature_vars = standard_scale(X_train)
                X_val = standard_scale(X_val, self.feature_means, self.feature_vars)
                y_train, self.target_means, self.target_vars = standard_scale(y_train)
                y_val = standard_scale(y_val, self.target_means, self.target_vars)

                train_dataset = fastpropDataset(X_train, y_train)
                train_dataloader = fastpropDataLoader(
                    train_dataset,
                    shuffle=True,
                    batch_size=self.batch_size,
                    num_workers=0,
                    persistent_workers=False,
                )
                val_dataset = fastpropDataset(X_val, y_val)
                val_dataloader = fastpropDataLoader(
                    val_dataset,
                    shuffle=False,
                    batch_size=self.batch_size,
                    num_workers=0,
                    persistent_workers=False,
                )
                callbacks.append(
                    EarlyStopping(
                        monitor='validation_mse_scaled_loss',
                        patience=self.early_stopping_patience,
                        mode='min',
                        verbose=False,
                    )
                )
            else:
                if self.val_fraction > 0 and n_samples < min_samples_for_split:
                    logger.warning(
                        f'{self.get_name()}: {n_samples} samples is below '
                        f'min_samples_for_split={min_samples_for_split}. '
                        f'Skipping validation split; training for max_epochs={self.max_epochs}.'
                    )
                X, self.feature_means, self.feature_vars = standard_scale(X)
                y, self.target_means, self.target_vars = standard_scale(y)

                train_dataset = fastpropDataset(X, y)
                train_dataloader = fastpropDataLoader(
                    train_dataset,
                    shuffle=True,
                    batch_size=self.batch_size,
                    num_workers=0,
                    persistent_workers=False,
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

            enable_verbose = logging.getLogger().level <= logging.DEBUG

            self.trainer = Trainer(
                max_epochs=self.max_epochs,
                precision=self.precision,
                enable_progress_bar=enable_verbose,
                enable_model_summary=False,
                callbacks=callbacks,
                accelerator='auto',
                devices=1,
                logger=False,
            )

            logger.debug(
                f'Training {self.get_name()} for up to {self.max_epochs} epochs '
                f'on {len(features)} samples (precision={self.precision}, '
                f'val_split={"yes" if use_val else "no"})'
            )
            self.trainer.fit(self.model, train_dataloader, val_dataloader)

            self.is_trained = True
            train_time = time.time() - start_time
            logger.debug(f'Trained {self.get_name()} on {len(features)} samples in {train_time:.2f}s')

            _cleanup_gpu_memory(self.enable_aggressive_gc, 'after training')

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f'Failed to train {self.get_name()}: {e}')
            raise LearnerError(
                f'Training failed for {self.get_name()}: {e}. '
                f'Check that features are valid and training data is sufficient.'
            ) from None
        finally:
            for lg, lvl in zip(_pl_loggers, _orig_levels, strict=True):
                lg.setLevel(lvl)

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
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
            raise LearnerError('Model must be trained before prediction')

        logger.debug(f'Predicting with {self.get_name()} on {len(features)} samples')

        _pl_loggers = [
            logging.getLogger('pytorch_lightning'),
            logging.getLogger('lightning'),
            logging.getLogger('lightning.pytorch'),
        ]
        _orig_levels = [lg.level for lg in _pl_loggers]
        for lg in _pl_loggers:
            lg.setLevel(logging.WARNING)

        try:
            X = torch.tensor(features, dtype=torch.float32)

            dataset = TensorDataset(X)
            predict_dataloader = fastpropDataLoader(
                dataset,
                batch_size=self.predict_batch_size,
                num_workers=0,
                persistent_workers=False,
            )

            predict_trainer = Trainer(
                precision=self.precision,
                accelerator='auto',
                devices=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )

            logger.debug(
                f'Starting {self.get_name()} prediction on {len(features)} samples '
                f'(batch_size={self.predict_batch_size}, precision={self.precision})'
            )
            predictions = predict_trainer.predict(self.model, predict_dataloader)
            logger.debug(f'{self.get_name()} prediction completed, processing results')
            predictions = torch.cat(predictions).cpu().numpy().squeeze()

            if predictions.ndim == 0:
                predictions = np.array([predictions.item()])
            elif np.isscalar(predictions):
                predictions = np.array([predictions])

            _cleanup_gpu_memory(self.enable_aggressive_gc, 'after prediction')

            logger.debug(f'Predicted {len(predictions)} samples with {self.get_name()}')

            return predictions, None

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f'Failed to predict with {self.get_name()}: {e}')
            raise LearnerError(
                f'Prediction failed for {self.get_name()}: {e}. '
                f'Check that features match the training feature dimensions.'
            ) from None
        finally:
            for lg, lvl in zip(_pl_loggers, _orig_levels, strict=True):
                lg.setLevel(lvl)

    def get_name(self) -> str:
        """Return descriptive name for this learner."""
        return f'Fastprop(layers={self.fnn_layers},hidden={self.hidden_size})'

    def supports_uncertainty(self) -> bool:
        """Return True if this learner can provide uncertainty estimates.

        Single Fastprop model does not provide uncertainty.
        For uncertainty, use FastpropEnsemble (future implementation).
        """
        return False

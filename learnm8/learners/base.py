"""Base learner classes for the LearnM8 active learning framework.

This module provides the foundational base classes for sklearn and PyTorch models,
implementing dependency injection and clean interfaces as specified in the new
architecture design.
"""

import time
import logging
from abc import abstractmethod
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import numpy as np

# Core imports
from learnm8.core.interfaces import Learner
from learnm8.exceptions import LearnerError

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


ZERO_VARIANCE_THRESHOLD = 1e-10


def _preprocess_features(
    features: np.ndarray,
    valid_feature_mask: Optional[np.ndarray] = None,
    remove_zero_variance: bool = True,
    is_training: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Preprocess features by handling NaN/inf and optionally removing zero-variance columns.

    Args:
        features: Input feature matrix (n_samples, n_features)
        valid_feature_mask: Boolean mask from training (required if is_training=False)
        remove_zero_variance: Whether to remove zero-variance columns
        is_training: If True, compute mask; if False, apply existing mask

    Returns:
        Tuple of (preprocessed_features, valid_feature_mask)
        - During training: returns computed mask
        - During prediction: returns the input mask unchanged
    """
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if not remove_zero_variance:
        return features, None

    if is_training:
        variance = np.var(features, axis=0)
        valid_feature_mask = variance > ZERO_VARIANCE_THRESHOLD
        n_removed = (~valid_feature_mask).sum()
        if n_removed > 0:
            logger.debug(f"Removing {n_removed} zero-variance features "
                        f"({features.shape[1]} -> {valid_feature_mask.sum()})")
        features = features[:, valid_feature_mask]
    else:
        if valid_feature_mask is not None:
            features = features[:, valid_feature_mask]

    return features, valid_feature_mask


class SklearnLearner(Learner):
    """Base class for scikit-learn compatible models.

    This class implements the featurizer-agnostic architecture where learners
    work with numpy feature matrices, promoting clean separation of concerns
    between ML algorithms and molecular feature extraction.
    """

    def __init__(self,
                 model: BaseEstimator,
                 random_state: int = 42,
                 remove_zero_variance: bool = True):
        """Initialize sklearn learner.

        Args:
            model: Scikit-learn compatible model instance
            random_state: Random seed for reproducibility
            remove_zero_variance: Remove zero-variance features during training (default: True).
                                 NaN/inf values are always replaced with 0 regardless of this setting.
        """
        self.model = model
        self.random_state = random_state
        self.remove_zero_variance = remove_zero_variance
        self.is_trained = False
        self._valid_feature_mask = None

        if hasattr(self.model, 'random_state'):
            self.model.random_state = random_state
    
    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train sklearn model on feature matrix.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)

        Raises:
            ValueError: If input shapes invalid
            RuntimeError: If training fails
        """
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got {features.ndim}D")

        if features.shape[0] != targets.shape[0]:
            raise LearnerError(
                f"Feature and target arrays have mismatched lengths: "
                f"{features.shape[0]} features vs {targets.shape[0]} targets. "
                f"Ensure each sample has exactly one target value."
            )

        if features.shape[0] == 0:
            raise LearnerError(
                f"Cannot train {self.get_name()} on an empty dataset (0 samples). "
                f"This usually means no labeled compounds are available. "
                f"Check that batch_fraction is large enough to select at least one compound, "
                f"or increase the initial pool size."
            )

        logger.debug(f"Training {self.get_name()} on features shape: {features.shape}, targets shape: {targets.shape}")

        start_time = time.time()

        try:
            features, self._valid_feature_mask = _preprocess_features(
                features,
                remove_zero_variance=self.remove_zero_variance,
                is_training=True
            )

            self.model.fit(features, targets)
            self.is_trained = True

            train_time = time.time() - start_time
            logger.debug(f"Trained {self.get_name()} on {len(features)} samples in {train_time:.2f}s")

        except LearnerError:
            raise
        except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as e:
            logger.error(f"Failed to train {self.get_name()}: {e}")
            raise LearnerError(
                f"Training failed for {self.get_name()} with {features.shape[0]} samples: {e}. "
                f"Check that the training data is valid, the featurizer is compatible "
                f"with the learner, and there are enough labeled compounds."
            ) from e

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict on feature matrix.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (predictions, uncertainties).
            uncertainties is None for base sklearn models.

        Raises:
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise LearnerError(
                f"{self.get_name()} must be trained before prediction. "
                f"Call train() with labeled data first. If using the active learning API, "
                f"ensure at least one training cycle has completed before predicting."
            )

        logger.debug(f"Predicting with {self.get_name()} on {len(features)} samples")

        start_time = time.time()

        try:
            features, _ = _preprocess_features(
                features,
                valid_feature_mask=self._valid_feature_mask,
                remove_zero_variance=self.remove_zero_variance,
                is_training=False
            )

            predictions = self.model.predict(features)

            pred_time = time.time() - start_time
            logger.debug(f"Predicted {len(predictions)} samples with {self.get_name()} in {pred_time:.2f}s")

            return predictions, None

        except LearnerError:
            raise
        except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as e:
            logger.error(f"Failed to predict with {self.get_name()}: {e}")
            raise LearnerError(
                f"Prediction failed for {self.get_name()} on {len(features)} samples: {e}. "
                f"Check that the input features have the same shape as training features "
                f"and that the featurizer is compatible with the model."
            ) from e

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f"Sklearn{self.model.__class__.__name__}"
    
    def supports_uncertainty(self) -> bool:
        """Return True if this learner can provide uncertainty estimates."""
        # Base sklearn models don't provide uncertainty
        return False


class TorchLearner(Learner):
    """Base class for PyTorch models with GPU support.

    This class provides common PyTorch functionality including device management,
    training loops, and model persistence while following the featurizer-agnostic
    architecture where learners work with numpy feature matrices.
    """

    def __init__(self,
                 device: str = 'auto',
                 batch_size: int = 1024,
                 max_epochs: int = 100,
                 learning_rate: float = 0.001,
                 early_stopping_patience: int = 10,
                 random_state: int = 42,
                 remove_zero_variance: bool = True):
        """Initialize PyTorch learner.

        Args:
            device: Device for computation ('auto', 'cpu', 'cuda')
            batch_size: Batch size for training and prediction
            max_epochs: Maximum training epochs
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Patience for early stopping
            random_state: Random seed for reproducibility
            remove_zero_variance: Remove zero-variance features during training (default: True).
                                 NaN/inf values are always replaced with 0 regardless of this setting.
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.remove_zero_variance = remove_zero_variance

        self.model = None
        self.optimizer = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        self._valid_feature_mask = None
        self._input_dim = None

        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)

        logger.debug(f"Initialized TorchLearner on device: {self.device}")
    
    @abstractmethod
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the PyTorch model architecture.
        
        Args:
            input_size: Number of input features
            
        Returns:
            PyTorch model instance
        """
        pass
    
    def _train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train for one epoch.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        
        for batch_X, batch_y in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _validate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Validate model on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Validation loss
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            y_tensor = torch.FloatTensor(y_val).to(self.device)
            
            outputs = self.model(X_tensor).squeeze()
            criterion = nn.MSELoss()
            val_loss = criterion(outputs, y_tensor).item()
        
        return val_loss
    
    def _split_validation(self, X: np.ndarray, y: np.ndarray, 
                         val_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets.
        
        Args:
            X: Feature array
            y: Target array
            val_fraction: Fraction of data for validation
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        n_samples = len(X)
        
        # Handle edge case of very small datasets
        if n_samples <= 2:
            # For very small datasets, use all data for both training and validation
            logger.warning(f"Dataset too small ({n_samples} samples) for proper train/val split. Using all data for both.")
            return X, X, y, y
        
        n_val = max(1, int(n_samples * val_fraction))
        
        # Ensure we have at least one training sample
        if n_val >= n_samples:
            n_val = n_samples - 1
        
        # Random indices for validation
        np.random.seed(self.random_state)
        val_indices = np.random.choice(n_samples, n_val, replace=False)
        train_indices = np.setdiff1d(np.arange(n_samples), val_indices)
        
        return X[train_indices], X[val_indices], y[train_indices], y[val_indices]
    
    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train PyTorch model on feature matrix.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)

        Raises:
            ValueError: If input shapes invalid
            RuntimeError: If training fails
        """
        if features.shape[0] != targets.shape[0]:
            raise LearnerError(
                f"Feature and target arrays have mismatched lengths: "
                f"{features.shape[0]} features vs {targets.shape[0]} targets. "
                f"Ensure each sample has exactly one target value."
            )

        if features.shape[0] == 0:
            raise LearnerError(
                f"Cannot train {self.get_name()} on an empty dataset (0 samples). "
                f"This usually means no labeled compounds are available. "
                f"Check that batch_fraction is large enough to select at least one compound, "
                f"or increase the initial pool size."
            )

        start_time = time.time()

        try:
            features, self._valid_feature_mask = _preprocess_features(
                features,
                remove_zero_variance=self.remove_zero_variance,
                is_training=True
            )

            input_dim = features.shape[1]
            if self.model is None or self._input_dim != input_dim:
                self._input_dim = input_dim
                self.model = self._create_model(input_dim).to(self.device)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features)

            X_train, X_val, y_train, y_val = self._split_validation(X_scaled, targets)

            best_val_loss = float('inf')
            patience_counter = 0
            self.training_history = []

            logger.debug(f"Training {self.get_name()} for up to {self.max_epochs} epochs")

            for epoch in range(self.max_epochs):
                train_loss = self._train_epoch(X_train, y_train)
                val_loss = self._validate(X_val, y_val)

                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.debug(f"Early stopping triggered at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.debug(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            self.is_trained = True
            train_time = time.time() - start_time
            logger.debug(f"Trained {self.get_name()} on {len(features)} samples in {train_time:.2f}s")

        except LearnerError:
            raise
        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Failed to train {self.get_name()}: {e}")
            raise LearnerError(
                f"Training failed for {self.get_name()} on device '{self.device}' "
                f"with {features.shape[0]} samples: {e}. "
                f"Check that the training data is valid, the featurizer is compatible "
                f"with the learner, and there are enough labeled compounds."
            ) from e

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict on feature matrix.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (predictions, uncertainties).
            uncertainties is None for base PyTorch models.

        Raises:
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise LearnerError(
                f"{self.get_name()} must be trained before prediction. "
                f"Call train() with labeled data first. If using the active learning API, "
                f"ensure at least one training cycle has completed before predicting."
            )

        start_time = time.time()

        try:
            features, _ = _preprocess_features(
                features,
                valid_feature_mask=self._valid_feature_mask,
                remove_zero_variance=self.remove_zero_variance,
                is_training=False
            )

            X_scaled = self.scaler.transform(features)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy().squeeze()

            if np.isscalar(predictions):
                predictions = np.array([predictions])
            elif predictions.ndim == 0:
                predictions = np.array([predictions.item()])

            pred_time = time.time() - start_time
            logger.debug(f"Predicted {len(predictions)} samples with {self.get_name()} in {pred_time:.2f}s")

            return predictions, None

        except LearnerError:
            raise
        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Failed to predict with {self.get_name()}: {e}")
            raise LearnerError(
                f"Prediction failed for {self.get_name()} on {len(features)} samples "
                f"(device: {self.device}): {e}. "
                f"Check that the input features have the same shape as training features "
                f"and that the featurizer is compatible with the model."
            ) from e
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f"Torch{self.__class__.__name__}"
    
    def supports_uncertainty(self) -> bool:
        """Return True if this learner can provide uncertainty estimates."""
        # Base PyTorch models don't provide uncertainty by default
        return False
    
    def get_training_history(self) -> list:
        """Get training history for analysis.
        
        Returns:
            List of dictionaries containing training metrics per epoch
        """
        return self.training_history.copy()
    
    def save_model(self, path: Path) -> None:
        """Save model state to file.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise LearnerError(
                f"No model to save for {self.get_name()}. "
                f"The model has not been created yet. Call train() first to create and train the model."
            )
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'config': {
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'learning_rate': self.learning_rate,
                'early_stopping_patience': self.early_stopping_patience,
                'random_state': self.random_state
            }
        }
        
        torch.save(state, path)
        logger.debug(f"Saved {self.get_name()} model to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load model state from file.
        
        Args:
            path: Path to load model from
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. "
                f"Verify the path is correct and that the model was previously saved with save_model()."
            )
        
        state = torch.load(path, map_location=self.device)
        
        # Restore configuration
        config = state.get('config', {})
        for key, value in config.items():
            setattr(self, key, value)
        
        # Create model if needed (requires input size from first use)
        # Note: Model creation is deferred until first training or explicit creation
        
        # Restore training state
        self.scaler = state.get('scaler')
        self.training_history = state.get('training_history', [])
        self.is_trained = state.get('is_trained', False)
        
        logger.debug(f"Loaded {self.get_name()} model from {path}")
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
import pandas as pd
import numpy as np

# Core imports
from learnm8.core.interfaces import Learner

# Optional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    from sklearn.base import BaseEstimator
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    BaseEstimator = object
    StandardScaler = None
    SKLEARN_AVAILABLE = False


logger = logging.getLogger(__name__)


class SklearnLearner(Learner):
    """Base class for scikit-learn compatible models with dependency injection.
    
    This class implements the new architecture pattern where learners receive
    a DataManager instance for all feature extraction needs, promoting
    clean separation of concerns and testability.
    """
    
    def __init__(self,
                 model: BaseEstimator,
                 featurizer_type: str = None,
                 random_state: int = 42):
        """Initialize sklearn learner with dependency injection pattern.

        Args:
            model: Scikit-learn compatible model instance
            featurizer_type: Type of molecular features to use
            random_state: Random seed for reproducibility
        """
        if featurizer_type is None:
            raise ValueError("featurizer_type is required")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for SklearnLearner")
        
        self.model = model
        self.featurizer_type = featurizer_type
        self.random_state = random_state
        self.is_trained = False
        self.training_data = pd.DataFrame()
        self.target_column = None
        
        # Set random state on model if it supports it
        if hasattr(self.model, 'random_state'):
            self.model.random_state = random_state
    
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: 'DataManager') -> None:
        """Train sklearn model using DataManager for features.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', and target columns
            target_column: Name of the target property column  
            data_manager: Central data manager for feature extraction and caching
            
        Raises:
            ValueError: If compounds DataFrame is malformed or target_column missing
            RuntimeError: If training fails
        """
        start_time = time.time()
        
        # Validate input
        required_cols = ['ID', 'SMILES', target_column]
        missing_cols = set(required_cols) - set(compounds.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if compounds.empty:
            raise ValueError("compounds DataFrame is empty")
        
        try:
            # Use DataManager to prepare training data
            valid_compounds, X, y = data_manager.prepare_training_data(compounds, target_column, self.featurizer_type)

            # Train the model
            self.model.fit(X, y)

            # Store training data reference for potential retraining
            self.training_data = compounds.copy()
            self.target_column = target_column
            self.is_trained = True

            train_time = time.time() - start_time
            logger.info(f"Trained {self.get_name()} on {len(valid_compounds)} compounds in {train_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to train {self.get_name()}: {e}")
            raise RuntimeError(f"Training failed: {e}") from e
    
    def predict(self, compounds: pd.DataFrame, data_manager: 'DataManager') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict using sklearn model.

        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            data_manager: Central data manager for feature extraction

        Returns:
            Tuple of (predictions, uncertainties).
            Base sklearn models return None for uncertainties.
            The predictions align with the valid compounds returned
            by data_manager.prepare_prediction_data().

        Raises:
            ValueError: If compounds DataFrame is malformed
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        start_time = time.time()

        try:
            # Use DataManager to prepare prediction data (filters invalid compounds)
            valid_compounds, X = data_manager.prepare_prediction_data(compounds, self.featurizer_type)

            if len(valid_compounds) == 0:
                logger.warning("No compounds could generate valid features for prediction")
                return np.array([]), None

            # Make predictions
            predictions = self.model.predict(X)

            pred_time = time.time() - start_time
            logger.debug(f"Predicted {len(predictions)} compounds with {self.get_name()} in {pred_time:.2f}s")

            return predictions, None  # Base sklearn models don't provide uncertainty

        except Exception as e:
            logger.error(f"Failed to predict with {self.get_name()}: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f"Sklearn{self.model.__class__.__name__}"
    
    def supports_uncertainty(self) -> bool:
        """Return True if this learner can provide uncertainty estimates."""
        # Base sklearn models don't provide uncertainty
        return False


class TorchLearner(Learner):
    """Base class for PyTorch models with GPU support and dependency injection.
    
    This class provides common PyTorch functionality including device management,
    training loops, and model persistence while following the new architecture
    pattern with DataManager dependency injection.
    """
    
    def __init__(self,
                 featurizer_type: str = None,
                 device: str = 'auto',
                 batch_size: int = 1024,
                 max_epochs: int = 100,
                 learning_rate: float = 0.001,
                 early_stopping_patience: int = 10,
                 random_state: int = 42):
        """Initialize PyTorch learner with dependency injection pattern.

        Args:
            featurizer_type: Type of molecular features to use
            device: Device for computation ('auto', 'cpu', 'cuda')
            batch_size: Batch size for training and prediction
            max_epochs: Maximum training epochs
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Patience for early stopping
            random_state: Random seed for reproducibility
        """
        if featurizer_type is None:
            raise ValueError("featurizer_type is required")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TorchLearner")
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Training configuration
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.featurizer_type = featurizer_type
        self.random_state = random_state
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        
        logger.info(f"Initialized TorchLearner on device: {self.device}")
    
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
    
    def train(self, compounds: pd.DataFrame, target_column: str, data_manager: 'DataManager') -> None:
        """Train PyTorch model using DataManager for features.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', and target columns
            target_column: Name of the target property column  
            data_manager: Central data manager for feature extraction and caching
            
        Raises:
            ValueError: If compounds DataFrame is malformed or target_column missing
            RuntimeError: If training fails
        """
        start_time = time.time()
        
        try:
            # Use DataManager to prepare training data
            valid_compounds, X, y = data_manager.prepare_training_data(compounds, target_column, self.featurizer_type)

            # Initialize model if needed
            if self.model is None:
                self.model = self._create_model(X.shape[1]).to(self.device)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Feature normalization
            if StandardScaler is None:
                raise ImportError("scikit-learn is required for feature scaling")

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Split into train/validation
            X_train, X_val, y_train, y_val = self._split_validation(X_scaled, y)

            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            self.training_history = []

            logger.info(f"Training {self.get_name()} for up to {self.max_epochs} epochs")

            for epoch in range(self.max_epochs):
                # Train for one epoch
                train_loss = self._train_epoch(X_train, y_train)

                # Validate
                val_loss = self._validate(X_val, y_val)

                # Track history
                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.debug(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            self.is_trained = True
            train_time = time.time() - start_time
            logger.info(f"Trained {self.get_name()} on {len(valid_compounds)} compounds in {train_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to train {self.get_name()}: {e}")
            raise RuntimeError(f"Training failed: {e}") from e
    
    def predict(self, compounds: pd.DataFrame, data_manager: 'DataManager') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict using PyTorch model.

        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            data_manager: Central data manager for feature extraction

        Returns:
            Tuple of (predictions, uncertainties).
            Base PyTorch models return None for uncertainties.
            The predictions align with the valid compounds returned
            by data_manager.prepare_prediction_data().

        Raises:
            ValueError: If compounds DataFrame is malformed
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        start_time = time.time()

        try:
            # Use DataManager to prepare prediction data (filters invalid compounds)
            valid_compounds, X = data_manager.prepare_prediction_data(compounds, self.featurizer_type)

            if len(valid_compounds) == 0:
                logger.warning("No compounds could generate valid features for prediction")
                return np.array([]), None

            # Scale features using training scaler
            X_scaled = self.scaler.transform(X)

            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            # Make predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy().squeeze()

            # Ensure predictions is always an array, even for single predictions
            if np.isscalar(predictions):
                predictions = np.array([predictions])
            elif predictions.ndim == 0:  # Handle 0-dimensional arrays
                predictions = np.array([predictions.item()])

            pred_time = time.time() - start_time
            logger.debug(f"Predicted {len(predictions)} compounds with {self.get_name()} in {pred_time:.2f}s")

            return predictions, None  # Base PyTorch models don't provide uncertainty

        except Exception as e:
            logger.error(f"Failed to predict with {self.get_name()}: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
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
            raise RuntimeError("No model to save")
        
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
                'featurizer_type': self.featurizer_type,
                'random_state': self.random_state
            }
        }
        
        torch.save(state, path)
        logger.info(f"Saved {self.get_name()} model to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load model state from file.
        
        Args:
            path: Path to load model from
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
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
        
        logger.info(f"Loaded {self.get_name()} model from {path}")
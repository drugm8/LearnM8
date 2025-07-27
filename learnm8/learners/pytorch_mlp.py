"""PyTorch Multi-Layer Perceptron learner with GPU acceleration."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from learnm8.core.interfaces import Learner
from learnm8.utils.featurizers import get_fingerprints, get_descriptors


class MLPNetwork(nn.Module):
    """Simple MLP network with configurable architecture."""
    
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...] = (128, 128), dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class PyTorchMLPLearner(Learner):
    """PyTorch-based MLP learner with GPU acceleration."""
    
    def __init__(self, hidden_sizes: Tuple[int, ...] = (128, 128), dropout: float = 0.2,
                 learning_rate: float = 0.001, batch_size: int = 32, epochs: int = 100,
                 early_stopping_patience: int = 10, random_state: int = 42,
                 results_dir: Path = None, featurizer_type: str = 'morgan'):
        """
        Initialize PyTorch MLP learner.
        
        Args:
            hidden_sizes: Tuple of hidden layer sizes
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            early_stopping_patience: Early stopping patience
            random_state: Random seed for reproducibility
            results_dir: Directory for storing representations
            featurizer_type: Type of molecular featurizer ('morgan', 'maccs', 'ecfp6', or 'descriptors')
        """
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.results_dir = results_dir
        self.featurizer_type = featurizer_type
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        # Training state
        self.is_trained = False
        self.training_data = pd.DataFrame()
        self.target_column = None
        
        # Model components
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_mean = None
        self.scaler_std = None
    
    def _get_features(self, compounds: pd.DataFrame) -> np.ndarray:
        """Get molecular features for compounds."""
        if self.featurizer_type in ['morgan', 'maccs', 'ecfp6']:
            return get_fingerprints(compounds['ID'].tolist(), self.results_dir, self.featurizer_type)
        elif self.featurizer_type == 'descriptors':
            return get_descriptors(compounds['ID'].tolist(), self.results_dir)
        else:
            raise ValueError(f"Unknown featurizer type: {self.featurizer_type}")
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using z-score normalization."""
        if fit:
            self.scaler_mean = np.nanmean(X, axis=0)
            self.scaler_std = np.nanstd(X, axis=0)
            # Avoid division by zero
            self.scaler_std[self.scaler_std == 0] = 1.0
        
        return (X - self.scaler_mean) / self.scaler_std
    
    def _create_data_loader(self, X: np.ndarray, y: np.ndarray = None, shuffle: bool = False):
        """Create PyTorch DataLoader."""
        # Handle NaN values by replacing with zeros
        X = np.nan_to_num(X, nan=0.0)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(X_tensor)
        
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )
    
    def train(self, compounds: pd.DataFrame, target_column: str) -> None:
        """
        Train the PyTorch MLP model.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES', and target columns
            target_column: Name of column containing target values
        """
        if target_column not in compounds.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Accumulate training data
        if self.training_data.empty:
            self.training_data = compounds.copy()
        else:
            self.training_data = pd.concat([self.training_data, compounds], ignore_index=True)
        
        self.target_column = target_column
        
        # Get features and targets
        X = self._get_features(self.training_data)
        y = self.training_data[target_column].values
        
        # Normalize features
        X_norm = self._normalize_features(X, fit=True)
        
        # Initialize model
        input_size = X_norm.shape[1]
        self.model = MLPNetwork(input_size, self.hidden_sizes, self.dropout).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create data loader
        train_loader = self._create_data_loader(X_norm, y, shuffle=True)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break
        
        self.is_trained = True
    
    def predict(self, compounds: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict scores for compounds.
        
        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            
        Returns:
            Tuple of (predictions, None) - PyTorch MLP doesn't provide uncertainty
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Get features
        X = self._get_features(compounds)
        X_norm = self._normalize_features(X, fit=False)
        
        # Create data loader
        test_loader = self._create_data_loader(X_norm, shuffle=False)
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, in test_loader:
                batch_pred = self.model(batch_X)
                predictions.append(batch_pred.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        
        return predictions, None
    
    def get_name(self) -> str:
        """Return the model name."""
        return "PyTorch MLP"
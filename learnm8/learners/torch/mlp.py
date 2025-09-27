"""Multi-Layer Perceptron learner implementation for the LearnM8 framework.

This module provides PyTorch-based MLP with configurable architecture
for molecular property prediction tasks.
"""

import logging
from typing import Tuple

# Base class import
from ..base import TorchLearner

# Optional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


class MLPLearner(TorchLearner):
    """Multi-Layer Perceptron with configurable architecture.
    
    This learner provides a standard feedforward neural network with
    configurable hidden layers, activation functions, and regularization
    for molecular property prediction.
    """
    
    def __init__(self,
                 hidden_sizes: Tuple[int, ...] = (512, 256, 128),
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 batch_norm: bool = True,
                 featurizer_type: str = None,
                 **kwargs):
        """Initialize MLP learner.

        Args:
            hidden_sizes: Tuple of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'gelu')
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            featurizer_type: Type of molecular features to use
            **kwargs: Additional arguments passed to TorchLearner
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MLPLearner")
        
        super().__init__(featurizer_type=featurizer_type, **kwargs)
        
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Activation function mapping
        self.activation_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU
        }.get(activation, nn.ReLU)
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create MLP model architecture.
        
        Args:
            input_size: Number of input features
            
        Returns:
            PyTorch MLP model
        """
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(self.activation_fn())
            
            # Dropout
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights(model)
        
        return model
    
    def _initialize_weights(self, model: nn.Module) -> None:
        """Initialize model weights using Xavier/He initialization.
        
        Args:
            model: PyTorch model to initialize
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'relu':
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        hidden_str = '-'.join(map(str, self.hidden_sizes))
        return f"MLP({hidden_str},{self.activation},dropout={self.dropout_rate})"
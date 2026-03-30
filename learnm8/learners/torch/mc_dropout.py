"""Monte Carlo Dropout learner implementation for the LearnM8 framework.

This module provides MLP with Monte Carlo Dropout for uncertainty estimation
in molecular property prediction tasks.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

from learnm8.exceptions import LearnerError

# Base class import
from ..base import TorchLearner, _preprocess_features

logger = logging.getLogger(__name__)


class MCDropoutLearner(TorchLearner):
    """MLP with Monte Carlo Dropout for uncertainty estimation.

    This learner provides uncertainty quantification through Monte Carlo
    Dropout, where multiple forward passes with dropout enabled provide
    an ensemble of predictions for uncertainty estimation.
    """

    def __init__(self,
                 hidden_sizes: tuple[int, ...] = (256, 128),
                 dropout_rate: float = 0.2,
                 n_dropout_samples: int = 100,
                 activation: str = 'relu',
                 batch_norm: bool = False,
                 **kwargs):
        """Initialize Monte Carlo Dropout learner.

        Args:
            hidden_sizes: Tuple of hidden layer sizes
            dropout_rate: Dropout rate for uncertainty estimation
            n_dropout_samples: Number of dropout samples for uncertainty
            activation: Activation function
            batch_norm: Whether to use batch normalization
            **kwargs: Additional arguments passed to TorchLearner
        """
        super().__init__(**kwargs)

        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.n_dropout_samples = n_dropout_samples
        self.activation = activation
        self.batch_norm = batch_norm

        # Activation function mapping
        self.activation_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU
        }.get(activation, nn.ReLU)

    def _create_model(self, input_size: int) -> nn.Module:
        """Create MLP with dropout layers for uncertainty estimation.

        Args:
            input_size: Number of input features

        Returns:
            PyTorch MLP model with dropout
        """
        layers = []
        prev_size = input_size

        # Hidden layers with dropout for uncertainty
        for hidden_size in self.hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # Batch normalization
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # Activation
            layers.append(self.activation_fn())

            # Dropout (always applied for MC Dropout)
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

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict on feature matrix with uncertainty via Monte Carlo Dropout.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (predictions, uncertainties) where uncertainties are
            estimated using Monte Carlo Dropout sampling.

        Raises:
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise LearnerError(
                f"{self.get_name()} must be trained before prediction. "
                f"Call train() with labeled data first."
            )

        try:
            features, _, _ = _preprocess_features(
                features,
                valid_feature_mask=self._valid_feature_mask,
                remove_zero_variance=self.remove_zero_variance,
                is_training=False,
                feature_type=self._feature_type,
                imputer=self._feature_imputer,
            )

            X_scaled = self.scaler.transform(features)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            self.model.train()
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

            predictions_list = []
            with torch.no_grad():
                for _ in range(self.n_dropout_samples):
                    pred = self.model(X_tensor).cpu().numpy().squeeze()
                    predictions_list.append(pred)

            predictions_array = np.array(predictions_list)
            mean_predictions = np.mean(predictions_array, axis=0)
            uncertainties = np.std(predictions_array, axis=0)

            if np.isscalar(mean_predictions):
                mean_predictions = np.array([mean_predictions])
            if np.isscalar(uncertainties):
                uncertainties = np.array([uncertainties])

            mean_predictions, uncertainties = self._inverse_transform_predictions(
                mean_predictions, uncertainties
            )

            logger.debug(f"Predicted {len(mean_predictions)} samples with {self.get_name()} "
                        f"using {self.n_dropout_samples} MC samples")

            return mean_predictions, uncertainties

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Failed to predict with {self.get_name()}: {e}")
            raise LearnerError(
                f"Prediction failed for {self.get_name()} on {len(features)} samples: {e}. "
                f"Check that the input features have the same shape as training features."
            ) from e

    def supports_uncertainty(self) -> bool:
        """Return True since MC Dropout provides uncertainty estimates."""
        return True

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        hidden_str = '-'.join(map(str, self.hidden_sizes))
        return f"MCDropout({hidden_str},samples={self.n_dropout_samples},dropout={self.dropout_rate})"

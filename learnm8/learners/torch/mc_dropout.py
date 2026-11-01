"""Monte Carlo Dropout learner implementation for the LearnM8 framework.

This module provides MLP with Monte Carlo Dropout for uncertainty estimation
in molecular property prediction tasks.
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from learnm8.exceptions import LearnerError

# Base class import
from ..base import TorchLearner, _preprocess_features

logger = logging.getLogger(__name__)

MCDROPOUT_EFFECTIVE_MIN = 256
MCDROPOUT_MAX_RETRIES = 3

_ROW_HASH_BLOCK = 256


def _i32(value: int) -> int:
    """The 32-bit pattern of ``value`` as a signed int32."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


_GOLDEN = _i32(0x9E3779B9)
_STREAM = _i32(0x85EBCA6B)
_MUL1 = _i32(0x7FEB352D)
_MUL2 = _i32(0x846CA68B)
_INT32_MIN = -(1 << 31)


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 integer finalizer on int32 tensors (bit patterns, wrapping).

    The shifts are masked to act as logical (unsigned) shifts.
    """
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * _MUL1
    x = x ^ ((x >> 15) & 0x1FFFF)
    x = x * _MUL2
    return x ^ ((x >> 16) & 0xFFFF)


def _row_keys(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """64-bit content hash of each float32 feature row, as int32 (low, high) halves.

    The key depends only on the row's values, so a compound draws the same
    dropout masks whatever chunk, pool or position it is predicted in.
    """
    words = np.ascontiguousarray(X, dtype=np.float32).view(np.uint32)
    coeffs = np.random.default_rng(0x5EED).integers(
        1, 2**63, size=words.shape[1], dtype=np.uint64
    ) | np.uint64(1)
    h = np.zeros(len(words), dtype=np.uint64)
    for start in range(0, words.shape[1], _ROW_HASH_BLOCK):
        block = words[:, start : start + _ROW_HASH_BLOCK].astype(np.uint64)
        h += (block * coeffs[start : start + _ROW_HASH_BLOCK]).sum(
            axis=1, dtype=np.uint64
        )
    halves = h.view(np.uint32).reshape(-1, 2).view(np.int32)
    return np.ascontiguousarray(halves[:, 0]), np.ascontiguousarray(halves[:, 1])


def _dropout_mask(
    row_low: torch.Tensor,
    row_high: torch.Tensor,
    *,
    seed: int,
    layer: int,
    sample: int,
    width: int,
    p: float,
) -> torch.Tensor:
    """Scaled Bernoulli(1 - p) dropout mask, a pure function of its arguments.

    Matches ``nn.Dropout``: kept units are scaled by ``1 / (1 - p)``.
    """
    device = row_low.device
    if p >= 1.0:
        return torch.zeros(len(row_low), width, device=device)
    stream = torch.tensor(
        _i32(seed + layer * 0x9E3779B9 + sample * 0x85EBCA6B),
        dtype=torch.int32,
        device=device,
    )
    base = _mix32(row_low ^ _mix32(row_high ^ _mix32(stream)))
    units = torch.arange(width, dtype=torch.int32, device=device) * _GOLDEN
    bits = _mix32(base[:, None] + units[None, :])
    # Unsigned comparison bits >= p * 2**32, done on signed int32 by flipping
    # the sign bit of both sides.
    keep = (bits ^ _INT32_MIN) >= round(p * 2**32) + _INT32_MIN
    return keep.float() / (1.0 - p)


class MCDropoutLearner(TorchLearner):
    """MLP with Monte Carlo Dropout for uncertainty estimation.

    This learner provides uncertainty quantification through Monte Carlo
    Dropout, where multiple forward passes with dropout enabled provide
    an ensemble of predictions for uncertainty estimation.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256, 128),
        dropout_rate: float = 0.2,
        n_dropout_samples: int = 100,
        activation: str = 'relu',
        batch_norm: bool = False,
        predict_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
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
        self.predict_batch_size = predict_batch_size

        # Activation function mapping
        self.activation_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU,
        }.get(activation, nn.ReLU)

    def _create_model(self, input_size: int) -> nn.Module:
        """Create MLP with dropout layers for uncertainty estimation.

        Args:
            input_size: Number of input features

        Returns:
            PyTorch MLP model with dropout
        """
        layers: list[nn.Module] = []
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
                    nn.init.kaiming_normal_(
                        module.weight, mode='fan_in', nonlinearity='relu'
                    )
                else:
                    nn.init.xavier_normal_(module.weight)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _mc_forward(
        self,
        X: torch.Tensor,
        row_low: torch.Tensor,
        row_high: torch.Tensor,
        sample: int,
    ) -> torch.Tensor:
        seed = self.random_state or 0
        h = X
        layer = 0
        for module in self.model:
            if isinstance(module, nn.Dropout):
                h = h * _dropout_mask(
                    row_low,
                    row_high,
                    seed=seed,
                    layer=layer,
                    sample=sample,
                    width=h.shape[1],
                    p=module.p,
                )
                layer += 1
            else:
                h = module(h)
        return h

    def _predict_chunk(self, X_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_gpu = torch.FloatTensor(X_chunk).to(self.device)

        # Eval mode keeps BatchNorm on its running statistics; dropout is applied
        # explicitly in _mc_forward with masks keyed on each row's content, so
        # results do not depend on chunking, row order or the global torch RNG.
        self.model.eval()
        low, high = _row_keys(X_chunk)
        row_low = torch.from_numpy(low).to(self.device)
        row_high = torch.from_numpy(high).to(self.device)

        predictions_list = []
        with torch.no_grad():
            for sample in range(self.n_dropout_samples):
                pred = self._mc_forward(X_gpu, row_low, row_high, sample)
                predictions_list.append(pred.cpu().numpy().squeeze())

        predictions_array = np.array(predictions_list)
        mean_predictions = np.mean(predictions_array, axis=0)
        uncertainties = np.std(predictions_array, axis=0)

        if np.isscalar(mean_predictions):
            mean_predictions = np.array([mean_predictions])
        if np.isscalar(uncertainties):
            uncertainties = np.array([uncertainties])

        del X_gpu
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        return mean_predictions, uncertainties

    def _predict_chunk_with_oom_retry(
        self,
        X_chunk: np.ndarray,
        retry_count: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            return self._predict_chunk(X_chunk)
        except torch.cuda.OutOfMemoryError as e:
            oom_error: RuntimeError = e
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            oom_error = e

        retry_count += 1
        if retry_count > MCDROPOUT_MAX_RETRIES:
            raise LearnerError(
                'MCDropout chunk OOM after 3 retries; possible non-size OOM'
            ) from oom_error

        chunk_size = len(X_chunk)
        new_chunk_size = chunk_size // 2
        if new_chunk_size < MCDROPOUT_EFFECTIVE_MIN:
            raise LearnerError(
                'MCDropout chunk OOM at minimum size; increase GPU memory or reduce T'
            ) from oom_error

        logger.warning(
            'MCDropout OOM on chunk size %d, retrying with %d (retry %d/%d)',
            chunk_size,
            new_chunk_size,
            retry_count,
            MCDROPOUT_MAX_RETRIES,
        )
        torch.cuda.empty_cache()

        half1 = X_chunk[:new_chunk_size]
        half2 = X_chunk[new_chunk_size:]

        mean1, std1 = self._predict_chunk_with_oom_retry(half1, retry_count)
        mean2, std2 = self._predict_chunk_with_oom_retry(half2, retry_count)

        mean = np.concatenate([mean1, mean2])
        std = np.concatenate([std1, std2])
        return mean, std

    def predict(
        self,
        features: np.ndarray,
        smiles: list[str] | None = None,
        *,
        compute_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict on feature matrix with uncertainty via Monte Carlo Dropout.

        MC Dropout stays *uniform-contract*. When
        ``compute_uncertainty=False`` the N stochastic forward passes still
        execute (skipping them would change the mean semantics per Gal &
        Ghahramani 2016); only the std return is discarded.

        Args:
            features: Feature matrix (n_samples, n_features)
            smiles: Ignored; present for interface compatibility.
            compute_uncertainty: Keyword-only. When False, returns
                ``(mean_predictions, None)`` — but the N stochastic forward
                passes still execute.

        Returns:
            Tuple of (predictions, uncertainties) where uncertainties is
            ``None`` if ``compute_uncertainty=False``.

        Raises:
            RuntimeError: If model is not trained or prediction fails
        """
        del smiles
        if not self.is_trained:
            raise LearnerError(
                f'{self.get_name()} must be trained before prediction. '
                f'Call train() with labeled data first.'
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

            batch_size = self._compute_predict_batch_size(
                len(X_scaled), X_scaled.shape[1]
            )

            all_means = []
            all_stds = []
            for start_idx in range(0, len(X_scaled), batch_size):
                chunk = X_scaled[start_idx : start_idx + batch_size]
                mean_chunk, std_chunk = self._predict_chunk_with_oom_retry(chunk)
                all_means.append(mean_chunk)
                all_stds.append(std_chunk)

            mean_predictions = np.concatenate(all_means)
            uncertainties = np.concatenate(all_stds)

            mean_predictions, uncertainties = self._inverse_transform_predictions(
                mean_predictions, uncertainties
            )

            logger.debug(
                f'Predicted {len(mean_predictions)} samples with {self.get_name()} '
                f'using {self.n_dropout_samples} MC samples'
            )

            # Uniform contract — discard uncertainty when
            # the cycle opted out. The N stochastic passes still ran above
            # (skipping them would change mean semantics).
            if not compute_uncertainty:
                return mean_predictions, None
            return mean_predictions, uncertainties

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f'Failed to predict with {self.get_name()}: {e}')
            raise LearnerError(
                f'Prediction failed for {self.get_name()} on {len(features)} samples: {e}. '
                f'Check that the input features have the same shape as training features.'
            ) from e

    def supports_uncertainty(self) -> bool:
        """Return True since MC Dropout provides uncertainty estimates."""
        return True

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        hidden_str = '-'.join(map(str, self.hidden_sizes))
        return f'MCDropout({hidden_str},samples={self.n_dropout_samples},dropout={self.dropout_rate})'

    def _compute_predict_batch_size(self, n_samples: int, n_features: int) -> int:
        if self.predict_batch_size is not None:
            return self.predict_batch_size

        try:
            from learnm8.core.batching import estimate_batch_size

            return estimate_batch_size(
                self,
                n_samples,
                n_features,
                device=str(self.device),
                memory_safety_factor=0.5,
            )
        except Exception:
            base_batch = getattr(self, 'batch_size', 32)
            batch_size = 4 * base_batch

        return max(1, min(batch_size, n_samples))

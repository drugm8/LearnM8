"""Fastprop ensemble learner for deep learning on molecular features."""

from typing import List, Optional
import numpy as np
from .ensemble import EnsembleLearner
from ..torch.fastprop_learner import FastpropLearner
from learnm8.exceptions import LearnerError


class FastpropEnsemble(EnsembleLearner):
    """Ensemble of 3 Fastprop learners with different random states."""

    def __init__(self,
                 fnn_layers: int = 2,
                 hidden_size: int = 1800,
                 max_epochs: int = 50,
                 learning_rate: float = 0.0001,
                 batch_size: int = 32,
                 predict_batch_size: Optional[int] = None,
                 precision: str = 'auto',
                 pin_memory: bool = False,
                 clamp_input: bool = True,
                 early_stopping_patience: int = 5,
                 random_states: Optional[List[int]] = None,
                 device: str = 'auto',
                 enable_aggressive_gc: bool = True,
                 **kwargs):
        """Initialize Fastprop ensemble.

        Args:
            fnn_layers: Number of hidden layers per learner (default: 2)
            hidden_size: Hidden layer size per learner (default: 1800)
            max_epochs: Maximum training epochs per learner (default: 50)
            learning_rate: Learning rate per learner (default: 0.0001)
            batch_size: Training batch size per learner (default: 32)
            predict_batch_size: Prediction batch size per learner (default: None, uses 4x batch_size)
            precision: Precision mode for all members - 'auto', '16-mixed', '32-true' (default: 'auto')
            pin_memory: Enable pinned memory for all members (may not work with fastpropDataLoader, default: False)
            clamp_input: Apply input winsorization per learner (default: True)
            early_stopping_patience: Early stopping patience per learner (default: 5)
            random_states: List of random states for diversity (default: [42, 123, 456])
            device: Device for computation ('auto', 'cpu', 'cuda') (default: 'auto')
            enable_aggressive_gc: Enable automatic GPU memory cleanup for all
                ensemble members (default: True)
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        if random_states is None:
            random_states = [42, 123, 456]

        learners = []
        for rs in random_states:
            fastprop = FastpropLearner(
                fnn_layers=fnn_layers,
                hidden_size=hidden_size,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                predict_batch_size=predict_batch_size,
                precision=precision,
                pin_memory=pin_memory,
                clamp_input=clamp_input,
                early_stopping_patience=early_stopping_patience,
                random_state=rs,
                device=device,
                enable_aggressive_gc=enable_aggressive_gc
            )
            learners.append(fastprop)

        kwargs.setdefault('aggregation_method', 'mean')
        kwargs.setdefault('uncertainty_method', 'std')

        super().__init__(learners, **kwargs)

        self.fnn_layers = fnn_layers
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.precision = precision
        self.pin_memory = pin_memory
        self.clamp_input = clamp_input
        self.early_stopping_patience = early_stopping_patience
        self.random_states = random_states
        self.device = device
        self.enable_aggressive_gc = enable_aggressive_gc

    def train(self, features, targets):
        """Train ensemble with feature matrix.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
        """
        if features.shape[0] != targets.shape[0]:
            raise ValueError(f"Features and targets must have same length: {features.shape[0]} vs {targets.shape[0]}")

        if features.shape[0] == 0:
            raise ValueError("Cannot train on empty dataset")

        # Train each learner with features
        for i, learner in enumerate(self.learners):
            learner.train(features, targets)

        self.is_trained = True

        self._cleanup_gpu_memory("after ensemble training")

    def predict(self, features):
        """Predict with feature matrix.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise LearnerError("Ensemble must be trained before prediction")

        # Get predictions from each learner
        predictions_list = []
        for learner in self.learners:
            pred, _ = learner.predict(features)
            predictions_list.append(pred)

        predictions_array = np.array(predictions_list)

        # Aggregate predictions
        ensemble_predictions = self._aggregate_predictions(predictions_array)
        uncertainties = self._calculate_uncertainty(predictions_array)

        self._cleanup_gpu_memory("after ensemble prediction")

        return ensemble_predictions, uncertainties

    def get_individual_predictions(self, features):
        """Get predictions from individual ensemble members.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Dictionary mapping learner names to their predictions
        """
        if not self.is_trained:
            raise LearnerError("Ensemble must be trained before prediction")

        individual_predictions = {}
        for i, learner in enumerate(self.learners):
            pred, _ = learner.predict(features)
            individual_predictions[f"{learner.get_name()}_{i}"] = pred

        return individual_predictions

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f"FastpropEnsemble(3xFastprop,layers={self.fnn_layers},hidden={self.hidden_size})"

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
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"GPU memory cleanup: {context}")

        except (RuntimeError, OSError, ImportError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"GPU memory cleanup failed ({context}): {e}")

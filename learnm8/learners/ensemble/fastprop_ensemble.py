"""Fastprop ensemble learner for deep learning on molecular features."""

from typing import List, Optional
from .ensemble import EnsembleLearner
from ..torch.fastprop_learner import FastpropLearner


class FastpropEnsemble(EnsembleLearner):
    """Ensemble of 3 Fastprop learners with different random states."""

    def __init__(self,
                 fnn_layers: int = 2,
                 hidden_size: int = 1800,
                 max_epochs: int = 50,
                 learning_rate: float = 0.0001,
                 batch_size: int = 32,
                 clamp_input: bool = True,
                 early_stopping_patience: int = 5,
                 random_states: Optional[List[int]] = None,
                 device: str = 'auto',
                 **kwargs):
        """Initialize Fastprop ensemble.

        Args:
            fnn_layers: Number of hidden layers per learner (default: 2)
            hidden_size: Hidden layer size per learner (default: 1800)
            max_epochs: Maximum training epochs per learner (default: 50)
            learning_rate: Learning rate per learner (default: 0.0001)
            batch_size: Batch size per learner (default: 32)
            clamp_input: Apply input winsorization per learner (default: True)
            early_stopping_patience: Early stopping patience per learner (default: 5)
            random_states: List of random states for diversity (default: [42, 123, 456])
            device: Device for computation ('auto', 'cpu', 'cuda') (default: 'auto')
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
                clamp_input=clamp_input,
                early_stopping_patience=early_stopping_patience,
                random_state=rs,
                device=device
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
        self.clamp_input = clamp_input
        self.early_stopping_patience = early_stopping_patience
        self.random_states = random_states
        self.device = device

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f"FastpropEnsemble(3xFastprop,layers={self.fnn_layers},hidden={self.hidden_size})"

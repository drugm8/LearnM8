"""Multi-Layer Perceptron learner implementation."""

from sklearn.neural_network import MLPRegressor
from learnm8.learners.base import SklearnLearner
from pathlib import Path


class MLPLearner(SklearnLearner):
    """Multi-Layer Perceptron (Neural Network) based active learner."""
    
    def __init__(self, random_state: int = 42, results_dir: Path = None, featurizer_type: str = 'morgan'):
        """
        Initialize MLP learner with 2 hidden layers of 128 neurons each.
        
        Args:
            random_state: Random seed for reproducibility
            results_dir: Directory for storing representations
            featurizer_type: Type of molecular featurizer ('morgan' or 'descriptors')
        """
        model = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=random_state,
            n_iter_no_change=10
        )
        
        super().__init__(model, results_dir, featurizer_type)
    
    def get_name(self) -> str:
        """Return the model name."""
        return "Multi-Layer Perceptron"
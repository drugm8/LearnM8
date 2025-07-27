"""Decision Tree learner implementation."""

from sklearn.tree import DecisionTreeRegressor
from learnm8.learners.base import SklearnLearner
from pathlib import Path


class DecisionTreeLearner(SklearnLearner):
    """Decision Tree based active learner."""
    
    def __init__(self, random_state: int = 42, results_dir: Path = None, featurizer_type: str = 'morgan'):
        """
        Initialize Decision Tree learner.
        
        Args:
            random_state: Random seed for reproducibility
            results_dir: Directory for storing representations
            featurizer_type: Type of molecular featurizer ('morgan' or 'descriptors')
        """
        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state
        )
        
        super().__init__(model, results_dir, featurizer_type)
    
    def get_name(self) -> str:
        """Return the model name."""
        return "Decision Tree"
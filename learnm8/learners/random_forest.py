"""Random Forest learner implementation."""

from sklearn.ensemble import RandomForestRegressor
from learnm8.learners.base import SklearnLearner
from pathlib import Path
import os


class RandomForestLearner(SklearnLearner):
    """Random Forest based active learner."""
    
    def __init__(self, n_jobs: int = -1, random_state: int = 42, results_dir: Path = None, featurizer_type: str = 'morgan'):
        """
        Initialize Random Forest learner.
        
        Args:
            n_jobs: Number of parallel jobs (-1 uses all CPUs)
            random_state: Random seed for reproducibility
            results_dir: Directory for storing representations
            featurizer_type: Type of molecular featurizer ('morgan' or 'descriptors')
        """
        if n_jobs == -1:
            n_jobs = min(os.cpu_count() or 1, 32)
        
        model = RandomForestRegressor(
            n_estimators=100,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        super().__init__(model, results_dir, featurizer_type)
    
    def get_name(self) -> str:
        """Return the model name."""
        return "Random Forest"
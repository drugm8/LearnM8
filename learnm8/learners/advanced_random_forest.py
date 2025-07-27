"""Advanced Random Forest learner implementation with optimized hyperparameters."""

from sklearn.ensemble import RandomForestRegressor
from learnm8.learners.base import SklearnLearner
from pathlib import Path
import os


class AdvancedRandomForestLearner(SklearnLearner):
    """Advanced Random Forest with optimized hyperparameters for better performance."""
    
    def __init__(self, n_jobs: int = -1, random_state: int = 42, results_dir: Path = None, featurizer_type: str = 'morgan'):
        """
        Initialize Advanced Random Forest learner with optimized hyperparameters.
        
        Args:
            n_jobs: Number of parallel jobs (-1 uses all CPUs)
            random_state: Random seed for reproducibility
            results_dir: Directory for storing representations
            featurizer_type: Type of molecular featurizer ('morgan' or 'descriptors')
        """
        if n_jobs == -1:
            n_jobs = min(os.cpu_count() or 1, 32)
        
        model = RandomForestRegressor(
            n_estimators=300,  # Increased from 100 for better performance
            max_depth=15,  # Limited depth to prevent overfitting
            min_samples_split=5,  # Require more samples to split nodes
            min_samples_leaf=2,  # Require more samples in leaf nodes
            max_features='sqrt',  # Use sqrt of features for better generalization
            bootstrap=True,  # Enable bootstrap sampling
            oob_score=True,  # Enable out-of-bag scoring
            max_samples=0.8,  # Use 80% of samples for each tree
            min_impurity_decrease=0.0001,  # Minimum improvement required to split
            ccp_alpha=0.001,  # Cost complexity pruning parameter
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        super().__init__(model, results_dir, featurizer_type)
    
    def get_name(self) -> str:
        """Return the model name."""
        return "Advanced Random Forest"
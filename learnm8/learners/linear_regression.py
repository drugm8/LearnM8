"""Linear Regression learner implementation."""

from sklearn.linear_model import LinearRegression
from learnm8.learners.base import SklearnLearner
from pathlib import Path


class LinearRegressionLearner(SklearnLearner):
    """Linear Regression based active learner."""
    
    def __init__(self, random_state: int = 42, results_dir: Path = None, featurizer_type: str = 'morgan'):
        """
        Initialize Linear Regression learner.
        
        Args:
            random_state: Random seed (ignored for LinearRegression, kept for interface consistency)
            results_dir: Directory for storing representations
            featurizer_type: Type of molecular featurizer ('morgan' or 'descriptors')
        """
        model = LinearRegression(
            fit_intercept=True,  # Fit the intercept term
			n_jobs=-1  # Use all available CPUs for parallel processing
		)
        
        super().__init__(model, results_dir, featurizer_type)
    
    def get_name(self) -> str:
        """Return the model name."""
        return "Linear Regression"
"""Linear Regression ensemble learner for LearnM8 framework."""

from typing import List, Optional
from .ensemble import EnsembleLearner
from ..sklearn.linear_regression import LinearRegressionLearner


class LREnsemble(EnsembleLearner):
    """Ensemble of 3 Linear Regression learners with different regularization strengths."""
    
    def __init__(self,
                 featurizer_type: str = None,
                 regularization_strengths: Optional[List[float]] = None,
                 random_states: Optional[List[int]] = None,
                 **kwargs):
        """Initialize LR ensemble.

        Args:
            featurizer_type: Type of molecular features to use
            regularization_strengths: List of alpha values for Ridge regression (default: [0.1, 1.0, 10.0])
            random_states: List of random states for diversity (default: [42, 123, 456])
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        if featurizer_type is None:
            raise ValueError("featurizer_type is required")
        if regularization_strengths is None:
            regularization_strengths = [0.1, 1.0, 10.0]
        if random_states is None:
            random_states = [42, 123, 456]

        learners = []
        for alpha, rs in zip(regularization_strengths, random_states):
            lr = LinearRegressionLearner(
                alpha=alpha,
                random_state=rs,
                featurizer_type=featurizer_type
            )
            learners.append(lr)
        
        # Set default ensemble parameters
        kwargs.setdefault('aggregation_method', 'mean')
        kwargs.setdefault('uncertainty_method', 'std')
        
        super().__init__(learners, **kwargs)
        
        self.regularization_strengths = regularization_strengths
        self.random_states = random_states
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        alphas_str = ','.join(f"{a:.1f}" for a in self.regularization_strengths)
        return f"LREnsemble(3xRidge,α=[{alphas_str}])"
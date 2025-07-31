"""Random Forest ensemble learner for LearnM8 framework."""

from typing import List, Optional
from .ensemble import EnsembleLearner
from ..sklearn.random_forest import RandomForestLearner


class RFEnsemble(EnsembleLearner):
    """Ensemble of 3 Random Forest learners with different random states."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 random_states: Optional[List[int]] = None,
                 **kwargs):
        """Initialize RF ensemble.
        
        Args:
            n_estimators: Number of trees per forest (default: 100)
            random_states: List of random states for diversity (default: [42, 123, 456])
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        if random_states is None:
            random_states = [42, 123, 456]
        
        learners = []
        for rs in random_states:
            rf = RandomForestLearner(n_estimators=n_estimators, random_state=rs)
            learners.append(rf)
        
        # Set default ensemble parameters
        kwargs.setdefault('aggregation_method', 'mean')
        kwargs.setdefault('uncertainty_method', 'std')
        
        super().__init__(learners, **kwargs)
        
        self.n_estimators = n_estimators
        self.random_states = random_states
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return f"RFEnsemble(3xRF,n_est={self.n_estimators})"
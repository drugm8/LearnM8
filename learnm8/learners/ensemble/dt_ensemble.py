"""Decision Tree ensemble learner for LearnM8 framework."""

from typing import List, Optional
from .ensemble import EnsembleLearner
from ..sklearn.decision_tree import DecisionTreeLearner


class DTEnsemble(EnsembleLearner):
    """Ensemble of 3 Decision Tree learners with different max depths."""
    
    def __init__(self,
                 featurizer_type: str = None,
                 max_depths: Optional[List[int]] = None,
                 random_states: Optional[List[int]] = None,
                 **kwargs):
        """Initialize DT ensemble.

        Args:
            featurizer_type: Type of molecular features to use
            max_depths: List of max depths for diversity (default: [5, 10, 15])
            random_states: List of random states for diversity (default: [42, 123, 456])
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        if featurizer_type is None:
            raise ValueError("featurizer_type is required")
        if max_depths is None:
            max_depths = [5, 10, 15]
        if random_states is None:
            random_states = [42, 123, 456]

        learners = []
        for depth, rs in zip(max_depths, random_states):
            dt = DecisionTreeLearner(
                max_depth=depth,
                random_state=rs,
                featurizer_type=featurizer_type
            )
            learners.append(dt)
        
        # Set default ensemble parameters
        kwargs.setdefault('aggregation_method', 'mean')
        kwargs.setdefault('uncertainty_method', 'std')
        
        super().__init__(learners, **kwargs)
        
        self.max_depths = max_depths
        self.random_states = random_states
    
    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        depths_str = ','.join(str(d) for d in self.max_depths)
        return f"DTEnsemble(3xDT,depth=[{depths_str}])"
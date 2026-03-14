"""XGBoost ensemble learner for LearnM8 framework."""

from ..sklearn.xgboost_learner import XGBoostLearner
from .ensemble import EnsembleLearner


class XGBEnsemble(EnsembleLearner):
    """Ensemble of 3 XGBoost learners with different learning rates."""

    def __init__(self,
                 learning_rates: list[float] | None = None,
                 random_states: list[int] | None = None,
                 **kwargs):
        """Initialize XGB ensemble.

        Args:
            learning_rates: List of learning rates for diversity (default: [0.05, 0.1, 0.2])
            random_states: List of random states for diversity (default: [42, 123, 456])
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        if learning_rates is None:
            learning_rates = [0.05, 0.1, 0.2]
        if random_states is None:
            random_states = [42, 123, 456]

        learners = []
        for lr, rs in zip(learning_rates, random_states):
            xgb = XGBoostLearner(
                learning_rate=lr,
                random_state=rs
            )
            learners.append(xgb)

        # Set default ensemble parameters
        kwargs.setdefault('aggregation_method', 'mean')
        kwargs.setdefault('uncertainty_method', 'std')

        super().__init__(learners, **kwargs)

        self.learning_rates = learning_rates
        self.random_states = random_states

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        lrs_str = ','.join(f"{lr:.2f}" for lr in self.learning_rates)
        return f"XGBEnsemble(3xXGB,lr=[{lrs_str}])"

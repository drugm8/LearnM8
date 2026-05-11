"""XGBoost ensemble learner for LearnM8 framework."""

from ..sklearn.xgboost_learner import XGBoostLearner
from .ensemble import EnsembleLearner, _derive_random_states


class XGBEnsemble(EnsembleLearner):
    """Ensemble of 3 XGBoost learners with different learning rates."""

    def __init__(self,
                 learning_rates: list[float] | None = None,
                 random_states: list[int] | None = None,
                 random_state: int = 42,
                 device: str = 'cpu',
                 **kwargs):
        """Initialize XGB ensemble.

        Args:
            learning_rates: List of learning rates for diversity (default: [0.05, 0.1, 0.2])
            random_states: Explicit list of member seeds. When None (default),
                seeds are derived from ``random_state`` via the unified offset
                scheme ``[base, base+81, base+314, ...]``.
            random_state: Base seed for member-seed derivation (default: 42).
                Ignored when ``random_states`` is provided explicitly.
            device: Compute device for XGBoost members ('cpu', 'cuda', or 'auto').
                'auto' resolves to 'cpu' — explicit GPU selection requires 'cuda'.
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        if learning_rates is None:
            learning_rates = [0.05, 0.1, 0.2]
        random_states = _derive_random_states(
            random_state, len(learning_rates), override=random_states
        )

        resolved_device = 'cpu' if device == 'auto' else device

        learners = []
        for lr, rs in zip(learning_rates, random_states):  # noqa: B905
            xgb = XGBoostLearner(
                learning_rate=lr,
                random_state=rs,
                device=resolved_device,
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

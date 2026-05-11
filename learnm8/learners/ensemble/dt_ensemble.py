"""Decision Tree ensemble learner for LearnM8 framework."""

from ..sklearn.decision_tree import DecisionTreeLearner
from .ensemble import EnsembleLearner, _derive_random_states


class DTEnsemble(EnsembleLearner):
    """Ensemble of 3 Decision Tree learners with different max depths."""

    def __init__(self,
                 max_depths: list[int] | None = None,
                 random_states: list[int] | None = None,
                 random_state: int = 42,
                 **kwargs):
        """Initialize DT ensemble.

        Args:
            max_depths: List of max depths for diversity (default: [5, 10, 15])
            random_states: Explicit list of member seeds. When None (default),
                seeds are derived from ``random_state`` via the unified offset
                scheme ``[base, base+81, base+314, ...]``.
            random_state: Base seed for member-seed derivation (default: 42).
                Ignored when ``random_states`` is provided explicitly.
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        if max_depths is None:
            max_depths = [5, 10, 15]
        random_states = _derive_random_states(
            random_state, len(max_depths), override=random_states
        )

        learners = []
        for depth, rs in zip(max_depths, random_states):
            dt = DecisionTreeLearner(
                max_depth=depth,
                random_state=rs
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

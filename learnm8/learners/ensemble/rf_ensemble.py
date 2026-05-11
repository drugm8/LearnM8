"""Random Forest ensemble learner for LearnM8 framework."""

import warnings

from ..sklearn.random_forest import RandomForestLearner
from .ensemble import EnsembleLearner, _derive_random_states


class RFEnsemble(EnsembleLearner):
    """Ensemble of 3 Random Forest learners with different random states."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_states: list[int] | None = None,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize RF ensemble.

        Args:
            n_estimators: Number of trees per forest (default: 100)
            random_states: Explicit list of member seeds. When None (default),
                seeds are derived from ``random_state`` via the unified offset
                scheme ``[base, base+81, base+314]``.
            random_state: Base seed for member-seed derivation (default: 42).
                Ignored when ``random_states`` is provided explicitly.
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        random_states = _derive_random_states(random_state, 3, override=random_states)

        total_trees = n_estimators * len(random_states)
        warnings.warn(
            f'RFEnsemble is deprecated. Use RandomForestLearner(n_estimators={total_trees}) '
            f'instead, where N = n_estimators * len(random_states). '
            f'Will be removed in v0.12.0.',
            DeprecationWarning,
            stacklevel=2,
        )

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
        return f'RFEnsemble(3xRF,n_est={self.n_estimators})'

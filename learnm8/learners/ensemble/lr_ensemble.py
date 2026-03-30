"""Linear Regression ensemble learner for LearnM8 framework."""

from ..sklearn.linear_regression import LinearRegressionLearner
from .ensemble import EnsembleLearner


class LREnsemble(EnsembleLearner):
    """Ensemble of 3 Linear Regression learners with different regularization strengths."""

    def __init__(self,
                 regularization_strengths: list[float] | None = None,
                 random_states: list[int] | None = None,
                 device: str = 'cpu',
                 **kwargs):
        """Initialize LR ensemble.

        Args:
            regularization_strengths: List of alpha values for Ridge regression (default: [0.1, 1.0, 10.0])
            random_states: List of random states for diversity (default: [42, 123, 456])
            device: Compute device for members ('cpu', 'cuda', or 'auto').
                'auto' resolves to CPU (LinearRegressionLearner).
                'cuda' creates RidgeCumlLearner members with GPU acceleration.
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        if regularization_strengths is None:
            regularization_strengths = [0.1, 1.0, 10.0]
        if random_states is None:
            random_states = [42, 123, 456]

        learners = []
        if device == 'cuda':
            from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
            for alpha, rs in zip(regularization_strengths, random_states):  # noqa: B905
                member = RidgeCumlLearner(alpha=alpha, random_state=rs)
                learners.append(member)
        else:
            for alpha, rs in zip(regularization_strengths, random_states):  # noqa: B905
                lr = LinearRegressionLearner(
                    alpha=alpha,
                    random_state=rs
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
        return f"LREnsemble(3xRidge,alpha=[{alphas_str}])"

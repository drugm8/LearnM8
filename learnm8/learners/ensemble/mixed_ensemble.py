"""Mixed ensemble learner for LearnM8 framework."""

from ..sklearn.linear_regression import LinearRegressionLearner
from ..sklearn.random_forest import RandomForestLearner
from ..sklearn.xgboost_learner import XGBoostLearner
from .ensemble import EnsembleLearner


class MixedEnsemble(EnsembleLearner):
    """Mixed ensemble with RF, Linear Regression, and XGBoost learners."""

    def __init__(self, random_state: int = 42, device: str = 'cpu', **kwargs):
        """Initialize mixed ensemble.

        Args:
            random_state: Random state for reproducibility (default: 42)
            device: Compute device for GPU-accelerated members ('cpu', 'cuda', or 'auto').
                'cuda' replaces RF with RfFilLearner, LR with RidgeCumlLearner (alpha=1.0),
                and uses XGBoostLearner with device='cuda'.
                'auto' resolves to CPU member types.
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        if device == 'cuda':
            from learnm8.learners.gpu.rf_fil import RfFilLearner
            from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner
            learners = [
                RfFilLearner(n_estimators=100, random_state=random_state),
                RidgeCumlLearner(alpha=1.0, random_state=random_state),
                XGBoostLearner(learning_rate=0.1, random_state=random_state, device='cuda'),
            ]
        else:
            learners = [
                RandomForestLearner(
                    n_estimators=100,
                    random_state=random_state
                ),
                LinearRegressionLearner(
                    alpha=1.0,
                    random_state=random_state
                ),
                XGBoostLearner(
                    learning_rate=0.1,
                    random_state=random_state
                )
            ]

        # Set default ensemble parameters
        kwargs.setdefault('aggregation_method', 'mean')
        kwargs.setdefault('uncertainty_method', 'std')

        super().__init__(learners, **kwargs)

        self.random_state = random_state

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return "MixedEnsemble(RF+LR+XGB)"

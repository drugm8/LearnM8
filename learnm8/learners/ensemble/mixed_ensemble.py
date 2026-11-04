"""Mixed ensemble learner for LearnM8 framework."""

from typing import Any

from ..sklearn.linear_regression import LinearRegressionLearner
from ..sklearn.random_forest import RandomForestLearner
from ..sklearn.xgboost_learner import XGBoostLearner
from .ensemble import (
    EnsembleLearner,
    _derive_random_states,
    _is_cuda,
    _resolve_member_device,
)


class MixedEnsemble(EnsembleLearner):
    """Mixed ensemble with RF, Linear Regression, and XGBoost learners."""

    def __init__(
        self, random_state: int = 42, device: str = 'cpu', **kwargs: Any
    ) -> None:
        """Initialize mixed ensemble.

        Args:
            random_state: Base seed for member-seed derivation (default: 42).
                Each of the 3 members gets a distinct seed via the unified
                offset scheme ``[base, base+81, base+314]``.
            device: Compute device for GPU-accelerated members ('cpu', 'cuda',
                'cuda:N' or 'auto'). 'cuda' / 'cuda:N' replace RF with
                RfFilLearner, LR with RidgeCumlLearner (alpha=1.0), and use
                XGBoostLearner with the same device (the cuML members always
                use the current CUDA device). 'auto' resolves to 'cuda' when
                CUDA is available and cuML is importable, else CPU members.
            **kwargs: Additional arguments passed to EnsembleLearner
        """
        rs_rf, rs_lr, rs_xgb = _derive_random_states(random_state, 3)

        device = _resolve_member_device(device, needs_cuml=True)
        if _is_cuda(device):
            from learnm8.learners.gpu.rf_fil import RfFilLearner
            from learnm8.learners.gpu.ridge_cuml import RidgeCumlLearner

            learners = [
                RfFilLearner(n_estimators=100, random_state=rs_rf),
                RidgeCumlLearner(alpha=1.0, random_state=rs_lr),
                XGBoostLearner(learning_rate=0.1, random_state=rs_xgb, device=device),
            ]
        else:
            learners = [
                RandomForestLearner(n_estimators=100, random_state=rs_rf),
                LinearRegressionLearner(alpha=1.0, random_state=rs_lr),
                XGBoostLearner(learning_rate=0.1, random_state=rs_xgb),
            ]

        # Set default ensemble parameters
        kwargs.setdefault('aggregation_method', 'mean')
        kwargs.setdefault('uncertainty_method', 'std')

        super().__init__(learners, **kwargs)

        self.random_state = random_state
        self.random_states = [rs_rf, rs_lr, rs_xgb]

    def get_name(self) -> str:
        """Return a descriptive name for this learner."""
        return 'MixedEnsemble(RF+LR+XGB)'

"""GPU-accelerated learner implementations.

Provides explicit GPU learner classes:
- RfFilLearner: sklearn RF training + RAPIDS FIL GPU inference
- RidgeCumlLearner: cuML Ridge GPU training + leverage-based uncertainty
"""

from .rf_fil import RfFilLearner
from .ridge_cuml import RidgeCumlLearner

__all__ = ['RfFilLearner', 'RidgeCumlLearner']

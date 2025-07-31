"""Legacy sklearn-based learners."""

from .gradient_boosting import XGBoostLearner as LegacyXGBoostLearner

__all__ = [
    'LegacyXGBoostLearner'
]
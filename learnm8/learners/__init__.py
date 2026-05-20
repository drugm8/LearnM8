# learnm8/learners/__init__.py
"""Machine learning models for active learning.

New Architecture (v1.0.0):
- Featurizer-agnostic learners working with numpy arrays
- Composition-based ensemble methods
- PyTorch and scikit-learn integration
- Uncertainty quantification capabilities
"""

import contextlib

from .base import SklearnLearner, TorchLearner
from .ensemble import (
    DTEnsemble,
    EnsembleLearner,
    FastpropEnsemble,
    LREnsemble,
    MixedEnsemble,
    RFEnsemble,
    XGBEnsemble,
)
from .gpu import RfFilLearner, RidgeCumlLearner
from .sklearn import (
    DecisionTreeLearner,
    GaussianProcessLearner,
    LinearRegressionLearner,
    RandomForestLearner,
    XGBoostLearner,
)
from .torch import ChempropLearner, FastpropLearner, MCDropoutLearner, MLPLearner

with contextlib.suppress(ImportError):
    from .gpytorch import GPyTorchGPLearner  # noqa: F401
with contextlib.suppress(ImportError):
    from .gpytorch import SVGPLearner  # noqa: F401

__all__ = [
    'ChempropLearner',
    'DTEnsemble',
    'DecisionTreeLearner',
    'EnsembleLearner',
    'FastpropEnsemble',
    'FastpropLearner',
    'GaussianProcessLearner',
    'LREnsemble',
    'LinearRegressionLearner',
    'MCDropoutLearner',
    'MLPLearner',
    'MixedEnsemble',
    'RFEnsemble',
    'RandomForestLearner',
    'RfFilLearner',
    'RidgeCumlLearner',
    'SklearnLearner',
    'TorchLearner',
    'XGBEnsemble',
    'XGBoostLearner',
]

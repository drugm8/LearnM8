# learnm8/learners/__init__.py
"""Machine learning models for active learning.

New Architecture (v1.0.0):
- Featurizer-agnostic learners working with numpy arrays
- Composition-based ensemble methods
- PyTorch and scikit-learn integration
- Uncertainty quantification capabilities
"""

import contextlib

# Base classes
from .base import SklearnLearner, TorchLearner

# Ensemble learners
from .ensemble import (
    DTEnsemble,
    EnsembleLearner,
    FastpropEnsemble,
    LREnsemble,
    MixedEnsemble,
    RFEnsemble,
    XGBEnsemble,
)

# Scikit-learn learners
from .sklearn import (
    AdvancedRandomForestLearner,
    DecisionTreeLearner,
    GaussianProcessLearner,
    LinearRegressionLearner,
    RandomForestLearner,
    XGBoostLearner,
)

# PyTorch learners
from .torch import ChempropLearner, FastpropLearner, MCDropoutLearner, MLPLearner

# GPyTorch learners (optional dependency)
with contextlib.suppress(ImportError):
    from .gpytorch import GPyTorchGPLearner  # noqa: F401
with contextlib.suppress(ImportError):
    from .gpytorch import SVGPLearner  # noqa: F401

__all__ = [
    'AdvancedRandomForestLearner',
    'ChempropLearner',
    'DTEnsemble',
    'DecisionTreeLearner',
    # Ensemble learners
    'EnsembleLearner',
    'FastpropEnsemble',
    'FastpropLearner',
    'GaussianProcessLearner',
    'LREnsemble',
    'LinearRegressionLearner',
    'MCDropoutLearner',
    # PyTorch learners
    'MLPLearner',
    'MixedEnsemble',
    'RFEnsemble',
    # Sklearn learners
    'RandomForestLearner',
    # Base classes
    'SklearnLearner',
    'TorchLearner',
    'XGBEnsemble',
    'XGBoostLearner',
]

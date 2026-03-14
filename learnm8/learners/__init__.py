# learnm8/learners/__init__.py
"""Machine learning models for active learning.

New Architecture (v1.0.0):
- Featurizer-agnostic learners working with numpy arrays
- Composition-based ensemble methods
- PyTorch and scikit-learn integration
- Uncertainty quantification capabilities
"""

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

__all__ = [
    # Base classes
    'SklearnLearner',
    'TorchLearner',

    # Sklearn learners
    'RandomForestLearner',
    'GaussianProcessLearner',
    'XGBoostLearner',
    'DecisionTreeLearner',
    'LinearRegressionLearner',
    'AdvancedRandomForestLearner',

    # PyTorch learners
    'MLPLearner',
    'MCDropoutLearner',
    'FastpropLearner',
    'ChempropLearner',

    # Ensemble learners
    'EnsembleLearner',
    'RFEnsemble',
    'LREnsemble',
    'XGBEnsemble',
    'DTEnsemble',
    'MixedEnsemble',
    'FastpropEnsemble',
]

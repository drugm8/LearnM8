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

# Scikit-learn learners
from .sklearn import (
    RandomForestLearner,
    GaussianProcessLearner,
    XGBoostLearner,
    DecisionTreeLearner,
    LinearRegressionLearner,
    AdvancedRandomForestLearner
)

# PyTorch learners
from .torch import (
    MLPLearner,
    MCDropoutLearner,
    FastpropLearner,
    ChempropLearner
)

# Ensemble learners
from .ensemble import (
    EnsembleLearner,
    RFEnsemble,
    LREnsemble,
    XGBEnsemble,
    DTEnsemble,
    MixedEnsemble,
    FastpropEnsemble
)


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

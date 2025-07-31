# learnm8/learners/__init__.py
"""Machine learning models for active learning.

New Architecture (v0.3.0):
- Dependency injection pattern with DataManager
- Composition-based ensemble methods
- PyTorch and scikit-learn integration
- Uncertainty quantification capabilities
"""

# Base classes
try:
    from .base import SklearnLearner, TorchLearner
except ImportError:
    pass

# New architecture scikit-learn learners
try:
    from .sklearn import (
        RandomForestLearner,
        GaussianProcessLearner,
        XGBoostLearner,
        DecisionTreeLearner,
        LinearRegressionLearner,
        AdvancedRandomForestLearner
    )
except ImportError:
    pass

# New architecture PyTorch learners
try:
    from .torch import (
        MLPLearner,
        MCDropoutLearner
    )
except ImportError:
    pass

# New architecture ensemble learners
try:
    from .ensemble import (
        EnsembleLearner,
        RFEnsemble,
        LREnsemble,
        XGBEnsemble,
        DTEnsemble,
        MixedEnsemble
    )
except ImportError:
    pass

# Legacy learners (for backward compatibility)
try:
    from .sklearn.legacy import (
        LegacyXGBoostLearner
    )
except ImportError:
    pass

try:
    from .torch.legacy import (
        GaussianPyTorchLearner,
        PyTorchMLPLearner
    )
except ImportError:
    pass

# Export all available learners
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
    
    # Ensemble learners
    'EnsembleLearner',
    'RFEnsemble',
    'LREnsemble',
    'XGBEnsemble',
    'DTEnsemble',
    'MixedEnsemble',
    
    # Legacy learners
    'LegacyXGBoostLearner',
    'GaussianPyTorchLearner',
    'PyTorchMLPLearner'
]
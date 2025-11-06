"""PyTorch-based learners."""

from .mlp import MLPLearner
from .mc_dropout import MCDropoutLearner
from .fastprop_learner import FastpropLearner

try:
    from .chemprop_learner import ChempropLearner
    _chemprop_available = True
except ImportError:
    _chemprop_available = False

__all__ = [
    'MLPLearner',
    'MCDropoutLearner',
    'FastpropLearner'
]

if _chemprop_available:
    __all__.append('ChempropLearner')
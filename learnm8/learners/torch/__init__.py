"""PyTorch-based learners."""

from .mlp import MLPLearner
from .mc_dropout import MCDropoutLearner
from .fastprop_learner import FastpropLearner

__all__ = [
    'MLPLearner',
    'MCDropoutLearner',
    'FastpropLearner'
]
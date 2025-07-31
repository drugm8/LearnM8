"""PyTorch-based learners."""

from .mlp import MLPLearner
from .mc_dropout import MCDropoutLearner

__all__ = [
    'MLPLearner',
    'MCDropoutLearner'
]
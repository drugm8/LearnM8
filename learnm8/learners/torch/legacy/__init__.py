"""Legacy PyTorch-based learners."""

from .gaussian_pytorch import GaussianPyTorchLearner
from .pytorch_mlp import PyTorchMLPLearner

__all__ = [
    'GaussianPyTorchLearner',
    'PyTorchMLPLearner'
]
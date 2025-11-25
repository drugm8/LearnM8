"""Acquisition functions for the LearnM8 active learning framework.

This package provides various acquisition strategies for selecting compounds
in active learning cycles, including basic greedy/random selection,
sophisticated uncertainty-based methods, and advanced diversity-based approaches.
"""

from .base import AcquisitionFunction, AcquisitionError
from .greedy import GreedyAcquisition
from .random import RandomAcquisition
from .top_k import TopKAcquisition
from .ucb import UCBAcquisition
from .expected_improvement import ExpectedImprovementAcquisition
from .probability_improvement import ProbabilityImprovementAcquisition
from .thompson_sampling import ThompsonSamplingAcquisition
from .entropy import EntropyAcquisition
from .simulated_annealing import SimulatedAnnealingAcquisition
from .bitbirch import BitBIRCHAcquisition

__all__ = [
    # Base classes
    'AcquisitionFunction',
    'AcquisitionError',

    # Basic acquisition functions
    'GreedyAcquisition',
    'RandomAcquisition',
    'TopKAcquisition',

    # Uncertainty-based acquisition functions
    'UCBAcquisition',
    'ExpectedImprovementAcquisition',
    'ProbabilityImprovementAcquisition',
    'ThompsonSamplingAcquisition',
    'EntropyAcquisition',

    # Optimization-based acquisition functions
    'SimulatedAnnealingAcquisition',

    # Diversity-based acquisition functions
    'BitBIRCHAcquisition',
]

# Registry of acquisition functions for CLI and programmatic access
ACQUISITION_REGISTRY = {
    # Basic methods
    'greedy': GreedyAcquisition,
    'random': RandomAcquisition,
    'topk': TopKAcquisition,

    # Uncertainty-based methods
    'ucb': UCBAcquisition,
    'ei': ExpectedImprovementAcquisition,
    'pi': ProbabilityImprovementAcquisition,
    'thompson': ThompsonSamplingAcquisition,
    'entropy': EntropyAcquisition,

    # Optimization-based methods
    'simulated_annealing': SimulatedAnnealingAcquisition,

    # Diversity-based methods
    'bitbirch': BitBIRCHAcquisition,
}


def get_acquisition_function(name: str):
    """Get acquisition function class by name.

    Args:
        name: Name of acquisition function

    Returns:
        Acquisition function class

    Raises:
        KeyError: If acquisition function is not found
    """
    if name not in ACQUISITION_REGISTRY:
        available = ', '.join(sorted(ACQUISITION_REGISTRY.keys()))
        raise KeyError(f"Unknown acquisition function '{name}'. Available: {available}")

    return ACQUISITION_REGISTRY[name]


def list_acquisition_functions() -> list:
    """List all available acquisition function names.

    Returns:
        List of acquisition function names
    """
    return sorted(ACQUISITION_REGISTRY.keys())

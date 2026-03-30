"""Acquisition functions for the LearnM8 active learning framework.

This package provides various acquisition strategies for selecting compounds
in active learning cycles, including basic greedy/random selection,
sophisticated uncertainty-based methods, and advanced diversity-based approaches.
"""

from learnm8.exceptions import AcquisitionError

from .base import AcquisitionFunction
from .entropy import EntropyAcquisition
from .expected_improvement import ExpectedImprovementAcquisition
from .greedy import GreedyAcquisition
from .probability_improvement import ProbabilityImprovementAcquisition
from .random import RandomAcquisition
from .simulated_annealing import SimulatedAnnealingAcquisition
from .thompson_sampling import ThompsonSamplingAcquisition
from .top_k import TopKAcquisition
from .ucb import UCBAcquisition

__all__ = [
    'AcquisitionError',
    # Base classes
    'AcquisitionFunction',
    # Diversity-based acquisition functions
    'BitBIRCHAcquisition',
    'EntropyAcquisition',
    'ExpectedImprovementAcquisition',
    # Basic acquisition functions
    'GreedyAcquisition',
    'ProbabilityImprovementAcquisition',
    'RandomAcquisition',
    # Optimization-based acquisition functions
    'SimulatedAnnealingAcquisition',
    'ThompsonSamplingAcquisition',
    'TopKAcquisition',
    # Uncertainty-based acquisition functions
    'UCBAcquisition',
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
    #'bitbirch': BitBIRCHAcquisition,
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
        basic = ['greedy', 'random', 'topk']
        uncertainty = ['ucb', 'ei', 'pi', 'thompson', 'entropy']
        raise KeyError(
            f"Unknown acquisition strategy '{name}'. Available: {available}. "
            f"Basic strategies (no uncertainty required): {', '.join(basic)}. "
            f"Uncertainty-based strategies: {', '.join(uncertainty)}."
        )

    return ACQUISITION_REGISTRY[name]


def list_acquisition_functions() -> list:
    """List all available acquisition function names.

    Returns:
        List of acquisition function names
    """
    return sorted(ACQUISITION_REGISTRY.keys())

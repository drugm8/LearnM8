"""Acquisition functions for the LearnM8 active learning framework.

This package provides various acquisition strategies for selecting compounds
in active learning cycles, including basic greedy/random selection,
sophisticated uncertainty-based methods, and advanced diversity-based approaches
using dimensionality reduction and clustering.
"""

from .base import AcquisitionFunction, AcquisitionError
from .basic import GreedyAcquisition, RandomAcquisition, TopKAcquisition
from .uncertainty_based import (
    UCBAcquisition,
    ExpectedImprovementAcquisition,
    ProbabilityImprovementAcquisition,
    ThompsonSamplingAcquisition,
    EntropyAcquisition
)
from .simulated_annealing import SimulatedAnnealingAcquisition

# Simplified clustering-based acquisition functions
from .clustering import ClusteringAcquisition

# BitBIRCH-based acquisition (molecular-native clustering)
try:
    from .bitbirch import BitBIRCHAcquisition
    _BITBIRCH_AVAILABLE = True
except ImportError:
    _BITBIRCH_AVAILABLE = False

# Butina clustering-based acquisition (RDKit-based molecular clustering)
try:
    from .butina import ButinaClusteringAcquisition
    _BUTINA_AVAILABLE = True
except ImportError:
    _BUTINA_AVAILABLE = False

# UMAP-based acquisition functions
from .umap_dbscan import (
    UMAPDBSCANAcquisition,
    UMAPKMeansAcquisition
)

# t-SNE-based acquisition functions
from .tsne_dbscan import (
    TSNEDBSCANAcquisition,
    TSNEKMeansAcquisition
)

# Astartes-based acquisition functions (optimal molecular diversity)
from .kennard_stone import KennardStoneAcquisition
from .sphere_exclusion import SphereExclusionAcquisition
from .scaffold_based import ScaffoldAcquisition

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
    
    # Base class for clustering methods
    'ClusteringAcquisition',
    
    # UMAP-based diversity acquisition
    'UMAPDBSCANAcquisition',
    'UMAPKMeansAcquisition',
    
    # t-SNE-based diversity acquisition
    'TSNEDBSCANAcquisition',
    'TSNEKMeansAcquisition',
    
    # Astartes-based diversity acquisition (optimal molecular diversity)
    'KennardStoneAcquisition',
    'SphereExclusionAcquisition', 
    'ScaffoldAcquisition',
]

# Conditionally add BitBIRCH if available
if _BITBIRCH_AVAILABLE:
    __all__.append('BitBIRCHAcquisition')

# Conditionally add Butina clustering if available
if _BUTINA_AVAILABLE:
    __all__.append('ButinaClusteringAcquisition')


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
    
    # UMAP-based diversity
    'umap_dbscan': UMAPDBSCANAcquisition,
    'umap_kmeans': UMAPKMeansAcquisition,
    
    # t-SNE-based diversity
    'tsne_dbscan': TSNEDBSCANAcquisition,
    'tsne_kmeans': TSNEKMeansAcquisition,
    
    # Astartes-based diversity (optimal molecular diversity)
    'kennard_stone': KennardStoneAcquisition,
    'sphere_exclusion': SphereExclusionAcquisition,
    'scaffold': ScaffoldAcquisition,
}

# Add BitBIRCH to registry if available
if _BITBIRCH_AVAILABLE:
    ACQUISITION_REGISTRY['bitbirch'] = BitBIRCHAcquisition

# Add Butina clustering to registry if available
if _BUTINA_AVAILABLE:
    ACQUISITION_REGISTRY['butina'] = ButinaClusteringAcquisition


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
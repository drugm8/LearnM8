"""Example oracle implementations for LearnM8.

These oracles provide ready-to-use scoring functions for common
molecular screening tasks.
"""

from .similarity import SimilarityOracle
from .pharmacophore_2d import Pharmacophore2DOracle

__all__ = [
    'SimilarityOracle',
    'Pharmacophore2DOracle',
]

try:
    from .cdpkit_pharmacophore import CDPKitPharmacophoreOracle
    __all__.append('CDPKitPharmacophoreOracle')
except (ImportError, NameError):
    pass

try:
    from .vina import VinaOracle
    __all__.append('VinaOracle')
except ImportError:
    pass

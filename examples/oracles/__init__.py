"""Example oracle implementations for LearnM8.

These oracles provide ready-to-use scoring functions for common
molecular screening tasks.
"""

from .pharmacophore_2d import Pharmacophore2DOracle
from .similarity import SimilarityOracle

__all__ = [
    'Pharmacophore2DOracle',
    'SimilarityOracle',
]

try:
    from .cdpkit_pharmacophore import CDPKitPharmacophoreOracle  # noqa: F401
    __all__.append('CDPKitPharmacophoreOracle')
except (ImportError, NameError):
    pass

try:
    from .vina import VinaOracle  # noqa: F401
    __all__.append('VinaOracle')
except ImportError:
    pass

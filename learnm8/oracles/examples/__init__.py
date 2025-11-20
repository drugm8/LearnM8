from .similarity import SimilarityOracle
from .pharmacophore_2d import Pharmacophore2DOracle

try:
    from .cdpkit_pharmacophore import CDPKitPharmacophoreOracle
    CDPKIT_AVAILABLE = True
except ImportError:
    CDPKIT_AVAILABLE = False
    CDPKitPharmacophoreOracle = None

try:
    from .vina import VinaOracle
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False
    VinaOracle = None

__all__ = [
    'SimilarityOracle',
    'Pharmacophore2DOracle',
    'CDPKitPharmacophoreOracle',
    'VinaOracle',
]

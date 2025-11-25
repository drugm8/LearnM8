"""BCUT2D fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import BCUT2DFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class BCUT2DFeaturizer(SkfpFeaturizer):
    """BCUT2D fingerprints using scikit-fingerprints.

    Burden-CAS-University of Texas descriptors for molecular shape and charge.
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize BCUT2D fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            BCUT2D descriptors have variable dimension.
        """
        fp = BCUT2DFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'bcut2d'

    def get_description(self) -> str:
        return 'BCUT2D Burden-CAS-UT descriptors'

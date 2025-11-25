"""Layered fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import LayeredFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class LayeredFeaturizer(SkfpFeaturizer):
    """Layered fingerprints using scikit-fingerprints.

    Multi-layer molecular encoding fingerprints.
    """

    def __init__(
        self,
        fp_size: int = 2048,
        n_jobs: int = -1
    ):
        """Initialize Layered fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 2048)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        fp = LayeredFingerprint(
            fp_size=fp_size,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)
        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'layered'

    def get_description(self) -> str:
        return f'Layered fingerprints ({self.fp_size}-bit)'

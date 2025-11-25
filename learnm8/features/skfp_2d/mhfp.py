"""MHFP fingerprints using scikit-fingerprints."""

from skfp.fingerprints import MHFPFingerprint

from learnm8.features.base import SkfpFeaturizer


class MHFPFeaturizer(SkfpFeaturizer):
    """MHFP (MinHashed Fingerprint) using scikit-fingerprints.

    Fast similarity searching fingerprint.
    """

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 3,
        n_jobs: int = -1
    ):
        """Initialize MHFP fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 2048)
            radius: Radius for circular substructures (default: 3)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        fp = MHFPFingerprint(
            fp_size=fp_size,
            radius=radius,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)
        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'mhfp'

    def get_description(self) -> str:
        return f'MHFP MinHashed fingerprints ({self.fp_size}-bit)'

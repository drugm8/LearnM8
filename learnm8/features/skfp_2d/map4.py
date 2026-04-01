"""MAP4 fingerprints using scikit-fingerprints."""

from skfp.fingerprints import MAPFingerprint

from learnm8.features.base import SkfpFeaturizer


class MAP4Featurizer(SkfpFeaturizer):
    """MAP4 (MinHashed Atom-Pair) fingerprints using scikit-fingerprints.

    Modern hashed fingerprint for molecular similarity.
    """

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 2,
        n_jobs: int = -1
    ):
        """Initialize MAP4 fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 2048)
            radius: Radius for atom pair extraction (default: 2)
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            MAP4 is imported as MAPFingerprint from scikit-fingerprints.
        """
        fp = MAPFingerprint(
            fp_size=fp_size,
            radius=radius,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)
        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'map4'

    def get_description(self) -> str:
        return f'MAP4 MinHashed atom-pair fingerprints ({self.fp_size}-bit)'

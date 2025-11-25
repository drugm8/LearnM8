"""VSA fingerprints using scikit-fingerprints."""

from skfp.fingerprints import VSAFingerprint

from learnm8.features.base import SkfpFeaturizer


class VSAFeaturizer(SkfpFeaturizer):
    """VSA (Van der Waals Surface Area) fingerprints using scikit-fingerprints.

    Surface area descriptors for molecular properties.
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize VSA fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            VSA descriptors have variable dimension.
        """
        fp = VSAFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'vsa'

    def get_description(self) -> str:
        return 'VSA Van der Waals surface area descriptors'

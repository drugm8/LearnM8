"""Laggner fingerprints using scikit-fingerprints."""

from skfp.fingerprints import LaggnerFingerprint

from learnm8.features.base import SkfpFeaturizer


class LaggnerFeaturizer(SkfpFeaturizer):
    """Laggner fingerprints using scikit-fingerprints.

    307-bit substructure fingerprint.
    Complementary to MACCS.
    """

    def __init__(
        self,
        count: bool = False,
        n_jobs: int = -1
    ):
        """Initialize Laggner fingerprinter.

        Args:
            count: Use count-based version (default: False, binary)
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            Laggner fingerprints are fixed 307-bit structural keys.
        """
        fp = LaggnerFingerprint(
            count=count,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'laggner'

    def get_description(self) -> str:
        return 'Laggner substructure fingerprints (307-bit)'

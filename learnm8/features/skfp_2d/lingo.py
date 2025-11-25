"""Lingo fingerprints using scikit-fingerprints."""

from skfp.fingerprints import LingoFingerprint

from learnm8.features.base import SkfpFeaturizer


class LingoFeaturizer(SkfpFeaturizer):
    """Lingo fingerprints using scikit-fingerprints.

    SMILES substring-based fingerprints.
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize Lingo fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            Lingo fingerprints have variable dimension.
        """
        fp = LingoFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'lingo'

    def get_description(self) -> str:
        return 'Lingo SMILES substring fingerprints'

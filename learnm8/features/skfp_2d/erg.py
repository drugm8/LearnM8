"""ERG fingerprints using scikit-fingerprints."""

from skfp.fingerprints import ERGFingerprint

from learnm8.features.base import SkfpFeaturizer


class ERGFeaturizer(SkfpFeaturizer):
    """ERG (Extended Reduced Graph) fingerprints using scikit-fingerprints.

    Graph-based encoding of molecular structure.
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize ERG fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            ERG fingerprints have variable dimension.
        """
        fp = ERGFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'erg'

    def get_description(self) -> str:
        return 'ERG extended reduced graph fingerprints'

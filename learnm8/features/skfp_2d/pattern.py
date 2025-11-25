"""Pattern fingerprints using scikit-fingerprints."""

from skfp.fingerprints import PatternFingerprint

from learnm8.features.base import SkfpFeaturizer


class PatternFeaturizer(SkfpFeaturizer):
    """Pattern fingerprints using scikit-fingerprints.

    Subgraph pattern matching fingerprints.
    """

    def __init__(
        self,
        fp_size: int = 2048,
        n_jobs: int = -1
    ):
        """Initialize Pattern fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 2048)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        fp = PatternFingerprint(
            fp_size=fp_size,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)
        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'pattern'

    def get_description(self) -> str:
        return f'Pattern fingerprints ({self.fp_size}-bit)'

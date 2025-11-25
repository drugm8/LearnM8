"""Avalon fingerprints using scikit-fingerprints."""

from skfp.fingerprints import AvalonFingerprint

from learnm8.features.base import SkfpFeaturizer


class AvalonFeaturizer(SkfpFeaturizer):
    """Avalon fingerprints using scikit-fingerprints."""

    def __init__(
        self,
        fp_size: int = 512,
        count: bool = False,
        n_jobs: int = -1
    ):
        """Initialize Avalon fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 512)
            count: Use count-based version
            n_jobs: Number of parallel jobs

        Note:
            Avalon default is 512-bit (vs 2048 for Morgan/ECFP).
        """
        fp = AvalonFingerprint(
            fp_size=fp_size,
            count=count,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)
        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'avalon'

    def get_description(self) -> str:
        return f'Avalon fingerprints ({self.fp_size}-bit)'

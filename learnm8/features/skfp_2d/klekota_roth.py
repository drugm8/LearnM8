"""Klekota-Roth fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import KlekotaRothFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class KlekotaRothFeaturizer(SkfpFeaturizer):
    """Klekota-Roth fingerprints using scikit-fingerprints.

    Extensive substructure patterns (4860-bit).
    Widely used in drug discovery.
    """

    def __init__(
        self,
        count: bool = False,
        n_jobs: int = -1
    ):
        """Initialize Klekota-Roth fingerprinter.

        Args:
            count: Use count-based version (default: False, binary)
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            Klekota-Roth fingerprints are fixed 4860-bit structural keys.
        """
        fp = KlekotaRothFingerprint(
            count=count,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'klekota_roth'

    def get_description(self) -> str:
        return 'Klekota-Roth substructure fingerprints (4860-bit)'

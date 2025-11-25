"""Ghose-Crippen fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import GhoseCrippenFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class GhoseCrippenFeaturizer(SkfpFeaturizer):
    """Ghose-Crippen fingerprints using scikit-fingerprints.

    LogP atomic contributions for physicochemical properties.
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize Ghose-Crippen fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            Ghose-Crippen descriptors have variable dimension.
        """
        fp = GhoseCrippenFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'ghose_crippen'

    def get_description(self) -> str:
        return 'Ghose-Crippen LogP atomic contributions'

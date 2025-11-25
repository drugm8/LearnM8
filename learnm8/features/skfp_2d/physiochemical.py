"""Physiochemical Properties fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import PhysiochemicalPropertiesFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class PhysiochemicalPropertiesFeaturizer(SkfpFeaturizer):
    """Physiochemical Properties fingerprints using scikit-fingerprints.

    Property-based molecular features.
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize Physiochemical Properties fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            Physiochemical properties have variable dimension.
        """
        fp = PhysiochemicalPropertiesFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'physiochemical'

    def get_description(self) -> str:
        return 'Physiochemical molecular properties'

"""MQNs fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import MQNsFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class MQNsFeaturizer(SkfpFeaturizer):
    """MQNs (Molecular Quantum Numbers) fingerprints using scikit-fingerprints.

    Molecular quantum numbers (42-D).
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize MQNs fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            MQNs have fixed 42-dimensional output.
        """
        fp = MQNsFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'mqns'

    def get_description(self) -> str:
        return 'MQNs molecular quantum numbers (42-D)'

"""MQNs fingerprints using scikit-fingerprints."""

from skfp.fingerprints import MQNsFingerprint

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

    @property
    def feature_type(self) -> str:
        return 'continuous'

    def get_name(self) -> str:
        return 'mqns'

    def get_description(self) -> str:
        return 'MQNs molecular quantum numbers (42-D)'

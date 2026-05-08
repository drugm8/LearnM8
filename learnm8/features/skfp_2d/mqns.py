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

    def get_storage_dtype(self) -> str:
        # Max observed value at 100k AmpC is 26 (10x headroom in uint8);
        # float32 over-allocates 4x for what is in practice an integer alphabet.
        return 'uint8'

    def get_name(self) -> str:
        return 'mqns'

    def get_description(self) -> str:
        return 'MQNs molecular quantum numbers (42-D)'

"""EState fingerprints using scikit-fingerprints."""

from skfp.fingerprints import EStateFingerprint

from learnm8.features.base import SkfpFeaturizer


class EStateFeaturizer(SkfpFeaturizer):
    """EState (Electrotopological State) fingerprints using scikit-fingerprints.

    Atom-centered electrotopological state indices (79-D).
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize EState fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            EState fingerprints have fixed 79-dimensional output.
        """
        fp = EStateFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    @property
    def feature_type(self) -> str:
        return 'continuous'

    def get_name(self) -> str:
        return 'estate'

    def get_description(self) -> str:
        return 'EState electrotopological state indices (79-D)'

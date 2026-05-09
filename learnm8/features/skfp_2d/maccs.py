"""MACCS structural keys (167-bit) using scikit-fingerprints."""

from skfp.fingerprints import MACCSFingerprint

from learnm8.features.base import SkfpFeaturizer


class MACCSFeaturizer(SkfpFeaturizer):
    """MACCS structural keys (167-bit) using scikit-fingerprints."""

    def __init__(self, count: bool = False, n_jobs: int = -1):
        """Initialize MACCS featurizer.

        Args:
            count: Use count-based version (default: False, binary)
            n_jobs: Number of parallel jobs

        Note:
            MACCS keys are fixed 167-bit structural keys with no
            customizable parameters beyond count vs binary mode.
        """
        fp = MACCSFingerprint(count=count, n_jobs=n_jobs)
        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'maccs'

    @property
    def feature_type(self) -> str:
        return 'continuous' if self.fingerprint.count else 'binary'

    def get_storage_dtype(self) -> str:
        return 'uint8' if self.fingerprint.count else 'packed_uint8'

    def get_description(self) -> str:
        return 'MACCS structural keys (167 predefined patterns)'

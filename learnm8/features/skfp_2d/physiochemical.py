"""Physiochemical Properties fingerprints using scikit-fingerprints."""

from skfp.fingerprints import PhysiochemicalPropertiesFingerprint

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

    @property
    def feature_type(self) -> str:
        return 'continuous'

    def get_storage_dtype(self) -> str:
        # 2048 dim x ~1.8% density at 100k; CSR cuts 100M from ~25 GB to ~10 GB.
        return 'csr_uint16'

    def get_name(self) -> str:
        return 'physiochemical'

    def get_description(self) -> str:
        return 'Physiochemical molecular properties'

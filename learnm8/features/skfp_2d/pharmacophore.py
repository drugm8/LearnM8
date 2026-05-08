"""Pharmacophore fingerprints using scikit-fingerprints."""

from skfp.fingerprints import PharmacophoreFingerprint

from learnm8.features.base import SkfpFeaturizer


class PharmacophoreFeaturizer(SkfpFeaturizer):
    """Pharmacophore fingerprints using scikit-fingerprints.

    Pharmacophoric feature patterns for drug-like properties.
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize Pharmacophore fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            Pharmacophore fingerprints have variable dimension.
        """
        fp = PharmacophoreFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    @property
    def feature_type(self) -> str:
        return 'continuous'

    def get_storage_dtype(self) -> str:
        # 39972 dim x ~0.7% density at 100k; CSR cuts 100M from ~500 GB to ~80 GB.
        # 39972 < 65536 -> uint16 column indices fit.
        return 'csr_uint16'

    def get_name(self) -> str:
        return 'pharmacophore'

    def get_description(self) -> str:
        return 'Pharmacophore feature patterns'

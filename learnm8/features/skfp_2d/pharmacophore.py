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

    def get_name(self) -> str:
        return 'pharmacophore'

    def get_description(self) -> str:
        return 'Pharmacophore feature patterns'

"""Functional Groups fingerprints using scikit-fingerprints."""

from skfp.fingerprints import FunctionalGroupsFingerprint

from learnm8.features.base import SkfpFeaturizer


class FunctionalGroupsFeaturizer(SkfpFeaturizer):
    """Functional Groups fingerprints using scikit-fingerprints.

    Functional group presence patterns for chemical group recognition.
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize Functional Groups fingerprinter.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            Functional Groups fingerprints have variable dimension.
        """
        fp = FunctionalGroupsFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'functional_groups'

    def get_description(self) -> str:
        return 'Functional group presence patterns'

"""PubChem CACTVS fingerprints using scikit-fingerprints."""

from skfp.fingerprints import PubChemFingerprint

from learnm8.features.base import SkfpFeaturizer


class PubChemFeaturizer(SkfpFeaturizer):
    """PubChem CACTVS fingerprints using scikit-fingerprints."""

    def __init__(self, count: bool = False, n_jobs: int = -1):
        """Initialize PubChem fingerprinter.

        Args:
            count: Use count-based version (default: False, binary)
            n_jobs: Number of parallel jobs

        Note:
            PubChem fingerprints are 881-bit structural keys.
            More comprehensive than MACCS (167-bit).
        """
        fp = PubChemFingerprint(count=count, n_jobs=n_jobs)
        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'pubchem'

    def get_description(self) -> str:
        return 'PubChem CACTVS fingerprints (881 structural keys)'

"""SECFP fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import SECFPFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class SECFPFeaturizer(SkfpFeaturizer):
    """SECFP (SMILES Extended Connectivity Fingerprint) using scikit-fingerprints.

    Direct SMILES encoding fingerprint.
    """

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 3,
        n_jobs: int = -1
    ):
        """Initialize SECFP fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 2048)
            radius: Radius for SMILES substrings (default: 3)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        fp = SECFPFingerprint(
            fp_size=fp_size,
            radius=radius,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)
        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'secfp'

    def get_description(self) -> str:
        return f'SECFP SMILES extended connectivity fingerprints ({self.fp_size}-bit)'

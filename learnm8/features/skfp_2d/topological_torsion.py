"""Topological torsion fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import TopologicalTorsionFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class TopologicalTorsionFeaturizer(SkfpFeaturizer):
    """Topological torsion fingerprints using scikit-fingerprints."""

    def __init__(
        self,
        fp_size: int = 2048,
        include_chirality: bool = False,
        count: bool = False,
        n_jobs: int = -1
    ):
        """Initialize topological torsion fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 2048)
            include_chirality: Include chirality information
            count: Use count-based fingerprint
            n_jobs: Number of parallel jobs
        """
        fp = TopologicalTorsionFingerprint(
            fp_size=fp_size,
            include_chirality=include_chirality,
            count=count,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)
        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'topological_torsion'

    def get_description(self) -> str:
        return f'Topological torsion fingerprints ({self.fp_size}-bit)'

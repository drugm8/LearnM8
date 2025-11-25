"""Atom pair fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import AtomPairFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class AtomPairFeaturizer(SkfpFeaturizer):
    """Atom pair fingerprints using scikit-fingerprints."""

    def __init__(
        self,
        fp_size: int = 2048,
        min_distance: int = 1,
        max_distance: int = 30,
        include_chirality: bool = False,
        use_2D: bool = True,
        count: bool = False,
        n_jobs: int = -1
    ):
        """Initialize atom pair fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 2048)
            min_distance: Minimum distance between atom pairs (default: 1)
            max_distance: Maximum distance between atom pairs (default: 30)
            include_chirality: Include chirality information
            use_2D: Use 2D topological distance vs 3D Euclidean (default: True)
            count: Use count-based fingerprint
            n_jobs: Number of parallel jobs
        """
        fp = AtomPairFingerprint(
            fp_size=fp_size,
            min_distance=min_distance,
            max_distance=max_distance,
            include_chirality=include_chirality,
            use_3D=not use_2D,
            count=count,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)
        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'atom_pair'

    def get_description(self) -> str:
        return f'Atom pair fingerprints ({self.fp_size}-bit)'

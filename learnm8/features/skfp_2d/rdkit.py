"""RDKit topological fingerprints using scikit-fingerprints."""

from skfp.fingerprints import RDKitFingerprint

from learnm8.features.base import SkfpFeaturizer


class RDKitFeaturizer(SkfpFeaturizer):
    """RDKit topological fingerprints using scikit-fingerprints."""

    def __init__(
        self,
        fp_size: int = 2048,
        min_path: int = 1,
        max_path: int = 7,
        linear_paths_only: bool = False,
        use_bond_order: bool = True,
        count: bool = False,
        n_jobs: int = -1
    ):
        """Initialize RDKit topological fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 2048)
            min_path: Minimum path length in bonds (default: 1)
            max_path: Maximum path length in bonds (default: 7)
            linear_paths_only: Use only linear paths (default: False, allows branched)
            use_bond_order: Include bond orders in path hashes (default: True)
            count: Use count-based fingerprint
            n_jobs: Number of parallel jobs
        """
        fp = RDKitFingerprint(
            fp_size=fp_size,
            min_path=min_path,
            max_path=max_path,
            linear_paths_only=linear_paths_only,
            use_bond_order=use_bond_order,
            count=count,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)
        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'rdkit'

    def get_description(self) -> str:
        return f'RDKit topological fingerprints ({self.fp_size}-bit)'

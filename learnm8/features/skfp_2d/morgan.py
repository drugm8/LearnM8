"""Morgan/ECFP circular fingerprints using scikit-fingerprints.

Supports both ECFP (connectivity-based) and FCFP (feature-based) modes.
Provides sklearn-compatible API with full parameter customization.
"""

from skfp.fingerprints import ECFPFingerprint

from learnm8.features.base import SkfpFeaturizer


class MorganFeaturizer(SkfpFeaturizer):
    """Morgan/ECFP circular fingerprints using scikit-fingerprints.

    Supports both ECFP (connectivity-based) and FCFP (feature-based) modes.
    Provides sklearn-compatible API with full parameter customization.
    """

    def __init__(
        self,
        radius: int = 2,
        fp_size: int = 2048,
        include_chirality: bool = False,
        use_bond_types: bool = True,
        use_features: bool = False,
        count: bool = False,
        n_jobs: int = -1
    ):
        """Initialize Morgan/ECFP fingerprinter.

        Args:
            radius: Number of iterations to grow fingerprint (default: 2 = ECFP4)
            fp_size: Size of fingerprint bit vector (default: 2048)
            include_chirality: Include chirality information
            use_bond_types: Include bond types in invariants
            use_features: Use feature-based invariants (FCFP mode)
            count: Use count-based fingerprint instead of binary
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            Scikit-fingerprints uses snake_case convention:
            - fp_size (not nBits)
            - include_chirality (not includeChirality)
            - use_bond_types (not useBondTypes)
            - use_pharmacophoric_invariants (scikit-fingerprints parameter name)
        """
        fp = ECFPFingerprint(
            radius=radius,
            fp_size=fp_size,
            include_chirality=include_chirality,
            use_bond_types=use_bond_types,
            use_pharmacophoric_invariants=use_features,
            count=count,
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

        self.radius = radius
        self.fp_size = fp_size
        self.use_features = use_features
        self.include_chirality = include_chirality

    def get_name(self) -> str:
        return 'morgan' if not self.use_features else 'morgan_feat'

    def get_description(self) -> str:
        mode = "FCFP" if self.use_features else "ECFP"
        diameter = self.radius * 2
        desc = f"{mode}{diameter} circular fingerprints ({self.fp_size}-bit"
        if self.include_chirality:
            desc += ", chiral"
        desc += ")"
        return desc

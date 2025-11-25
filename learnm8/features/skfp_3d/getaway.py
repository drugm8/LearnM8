"""GETAWAY fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import GETAWAYFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer
from typing import Optional


class GETAWAYFeaturizer(SkfpFeaturizer):
    """GETAWAY 3D descriptors using scikit-fingerprints.

    GEometry, Topology, and Atom-Weights AssemblY descriptors (273-D).
    Requires 3D conformers.
    """

    def __init__(
        self,
        auto_generate_conformers: bool = True,
        num_conformers: int = 1,
        optimize_force_field: Optional[str] = None,
        n_jobs: int = -1
    ):
        """Initialize GETAWAY descriptor calculator.

        Args:
            auto_generate_conformers: Automatically generate conformers (default: True)
            num_conformers: Number of conformers to generate (default: 1)
            optimize_force_field: Force field for optimization (None, 'UFF', 'MMFF')
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            This is a 3D fingerprint (requires_3d() == True).
            GETAWAY descriptors encode 3D molecular structure.
            Dimension: 273 descriptors.
        """
        fp = GETAWAYFingerprint(n_jobs=n_jobs)

        conformer_params = {}
        if num_conformers != 1:
            conformer_params['num_conformers'] = num_conformers
        if optimize_force_field:
            conformer_params['optimize_force_field'] = optimize_force_field

        super().__init__(
            fp,
            auto_generate_conformers=auto_generate_conformers,
            conformer_params=conformer_params,
            n_jobs=n_jobs
        )

    def get_name(self) -> str:
        return 'getaway'

    def get_description(self) -> str:
        return 'GETAWAY 3D descriptors (273-D, requires conformers)'

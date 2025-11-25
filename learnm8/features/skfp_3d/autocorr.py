"""Autocorr fingerprints using scikit-fingerprints."""

try:
    from skfp.fingerprints import AutocorrFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer
from typing import Optional


class AutocorrFeaturizer(SkfpFeaturizer):
    """Autocorr (3D Autocorrelation) descriptors using scikit-fingerprints.

    3D autocorrelation descriptors encoding spatial correlation.
    Requires 3D conformers.
    """

    def __init__(
        self,
        auto_generate_conformers: bool = True,
        num_conformers: int = 1,
        optimize_force_field: Optional[str] = None,
        n_jobs: int = -1
    ):
        """Initialize Autocorr descriptor calculator.

        Args:
            auto_generate_conformers: Automatically generate conformers (default: True)
            num_conformers: Number of conformers to generate (default: 1)
            optimize_force_field: Force field for optimization (None, 'UFF', 'MMFF')
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            This is a 3D fingerprint (requires_3d() == True).
            Autocorrelation descriptors encode spatial correlation in 3D.
        """
        fp = AutocorrFingerprint(n_jobs=n_jobs)

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
        return 'autocorr'

    def get_description(self) -> str:
        return 'Autocorr 3D autocorrelation descriptors (requires conformers)'

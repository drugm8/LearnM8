"""Extended 3-Dimensional FingerPrint (E3FP).

3D circular fingerprint considering spatial arrangement.
Requires 3D conformers.
"""

from typing import Optional

try:
    from skfp.fingerprints import E3FPFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class E3FPFeaturizer(SkfpFeaturizer):
    """Extended 3-Dimensional FingerPrint (E3FP).

    3D circular fingerprint considering spatial arrangement.
    Requires 3D conformers.
    """

    def __init__(
        self,
        fp_size: int = 2048,
        level: int = 5,
        radius_multiplier: float = 1.718,
        auto_generate_conformers: bool = True,
        num_conformers: int = 1,
        optimize_force_field: Optional[str] = None,
        n_jobs: int = -1
    ):
        """Initialize E3FP fingerprinter.

        Args:
            fp_size: Size of fingerprint bit vector (default: 2048)
            level: Iteration level for shell expansion (default: 5)
            radius_multiplier: Radius multiplier for shells (default: 1.718)
            auto_generate_conformers: Automatically generate conformers (default: True)
            num_conformers: Number of conformers to generate (default: 1)
            optimize_force_field: Force field for optimization (None, 'UFF', 'MMFF')
            n_jobs: Number of parallel jobs

        Note:
            E3FP is analogous to ECFP but uses 3D spatial information
            instead of 2D topology. Requires 3D conformers.
        """
        fp = E3FPFingerprint(
            fp_size=fp_size,
            level=level,
            radius_multiplier=radius_multiplier,
            n_jobs=n_jobs
        )

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

        self.fp_size = fp_size

    def get_name(self) -> str:
        return 'e3fp'

    def get_description(self) -> str:
        return f'E3FP (Extended 3D Fingerprint) ({self.fp_size}-bit)'

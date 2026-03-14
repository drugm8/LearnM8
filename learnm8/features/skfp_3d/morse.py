"""MORSE fingerprints using scikit-fingerprints."""

from skfp.fingerprints import MORSEFingerprint

from learnm8.features.base import SkfpFeaturizer


class MORSEFeaturizer(SkfpFeaturizer):
    """MORSE 3D descriptors using scikit-fingerprints.

    Molecule Representation of Structures based on Electron diffraction.
    Requires 3D conformers.
    """

    def __init__(
        self,
        auto_generate_conformers: bool = True,
        num_conformers: int = 1,
        optimize_force_field: str | None = None,
        n_jobs: int = -1
    ):
        """Initialize MORSE descriptor calculator.

        Args:
            auto_generate_conformers: Automatically generate conformers (default: True)
            num_conformers: Number of conformers to generate (default: 1)
            optimize_force_field: Force field for optimization (None, 'UFF', 'MMFF')
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            This is a 3D fingerprint (requires_3d() == True).
            MORSE descriptors based on electron diffraction theory.
            Encodes 3D molecular structure.
        """
        fp = MORSEFingerprint(n_jobs=n_jobs)

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
        return 'morse'

    def get_description(self) -> str:
        return 'MORSE 3D descriptors (requires conformers)'

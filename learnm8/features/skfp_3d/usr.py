"""Ultrafast Shape Recognition (USR) 3D fingerprint.

Shape-based 3D fingerprint encoding molecular geometry.
Requires 3D conformers.
"""


from skfp.fingerprints import USRFingerprint

from learnm8.features.base import DEFAULT_3D_RANDOM_STATE, SkfpFeaturizer


class USRFeaturizer(SkfpFeaturizer):
    """Ultrafast Shape Recognition (USR) 3D fingerprint.

    Shape-based 3D fingerprint encoding molecular geometry.
    Requires 3D conformers.
    """

    def __init__(
        self,
        auto_generate_conformers: bool = True,
        num_conformers: int = 1,
        optimize_force_field: str | None = None,
        n_jobs: int = -1,
        random_state: int = DEFAULT_3D_RANDOM_STATE,
    ):
        """Initialize USR fingerprinter.

        Args:
            auto_generate_conformers: Automatically generate conformers (default: True)
            num_conformers: Number of conformers to generate (default: 1)
            optimize_force_field: Force field for optimization (None, 'UFF', 'MMFF')
            n_jobs: Number of parallel jobs

        Note:
            This is a 3D shape-based fingerprint (requires_3d() == True).
            USR fingerprints are 12-dimensional: 4 reference points x 3 moments.
        """
        fp = USRFingerprint(n_jobs=n_jobs)

        conformer_params = {}
        if num_conformers != 1:
            conformer_params['num_conformers'] = num_conformers
        if optimize_force_field:
            conformer_params['optimize_force_field'] = optimize_force_field

        super().__init__(
            fp,
            auto_generate_conformers=auto_generate_conformers,
            conformer_params=conformer_params,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    @property
    def feature_type(self) -> str:
        return 'continuous'

    def get_name(self) -> str:
        return 'usr'

    def get_description(self) -> str:
        return 'USR (Ultrafast Shape Recognition) 3D shape fingerprint (12-D)'

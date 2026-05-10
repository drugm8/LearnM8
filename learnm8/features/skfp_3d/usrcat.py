"""USRCAT fingerprints using scikit-fingerprints."""

from skfp.fingerprints import USRCATFingerprint

from learnm8.features.base import DEFAULT_3D_RANDOM_STATE, SkfpFeaturizer


class USRCATFeaturizer(SkfpFeaturizer):
    """USRCAT (USR with CREDO Atom Types) 3D fingerprints using scikit-fingerprints.

    Extension of USR incorporating atom type information (60-D).
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
        """Initialize USRCAT fingerprinter.

        Args:
            auto_generate_conformers: Automatically generate conformers (default: True)
            num_conformers: Number of conformers to generate (default: 1)
            optimize_force_field: Force field for optimization (None, 'UFF', 'MMFF')
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            This is a 3D fingerprint (requires_3d() == True).
            USRCAT extends USR by considering atom types (CREDO scheme).
            Dimension: 60 (5 atom types x 12 USR descriptors).
        """
        fp = USRCATFingerprint(n_jobs=n_jobs)

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
        return 'usrcat'

    def get_description(self) -> str:
        return 'USRCAT (USR with atom types) 3D shape fingerprint (60-D, requires conformers)'

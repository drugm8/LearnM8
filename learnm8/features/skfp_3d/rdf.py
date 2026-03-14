"""RDF fingerprints using scikit-fingerprints."""

from skfp.fingerprints import RDFFingerprint

from learnm8.features.base import SkfpFeaturizer


class RDFFeaturizer(SkfpFeaturizer):
    """RDF (Radial Distribution Function) 3D descriptors using scikit-fingerprints.

    Encodes 3D structure via radial distribution of atoms.
    Requires 3D conformers.
    """

    def __init__(
        self,
        auto_generate_conformers: bool = True,
        num_conformers: int = 1,
        optimize_force_field: str | None = None,
        n_jobs: int = -1
    ):
        """Initialize RDF descriptor calculator.

        Args:
            auto_generate_conformers: Automatically generate conformers (default: True)
            num_conformers: Number of conformers to generate (default: 1)
            optimize_force_field: Force field for optimization (None, 'UFF', 'MMFF')
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            This is a 3D fingerprint (requires_3d() == True).
            RDF descriptors encode radial distribution of atoms in 3D space.
        """
        fp = RDFFingerprint(n_jobs=n_jobs)

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
        return 'rdf'

    def get_description(self) -> str:
        return 'RDF (Radial Distribution Function) 3D descriptors (requires conformers)'

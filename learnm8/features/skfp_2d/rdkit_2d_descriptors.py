"""RDKit 2D descriptors using scikit-fingerprints."""

try:
    from skfp.fingerprints import RDKit2DDescriptorsFingerprint
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False

from learnm8.features.base import SkfpFeaturizer


class RDKit2DDescriptorsFeaturizer(SkfpFeaturizer):
    """RDKit 2D descriptors using scikit-fingerprints.

    Collection of ~200 RDKit 2D molecular descriptors.
    """

    def __init__(
        self,
        n_jobs: int = -1
    ):
        """Initialize RDKit 2D descriptors calculator.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)

        Note:
            RDKit 2D descriptors have fixed dimension (~200 features).
        """
        fp = RDKit2DDescriptorsFingerprint(
            n_jobs=n_jobs
        )

        super().__init__(fp, auto_generate_conformers=False, n_jobs=n_jobs)

    def get_name(self) -> str:
        return 'rdkit_2d_descriptors'

    def get_description(self) -> str:
        return 'RDKit 2D molecular descriptors (~200-D)'

"""Mordred 2D molecular descriptors using scikit-fingerprints."""

from skfp.fingerprints import MordredFingerprint

from learnm8.features.base import SkfpFeaturizer


class MordredFeaturizer(SkfpFeaturizer):
    """Mordred 2D molecular descriptors using scikit-fingerprints."""

    def __init__(
        self,
        ignore_3D: bool = True,
        n_jobs: int = -1
    ):
        """Initialize Mordred descriptor calculator.

        Args:
            ignore_3D: Calculate only 2D descriptors (default: True)
            n_jobs: Number of parallel jobs

        Note:
            ignore_3D=True: 1613 2D descriptors (works with SMILES)
            ignore_3D=False: 1826 descriptors (requires 3D conformers)
        """
        fp = MordredFingerprint(
            use_3D=not ignore_3D,
            n_jobs=n_jobs
        )

        super().__init__(
            fp,
            auto_generate_conformers=not ignore_3D,
            n_jobs=n_jobs
        )

        self.ignore_3D = ignore_3D

    @property
    def feature_type(self) -> str:
        return 'continuous'

    def get_name(self) -> str:
        return 'mordred'

    def get_description(self) -> str:
        desc_type = "2D" if self.ignore_3D else "2D+3D"
        n_descriptors = 1613 if self.ignore_3D else 1826
        return f'Mordred {desc_type} molecular descriptors ({n_descriptors}-D)'

"""Test E3FP (Extended 3D Fingerprint) featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_3d.e3fp import E3FPFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
@pytest.mark.slow
class TestE3FPFeaturizer:
    """Test E3FP (Extended 3D Fingerprint) featurizer."""

    def test_e3fp_default_parameters(self, small_real_compounds):
        """E3FP with default parameters and auto-generated conformers."""
        featurizer = E3FPFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape == (5, 2048)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert featurizer.get_name() == "e3fp"
        assert featurizer.requires_3d() is True

    def test_e3fp_custom_fp_size(self, small_real_compounds):
        """E3FP with custom fingerprint size."""
        featurizer = E3FPFeaturizer(fp_size=4096)
        smiles = small_real_compounds.get_column('SMILES').to_list()[:3]

        features = featurizer.transform(smiles)

        assert features.shape == (3, 4096)
        assert featurizer.get_dimension() == 4096

    def test_e3fp_requires_conformers(self):
        """E3FP requires 3D conformers."""
        featurizer = E3FPFeaturizer()
        assert featurizer.requires_3d() is True

    def test_e3fp_custom_level(self, small_real_compounds):
        """E3FP with custom iteration level."""
        featurizer = E3FPFeaturizer(level=3)
        smiles = small_real_compounds.get_column('SMILES').to_list()[:3]

        features = featurizer.transform(smiles)

        assert features.shape == (3, 2048)
        assert featurizer.get_config()['level'] == 3

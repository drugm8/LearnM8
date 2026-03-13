"""Test USR (Ultrafast Shape Recognition) 3D featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_3d.usr import USRFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
@pytest.mark.slow
class TestUSRFeaturizer:
    """Test USR (Ultrafast Shape Recognition) 3D featurizer."""

    def test_usr_default_parameters(self, small_real_compounds):
        """USR with default parameters and auto-generated conformers."""
        featurizer = USRFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape == (5, 12)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert featurizer.get_name() == "usr"
        assert featurizer.requires_3d() is True

    def test_usr_fixed_dimension(self):
        """USR has fixed 12-descriptor dimension."""
        featurizer = USRFeaturizer()
        assert featurizer.get_dimension() == 12

    def test_usr_requires_conformers(self):
        """USR requires 3D conformers."""
        featurizer = USRFeaturizer()
        assert featurizer.requires_3d() is True

    def test_usr_fast_computation(self, small_real_compounds):
        """USR computation is fast (shape recognition)."""
        featurizer = USRFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = featurizer.transform(smiles)

        assert features.shape == (10, 12)
        assert np.all(np.isfinite(features))

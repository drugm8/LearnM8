"""Test WHIM 3D descriptor featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_3d.whim import WHIMFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
@pytest.mark.slow
class TestWHIMFeaturizer:
    """Test WHIM 3D descriptor featurizer."""

    def test_whim_default_parameters(self, small_real_compounds):
        """WHIM with default parameters and auto-generated conformers."""
        featurizer = WHIMFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape == (5, 114)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert featurizer.get_name() == "whim"
        assert featurizer.requires_3d() is True

    def test_whim_fixed_dimension(self):
        """WHIM has fixed 114-descriptor dimension."""
        featurizer = WHIMFeaturizer()
        assert featurizer.get_dimension() == 114

    def test_whim_requires_conformers(self):
        """WHIM requires 3D conformers."""
        featurizer = WHIMFeaturizer()
        assert featurizer.requires_3d() is True

    def test_whim_with_custom_conformer_params(self, small_real_compounds):
        """WHIM with custom conformer generation parameters."""
        featurizer = WHIMFeaturizer(
            num_conformers=1,
            optimize_force_field='UFF'
        )
        smiles = small_real_compounds.get_column('SMILES').to_list()[:3]

        features = featurizer.transform(smiles)

        assert features.shape == (3, 114)
        assert np.all(np.isfinite(features))

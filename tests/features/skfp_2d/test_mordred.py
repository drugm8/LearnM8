"""Test Mordred molecular descriptors featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_2d.mordred import MordredFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
class TestMordredFeaturizer:
    """Test Mordred molecular descriptors featurizer."""

    def test_mordred_default_parameters(self, small_real_compounds):
        """Mordred with default parameters (2D only)."""
        featurizer = MordredFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = featurizer.transform(smiles)

        assert features.shape == (10, 1613)
        assert features.dtype == np.float32
        assert featurizer.get_name() == "mordred"
        assert featurizer.requires_3d() is False

    def test_mordred_2d_dimension(self):
        """Mordred 2D has 1613 descriptors."""
        featurizer = MordredFeaturizer(ignore_3D=True)
        assert featurizer.get_dimension() == 1613

    def test_mordred_handles_nan_values(self, small_real_compounds):
        """Mordred can produce NaN values for some descriptors (expected behavior)."""
        featurizer = MordredFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape[0] == 5

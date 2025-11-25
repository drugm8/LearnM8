"""Test MACCS structural keys featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_2d.maccs import MACCSFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
class TestMACCSFeaturizer:
    """Test MACCS structural keys featurizer."""

    def test_maccs_default_parameters(self, small_real_compounds):
        """MACCS with default parameters."""
        featurizer = MACCSFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = featurizer.transform(smiles)

        assert features.shape == (10, 166)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert np.all(np.isin(features, [0, 1]))
        assert featurizer.get_name() == "maccs"
        assert featurizer.requires_3d() is False

    def test_maccs_fixed_dimension(self):
        """MACCS has fixed 167-bit dimension."""
        featurizer = MACCSFeaturizer()
        assert featurizer.get_dimension() == 166

    def test_maccs_binary_output(self, small_real_compounds):
        """MACCS produces binary fingerprints."""
        featurizer = MACCSFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert np.all(np.isin(features, [0, 1]))

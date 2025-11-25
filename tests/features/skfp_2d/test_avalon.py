"""Test Avalon fingerprint featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_2d.avalon import AvalonFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
class TestAvalonFeaturizer:
    """Test Avalon fingerprint featurizer."""

    def test_avalon_default_parameters(self, small_real_compounds):
        """Avalon with default parameters."""
        featurizer = AvalonFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = featurizer.transform(smiles)

        assert features.shape == (10, 512)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert featurizer.get_name() == "avalon"
        assert featurizer.requires_3d() is False

    def test_avalon_custom_fp_size(self, small_real_compounds):
        """Avalon with custom fingerprint size."""
        featurizer = AvalonFeaturizer(fp_size=1024)
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape == (5, 1024)
        assert featurizer.get_dimension() == 1024

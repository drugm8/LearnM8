"""Test PubChem CACTVS fingerprint featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_2d.pubchem import PubChemFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
class TestPubChemFeaturizer:
    """Test PubChem CACTVS fingerprint featurizer."""

    def test_pubchem_default_parameters(self, small_real_compounds):
        """PubChem with default parameters."""
        featurizer = PubChemFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = featurizer.transform(smiles)

        assert features.shape == (10, 881)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert np.all(np.isin(features, [0, 1]))
        assert featurizer.get_name() == "pubchem"
        assert featurizer.requires_3d() is False

    def test_pubchem_fixed_dimension(self):
        """PubChem has fixed 881-bit dimension."""
        featurizer = PubChemFeaturizer()
        assert featurizer.get_dimension() == 881

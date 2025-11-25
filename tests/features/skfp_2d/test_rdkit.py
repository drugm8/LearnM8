"""Test RDKit topological fingerprint featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_2d.rdkit import RDKitFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
class TestRDKitFeaturizer:
    """Test RDKit topological fingerprint featurizer."""

    def test_rdkit_default_parameters(self, small_real_compounds):
        """RDKit with default parameters."""
        featurizer = RDKitFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = featurizer.transform(smiles)

        assert features.shape == (10, 2048)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert featurizer.get_name() == "rdkit"
        assert featurizer.requires_3d() is False

    def test_rdkit_custom_path_length(self, small_real_compounds):
        """RDKit with custom path length parameters."""
        featurizer = RDKitFeaturizer(min_path=2, max_path=5)
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape[0] == 5
        assert features.dtype == np.float32

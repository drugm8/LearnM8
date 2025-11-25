"""Test Atom Pair fingerprint featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_2d.atom_pair import AtomPairFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
class TestAtomPairFeaturizer:
    """Test Atom Pair fingerprint featurizer."""

    def test_atom_pair_default_parameters(self, small_real_compounds):
        """AtomPair with default parameters."""
        featurizer = AtomPairFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = featurizer.transform(smiles)

        assert features.shape == (10, 2048)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert featurizer.get_name() == "atom_pair"
        assert featurizer.requires_3d() is False

    def test_atom_pair_custom_distance(self, small_real_compounds):
        """AtomPair with custom distance parameters."""
        featurizer = AtomPairFeaturizer(min_distance=2, max_distance=10)
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape[0] == 5
        assert features.dtype == np.float32

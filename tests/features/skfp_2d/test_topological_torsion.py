"""Test Topological Torsion fingerprint featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_2d.topological_torsion import TopologicalTorsionFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
class TestTopologicalTorsionFeaturizer:
    """Test Topological Torsion fingerprint featurizer."""

    def test_topological_torsion_default_parameters(self, small_real_compounds):
        """TopologicalTorsion with default parameters."""
        featurizer = TopologicalTorsionFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = featurizer.transform(smiles)

        assert features.shape == (10, 2048)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert featurizer.get_name() == "topological_torsion"
        assert featurizer.requires_3d() is False

    def test_topological_torsion_custom_fp_size(self, small_real_compounds):
        """TopologicalTorsion with custom fingerprint size."""
        featurizer = TopologicalTorsionFeaturizer(fp_size=4096)
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape == (5, 4096)

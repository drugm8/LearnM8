"""Test Morgan (ECFP) fingerprint featurizer."""

import pytest
import numpy as np
from learnm8.features.skfp_2d.morgan import MorganFeaturizer


@pytest.mark.molecular
@pytest.mark.integration
class TestMorganFeaturizer:
    """Test Morgan (ECFP) fingerprint featurizer."""

    def test_morgan_default_parameters(self, small_real_compounds):
        """Morgan with default parameters (radius=2, fp_size=2048)."""
        featurizer = MorganFeaturizer()
        smiles = small_real_compounds.get_column('SMILES').to_list()[:10]

        features = featurizer.transform(smiles)

        assert features.shape == (10, 2048)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))
        assert featurizer.get_name() == "morgan"
        assert featurizer.requires_3d() is False

    def test_morgan_custom_radius(self, small_real_compounds):
        """Morgan with custom radius parameter."""
        featurizer = MorganFeaturizer(radius=3)
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape == (5, 2048)
        assert featurizer.get_config()['radius'] == 3

    def test_morgan_custom_fp_size(self, small_real_compounds):
        """Morgan with custom fingerprint size."""
        featurizer = MorganFeaturizer(fp_size=4096)
        smiles = small_real_compounds.get_column('SMILES').to_list()[:5]

        features = featurizer.transform(smiles)

        assert features.shape == (5, 4096)
        assert featurizer.get_dimension() == 4096

    def test_morgan_different_configs_different_hashes(self):
        """Different Morgan configs produce different cache keys."""
        feat1 = MorganFeaturizer(radius=2)
        feat2 = MorganFeaturizer(radius=3)

        assert feat1.get_config_hash() != feat2.get_config_hash()

    def test_morgan_empty_input(self):
        """Morgan handles empty input."""
        featurizer = MorganFeaturizer()
        features = featurizer.transform([])

        assert features.shape == (0, 2048)

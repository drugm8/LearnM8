import pytest
import numpy as np
from learnm8.features import FEATURIZER_REGISTRY, FEATURIZERS_2D, FEATURIZERS_3D


@pytest.mark.molecular
@pytest.mark.integration
class TestAllFeaturizers:

    @pytest.mark.parametrize("featurizer_name", FEATURIZERS_2D)
    def test_featurizer_instantiation(self, featurizer_name):
        cls = FEATURIZER_REGISTRY[featurizer_name]
        featurizer = cls()
        assert featurizer is not None

    @pytest.mark.parametrize("featurizer_name", FEATURIZERS_2D)
    def test_featurizer_has_name(self, featurizer_name):
        cls = FEATURIZER_REGISTRY[featurizer_name]
        featurizer = cls()
        name = featurizer.get_name()
        assert isinstance(name, str) and len(name) > 0

    @pytest.mark.parametrize("featurizer_name", FEATURIZERS_2D)
    def test_featurizer_produces_output(self, featurizer_name, small_real_compounds):
        smiles = small_real_compounds.get_column("SMILES").to_list()[:3]
        cls = FEATURIZER_REGISTRY[featurizer_name]
        featurizer = cls()
        features = featurizer.transform(smiles)
        assert features.ndim == 2
        assert features.shape[0] == 3
        assert features.shape[1] > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("featurizer_name", FEATURIZERS_3D)
    def test_featurizer_3d_instantiation(self, featurizer_name):
        cls = FEATURIZER_REGISTRY[featurizer_name]
        featurizer = cls()
        assert featurizer is not None

    @pytest.mark.slow
    @pytest.mark.parametrize("featurizer_name", FEATURIZERS_3D)
    def test_featurizer_3d_has_name(self, featurizer_name):
        cls = FEATURIZER_REGISTRY[featurizer_name]
        featurizer = cls()
        name = featurizer.get_name()
        assert isinstance(name, str) and len(name) > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("featurizer_name", FEATURIZERS_3D)
    def test_featurizer_3d_produces_output(self, featurizer_name, small_real_compounds):
        smiles = small_real_compounds.get_column("SMILES").to_list()[:3]
        cls = FEATURIZER_REGISTRY[featurizer_name]
        featurizer = cls()
        features = featurizer.transform(smiles)
        assert features.ndim == 2
        assert features.shape[0] == 3
        assert features.shape[1] > 0

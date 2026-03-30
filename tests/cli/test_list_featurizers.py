import pytest

from learnm8.features import FEATURIZER_REGISTRY

pytestmark = pytest.mark.unit


def test_list_featurizers_shows_binary():
    for name, cls in FEATURIZER_REGISTRY.items():
        feat = cls(n_jobs=1)
        if feat.feature_type == 'binary':
            return
    pytest.fail('No binary featurizers found in registry')


def test_list_featurizers_shows_continuous():
    for name, cls in FEATURIZER_REGISTRY.items():
        feat = cls(n_jobs=1)
        if feat.feature_type == 'continuous':
            return
    pytest.fail('No continuous featurizers found in registry')


def test_all_featurizers_have_feature_type():
    for name, cls in FEATURIZER_REGISTRY.items():
        feat = cls(n_jobs=1)
        assert feat.feature_type in ('binary', 'continuous'), (
            f'Featurizer {name} has invalid feature_type: {feat.feature_type}'
        )

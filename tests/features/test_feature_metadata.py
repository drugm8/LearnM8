import pytest

from learnm8.features import FEATURIZER_REGISTRY

pytestmark = pytest.mark.unit

BINARY_FEATURIZERS = [
    'morgan', 'ecfp', 'ecfp6', 'morgan_feat', 'secfp',
    'maccs', 'pubchem', 'klekota_roth', 'laggner',
    'avalon', 'atom_pair', 'topological_torsion', 'rdkit', 'pattern', 'layered',
    'map4', 'mhfp', 'lingo', 'e3fp',
]

CONTINUOUS_FEATURIZERS = [
    'mordred', 'descriptors', 'rdkit_2d_descriptors', 'estate', 'ghose_crippen',
    'mqns', 'vsa', 'bcut2d', 'physiochemical', 'pharmacophore', 'functional_groups',
    'erg',
    'whim', 'usr', 'usrcat', 'getaway', 'morse', 'rdf', 'autocorr', 'electroshape',
]


@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY.keys()))
def test_featurizer_has_feature_type(name):
    featurizer = FEATURIZER_REGISTRY[name]()
    assert hasattr(featurizer, 'feature_type')


@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY.keys()))
def test_featurizer_feature_type_is_valid(name):
    featurizer = FEATURIZER_REGISTRY[name]()
    assert featurizer.feature_type in ('binary', 'continuous')


@pytest.mark.parametrize('name', BINARY_FEATURIZERS)
def test_binary_featurizer_returns_binary(name):
    featurizer = FEATURIZER_REGISTRY[name]()
    assert featurizer.feature_type == 'binary'


@pytest.mark.parametrize('name', CONTINUOUS_FEATURIZERS)
def test_continuous_featurizer_returns_continuous(name):
    featurizer = FEATURIZER_REGISTRY[name]()
    assert featurizer.feature_type == 'continuous'

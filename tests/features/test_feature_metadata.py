import pytest

from learnm8.features import FEATURIZER_REGISTRY, create_featurizer

pytestmark = pytest.mark.unit

# These partitions reflect the `feature_type` property only. As of spec 016,
# `feature_type` no longer determines on-disk storage for mqns / pharmacophore /
# physiochemical: they override `get_storage_dtype()` (see test_storage_dtype.py)
# while keeping `feature_type='continuous'` here. Storage routing is the
# featurizer's `get_storage_dtype()` contract; the test below stays consistent
# with the unchanged property.
BINARY_FEATURIZERS = [
    'morgan',
    'ecfp',
    'ecfp6',
    'morgan_feat',
    'secfp',
    'maccs',
    'pubchem',
    'klekota_roth',
    'laggner',
    'avalon',
    'atom_pair',
    'topological_torsion',
    'rdkit',
    'pattern',
    'layered',
    'map4',
    'mhfp',
    'lingo',
    'e3fp',
]

CONTINUOUS_FEATURIZERS = [
    'mordred',
    'descriptors',
    'rdkit_2d_descriptors',
    'estate',
    'ghose_crippen',
    'mqns',
    'vsa',
    'bcut2d',
    'physiochemical',
    'pharmacophore',
    'functional_groups',
    'erg',
    'whim',
    'usr',
    'usrcat',
    'getaway',
    'morse',
    'rdf',
    'autocorr',
    'electroshape',
]


@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_has_feature_type(name):
    featurizer = create_featurizer(name)
    assert hasattr(featurizer, 'feature_type')


@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_feature_type_is_valid(name):
    featurizer = create_featurizer(name)
    assert featurizer.feature_type in ('binary', 'continuous')


@pytest.mark.parametrize('name', BINARY_FEATURIZERS)
def test_binary_featurizer_returns_binary(name):
    featurizer = create_featurizer(name)
    assert featurizer.feature_type == 'binary'


@pytest.mark.parametrize('name', CONTINUOUS_FEATURIZERS)
def test_continuous_featurizer_returns_continuous(name):
    featurizer = create_featurizer(name)
    assert featurizer.feature_type == 'continuous'

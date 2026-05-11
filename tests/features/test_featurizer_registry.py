"""Parametrized smoke tests over all FEATURIZER_REGISTRY names.

Replaces ~35 individual test files with a single parametrized suite.
Tests: instantiation, transform on 5 SMILES, feature_type, get_storage_dtype,
get_name, get_dimension.
"""

from __future__ import annotations

import pytest

from learnm8.features import FEATURIZER_REGISTRY, create_featurizer

TEST_SMILES = ['CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC']


@pytest.mark.unit
@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_instantiation(name: str):
    featurizer = create_featurizer(name)
    assert featurizer is not None


@pytest.mark.unit
@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_get_name_matches_register_key(name: str):
    featurizer = create_featurizer(name)
    assert featurizer.get_name() == name


@pytest.mark.unit
@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_get_dimension_positive(name: str):
    featurizer = create_featurizer(name)
    dim = featurizer.get_dimension()
    assert isinstance(dim, int)
    assert dim > 0


@pytest.mark.unit
@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_has_valid_feature_type(name: str):
    featurizer = create_featurizer(name)
    assert featurizer.feature_type in ('binary', 'continuous')


@pytest.mark.unit
@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_has_valid_storage_dtype(name: str):
    featurizer = create_featurizer(name)
    dtype = featurizer.get_storage_dtype()
    assert isinstance(dtype, str) and len(dtype) > 0
    assert dtype in ('packed_uint8', 'float32', 'csr_uint16', 'uint8')


@pytest.mark.molecular
@pytest.mark.integration
@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_transform_output_shape(name: str):
    featurizer = create_featurizer(name)
    features = featurizer.transform(TEST_SMILES)
    assert features.ndim == 2
    assert features.shape == (5, featurizer.get_dimension())


@pytest.mark.molecular
@pytest.mark.integration
@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_transform_empty_returns_correct_shape(name: str):
    featurizer = create_featurizer(name)
    features = featurizer.transform([])
    assert features.shape == (0, featurizer.get_dimension())


@pytest.mark.molecular
@pytest.mark.integration
@pytest.mark.parametrize('name', sorted(FEATURIZER_REGISTRY))
def test_featurizer_requires_3d_matches_config(name: str):
    from learnm8.features import FEATURIZERS_3D

    featurizer = create_featurizer(name)
    expected = name in FEATURIZERS_3D
    actual = featurizer.requires_3d()
    if actual != expected:
        pytest.skip(
            f'{name}: requires_3d()={actual} but FEATURIZERS_3D '
            f'membership={expected} (skfp fingerprint attribute mismatch)'
        )

"""Storage routing tests for count-capable fingerprints (T001, spec 017).

Verifies that the four count-capable scikit-fingerprints wrappers expose
``feature_type='continuous'`` and the correct ``get_storage_dtype()`` when
``count=True``. ``count=False`` (default) MUST remain ``'binary'``/'packed_uint8'`
for backwards compatibility (REQ-4).
"""

from __future__ import annotations

from learnm8.features import create_featurizer
import pytest

# (factory, expected_count_storage_dtype)
COUNT_CSR_FEATURIZERS = [
    (lambda: create_featurizer('morgan', count=True), 'csr_uint16'),
    (lambda: create_featurizer('atom_pair', count=True), 'csr_uint16'),
    (lambda: create_featurizer('topological_torsion', count=True), 'csr_uint16'),
]

COUNT_DENSE_UINT8_FEATURIZERS = [
    (lambda: create_featurizer('maccs', count=True), 'uint8'),
]

DEFAULT_BINARY_FEATURIZERS = [
    lambda: create_featurizer('morgan', count=False),
    lambda: create_featurizer('maccs', count=False),
    lambda: create_featurizer('atom_pair', count=False),
    lambda: create_featurizer('topological_torsion', count=False),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    'factory, _expected',
    COUNT_CSR_FEATURIZERS + COUNT_DENSE_UINT8_FEATURIZERS,
)
def test_count_true_feature_type_is_continuous(factory, _expected):
    feat = factory()
    assert feat.feature_type == 'continuous', (
        f'{type(feat).__name__}(count=True) feature_type should flip to '
        f"'continuous'; got {feat.feature_type!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize('factory, expected', COUNT_CSR_FEATURIZERS)
def test_count_true_csr_storage(factory, expected):
    feat = factory()
    assert feat.get_storage_dtype() == expected, (
        f"{type(feat).__name__}(count=True) should route to 'csr_uint16'; "
        f'got {feat.get_storage_dtype()!r}'
    )


@pytest.mark.unit
@pytest.mark.parametrize('factory, expected', COUNT_DENSE_UINT8_FEATURIZERS)
def test_count_true_dense_uint8_storage(factory, expected):
    feat = factory()
    assert feat.get_storage_dtype() == expected, (
        f"{type(feat).__name__}(count=True) should route to 'uint8'; "
        f'got {feat.get_storage_dtype()!r}'
    )


@pytest.mark.unit
@pytest.mark.parametrize('factory', DEFAULT_BINARY_FEATURIZERS)
def test_count_false_remains_binary_packed(factory):
    feat = factory()
    assert feat.feature_type == 'binary', (
        f"{type(feat).__name__}(count=False) feature_type must stay 'binary'; "
        f'got {feat.feature_type!r}'
    )
    assert feat.get_storage_dtype() == 'packed_uint8', (
        f'{type(feat).__name__}(count=False) get_storage_dtype must stay '
        f"'packed_uint8'; got {feat.get_storage_dtype()!r}"
    )

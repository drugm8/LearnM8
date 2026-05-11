"""Targeted tests for count=True flip behavior (the 017 fix).

Verifies that the 7 count-override featurizers correctly switch:
- storage_dtype when count=True vs count=False
- feature_type when count=True vs count=False
"""

from __future__ import annotations

import pytest

from learnm8.features import create_featurizer

COUNT_OVERRIDE_FEATURIZERS = [
    ('morgan', 'packed_uint8', 'binary', 'csr_uint16', 'continuous'),
    ('ecfp', 'packed_uint8', 'binary', 'csr_uint16', 'continuous'),
    ('ecfp6', 'packed_uint8', 'binary', 'csr_uint16', 'continuous'),
    ('morgan_feat', 'packed_uint8', 'binary', 'csr_uint16', 'continuous'),
    ('maccs', 'packed_uint8', 'binary', 'uint8', 'continuous'),
    ('atom_pair', 'packed_uint8', 'binary', 'csr_uint16', 'continuous'),
    ('topological_torsion', 'packed_uint8', 'binary', 'csr_uint16', 'continuous'),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    'name, false_dtype, false_ftype, true_dtype, true_ftype',
    COUNT_OVERRIDE_FEATURIZERS,
)
def test_count_false_defaults(name, false_dtype, false_ftype, true_dtype, true_ftype):
    featurizer = create_featurizer(name)
    assert featurizer.get_storage_dtype() == false_dtype
    assert featurizer.feature_type == false_ftype


@pytest.mark.unit
@pytest.mark.parametrize(
    'name, false_dtype, false_ftype, true_dtype, true_ftype',
    COUNT_OVERRIDE_FEATURIZERS,
)
def test_count_true_flips_storage_dtype(
    name, false_dtype, false_ftype, true_dtype, true_ftype
):
    featurizer = create_featurizer(name, count=True)
    assert featurizer.get_storage_dtype() == true_dtype


@pytest.mark.unit
@pytest.mark.parametrize(
    'name, false_dtype, false_ftype, true_dtype, true_ftype',
    COUNT_OVERRIDE_FEATURIZERS,
)
def test_count_true_flips_feature_type(
    name, false_dtype, false_ftype, true_dtype, true_ftype
):
    featurizer = create_featurizer(name, count=True)
    assert featurizer.feature_type == true_ftype

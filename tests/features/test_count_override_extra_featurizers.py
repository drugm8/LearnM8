"""``count=True`` routes to CSR uint16 storage for pubchem/klekota_roth/laggner/avalon/rdkit."""

from pathlib import Path

import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.extraction import extract_features

NAMES = ['pubchem', 'klekota_roth', 'laggner', 'avalon', 'rdkit']
SMILES = [
    'CCCCCCCCCCCCCCCC',
    'c1ccccc1CCCCCCCC',
    'CC(C)CC(C)CC(C)CC(C)C',
    'OCC(O)C(O)C(O)C(O)CO',
]


@pytest.mark.unit
@pytest.mark.parametrize('name', NAMES)
def test_count_true_uses_csr_uint16_and_roundtrips_cache(name: str, tmp_path: Path):
    fz = create_featurizer(name, count=True)
    assert fz.get_storage_dtype() == 'csr_uint16'
    assert fz.feature_type == 'continuous'

    cold = extract_features(SMILES, fz, cache_dir=tmp_path)
    warm = extract_features(SMILES, fz, cache_dir=tmp_path)
    assert np.array_equal(cold, warm)
    assert cold.max() > 1


@pytest.mark.unit
@pytest.mark.parametrize('name', NAMES)
def test_count_false_stays_packed_binary(name: str):
    fz = create_featurizer(name)
    assert fz.get_storage_dtype() == 'packed_uint8'
    assert fz.feature_type == 'binary'

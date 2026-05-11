"""End-to-end count-fingerprint round-trip tests (T002, spec 017).

For each of the four count-capable featurizers (Morgan, MACCS, AtomPair,
TopologicalTorsion), feed SMILES that produce nonzero counts (≥ 2 for several
bits) through ``extract_features`` and assert the cached read matches
``featurizer.transform()`` exactly. On current main this test FAILS because
``np.packbits`` truncates count values to single bits (silent corruption).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.extraction import extract_features

# SMILES chosen to produce repeated substructures so count fingerprints
# accumulate values ≥ 2 in multiple bins.
HIGH_COUNT_SMILES = [
    'CCCCCCCCCCCCCCCCCC',  # n-octadecane: many repeated CH2 atoms
    'c1ccc2c(c1)ccc1ccccc12',  # phenanthrene-ish polycyclic aromatic
    'C1CCCCCCCCCC1',  # cycloundecane
    'CCCCCCCCCCCCCCCC(=O)O',  # palmitic acid
    'c1ccc(-c2ccc(-c3ccccc3)cc2)cc1',  # terphenyl
    'CC(C)(C)CC(C)(C)CC(C)(C)C',  # branched alkane
    'O=C(O)CC(=O)O',  # malonic acid (carboxyls)
    'NCCNCCNCCN',  # polyamine (repeated NH)
]


@pytest.mark.unit
@pytest.mark.parametrize(
    'factory_name',
    ['morgan_count', 'atom_pair_count', 'topological_torsion_count', 'maccs_count'],
)
def test_count_fingerprint_round_trip_exact(tmp_path: Path, factory_name: str):
    factories = {
        'morgan_count': create_featurizer('morgan', count=True),
        'atom_pair_count': create_featurizer('atom_pair', count=True),
        'topological_torsion_count': create_featurizer(
            'topological_torsion', count=True
        ),
        'maccs_count': create_featurizer('maccs', count=True),
    }
    feat = factories[factory_name]

    expected = feat.transform(HIGH_COUNT_SMILES)

    # Sanity: for COUNT featurizers we should observe values ≥ 2 in at least
    # one bin so the test actually exercises non-binary behaviour.
    assert int(expected.max()) >= 2, (
        f'{factory_name}: expected max value ≥ 2 from count featurizer on '
        f'chosen SMILES; got max={expected.max()} — choose denser SMILES'
    )

    cold = extract_features(HIGH_COUNT_SMILES, feat, cache_dir=tmp_path)
    warm = extract_features(HIGH_COUNT_SMILES, feat, cache_dir=tmp_path)

    np.testing.assert_array_equal(cold, expected)
    np.testing.assert_array_equal(warm, expected)

"""Storage-dtype-aware cache filenames: morgan count=False vs count=True must not collide."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.extraction import extract_features

_SMILES = ['CCO', 'CCC', 'CCN']


@pytest.mark.unit
def test_morgan_count_false_then_true_shares_cache_dir(tmp_path: Path):
    dense = extract_features(
        _SMILES, create_featurizer('morgan', count=False, n_jobs=1), cache_dir=tmp_path
    )
    counts = extract_features(
        _SMILES, create_featurizer('morgan', count=True, n_jobs=1), cache_dir=tmp_path
    )

    assert dense.shape == (3, 2048)
    assert counts.shape == (3, 2048)

    h5_files = sorted(p.name for p in tmp_path.glob('*.h5'))
    assert len(h5_files) == 2, h5_files


@pytest.mark.unit
def test_morgan_count_true_then_false_shares_cache_dir(tmp_path: Path):
    counts = extract_features(
        _SMILES, create_featurizer('morgan', count=True, n_jobs=1), cache_dir=tmp_path
    )
    dense = extract_features(
        _SMILES, create_featurizer('morgan', count=False, n_jobs=1), cache_dir=tmp_path
    )

    assert counts.shape == (3, 2048)
    assert dense.shape == (3, 2048)
    assert set(np.unique(dense).tolist()) <= {0.0, 1.0}

    h5_files = sorted(p.name for p in tmp_path.glob('*.h5'))
    assert len(h5_files) == 2, h5_files

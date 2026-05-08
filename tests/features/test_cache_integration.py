"""Learner-layer integration smoke tests (T021)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learnm8.features import MordredFeaturizer, MorganFeaturizer
from learnm8.features.extraction import extract_features


@pytest.mark.unit
def test_morgan_extract_features_returns_binary_float32(tmp_path: Path):
    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    out = extract_features(['CCO', 'CCC', 'CCN'], feat, cache_dir=tmp_path)
    assert out.dtype == np.float32
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}


@pytest.mark.unit
def test_mordred_extract_features_returns_continuous_float32(tmp_path: Path):
    feat = MordredFeaturizer(n_jobs=1)
    out = extract_features(['CCO'], feat, cache_dir=tmp_path)
    assert out.dtype == np.float32
    assert out.shape[1] > 100


@pytest.mark.unit
def test_extract_features_preserves_input_order(tmp_path: Path):
    """Critical for learner correctness: cache hits must align with input row order."""
    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    direct = feat.transform(['CCO', 'CCC', 'CCN', 'CCCC']).astype(np.float32)

    extract_features(['CCO', 'CCC'], feat, cache_dir=tmp_path)
    cached = extract_features(['CCO', 'CCC', 'CCN', 'CCCC'], feat, cache_dir=tmp_path)
    assert np.array_equal(cached, direct)

    reordered = extract_features(['CCCC', 'CCO', 'CCN', 'CCC'], feat, cache_dir=tmp_path)
    direct_reorder = feat.transform(['CCCC', 'CCO', 'CCN', 'CCC']).astype(np.float32)
    assert np.array_equal(reordered, direct_reorder)


@pytest.mark.unit
def test_extract_features_string_api_unchanged(tmp_path: Path):
    out = extract_features(['CCO', 'CCC'], 'morgan', cache_dir=tmp_path)
    assert out.shape == (2, 2048)
    assert out.dtype == np.float32


@pytest.mark.unit
def test_empty_input_short_circuits_no_files(tmp_path: Path):
    out = extract_features([], 'morgan', cache_dir=tmp_path)
    assert out.shape == (0, 2048)
    assert out.dtype == np.float32
    assert list(tmp_path.iterdir()) == []

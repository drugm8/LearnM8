"""``extract_features(..., preferred_dtype='uint8')`` tests (T011, spec 017).

Cache hits (and CSR-friendly misses) for binary featurizers should return a
``uint8`` matrix when the caller asks for it, skipping the float32 inflation.
For featurizers whose storage is ``float32`` (e.g. mordred) or ``csr_uint16``
(e.g. ``Morgan(count=True)``), ``preferred_dtype='uint8'`` MUST fall back to
``float32`` and emit a ``logger.debug`` line.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from learnm8.features import create_featurizer
from learnm8.features.extraction import extract_features

SMILES = ['CCO', 'CCC', 'CCN', 'CCCl', 'c1ccccc1', 'CCOC', 'CCOCC', 'CCS']


@pytest.mark.unit
def test_morgan_packed_returns_uint8_when_requested(tmp_path: Path):
    feat = create_featurizer('morgan', fp_size=512)
    extract_features(SMILES, feat, cache_dir=tmp_path)  # warm cache

    out_u8 = extract_features(SMILES, feat, cache_dir=tmp_path, preferred_dtype='uint8')
    out_f32 = extract_features(
        SMILES, feat, cache_dir=tmp_path, preferred_dtype='float32'
    )

    assert out_u8.dtype == np.uint8
    assert out_u8.shape == out_f32.shape
    np.testing.assert_array_equal(out_u8.astype(np.float32), out_f32)


@pytest.mark.unit
def test_morgan_count_csr_falls_back_to_float32(tmp_path: Path, caplog):
    feat = create_featurizer('morgan', count=True, fp_size=512)
    extract_features(SMILES, feat, cache_dir=tmp_path)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger='learnm8.features.cache'):
        out = extract_features(
            SMILES, feat, cache_dir=tmp_path, preferred_dtype='uint8'
        )
    assert out.dtype == np.float32
    assert any('uint8' in record.getMessage().lower() for record in caplog.records), (
        'expected a debug log mentioning the uint8→float32 fallback for csr_uint16 storage'
    )


@pytest.mark.integration
def test_mordred_continuous_falls_back_to_float32(tmp_path: Path, caplog):
    """Mordred storage_dtype is float32 — uint8 request must fall back."""
    pytest.importorskip('mordred')

    feat = create_featurizer('mordred')
    smiles_subset = SMILES[:3]
    extract_features(smiles_subset, feat, cache_dir=tmp_path)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger='learnm8.features.cache'):
        out = extract_features(
            smiles_subset, feat, cache_dir=tmp_path, preferred_dtype='uint8'
        )
    assert out.dtype == np.float32

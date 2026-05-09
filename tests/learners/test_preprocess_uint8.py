"""``_preprocess_features`` uint8 input path (T012, spec 017).

When the caller hands in a ``uint8`` binary matrix, ``_preprocess_features``
MUST: (1) skip ``np.isnan``/``np.nan_to_num`` (uint8 cannot represent NaN),
(2) compute and apply the zero-variance mask, and (3) preserve the input
dtype on the output.

Tests must FAIL on current main because ``np.isnan`` on uint8 raises
``TypeError`` and ``np.nan_to_num`` upcasts to float.
"""

from __future__ import annotations

import numpy as np
import pytest

from learnm8.learners.base import _preprocess_features


@pytest.mark.unit
def test_preprocess_uint8_training_preserves_dtype():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(50, 16), dtype=np.uint8)
    # Make column 3 zero-variance so the mask drops it.
    X[:, 3] = 0

    preprocessed, mask, imputer = _preprocess_features(
        X.copy(),
        remove_zero_variance=True,
        is_training=True,
        feature_type='binary',
    )

    assert mask is not None
    assert preprocessed.dtype == np.uint8
    assert imputer is None
    assert mask.shape == (16,)
    assert not bool(mask[3]), 'zero-variance column 3 should be masked out'
    assert preprocessed.shape == (50, int(mask.sum()))


@pytest.mark.unit
def test_preprocess_uint8_prediction_uses_saved_mask():
    rng = np.random.default_rng(1)
    X_train = rng.integers(0, 2, size=(50, 16), dtype=np.uint8)
    X_train[:, 3] = 0

    _, mask, _ = _preprocess_features(
        X_train.copy(),
        remove_zero_variance=True,
        is_training=True,
        feature_type='binary',
    )
    assert mask is not None

    X_pred = rng.integers(0, 2, size=(7, 16), dtype=np.uint8)
    preprocessed, _, _ = _preprocess_features(
        X_pred.copy(),
        valid_feature_mask=mask,
        remove_zero_variance=True,
        is_training=False,
        feature_type='binary',
    )
    assert preprocessed.dtype == np.uint8
    assert preprocessed.shape == (7, int(mask.sum()))


@pytest.mark.unit
def test_preprocess_uint8_skips_nan_check_no_warning(caplog):
    """uint8 cannot carry NaN; the warn path must not trigger."""
    import logging

    X = np.zeros((4, 8), dtype=np.uint8)
    X[:, 0] = 1
    with caplog.at_level(logging.WARNING, logger='learnm8.learners.base'):
        _preprocess_features(
            X,
            remove_zero_variance=False,
            is_training=True,
            feature_type='binary',
        )
    assert not any(
        'NaN' in record.getMessage() for record in caplog.records
    ), 'no NaN-warning expected on uint8 input'

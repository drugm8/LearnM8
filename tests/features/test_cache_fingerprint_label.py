"""Test fingerprint_used label extension with storage dtype suffix (T032, FR-013)."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_morgan_label_has_uint8packed_suffix():
    from learnm8.evaluation.metrics.similarity import _get_or_build_fps
    from learnm8.features import MorganFeaturizer

    feat = MorganFeaturizer(radius=2, fp_size=2048, n_jobs=1)
    fp_cache: dict = {}
    _, _, label, _ = _get_or_build_fps(
        ['CCO', 'CCC'], feat, None, fp_cache
    )
    assert label.endswith('_uint8packed'), f"got {label!r}"


@pytest.mark.unit
def test_mordred_label_has_float32_and_fallback_tokens():
    from learnm8.evaluation.metrics.similarity import _get_or_build_fps
    from learnm8.features import MordredFeaturizer

    feat = MordredFeaturizer(n_jobs=1)
    fp_cache: dict = {}
    _, _, label, _ = _get_or_build_fps(
        ['CCO'], feat, None, fp_cache
    )
    assert '_float32' in label and 'fallback' in label, f"got {label!r}"


@pytest.mark.unit
def test_no_featurizer_label_is_default_with_dtype_suffix():
    from learnm8.evaluation.metrics.similarity import _get_or_build_fps

    fp_cache: dict = {}
    _, _, label, _ = _get_or_build_fps(
        ['CCO'], None, None, fp_cache
    )
    assert label == 'morgan_2_2048_float32', f"got {label!r}"

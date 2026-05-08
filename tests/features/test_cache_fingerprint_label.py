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


@pytest.mark.unit
def test_mqns_label_has_uint8_suffix():
    """016: mqns featurizer emits the raw uint8 dtype token."""
    from learnm8.evaluation.metrics.similarity import _get_or_build_fps
    from learnm8.features import MQNsFeaturizer

    feat = MQNsFeaturizer(n_jobs=1)
    fp_cache: dict = {}
    _, _, label, _ = _get_or_build_fps(
        ['CCO'], feat, None, fp_cache
    )
    assert '_uint8' in label and '_uint8packed' not in label and '_float32' not in label, (
        f"got {label!r}"
    )


@pytest.mark.unit
def test_pharmacophore_label_has_csruint16_suffix():
    """016: pharmacophore featurizer emits the csruint16 dtype token."""
    from learnm8.evaluation.metrics.similarity import _get_or_build_fps
    from learnm8.features import PharmacophoreFeaturizer

    feat = PharmacophoreFeaturizer(n_jobs=1)
    fp_cache: dict = {}
    _, _, label, _ = _get_or_build_fps(
        ['CCO'], feat, None, fp_cache
    )
    assert '_csruint16' in label, f"got {label!r}"


@pytest.mark.unit
def test_physiochemical_label_has_csruint16_suffix():
    """016: physiochemical featurizer emits the csruint16 dtype token."""
    from learnm8.evaluation.metrics.similarity import _get_or_build_fps
    from learnm8.features import PhysiochemicalPropertiesFeaturizer

    feat = PhysiochemicalPropertiesFeaturizer(n_jobs=1)
    fp_cache: dict = {}
    _, _, label, _ = _get_or_build_fps(
        ['CCO'], feat, None, fp_cache
    )
    assert '_csruint16' in label, f"got {label!r}"

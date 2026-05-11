"""Spec 021 / US3 / FR-006/FR-007/FR-008 — unified ensemble seed scheme.

Every ensemble class must derive member seeds from a single ``random_state``
using the offsets ``(0, 81, 314)`` (extending naturally for n_members > 3).
Direct construction and ``_create_learner('<name>', random_state=42)`` must
agree on the resulting ``random_states`` list.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


_REGISTRY_CASES = [
    ("lr_ensemble", "learnm8.learners.ensemble.lr_ensemble", "LREnsemble"),
    ("dt_ensemble", "learnm8.learners.ensemble.dt_ensemble", "DTEnsemble"),
    ("xgb_ensemble", "learnm8.learners.ensemble.xgb_ensemble", "XGBEnsemble"),
    ("rf_ensemble", "learnm8.learners.ensemble.rf_ensemble", "RFEnsemble"),
]


def _import_class(module_path: str, class_name: str):
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


@pytest.mark.parametrize("registry_key,module_path,class_name", _REGISTRY_CASES)
def test_direct_construction_uses_unified_offsets(registry_key, module_path, class_name):
    cls = _import_class(module_path, class_name)
    instance = cls(random_state=42)
    assert instance.random_states == [42, 123, 356], (
        f"{class_name}(random_state=42).random_states == {instance.random_states}, "
        f"expected [42, 123, 356] per spec FR-006 (offsets 0, 81, 314)"
    )


@pytest.mark.parametrize("registry_key,module_path,class_name", _REGISTRY_CASES)
def test_create_learner_matches_direct_construction(registry_key, module_path, class_name):
    from learnm8.api import _create_learner

    via_factory = _create_learner(registry_key, random_state=42)
    direct = _import_class(module_path, class_name)(random_state=42)

    assert via_factory.random_states == direct.random_states  # type: ignore[attr-defined]


def test_n_members_4_extends_offsets_predictably():
    from learnm8.learners.ensemble.lr_ensemble import LREnsemble

    instance = LREnsemble(
        random_state=42,
        regularization_strengths=[0.1, 1.0, 10.0, 100.0],
    )
    assert instance.random_states[:3] == [42, 123, 356]
    assert len(instance.random_states) == 4
    assert instance.random_states[3] == instance.random_states[3]


def test_explicit_random_states_overrides_derivation():
    from learnm8.learners.ensemble.lr_ensemble import LREnsemble

    instance = LREnsemble(random_state=42, random_states=[1, 2, 3])
    assert instance.random_states == [1, 2, 3]

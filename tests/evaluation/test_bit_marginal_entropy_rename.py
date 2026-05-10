"""Rename tests for shannon_entropy_diversity_* → bit_marginal_entropy_* (feature 019).

Covers FR-010, FR-011 (US5).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def test_bit_marginal_entropy_keys_in_diversity_keys() -> None:
    from learnm8.evaluation.metrics.similarity import DIVERSITY_KEYS

    assert "bit_marginal_entropy_batch" in DIVERSITY_KEYS
    assert "bit_marginal_entropy_cumulative" in DIVERSITY_KEYS
    assert "shannon_entropy_diversity_batch" not in DIVERSITY_KEYS
    assert "shannon_entropy_diversity_cumulative" not in DIVERSITY_KEYS


def test_bit_marginal_entropy_function_exists() -> None:
    similarity = importlib.import_module("learnm8.evaluation.metrics.similarity")
    assert hasattr(similarity, "_bit_marginal_entropy_from_bit_sum")


def test_bit_marginal_entropy_value_unchanged_vs_explicit_compute() -> None:
    from learnm8.evaluation.metrics.similarity import _bit_marginal_entropy_from_bit_sum
    from scipy.stats import entropy as scipy_entropy

    bit_sum = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    n_compounds = 5
    expected = float(scipy_entropy(bit_sum.astype(np.float64) / n_compounds, base=2))
    actual = _bit_marginal_entropy_from_bit_sum(bit_sum, n_compounds)
    assert actual == pytest.approx(expected)


def test_old_symbol_dot_access_raises_attributeerror() -> None:
    similarity = importlib.import_module("learnm8.evaluation.metrics.similarity")
    with pytest.raises(AttributeError) as exc_info:
        _ = similarity._shannon_entropy_from_bit_sum  # noqa: SLF001
    msg = str(exc_info.value)
    assert "Renamed to '_bit_marginal_entropy_from_bit_sum'" in msg


def test_old_symbol_getattr_raises_attributeerror() -> None:
    similarity = importlib.import_module("learnm8.evaluation.metrics.similarity")
    with pytest.raises(AttributeError):
        getattr(similarity, "_shannon_entropy_from_bit_sum")


def test_old_dict_key_attribute_raises() -> None:
    similarity = importlib.import_module("learnm8.evaluation.metrics.similarity")
    with pytest.raises(AttributeError):
        getattr(similarity, "shannon_entropy_diversity_batch")


def test_old_symbol_from_import_raises_importerror() -> None:
    """`from X import Y` translates AttributeError → ImportError. CPython strips the hint
    from the from-import path, but the standard ImportError still names the missing symbol;
    the descriptive hint is available via direct attribute access (covered separately).
    """
    with pytest.raises(ImportError) as exc_info:
        from learnm8.evaluation.metrics.similarity import _shannon_entropy_from_bit_sum  # type: ignore[attr-defined]  # noqa: F401
    assert "_shannon_entropy_from_bit_sum" in str(exc_info.value)


def test_unknown_attribute_still_raises_normal_attribute_error() -> None:
    similarity = importlib.import_module("learnm8.evaluation.metrics.similarity")
    with pytest.raises(AttributeError) as exc_info:
        _ = similarity.this_attribute_does_not_exist
    assert "has no attribute" in str(exc_info.value).lower()

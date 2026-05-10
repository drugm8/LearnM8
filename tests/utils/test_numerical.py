"""Tests for learnm8.utils.numerical (feature 019).

Covers SIGMA_FLOOR, clamp_sigma, assert_no_nan, assert_no_inf_uncertainty
per contracts/sigma_clamp_helper.md.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from learnm8.exceptions import LearnerError
from learnm8.utils.numerical import (
    SIGMA_FLOOR,
    assert_no_inf_uncertainty,
    assert_no_nan,
    clamp_sigma,
)


pytestmark = pytest.mark.unit


def test_sigma_floor_constant_value() -> None:
    assert SIGMA_FLOOR == 1e-9


def test_clamp_sigma_promotes_float32_to_float64() -> None:
    arr = np.array([0.0], dtype=np.float32)
    out = clamp_sigma(arr)
    assert out.dtype == np.float64
    assert out[0] == SIGMA_FLOOR


def test_clamp_sigma_does_not_modify_input() -> None:
    arr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    snapshot = arr.copy()
    _ = clamp_sigma(arr)
    np.testing.assert_array_equal(arr, snapshot)


def test_clamp_sigma_passes_through_above_floor() -> None:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out = clamp_sigma(arr)
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == np.float64


def test_clamp_sigma_clamps_zero() -> None:
    arr = np.array([0.0])
    out = clamp_sigma(arr)
    assert out[0] == SIGMA_FLOOR


def test_clamp_sigma_clamps_subnormal() -> None:
    arr = np.array([np.finfo(float).tiny])
    out = clamp_sigma(arr)
    assert out[0] == SIGMA_FLOOR


def test_assert_no_nan_passes_clean() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert assert_no_nan(arr, ["a", "b", "c"], "foo") is None


def test_assert_no_nan_message_format_singular_row() -> None:
    arr = np.array([1.0, 2.0, np.nan])
    with pytest.raises(LearnerError) as exc_info:
        assert_no_nan(arr, ["a", "b", "c"], "foo")
    msg = str(exc_info.value)
    assert re.search(r"NaN foo detected: 1 row, first IDs: \['c'\]", msg), msg


def test_assert_no_nan_message_format_plural_rows() -> None:
    arr = np.array([1.0, np.nan, np.nan])
    with pytest.raises(LearnerError) as exc_info:
        assert_no_nan(arr, ["a", "b", "c"], "foo")
    msg = str(exc_info.value)
    assert re.search(r"NaN foo detected: 2 rows, first IDs: \['b', 'c'\]", msg), msg


def test_assert_no_nan_message_handles_id_with_quote() -> None:
    arr = np.array([np.nan])
    ids = ["smiles'frag"]
    with pytest.raises(LearnerError) as exc_info:
        assert_no_nan(arr, ids, "foo")
    msg = str(exc_info.value)
    bracket = msg.split("first IDs: ", 1)[1]
    parsed = eval(bracket)  # safe: known-shape literal list of strings
    assert parsed == ids


def test_assert_no_nan_truncates_at_10() -> None:
    n = 100
    arr = np.full(n, np.nan)
    ids = [f"ID{i}" for i in range(n)]
    with pytest.raises(LearnerError) as exc_info:
        assert_no_nan(arr, ids, "foo")
    msg = str(exc_info.value)
    bracket = msg.split("first IDs: ", 1)[1]
    parsed = eval(bracket)
    assert parsed == ids[:10]


def test_assert_no_inf_uncertainty_passes_clean() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert assert_no_inf_uncertainty(arr, ["a", "b", "c"]) is None


def test_assert_no_inf_uncertainty_message_format() -> None:
    arr = np.array([1.0, np.inf, 2.0])
    with pytest.raises(LearnerError) as exc_info:
        assert_no_inf_uncertainty(arr, ["a", "b", "c"])
    msg = str(exc_info.value)
    assert "Inf uncertainties not supported: 1 row" in msg
    assert "'b'" in msg


def test_assert_no_inf_uncertainty_negative_inf_also_caught() -> None:
    arr = np.array([1.0, -np.inf, 2.0])
    with pytest.raises(LearnerError):
        assert_no_inf_uncertainty(arr, ["a", "b", "c"])

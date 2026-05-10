"""Test that NaN predictions/uncertainties cause LearnerError immediately
from the cycle.py guard (FR-005, US2).
"""

from __future__ import annotations

import numpy as np
import pytest

from learnm8.exceptions import LearnerError
from learnm8.utils.numerical import (
    assert_no_inf_uncertainty,
    assert_no_nan,
)


pytestmark = pytest.mark.unit


def test_assert_no_nan_passes_when_clean() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert assert_no_nan(arr, ["a", "b", "c"], "predictions") is None


def test_assert_no_nan_raises_with_count_and_ids() -> None:
    predictions = np.array([1.0, 2.0, np.nan, 4.0])
    ids = ["ID1", "ID2", "ID3", "ID4"]
    with pytest.raises(LearnerError) as exc_info:
        assert_no_nan(predictions, ids, "predictions")
    msg = str(exc_info.value)
    assert "NaN predictions detected: 1 row" in msg
    assert "'ID3'" in msg


def test_assert_no_nan_truncates_to_10_ids() -> None:
    n = 100
    arr = np.full(n, np.nan)
    ids = [f"ID{i}" for i in range(n)]
    with pytest.raises(LearnerError) as exc_info:
        assert_no_nan(arr, ids, "predictions")
    msg = str(exc_info.value)
    bracket = msg.split("first IDs: ", 1)[1]
    parsed = eval(bracket)
    assert parsed == ids[:10]


def test_cycle_assert_no_nan_uncertainty_raises() -> None:
    """assert_no_nan applied to uncertainties array."""
    arr = np.array([0.1, np.nan, 0.3])
    with pytest.raises(LearnerError) as exc_info:
        assert_no_nan(arr, ["a", "b", "c"], "uncertainties")
    assert "NaN uncertainties detected" in str(exc_info.value)


def test_assert_no_inf_uncertainty_raises() -> None:
    arr = np.array([0.1, np.inf, 0.3])
    with pytest.raises(LearnerError) as exc_info:
        assert_no_inf_uncertainty(arr, ["a", "b", "c"])
    assert "Inf uncertainties not supported" in str(exc_info.value)

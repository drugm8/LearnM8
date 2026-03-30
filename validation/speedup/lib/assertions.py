from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

TOLERANCES = {
    "zero_impact": {"atol": 1e-6, "rtol": 1e-5, "min_spearman": 0.999},
    "near_zero": {"atol": 1e-4, "rtol": None, "min_spearman": 0.9999},
    "documented": {"atol": 1e-4, "rtol": None, "min_spearman": 0.95},
    "exact": {"atol": 0, "rtol": 0, "min_spearman": 1.0},
}


def assert_predictions_match(
    baseline: np.ndarray, optimized: np.ndarray, tolerance_tier: str
) -> float:
    tier = TOLERANCES[tolerance_tier]
    atol = tier["atol"]
    rtol = tier["rtol"]
    min_spearman = tier["min_spearman"]

    if tolerance_tier == "exact":
        if not np.array_equal(baseline, optimized):
            max_diff = float(np.max(np.abs(baseline - optimized)))
            raise AssertionError(
                f"Predictions not exactly equal. Max diff: {max_diff}"
            )
    else:
        effective_rtol = rtol if rtol is not None else 0
        if not np.allclose(baseline, optimized, atol=atol, rtol=effective_rtol):
            max_diff = float(np.max(np.abs(baseline - optimized)))
            raise AssertionError(
                f"Predictions exceed tolerance tier '{tolerance_tier}' "
                f"(atol={atol}, rtol={effective_rtol}). Max diff: {max_diff}"
            )

    rho, _ = spearmanr(baseline, optimized)
    if rho < min_spearman:
        raise AssertionError(
            f"Spearman rho {rho:.6f} < required {min_spearman} "
            f"for tolerance tier '{tolerance_tier}'"
        )
    return float(rho)


def assert_ranking_preserved(
    baseline: np.ndarray,
    optimized: np.ndarray,
    min_spearman: float = 0.95,
    label: str = "predictions",
) -> float:
    rho, _ = spearmanr(baseline, optimized)
    if rho < min_spearman:
        raise AssertionError(
            f"{label} ranking not preserved: "
            f"Spearman rho {rho:.6f} < required {min_spearman}"
        )
    return float(rho)


def assert_uncertainty_ranking_preserved(
    baseline_unc: np.ndarray,
    optimized_unc: np.ndarray,
    min_spearman: float = 0.95,
) -> float:
    rho, _ = spearmanr(baseline_unc, optimized_unc)
    if rho < min_spearman:
        raise AssertionError(
            f"Uncertainty ranking not preserved: "
            f"Spearman rho {rho:.6f} < required {min_spearman}"
        )
    return float(rho)

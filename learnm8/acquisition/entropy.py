"""Entropy acquisition function for the LearnM8 framework.

Computes the canonical Gaussian differential entropy of the predictive
distribution (units: nats). Refactored in feature 019 to replace the
mislabelled raw-σ / raw-σ² scoring with the proper differential-entropy
formula. Selection ranking is rank-equivalent to a σ-descending baseline
(NOT to UCB(β=0), which ranks by μ).
"""

from __future__ import annotations

import logging
import warnings
from typing import Final

import numpy as np
import polars as pl

from learnm8.exceptions import AcquisitionError, LearnM8Warning
from learnm8.utils.numerical import (
    SIGMA_FLOOR,
    assert_no_inf_uncertainty,
    assert_no_nan,
    clamp_sigma,
)

from .base import AcquisitionFunction, validate_uncertainty_inputs

logger = logging.getLogger(__name__)

# Pre-computed 0.5 · log(2πe) for the differential entropy of a unit-σ Gaussian.
_HALF_LOG_2_PI_E: Final[float] = 0.5 * float(np.log(2.0 * np.pi * np.e))


def _entropy_scores(uncertainties: np.ndarray, entropy_type: str) -> np.ndarray:
    """Higher-is-better Gaussian differential entropy in nats (single source).

    Expanded form (FR-007): numerically robust at σ < 1e-154. Monotone in σ, so
    rank-equivalent to σ-descending for both ``entropy_type`` values.
    """
    log_sigma = np.log(clamp_sigma(uncertainties))
    if entropy_type == 'uncertainty':
        return _HALF_LOG_2_PI_E + log_sigma
    return _HALF_LOG_2_PI_E + 2.0 * log_sigma


class EntropyAcquisition(AcquisitionFunction):
    """Differential entropy of Gaussian predictive (nats); rank-equivalent to σ-descending baseline (`np.argsort(σ)[::-1]`). NOT rank-equivalent to UCB(β=0).

    Computes ``H = 0.5 · log(2πe · σ²)`` for ``entropy_type='uncertainty'``
    (default), and ``H = 0.5 · log(2πe · σ⁴) = log(2πe) + 2·log(σ)`` for
    ``entropy_type='variance'``. Both are monotone-increasing in σ, so
    selection ranks are identical to ``np.argsort(σ)[::-1]`` (a σ-descending
    baseline) for either argument value.

    NOT rank-equivalent to UCBAcquisition(beta=0). UCB(β=0) ranks by
    predictions (μ) alone (verified at learnm8/acquisition/ucb.py:61-66);
    EntropyAcquisition ranks by σ. These are orthogonal axes.

    For epistemic-only entropy (BALD; mutual information between prediction
    and model parameters), see future spec.
    """

    def __init__(self, entropy_type: str = "uncertainty", **kwargs):
        """Initialize Entropy acquisition function.

        Args:
            entropy_type: ``'uncertainty'`` (compute H from σ; default) or
                ``'variance'`` (compute H from σ², differs from
                ``'uncertainty'`` by an additive ``log(σ)`` term;
                rank-equivalent in both cases).
            **kwargs: forwarded to ``AcquisitionFunction``.
        """
        super().__init__(**kwargs)
        if entropy_type not in {"uncertainty", "variance"}:
            raise ValueError("entropy_type must be 'uncertainty' or 'variance'")
        self.entropy_type = entropy_type

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select compounds with highest predictive differential entropy.

        Args:
            compounds: DataFrame with ``ID``, ``SMILES``, ``prediction``,
                ``uncertainty`` columns.
            n_select: Number of compounds to select.

        Returns:
            DataFrame subset with selected compounds; ``acquisition_score``
            column carries the differential entropy in nats.
        """
        self.validate_input(compounds, n_select)

        predictions, uncertainties = validate_uncertainty_inputs(compounds)

        # FR-004: defence-in-depth NaN/Inf guards. Lazy ID source — only
        # materialised on the error path inside numerical.py.
        ids = compounds.get_column("ID")
        assert_no_nan(predictions, ids, "predictions")
        assert_no_nan(uncertainties, ids, "uncertainties")
        assert_no_inf_uncertainty(uncertainties, ids)

        # FR-008: aggregated single warning per call when sub-floor σ values
        # are non-trivial (≥ 1% of input). Below threshold → debug log only.
        # The comparison works on any float dtype (SIGMA_FLOOR=1e-9 > float32 eps),
        # so no `.astype(float64)` copy is needed for the count alone.
        n_clamped = int(np.sum(uncertainties < SIGMA_FLOOR))
        if n_clamped > 0:
            msg = f"Clamped {n_clamped} σ values at 1e-9 floor"
            threshold = max(1, int(0.01 * len(uncertainties)))
            if n_clamped >= threshold:
                warnings.warn(msg, LearnM8Warning, stacklevel=2)
            else:
                logger.debug(msg)

        # Single source: identical math to score_chunk (FR-007).
        entropy_scores = _entropy_scores(uncertainties, self.entropy_type)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Entropy(nats) statistics: min=%.3f max=%.3f mean=%.3f",
                float(entropy_scores.min()),
                float(entropy_scores.max()),
                float(entropy_scores.mean()),
            )

        selected = self._safe_select_top_k(
            compounds, entropy_scores, n_select, ascending=False
        )

        logger.debug(
            "EntropyAcquisition selected %d compounds using %s entropy (nats)",
            len(selected),
            self.entropy_type,
        )

        return selected

    def requires_uncertainty(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def score_chunk(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray | None,
        *,
        global_offset: int,
        n_total: int,
    ) -> np.ndarray:
        if uncertainties is None:
            raise AcquisitionError(
                'Entropy acquisition requires uncertainty estimates, but the '
                'chunk provided none.'
            )
        return _entropy_scores(uncertainties, self.entropy_type)

    def get_name(self) -> str:
        return f"Entropy({self.entropy_type})"

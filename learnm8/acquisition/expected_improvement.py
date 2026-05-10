"""Expected Improvement acquisition function for the LearnM8 framework.

This module provides the Expected Improvement acquisition strategy which calculates
the expected improvement over the current best observed value.
"""

import logging
import warnings

import polars as pl
from scipy.stats import norm

from learnm8.utils.numerical import (
    assert_no_inf_uncertainty,
    assert_no_nan,
    clamp_sigma,
)

from .base import AcquisitionFunction, validate_uncertainty_inputs

logger = logging.getLogger(__name__)


class ExpectedImprovementAcquisition(AcquisitionFunction):
    """Expected Improvement acquisition function.

    EI calculates the expected improvement over the current best observed value,
    providing a principled way to balance exploration and exploitation.
    """

    def __init__(self,
                 xi: float = 0.01, minimize: bool = None, score_direction: str = 'higher',
                 current_best: float | None = None,
                 **kwargs):
        """Initialize Expected Improvement acquisition function.

        Args:
            xi: Exploration parameter. Small positive values encourage exploration.
            minimize: DEPRECATED. Use score_direction instead. If provided, overrides score_direction.
            score_direction: Direction to optimize ('higher' or 'lower'). Default 'higher'
            current_best: Current best observed value from labeled data. Required for correct EI calculation.
            **kwargs: Additional parameters for compatibility
        """
        # Handle backward compatibility with minimize parameter
        if minimize is not None:
            warnings.warn(
                "The 'minimize' parameter is deprecated. Use 'score_direction' instead.",
                DeprecationWarning, stacklevel=2
            )
            score_direction = 'lower' if minimize else 'higher'

        super().__init__(score_direction=score_direction, **kwargs)
        if xi < 0:
            raise ValueError("xi must be non-negative")

        self.xi = xi
        self.current_best = current_best

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select using Expected Improvement.

        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction', 'uncertainty' columns
            n_select: Number of compounds to select

        Returns:
            DataFrame subset with selected compounds

        Raises:
            ValueError: If uncertainty estimates are not available
        """
        # Validate input
        self.validate_input(compounds, n_select)

        # Extract predictions and uncertainties
        predictions, uncertainties = validate_uncertainty_inputs(compounds)

        # Require current_best from labeled data
        if self.current_best is None:
            raise ValueError(
                "Expected Improvement requires 'current_best' parameter with the best observed value "
                "from labeled training data. This should be passed via acquisition_params at the cycle level."
            )

        current_best = self.current_best

        logger.debug(f"EIAcquisition: current_best={current_best:.3f}, calculating expected improvement")

        # FR-004: defence-in-depth — cycle.py is the canonical guard, but this
        # acquisition can also be invoked directly (notebook / test harness).
        # The numerical helpers materialise IDs only on the error path, so the
        # clean path pays no `.to_list()` cost per call.
        ids = compounds.get_column("ID")
        assert_no_nan(predictions, ids, "predictions")
        assert_no_nan(uncertainties, ids, "uncertainties")
        assert_no_inf_uncertainty(uncertainties, ids)

        # Calculate improvement based on score direction
        if self.maximize:
            improvement = predictions - current_best - self.xi
        else:
            improvement = current_best - predictions - self.xi

        # FR-001: clamp σ at 1e-9 (float64-promoted) to avoid 0/0 → NaN.
        # Φ(±∞) ∈ {0, 1} and σ_clamped · φ(±∞) = 0, so the formula naturally
        # delivers the analytic limits at the σ → 0 boundary.
        sigma_clamped = clamp_sigma(uncertainties)
        z_scores = improvement / sigma_clamped
        ei_scores = improvement * norm.cdf(z_scores) + sigma_clamped * norm.pdf(z_scores)

        logger.debug(f"EI statistics: {(ei_scores > 0).sum()} compounds with positive EI")

        # Select top compounds
        selected = self._safe_select_top_k(
            compounds, ei_scores, n_select, ascending=False
        )

        logger.debug(f"ExpectedImprovementAcquisition selected {len(selected)} compounds "
                    f"with ξ={self.xi}, current_best={current_best:.3f}")

        return selected

    def requires_uncertainty(self) -> bool:
        """Return True since EI requires uncertainty estimates."""
        return True

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        direction = "max" if self.maximize else "min"
        return f"EI(ξ={self.xi},{direction})"

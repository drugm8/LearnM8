"""Expected Improvement acquisition function for the LearnM8 framework.

This module provides the Expected Improvement acquisition strategy which calculates
the expected improvement over the current best observed value.
"""

import logging

import numpy as np
import polars as pl
from scipy.special import ndtr

from .base import AcquisitionFunction, validate_uncertainty_inputs

logger = logging.getLogger(__name__)


class ExpectedImprovementAcquisition(AcquisitionFunction):
    """Expected Improvement acquisition function.

    EI calculates the expected improvement over the current best observed value,
    providing a principled way to balance exploration and exploitation.
    """

    def __init__(self,
                 xi: float = 0.01, minimize: bool | None = None, score_direction: str = 'higher',
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
            import warnings
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

        from learnm8.utils.numerical import assert_no_inf_uncertainty
        assert_no_inf_uncertainty(uncertainties, compounds.get_column("ID"))

        # Require current_best from labeled data
        if self.current_best is None:
            raise ValueError(
                "Expected Improvement requires 'current_best' parameter with the best observed value "
                "from labeled training data. This should be passed via acquisition_params at the cycle level."
            )

        current_best = self.current_best

        logger.debug(f"EIAcquisition: current_best={current_best:.3f}, calculating expected improvement")

        # Calculate improvement based on score direction
        if self.maximize:
            improvement = predictions - current_best - self.xi
        else:
            improvement = current_best - predictions - self.xi

        # Spec 022 FR-006/FR-007: inline EI with scipy.special.ndtr (C-level
        # Cephes; bit-exact equivalent of scipy.stats.norm.cdf) + symmetric
        # z-clipping for IEEE 754 safety + Botorch sigma=0 -> 0 convention.
        std_devs = uncertainties
        sigma_safe = np.where(std_devs > 0, std_devs, 1.0)
        z_scores = improvement / sigma_safe
        z_clipped = np.clip(z_scores, -37.0, 37.0)
        Phi = ndtr(z_clipped)
        phi = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
        ei_scores = improvement * Phi + std_devs * phi
        # Botorch convention: when sigma = 0 the model is fully certain -- no
        # improvement opportunity beyond what the deterministic prediction
        # already says, so EI = 0. This is behavioural-test-locked in
        # tests/acquisition/test_expected_improvement.py.
        ei_scores = np.where(std_devs > 0, ei_scores, 0.0)

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

"""Upper Confidence Bound acquisition function for the LearnM8 framework.

This module provides the UCB acquisition strategy which balances exploitation
and exploration using uncertainty estimates.
"""

import logging

import polars as pl

from .base import AcquisitionFunction, validate_uncertainty_inputs

logger = logging.getLogger(__name__)


class UCBAcquisition(AcquisitionFunction):
    """Upper Confidence Bound acquisition function.

    UCB balances exploitation (high predicted values) with exploration
    (high uncertainty) by computing upper confidence bounds using
    prediction + beta * uncertainty.
    """

    def __init__(self, beta: float = 2.0, **kwargs):
        """Initialize UCB acquisition function.

        Args:
            beta: Confidence parameter controlling exploration vs exploitation.
                  Higher values favor exploration.
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(**kwargs)
        if beta < 0:
            raise ValueError("beta must be non-negative")

        self.beta = beta

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select using Upper Confidence Bound.

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

        # Reject ±inf σ rather than silently propagating ±inf UCB scores,
        # matching the EI/PI/Entropy guard. A learner emitting inf σ is broken.
        from learnm8.utils.numerical import assert_no_inf_uncertainty

        assert_no_inf_uncertainty(uncertainties, compounds.get_column('ID'))

        logger.debug(f"UCBAcquisition: β={self.beta}, calculating UCB scores")

        # Calculate UCB scores based on score direction
        # Note: uncertainties are already standard deviations, not variances
        if self.maximize:
            # For maximization: select upper confidence bound
            ucb_scores = predictions + self.beta * uncertainties
        else:
            # For minimization: select lower confidence bound
            ucb_scores = predictions - self.beta * uncertainties

        logger.debug(f"UCB score statistics: min={ucb_scores.min():.3f}, max={ucb_scores.max():.3f}, mean={ucb_scores.mean():.3f}")

        # Select top compounds (always ascending=False because we want highest UCB scores)
        if self.maximize:
            ascending = False
        else:
            ascending = True
        selected = self._safe_select_top_k(
            compounds, ucb_scores, n_select, ascending=ascending
        )

        logger.debug(f"UCBAcquisition selected {len(selected)} compounds with β={self.beta}")

        return selected

    def requires_uncertainty(self) -> bool:
        """Return True since UCB requires uncertainty estimates."""
        return True

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"UCB(β={self.beta})"

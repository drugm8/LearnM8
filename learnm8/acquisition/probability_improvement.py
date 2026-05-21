"""Probability of Improvement acquisition function for the LearnM8 framework.

This module provides the Probability of Improvement acquisition strategy which calculates
the probability that a compound will improve over the current best observed value.
"""

import logging

import numpy as np
import polars as pl
from scipy.special import ndtr

from .base import AcquisitionFunction, validate_uncertainty_inputs

logger = logging.getLogger(__name__)


class ProbabilityImprovementAcquisition(AcquisitionFunction):
    """Probability of Improvement acquisition function.

    PI calculates the probability that a compound will improve over
    the current best observed value.
    """

    def __init__(
        self,
        xi: float = 0.01,
        minimize: bool | None = None,
        score_direction: str = 'higher',
        current_best: float | None = None,
        **kwargs,
    ):
        """Initialize Probability of Improvement acquisition function.

        Args:
            xi: Exploration parameter. Small positive values encourage exploration.
            minimize: DEPRECATED. Use score_direction instead. If provided, overrides score_direction.
            score_direction: Direction to optimize ('higher' or 'lower'). Default 'higher'
            current_best: Current best observed value from labeled data. Required for correct PI calculation.
            **kwargs: Additional parameters for compatibility
        """
        # Handle backward compatibility with minimize parameter
        if minimize is not None:
            import warnings

            warnings.warn(
                "The 'minimize' parameter is deprecated. Use 'score_direction' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            score_direction = 'lower' if minimize else 'higher'

        super().__init__(score_direction=score_direction, **kwargs)
        if xi < 0:
            raise ValueError('xi must be non-negative')

        self.xi = xi
        self.current_best = current_best

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select using Probability of Improvement.

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

        assert_no_inf_uncertainty(uncertainties, compounds.get_column('ID'))

        # Require current_best from labeled data
        if self.current_best is None:
            raise ValueError(
                "Probability of Improvement requires 'current_best' parameter with the best observed value "
                'from labeled training data. This should be passed via acquisition_params at the cycle level.'
            )

        current_best = self.current_best

        logger.debug(
            f'PIAcquisition: current_best={current_best:.3f}, calculating probability of improvement'
        )

        # Calculate improvement based on score direction
        if self.maximize:
            improvement = predictions - current_best - self.xi
        else:
            improvement = current_best - predictions - self.xi

        # Spec 022 FR-006/FR-007: inline PI with scipy.special.ndtr (C-level
        # Cephes; bit-exact equivalent of scipy.stats.norm.cdf) + symmetric
        # z-clipping + Botorch sigma=0 -> 0 convention.
        std_devs = uncertainties
        sigma_safe = np.where(std_devs > 0, std_devs, 1.0)
        z_scores = improvement / sigma_safe
        z_clipped = np.clip(z_scores, -37.0, 37.0)
        pi_scores = ndtr(z_clipped)
        # sigma -> 0 limit: a fully-certain compound has PI = 1.0 when it
        # improves over current_best, else 0.0. Behavioural-test-locked in
        # tests/acquisition/test_probability_improvement.py.
        pi_scores = np.where(
            std_devs > 0, pi_scores, np.where(improvement > 0, 1.0, 0.0)
        )

        # Select top compounds
        selected = self._safe_select_top_k(
            compounds, pi_scores, n_select, ascending=False
        )

        logger.debug(
            f'ProbabilityImprovementAcquisition selected {len(selected)} compounds '
            f'with ξ={self.xi}, current_best={current_best:.3f}'
        )

        return selected

    def requires_uncertainty(self) -> bool:
        """Return True since PI requires uncertainty estimates."""
        return True

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        direction = 'max' if self.maximize else 'min'
        return f'PI(ξ={self.xi},{direction})'

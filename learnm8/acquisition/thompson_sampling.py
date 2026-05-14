"""Thompson Sampling acquisition function for the LearnM8 framework.

This module provides the Thompson Sampling acquisition strategy which draws samples
from the posterior predictive distribution.
"""

import logging

import numpy as np
import polars as pl

from .base import AcquisitionFunction, validate_uncertainty_inputs

logger = logging.getLogger(__name__)


class ThompsonSamplingAcquisition(AcquisitionFunction):
    """Thompson Sampling acquisition function.

    Thompson sampling draws samples from the posterior predictive distribution
    and selects compounds with the highest sampled values, providing a
    stochastic exploration strategy.
    """

    def __init__(self, random_state: int = 42, **kwargs):
        """Initialize Thompson Sampling acquisition function.

        Args:
            random_state: Random seed for reproducible sampling
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(**kwargs)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select using Thompson Sampling.

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

        logger.debug(f"ThompsonAcquisition: sampling from posterior with random_state={self.random_state}")

        # Sample from posterior predictive distribution
        # Use uncertainties directly (already standard deviations, not variances)
        std_devs = uncertainties
        thompson_samples = self._rng.normal(predictions, std_devs)

        # Select top compounds based on samples and score direction
        selected = self._safe_select_top_k(
            compounds, thompson_samples, n_select, ascending=not self.maximize
        )

        logger.debug(f"ThompsonSamplingAcquisition selected {len(selected)} compounds "
                    f"with random_state={self.random_state}")

        return selected

    def requires_uncertainty(self) -> bool:
        """Return True since Thompson sampling requires uncertainty estimates."""
        return True

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"Thompson(seed={self.random_state})"

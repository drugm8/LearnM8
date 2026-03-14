"""Entropy acquisition function for the LearnM8 framework.

This module provides the Entropy acquisition strategy which selects compounds
that are expected to provide the most information.
"""

import logging

import polars as pl

from .base import AcquisitionFunction, validate_uncertainty_inputs

logger = logging.getLogger(__name__)


class EntropyAcquisition(AcquisitionFunction):
    """Entropy-based acquisition function for maximum information gain.

    This acquisition function selects compounds that are expected to provide
    the most information, measured by prediction entropy or uncertainty.
    """

    def __init__(self, entropy_type: str = 'uncertainty', **kwargs):
        """Initialize Entropy acquisition function.

        Args:
            entropy_type: Type of entropy measure ('uncertainty', 'variance')
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(**kwargs)
        if entropy_type not in ['uncertainty', 'variance']:
            raise ValueError("entropy_type must be 'uncertainty' or 'variance'")

        self.entropy_type = entropy_type

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select using entropy-based information criterion.

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

        logger.debug("EntropyAcquisition: calculating predictive entropy")

        if self.entropy_type == 'uncertainty':
            # Use uncertainty directly as information measure
            entropy_scores = uncertainties
        else:  # variance
            # Use variance (square of uncertainty) as information measure
            entropy_scores = uncertainties ** 2

        logger.debug(f"Entropy statistics: min={entropy_scores.min():.3f}, max={entropy_scores.max():.3f}, mean={entropy_scores.mean():.3f}")

        # Select compounds with highest entropy/information
        selected = self._safe_select_top_k(
            compounds, entropy_scores, n_select, ascending=False
        )

        logger.debug(f"EntropyAcquisition selected {len(selected)} compounds "
                    f"using {self.entropy_type} entropy")

        return selected

    def requires_uncertainty(self) -> bool:
        """Return True since entropy acquisition requires uncertainty estimates."""
        return True

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"Entropy({self.entropy_type})"

"""Top-K acquisition function for the LearnM8 framework.

This module provides the Top-K acquisition strategy which selects based on rank ordering.
"""

import logging
import polars as pl

from .base import AcquisitionFunction

logger = logging.getLogger(__name__)


class TopKAcquisition(AcquisitionFunction):
    """Top-K acquisition that selects based on rank ordering.

    This is a more flexible version of greedy selection that can be configured
    to select from different parts of the prediction distribution.
    """

    def __init__(self, k_fraction: float = 0.1, score_direction: str = 'higher', **kwargs):
        """Initialize Top-K acquisition function.

        Args:
            k_fraction: Fraction of compounds to consider from the top/bottom
            score_direction: Direction of score optimization ('higher' or 'lower' is better)
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(score_direction=score_direction, **kwargs)
        if not 0.0 < k_fraction <= 1.0:
            raise ValueError("k_fraction must be between 0 and 1")
        if score_direction not in ['higher', 'lower']:
            raise ValueError(f"score_direction must be 'higher' or 'lower', got '{score_direction}'")

        self.k_fraction = k_fraction
        self.score_direction = score_direction
        # Keep backward compatibility
        self.from_top = score_direction == 'higher'

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select compounds from top-K or bottom-K predictions.

        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
            n_select: Number of compounds to select

        Returns:
            DataFrame subset with selected compounds

        Raises:
            ValueError: If required columns are missing or n_select is invalid
        """
        # Validate input
        self.validate_input(compounds, n_select)

        # Adjust n_select if it exceeds available compounds
        actual_n_select = min(n_select, len(compounds))

        # Calculate K - the number of top compounds to consider
        k_consider = max(actual_n_select, int(len(compounds) * self.k_fraction))
        k_consider = min(k_consider, len(compounds))

        logger.debug(f"TopKAcquisition: k_fraction={self.k_fraction}, candidates to consider: {k_consider}")

        # Sort compounds by prediction based on score direction
        if self.score_direction == 'higher':
            # Sort descending (highest first)
            sorted_compounds = compounds.sort('prediction', descending=True)
        else:
            # Sort ascending (lowest first)
            sorted_compounds = compounds.sort('prediction', descending=False)

        # Take top-K compounds
        top_k_compounds = sorted_compounds.head(k_consider)

        # Randomly select from top-K
        if len(top_k_compounds) == actual_n_select:
            selected = top_k_compounds
        else:
            selected = top_k_compounds.sample(n=actual_n_select)

        # Add acquisition scores (use prediction values)
        selected = selected.with_columns(
            pl.col('prediction').alias('acquisition_score')
        )

        logger.debug(f"TopKAcquisition selected {len(selected)} compounds from "
                    f"{self.score_direction} {k_consider} candidates")

        return selected

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"TopK({self.score_direction}_{self.k_fraction:.2f})"

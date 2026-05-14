"""Random acquisition function for the LearnM8 framework.

This module provides the random acquisition strategy for baseline comparisons.
"""

import logging

import numpy as np
import polars as pl

from .base import AcquisitionFunction

logger = logging.getLogger(__name__)


class RandomAcquisition(AcquisitionFunction):
    """Random acquisition function for baseline comparisons.

    This strategy randomly selects compounds from the available pool,
    providing a baseline for evaluating the effectiveness of more
    sophisticated acquisition strategies.
    """

    def __init__(self, random_state: int = 42, **kwargs):
        """Initialize random acquisition function.

        Args:
            random_state: Random seed for reproducible selection
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(**kwargs)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Randomly select compounds from the pool.

        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
            n_select: Number of compounds to select

        Returns:
            DataFrame subset with randomly selected compounds

        Raises:
            ValueError: If required columns are missing or n_select is invalid
        """
        # Validate input (only basic validation needed)
        if len(compounds) == 0:
            raise ValueError("compounds DataFrame is empty")

        if n_select <= 0:
            raise ValueError("n_select must be positive")

        # Adjust n_select if it exceeds available compounds
        actual_n_select = min(n_select, len(compounds))

        if actual_n_select < n_select:
            logger.warning(f"n_select ({n_select}) exceeds available compounds ({len(compounds)}), "
                         f"will select all {len(compounds)} available compounds")

        logger.debug(f"RandomAcquisition: randomly selecting {actual_n_select} from {len(compounds)} compounds with random_state={self.random_state}")

        # Random selection using Polars
        selected = compounds.sample(n=actual_n_select, seed=self.random_state)

        # Add dummy acquisition scores for consistency
        selected = selected.with_columns(
            pl.Series('acquisition_score', self._rng.uniform(0, 1, size=len(selected)))
        )

        logger.debug(f"RandomAcquisition randomly selected {len(selected)} compounds")

        return selected

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"Random(seed={self.random_state})"

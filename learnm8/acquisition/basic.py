"""Basic acquisition functions for the LearnM8 framework.

This module provides simple acquisition strategies including greedy selection
based on model predictions and random sampling for baseline comparisons.
"""

import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
import polars as pl

from .base import AcquisitionFunction


logger = logging.getLogger(__name__)


class GreedyAcquisition(AcquisitionFunction):
    """Greedy acquisition function that selects compounds with highest predicted values.

    This strategy simply selects the compounds with the highest model predictions,
    representing pure exploitation without exploration. It serves as a strong
    baseline for comparison with more sophisticated acquisition functions.
    """

    def __init__(self, score_direction: str = 'higher', **kwargs):
        """Initialize greedy acquisition function.

        Args:
            score_direction: Direction of score optimization ('higher' or 'lower' is better)
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(score_direction=score_direction, **kwargs)
        if score_direction not in ['higher', 'lower']:
            raise ValueError(f"score_direction must be 'higher' or 'lower', got '{score_direction}'")
        self.score_direction = score_direction
        # Keep backward compatibility
        self.maximize = score_direction == 'higher'

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select compounds with highest/lowest predicted values.

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

        # Get prediction scores
        scores = compounds.get_column('prediction').to_numpy()

        logger.debug(f"GreedyAcquisition: selecting top {n_select} from {len(compounds)} compounds")
        logger.debug(f"Score statistics: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")

        # Adjust n_select if it exceeds available compounds
        actual_n_select = min(n_select, len(compounds))

        # Select top compounds (ascending=False for highest scores, True for lowest)
        selected = self._safe_select_top_k(
            compounds, scores, actual_n_select, ascending=not self.maximize
        )

        logger.debug(f"GreedyAcquisition selected {len(selected)} compounds "
                    f"({'maximizing' if self.maximize else 'minimizing'})")

        return selected

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"Greedy({self.score_direction})"


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
        self._rng = np.random.RandomState(random_state)

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

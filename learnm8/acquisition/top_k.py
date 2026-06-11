"""Top-K acquisition function for the LearnM8 framework.

This module provides the Top-K acquisition strategy which selects based on rank ordering.
"""

import logging

import numpy as np
import polars as pl

from .base import AcquisitionFunction

logger = logging.getLogger(__name__)


class TopKAcquisition(AcquisitionFunction):
    """Top-K acquisition that selects based on rank ordering.

    This is a more flexible version of greedy selection that can be configured
    to select from different parts of the prediction distribution.
    """

    def __init__(
        self,
        k_fraction: float = 0.1,
        score_direction: str = 'higher',
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize Top-K acquisition function.

        Args:
            k_fraction: Fraction of compounds to consider from the top/bottom
            score_direction: Direction of score optimization ('higher' or 'lower' is better)
            random_state: Random seed for the random draw from the top-K pool,
                ensuring reproducible selection when the candidate pool exceeds
                the requested batch size.
            **kwargs: Additional parameters for compatibility
        """
        super().__init__(score_direction=score_direction, **kwargs)
        if not 0.0 < k_fraction <= 1.0:
            raise ValueError("k_fraction must be between 0 and 1")
        if score_direction not in ['higher', 'lower']:
            raise ValueError(f"score_direction must be 'higher' or 'lower', got '{score_direction}'")

        self.k_fraction = k_fraction
        self.score_direction = score_direction
        self.random_state = random_state
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

        # Sort compounds by prediction based on score direction.
        # maintain_order=True gives a deterministic, stable tie ordering.
        descending = self.score_direction == 'higher'
        sorted_compounds = compounds.sort(
            'prediction', descending=descending, maintain_order=True
        )

        # Take top-K compounds
        top_k_compounds = sorted_compounds.head(k_consider)

        # Randomly select from top-K (seeded for reproducibility)
        if len(top_k_compounds) == actual_n_select:
            selected = top_k_compounds
        else:
            selected = top_k_compounds.sample(
                n=actual_n_select, seed=self.random_state
            )

        # Add acquisition scores (use prediction values)
        selected = selected.with_columns(
            pl.col('prediction').alias('acquisition_score')
        )

        logger.debug(f"TopKAcquisition selected {len(selected)} compounds from "
                    f"{self.score_direction} {k_consider} candidates")

        return selected

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
        """Higher-is-better = prediction (negated when ranking from the bottom).

        The streaming reducer keeps the top ``shortlist_size`` by this score
        (the top/bottom ``k_fraction`` of the pool); :meth:`finalize` then draws
        the batch randomly from that shortlist.
        """
        preds = np.asarray(predictions, dtype=np.float64)
        return preds if self.from_top else -preds

    def shortlist_size(self, n_select: int, pool_size: int) -> int:
        """Top/bottom ``k_fraction`` of the pool, never fewer than ``n_select``."""
        k_consider = max(n_select, int(pool_size * self.k_fraction))
        return min(k_consider, pool_size)

    def finalize(self, shortlist: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Randomly draw ``n_select`` from the shortlist (seeded); all if smaller."""
        n = min(n_select, len(shortlist))
        if len(shortlist) <= n:
            selected = shortlist
        else:
            selected = shortlist.sample(n=n, seed=self.random_state)
        return selected.with_columns(pl.col('prediction').alias('acquisition_score'))

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"TopK({self.score_direction}_{self.k_fraction:.2f})"

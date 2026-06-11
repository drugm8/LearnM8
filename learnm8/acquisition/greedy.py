"""Greedy acquisition function for the LearnM8 framework.

This module provides the greedy acquisition strategy which selects compounds
with the highest model predictions.
"""

import logging

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
        """Higher-is-better score = prediction (negated when minimising)."""
        preds = np.asarray(predictions, dtype=np.float64)
        return preds if self.maximize else -preds

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function."""
        return f"Greedy({self.score_direction})"

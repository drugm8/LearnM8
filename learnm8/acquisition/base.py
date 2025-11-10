"""Base acquisition function interface for the LearnM8 framework.

This module defines the abstract base class for all acquisition functions
following the new architecture design with clean interfaces and dependency injection.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
import polars as pl
import numpy as np


logger = logging.getLogger(__name__)


class AcquisitionFunction(ABC):
    """Base class for compound selection strategies.

    Acquisition functions determine which compounds to select for labeling
    in each active learning cycle based on model predictions and optionally
    uncertainty estimates.

    Args:
        score_direction: Direction to optimize ('higher' or 'lower'). Default 'higher'
        **kwargs: Additional parameters for specific acquisition methods
    """

    def __init__(self, score_direction: str = 'higher', **kwargs):
        """Initialize acquisition function.

        Args:
            score_direction: Direction to optimize ('higher' or 'lower'). Default 'higher'
            **kwargs: Additional parameters for specific acquisition methods
        """

        # Validate and store score direction
        if score_direction not in ['higher', 'lower']:
            raise ValueError(f"score_direction must be 'higher' or 'lower', got '{score_direction}'")

        self.score_direction = score_direction
        self.maximize = score_direction == 'higher'

    @abstractmethod
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """
        Select compounds for labeling.

        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
                      May also contain 'uncertainty' column if available
            n_select: Number of compounds to select

        Returns:
            DataFrame subset with selected compounds

        Raises:
            ValueError: If required columns are missing or n_select is invalid
            RuntimeError: If selection fails
        """
        pass

    def requires_uncertainty(self) -> bool:
        """Return True if this acquisition function requires uncertainty estimates.

        Returns:
            Boolean indicating if uncertainty column is required
        """
        return False

    def get_name(self) -> str:
        """Return a descriptive name for this acquisition function.

        Returns:
            String identifier for the acquisition function
        """
        return self.__class__.__name__

    def validate_input(self, compounds: pl.DataFrame, n_select: int) -> None:
        """Validate input parameters for acquisition function.

        Args:
            compounds: DataFrame to validate
            n_select: Number of compounds to select

        Raises:
            ValueError: If input is invalid
        """
        # Check basic DataFrame structure
        if len(compounds) == 0:
            raise ValueError("compounds DataFrame is empty")

        # Check required columns
        required_cols = ['ID', 'SMILES', 'prediction']
        missing_cols = set(required_cols) - set(compounds.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check uncertainty column if required
        if self.requires_uncertainty() and 'uncertainty' not in compounds.columns:
            raise ValueError(f"{self.get_name()} requires 'uncertainty' column")

        # Check n_select
        if n_select <= 0:
            raise ValueError("n_select must be positive")

        if n_select > len(compounds):
            logger.warning(f"n_select ({n_select}) exceeds available compounds ({len(compounds)}), "
                         f"will select all {len(compounds)} available compounds")

        # Check for NaN values in predictions
        if compounds.get_column('prediction').is_null().any():
            raise ValueError("Predictions contain NaN values")

        # Check for duplicate IDs
        if compounds.get_column('ID').n_unique() != len(compounds):
            raise ValueError("Duplicate compound IDs found in input data")

        # Check uncertainty values if present
        if 'uncertainty' in compounds.columns:
            if compounds.get_column('uncertainty').is_null().any():
                raise ValueError("Uncertainties contain NaN values")

            if (compounds.get_column('uncertainty') < 0).any():
                raise ValueError("Uncertainties must be non-negative")

    def _safe_select_top_k(self, compounds: pl.DataFrame, scores: np.ndarray,
                          n_select: int, ascending: bool = False) -> pl.DataFrame:
        """Safely select top-k compounds based on scores.

        Args:
            compounds: DataFrame of compounds
            scores: Array of scores for each compound
            n_select: Number of compounds to select
            ascending: If True, select lowest scores; if False, select highest

        Returns:
            DataFrame with selected compounds

        Raises:
            ValueError: If scores array length doesn't match compounds
        """
        if len(scores) != len(compounds):
            raise ValueError(f"Scores length ({len(scores)}) doesn't match compounds length ({len(compounds)})")

        # Handle infinite or NaN scores
        valid_mask = np.isfinite(scores)
        if not valid_mask.all():
            logger.warning(f"Found {(~valid_mask).sum()} invalid scores, setting to worst value")
            if ascending:
                scores[~valid_mask] = np.inf
            else:
                scores[~valid_mask] = -np.inf

        # Get top indices
        if ascending:
            top_indices = np.argsort(scores)[:n_select]
        else:
            top_indices = np.argsort(scores)[::-1][:n_select]

        # Get IDs for selected compounds
        all_ids = compounds.get_column('ID').to_numpy()
        selected_ids = all_ids[top_indices]

        # Filter by selected IDs
        selected_compounds = compounds.filter(pl.col('ID').is_in(selected_ids.tolist()))

        # Add acquisition scores for debugging/analysis
        score_dict = dict(zip(selected_ids, scores[top_indices]))
        selected_compounds = selected_compounds.with_columns(
            pl.col('ID').replace_strict(score_dict).alias('acquisition_score')
        )

        return selected_compounds


class AcquisitionError(Exception):
    """Exception raised when acquisition strategy encounters an error."""
    pass


# Utility functions for acquisition calculations
def validate_uncertainty_inputs(compounds: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and extract prediction means and uncertainties from compounds DataFrame.

    Args:
        compounds: DataFrame with 'prediction' and 'uncertainty' columns

    Returns:
        Tuple of (predictions, uncertainties) as numpy arrays

    Raises:
        AcquisitionError: If required columns are missing or contain invalid values
    """
    if 'prediction' not in compounds.columns:
        raise AcquisitionError("Compounds must have 'prediction' column for acquisition strategies")

    if 'uncertainty' not in compounds.columns:
        raise AcquisitionError("Compounds must have 'uncertainty' column for acquisition strategies")

    predictions = compounds.get_column('prediction').to_numpy()
    uncertainties = compounds.get_column('uncertainty').to_numpy()

    # Check for NaN values
    if np.any(np.isnan(predictions)):
        raise AcquisitionError("Predictions contain NaN values")

    if np.any(np.isnan(uncertainties)):
        raise AcquisitionError("Uncertainties contain NaN values")

    # Check for negative uncertainties
    if np.any(uncertainties < 0):
        raise AcquisitionError("Uncertainties must be non-negative")

    return predictions, uncertainties

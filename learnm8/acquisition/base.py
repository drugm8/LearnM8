"""Base acquisition function interface for the LearnM8 framework.

This module defines the abstract base class for all acquisition functions
following the new architecture design with clean interfaces and dependency injection.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from learnm8.exceptions import (
    AcquisitionError,
)

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
            raise ValueError(
                f"score_direction must be 'higher' or 'lower', got '{score_direction}'. "
                f"Use 'higher' to maximize the target property or 'lower' to minimize it."
            )

        self.score_direction = score_direction
        self.maximize = score_direction == 'higher'
        self._skip_unique_id_check = False

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
            raise ValueError(
                f"compounds DataFrame is empty. Cannot run {self.get_name()} on an empty compound pool. "
                f"All compounds may have been labeled or pruned. "
                f"Check n_cycles and pruning_fraction settings."
            )

        # Check required columns
        required_cols = ['ID', 'SMILES', 'prediction']
        missing_cols = set(required_cols) - set(compounds.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns for {self.get_name()}: expected {required_cols}, "
                f"but {missing_cols} are missing. "
                f"Available columns: {list(compounds.columns)}."
            )

        # Check uncertainty column if required
        if self.requires_uncertainty() and 'uncertainty' not in compounds.columns:
            uncertainty_strategies = ['greedy', 'random', 'topk']
            raise ValueError(
                f"{self.get_name()} requires an 'uncertainty' column, but the learner did not "
                f"provide uncertainty estimates. Use a learner that supports uncertainty "
                f"(gp, mc_dropout, or ensemble variants), or switch to an acquisition "
                f"strategy that doesn't require uncertainty (e.g., {', '.join(uncertainty_strategies)})."
            )

        # Check n_select
        if n_select <= 0:
            raise ValueError(
                f"n_select must be positive, got {n_select}. "
                f"This is usually caused by batch_fraction being too small for the pool size."
            )

        if n_select > len(compounds):
            logger.warning(f"n_select ({n_select}) exceeds available compounds ({len(compounds)}), "
                         f"will select all {len(compounds)} available compounds")

        # Check for NaN/null values in predictions — single combined pass (not 4 passes)
        pred_col = compounds.get_column('prediction')
        null_or_nan = pred_col.is_null() | pred_col.is_nan()
        if null_or_nan.any():
            raise ValueError(
                f"Predictions contain NaN values: found {null_or_nan.sum()} NaN/null values out of {len(compounds)} compounds. "
                f"This may indicate a featurizer or learner issue. "
                f"Check that all SMILES are valid and the model trained successfully."
            )

        # Check for duplicate IDs — skipped on internal call path where uniqueness is guaranteed
        if not self._skip_unique_id_check:
            n_unique_count = compounds.get_column('ID').n_unique()
            if n_unique_count != len(compounds):
                n_dupes = len(compounds) - n_unique_count
                raise ValueError(
                    f"Found {n_dupes} duplicate compound IDs in input data. "
                    f"Each compound must have a unique ID for acquisition selection."
                )

        # Check uncertainty values if present
        if 'uncertainty' in compounds.columns:
            unc_col = compounds.get_column('uncertainty')
            if unc_col.is_null().any() or unc_col.is_nan().any():
                nan_count = unc_col.is_null().sum() + unc_col.is_nan().sum()
                raise ValueError(
                    f"Uncertainties contain {nan_count} NaN/null values out of {len(compounds)} compounds. "
                    f"This may indicate an issue with the uncertainty estimation in the learner."
                )

            if (unc_col < 0).any():
                neg_count = (unc_col < 0).sum()
                raise ValueError(
                    f"Uncertainties contain {neg_count} negative values. "
                    f"Uncertainty estimates must be non-negative (>= 0)."
                )

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
            raise ValueError(
                f"Acquisition scores length ({len(scores)}) doesn't match "
                f"compounds length ({len(compounds)}). "
                f"This is an internal error in the acquisition function implementation."
            )

        # Handle invalid scores — copy only when bad values exist (avoids large allocation on normal path)
        # Note: scores may be modified in-place on this branch. Callers pass fresh .to_numpy() arrays
        # so this is safe. Do not reuse the array after calling this method if NaN/Inf were present.
        valid_mask = np.isfinite(scores)
        if not valid_mask.all():
            logger.warning(f"Found {(~valid_mask).sum()} invalid scores, setting to worst value")
            scores = scores.copy()
            scores[~valid_mask] = np.inf if ascending else -np.inf

        n = len(scores)
        actual_n_select = min(n_select, n)

        if ascending:
            if actual_n_select >= n:
                top_indices = np.argsort(scores)
            else:
                # O(n) partition + O(k log k) sort with index-stable tie-breaking
                top_indices = np.argpartition(scores, actual_n_select - 1)[:actual_n_select]
                top_indices = top_indices[np.lexsort((top_indices, scores[top_indices]))]
        else:
            if actual_n_select >= n:
                top_indices = np.argsort(scores)[::-1]
            else:
                # O(n) partition + O(k log k) sort with index-stable tie-breaking
                top_indices = np.argpartition(scores, -actual_n_select)[-actual_n_select:]
                top_indices = top_indices[np.lexsort((top_indices, -scores[top_indices]))]

        # O(k) positional row retrieval — requires Polars >=1.0 and 0-based positional alignment.
        # Invariant: scores were extracted from this same DataFrame via .to_numpy() in the same
        # execution context, so row i of compounds corresponds to scores[i].
        selected_compounds = compounds[top_indices]
        selected_compounds = selected_compounds.with_columns(
            pl.Series('acquisition_score', scores[top_indices])
        )

        return selected_compounds


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
        raise AcquisitionError(
            f"Compounds DataFrame is missing required 'prediction' column. "
            f"Available columns: {list(compounds.columns)}. "
            f"Ensure the learner has been trained and predictions were generated."
        )

    if 'uncertainty' not in compounds.columns:
        raise AcquisitionError(
            f"Compounds DataFrame is missing required 'uncertainty' column for "
            f"uncertainty-based acquisition. Available columns: {list(compounds.columns)}. "
            f"Use a learner that supports uncertainty (gp, mc_dropout, or ensemble variants), "
            f"or switch to an acquisition strategy that doesn't require uncertainty "
            f"(greedy, random, topk)."
        )

    predictions = compounds.get_column('prediction').to_numpy()
    uncertainties = compounds.get_column('uncertainty').to_numpy()

    # Check for NaN values
    if np.any(np.isnan(predictions)):
        nan_count = int(np.isnan(predictions).sum())
        raise AcquisitionError(
            f"Predictions contain {nan_count} NaN values out of {len(predictions)} compounds. "
            f"Check that the learner trained successfully and all SMILES are valid."
        )

    if np.any(np.isnan(uncertainties)):
        nan_count = int(np.isnan(uncertainties).sum())
        raise AcquisitionError(
            f"Uncertainties contain {nan_count} NaN values out of {len(uncertainties)} compounds. "
            f"This may indicate an issue with the uncertainty estimation in the learner."
        )

    # Check for negative uncertainties
    if np.any(uncertainties < 0):
        neg_count = int((uncertainties < 0).sum())
        raise AcquisitionError(
            f"Uncertainties contain {neg_count} negative values. "
            f"Uncertainty estimates must be non-negative (>= 0)."
        )

    return predictions, uncertainties

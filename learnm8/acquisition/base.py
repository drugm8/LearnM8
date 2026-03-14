"""Base acquisition function interface for the LearnM8 framework.

This module defines the abstract base class for all acquisition functions
following the new architecture design with clean interfaces and dependency injection.
"""

import logging
from abc import ABC, abstractmethod
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
            raise ValueError(
                f"score_direction must be 'higher' or 'lower', got '{score_direction}'. "
                f"Use 'higher' to maximize the target property or 'lower' to minimize it."
            )

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
            raise ValueError(
                f"Cannot run {self.get_name()} on an empty compound pool. "
                f"All compounds may have been labeled or pruned. "
                f"Check n_cycles and pruning_fraction settings."
            )

        # Check required columns
        required_cols = ['ID', 'SMILES', 'prediction']
        missing_cols = set(required_cols) - set(compounds.columns)
        if missing_cols:
            raise ValueError(
                f"{self.get_name()} requires columns {required_cols}, "
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

        # Check for NaN/null values in predictions
        pred_col = compounds.get_column('prediction')
        if pred_col.is_null().any() or pred_col.is_nan().any():
            nan_count = pred_col.is_null().sum() + pred_col.is_nan().sum()
            raise ValueError(
                f"Predictions contain {nan_count} NaN/null values out of {len(compounds)} compounds. "
                f"This may indicate a featurizer or learner issue. "
                f"Check that all SMILES are valid and the model trained successfully."
            )

        # Check for duplicate IDs
        if compounds.get_column('ID').n_unique() != len(compounds):
            n_dupes = len(compounds) - compounds.get_column('ID').n_unique()
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

        # Sort by acquisition score to preserve selection order
        selected_compounds = selected_compounds.sort('acquisition_score', descending=not ascending)

        return selected_compounds


from learnm8.exceptions import AcquisitionError  # noqa: E402 — re-exported for backward compat


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

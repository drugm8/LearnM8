"""Base classes for design space pruning in the LearnM8 framework.

This module defines the abstract interface for design space pruning strategies
that reduce the unlabeled compound pool by removing unlikely candidates.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class DesignSpacePruner(ABC):
    """Base class for design space pruning strategies.
    
    Design space pruners analyze model predictions and uncertainties to identify
    and remove compounds that are unlikely to be valuable for active learning,
    thereby reducing computational costs and focusing on promising regions.
    """

    @abstractmethod
    def prune(self,
              compounds: pl.DataFrame,
              predictions: np.ndarray,
              uncertainties: np.ndarray | None = None) -> pl.DataFrame:
        """
        Prune the compound pool based on predictions.
        
        Args:
            compounds: DataFrame with 'ID', 'SMILES' columns
            predictions: Model predictions for each compound
            uncertainties: Model uncertainties (optional, not used by score-based pruning)
            
        Returns:
            Pruned DataFrame subset with promising compounds
            
        Raises:
            PruningError: If pruning fails or inputs are invalid
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return a descriptive name for this pruning strategy.
        
        Returns:
            String identifier for the pruning strategy
        """
        pass

    def requires_uncertainty(self) -> bool:
        """
        Return True if this pruning strategy requires uncertainty estimates.
        
        Returns:
            Boolean indicating if uncertainty is required
        """
        return False

    def validate_inputs(self,
                       compounds: pl.DataFrame,
                       predictions: np.ndarray,
                       uncertainties: np.ndarray | None = None) -> None:
        """
        Validate inputs for pruning operation.
        
        Args:
            compounds: DataFrame to validate
            predictions: Predictions array to validate
            uncertainties: Uncertainties array to validate (optional)
            
        Raises:
            PruningError: If inputs are invalid
        """
        # Check DataFrame structure
        if len(compounds) == 0:
            raise PruningError(
                "Cannot prune an empty compound pool. "
                "All compounds may have been labeled or already pruned. "
                "Reduce pruning_fraction or n_cycles to retain more compounds."
            )

        required_cols = ['ID', 'SMILES']
        missing_cols = set(required_cols) - set(compounds.columns)
        if missing_cols:
            raise PruningError(
                f"Compounds DataFrame missing required columns: {missing_cols}. "
                f"Available columns: {list(compounds.columns)}."
            )

        # Check predictions array
        if len(predictions) != len(compounds):
            raise PruningError(
                f"Predictions length ({len(predictions)}) doesn't match compounds "
                f"length ({len(compounds)}). Ensure the model predicted all compounds."
            )

        if np.any(np.isnan(predictions)):
            nan_count = int(np.isnan(predictions).sum())
            raise PruningError(
                f"Predictions contain {nan_count} NaN values out of {len(predictions)} compounds. "
                f"Check that the model trained successfully and all compounds have valid features."
            )

        # Check uncertainties if provided
        if uncertainties is not None:
            if len(uncertainties) != len(compounds):
                raise PruningError(
                    f"Uncertainties length ({len(uncertainties)}) doesn't match "
                    f"compounds length ({len(compounds)})."
                )

            if np.any(np.isnan(uncertainties)):
                nan_count = int(np.isnan(uncertainties).sum())
                raise PruningError(
                    f"Uncertainties contain {nan_count} NaN values. "
                    f"Check the learner's uncertainty estimation."
                )

            if np.any(uncertainties < 0):
                neg_count = int((uncertainties < 0).sum())
                raise PruningError(
                    f"Uncertainties contain {neg_count} negative values. "
                    f"Uncertainty estimates must be non-negative (>= 0)."
                )

        # Check if uncertainties are required but not provided
        if self.requires_uncertainty() and uncertainties is None:
            raise PruningError(
                f"{self.get_name()} requires uncertainty estimates, but none were provided. "
                f"Use a learner that supports uncertainty (gp, mc_dropout, or ensemble variants)."
            )

    def _calculate_pruning_fraction(self, original_count: int, pruned_count: int) -> float:
        """Calculate the fraction of compounds pruned.
        
        Args:
            original_count: Original number of compounds
            pruned_count: Number of compounds after pruning
            
        Returns:
            Fraction of compounds removed (0.0 to 1.0)
        """
        if original_count == 0:
            return 0.0

        removed_count = original_count - pruned_count
        return removed_count / original_count

    def _safe_prune_by_indices(self,
                              compounds: pl.DataFrame,
                              keep_indices: np.ndarray) -> pl.DataFrame:
        """Safely prune compounds by keeping specified indices.
        
        Args:
            compounds: Original compounds DataFrame
            keep_indices: Boolean or integer array of indices to keep
            
        Returns:
            Pruned compounds DataFrame
            
        Raises:
            PruningError: If indices are invalid
        """
        try:
            if len(keep_indices) == 0:
                logger.warning("No compounds selected for retention during pruning")
                return compounds.head(0)  # Return empty DataFrame with same structure

            # Handle boolean indexing
            if keep_indices.dtype == bool:
                if len(keep_indices) != len(compounds):
                    raise PruningError(
                        f"Boolean index length ({len(keep_indices)}) doesn't match "
                        f"compounds length ({len(compounds)}). "
                        f"This is an internal error in the pruning strategy implementation."
                    )
                pruned_compounds = compounds.filter(pl.Series(keep_indices))
            else:
                # Handle integer indexing
                if np.any(keep_indices >= len(compounds)) or np.any(keep_indices < 0):
                    raise PruningError(
                        f"Invalid compound indices for pruning (out of range 0-{len(compounds) - 1}). "
                        f"This is an internal error in the pruning strategy implementation."
                    )
                pruned_compounds = compounds[keep_indices]

            return pruned_compounds

        except PruningError:
            raise
        except (ValueError, RuntimeError, TypeError) as e:
            raise PruningError(f"Failed to prune compounds: {e}") from e


from learnm8.exceptions import (
    PruningError,
)

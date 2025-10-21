"""Compound validation for early error detection.

This module validates all compounds upfront by attempting feature extraction,
catching invalid SMILES and feature generation errors before any active learning
cycles execute.

Key benefits:
- Catch errors early (before cycles start)
- Clear error messages for each invalid compound
- Benefits from caching (subsequent extraction is instant)
- Structured results with success rate calculation
"""

from dataclasses import dataclass
import pandas as pd
from typing import Dict
import logging
from pathlib import Path
from learnm8.features.extraction import extract_features

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from compound validation.

    Attributes:
        valid_compounds: Compounds that passed validation (ID, SMILES columns only)
        invalid_compounds: Compounds that failed validation (ID, SMILES columns only)
        validation_errors: Mapping of compound ID to error message string
    """
    valid_compounds: pd.DataFrame
    invalid_compounds: pd.DataFrame
    validation_errors: Dict[str, str]

    @property
    def success_rate(self) -> float:
        """Calculate validation success rate.

        Returns:
            Fraction of compounds that passed validation (0.0 to 1.0)
        """
        total = len(self.valid_compounds) + len(self.invalid_compounds)
        if total == 0:
            return 0.0
        return len(self.valid_compounds) / total


def validate_compound_pool(
    compound_pool: pd.DataFrame,
    featurizer_type: str,
    cache_dir: Path
) -> ValidationResult:
    """Validate all compounds by attempting feature extraction.

    This function validates every compound upfront to catch errors early,
    before any active learning cycles execute. It's intentionally slow but
    benefits from HDF5 caching - subsequent feature extraction will be instant.

    Args:
        compound_pool: DataFrame with 'ID' and 'SMILES' columns
        featurizer_type: Type of molecular featurizer ('morgan', 'descriptors', etc.)
        cache_dir: Directory for HDF5 feature cache

    Returns:
        ValidationResult with valid/invalid compounds and error messages

    Example:
        >>> result = validate_compound_pool(df, 'morgan', Path('.cache'))
        >>> print(f"Success rate: {result.success_rate:.1%}")
        >>> valid_df = result.valid_compounds
        >>> for cid, error in result.validation_errors.items():
        ...     print(f"Compound {cid}: {error}")
    """
    required = {'ID', 'SMILES'}
    missing = required - set(compound_pool.columns)
    if missing:
        logger.error("compound_pool missing required columns: %s", missing)
        invalid_df = compound_pool.copy()
        # Ensure invalid_compounds has required columns (create with NAs if missing)
        for col in required:
            if col not in invalid_df.columns:
                invalid_df[col] = pd.NA
        # Use index as error key if ID column missing, otherwise use ID
        if 'ID' in compound_pool.columns:
            errors = {str(row['ID']): f"Missing columns: {sorted(missing)}" for _, row in invalid_df.iterrows()}
        else:
            errors = {str(idx): f"Missing columns: {sorted(missing)}" for idx in invalid_df.index}
        return ValidationResult(pd.DataFrame(columns=list(required)), invalid_df, errors)

    valid_compounds = []
    invalid_compounds = []
    errors = {}

    logger.info(f"Validating {len(compound_pool)} compounds...")

    for i, (idx, row) in enumerate(compound_pool.iterrows(), start=1):
        try:
            extract_features([row['SMILES']], featurizer_type, cache_dir)
            valid_compounds.append(row)
        except Exception as e:
            invalid_compounds.append(row)
            # Use ID if available, otherwise use index
            error_key = str(row['ID']) if 'ID' in row and pd.notna(row['ID']) else str(idx)
            errors[error_key] = str(e)

        if i % 100 == 0:
            logger.debug(f"Validated {i}/{len(compound_pool)} compounds")

    valid_df = pd.DataFrame(valid_compounds) if valid_compounds else pd.DataFrame(columns=compound_pool.columns)
    invalid_df = pd.DataFrame(invalid_compounds) if invalid_compounds else pd.DataFrame(columns=compound_pool.columns)

    # Ensure both DataFrames have required columns (add with NAs if missing)
    for col in required:
        if col not in valid_df.columns:
            valid_df[col] = pd.NA
        if col not in invalid_df.columns:
            invalid_df[col] = pd.NA

    result = ValidationResult(valid_df, invalid_df, errors)
    logger.info(f"Validation complete: {len(valid_df)} valid, {len(invalid_df)} invalid ({result.success_rate:.1%} success rate)")

    return result

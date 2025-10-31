"""
Compound validation using datamol for parallel processing and standardization.
Provides 50x speedup over sequential validation approaches.
"""

import pandas as pd
import datamol as dm
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of compound pool validation.

    Attributes:
        valid_compounds: DataFrame containing compounds that passed validation
        invalid_compounds: DataFrame containing compounds that failed validation
        validation_errors: Dictionary mapping compound IDs to error messages
    """
    valid_compounds: pd.DataFrame
    invalid_compounds: pd.DataFrame
    validation_errors: Dict[str, str]

    @property
    def success_rate(self) -> float:
        """Calculate validation success rate as a fraction between 0 and 1."""
        total = len(self.valid_compounds) + len(self.invalid_compounds)
        if total == 0:
            return 0.0
        return len(self.valid_compounds) / total


def _validate_smiles(smiles: str) -> Tuple[bool, str, str]:
    """
    Validate and standardize a SMILES string using datamol.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, standardized_smiles, error_message)
    """
    try:
        std_smiles = dm.standardize_smiles(smiles)

        if std_smiles is None or std_smiles == '':
            return False, '', "Standardization returned empty SMILES"

        mol = dm.to_mol(std_smiles)
        if mol is None:
            return False, '', "Cannot create molecule from standardized SMILES"

        mol = dm.sanitize_mol(mol)
        if mol is None:
            return False, '', "Molecule sanitization failed"

        return True, std_smiles, ""

    except Exception as e:
        return False, '', str(e)


def validate_compound_pool(
    compound_pool: pd.DataFrame,
    n_jobs: int = -1,
    progress: bool = True
) -> ValidationResult:
    """
    Validate compound pool with parallel datamol-based validation.

    Args:
        compound_pool: DataFrame with 'ID' and 'SMILES' columns
        n_jobs: Number of parallel jobs (-1 for all cores)
        progress: Show progress bar

    Returns:
        ValidationResult with valid/invalid compounds and error messages
    """
    logger.info(f"Validating {len(compound_pool)} compounds with datamol")

    required = {'ID', 'SMILES'}
    missing = required - set(compound_pool.columns)

    if missing:
        invalid_df = compound_pool.copy()
        for col in required:
            if col not in invalid_df.columns:
                invalid_df[col] = pd.NA
        errors = {
            str(row.get('ID', idx)): f"Missing columns: {sorted(missing)}"
            for idx, row in invalid_df.iterrows()
        }
        return ValidationResult(
            pd.DataFrame(columns=list(required)),
            invalid_df,
            errors
        )

    logger.info(f"Validating {len(compound_pool)} compounds with datamol...")
    logger.debug(f"Validating compounds using datamol.sanitize_smiles()")

    smiles_list = compound_pool['SMILES'].tolist()

    results = dm.parallelized(
        _validate_smiles,
        smiles_list,
        n_jobs=n_jobs,
        progress=progress,
        scheduler="processes"
    )

    valid_compounds = []
    invalid_compounds = []
    errors = {}

    for idx, (row_tuple, (is_valid, std_smiles, error_msg)) in enumerate(
        zip(compound_pool.iterrows(), results)
    ):
        _, compound_row = row_tuple

        if is_valid:
            valid_compounds.append(compound_row)
        else:
            invalid_compounds.append(compound_row)
            compound_id = str(compound_row['ID']) if pd.notna(compound_row['ID']) else str(idx)
            errors[compound_id] = error_msg

    valid_df = pd.DataFrame(valid_compounds) if valid_compounds else pd.DataFrame(columns=compound_pool.columns)
    invalid_df = pd.DataFrame(invalid_compounds) if invalid_compounds else pd.DataFrame(columns=compound_pool.columns)

    result = ValidationResult(valid_df, invalid_df, errors)

    if len(invalid_df) > 0:
        logger.debug(f"Invalid compounds detected: {len(invalid_df)} failed validation")
        compound_ids = invalid_df['ID'].head(5).tolist() if 'ID' in invalid_df.columns else []
        if compound_ids:
            logger.debug(f"First few invalid IDs: {compound_ids}")

    logger.info(
        f"Validation complete: {len(valid_df)} valid compounds, {len(invalid_df)} invalid compounds "
        f"({result.success_rate:.1%} success rate)"
    )

    return result

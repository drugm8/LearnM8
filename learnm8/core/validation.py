"""
Compound validation using datamol for parallel processing and standardization.
Provides 50x speedup over sequential validation approaches.
"""

import logging
from dataclasses import dataclass

import datamol as dm
import polars as pl

from learnm8.exceptions import ValidationError

logger = logging.getLogger(__name__)


def _validate_target_dtype(valid_df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    """Enforce a numeric target column, raising on non-numeric input.

    Numeric columns pass through unchanged. Utf8 columns are strict-cast to
    Float64 (raising ``ValidationError`` if any value fails to parse). Boolean
    and all other dtypes are rejected — booleans must be cast to int 0/1 by the
    caller so regression-vs-classification intent stays explicit.
    """
    dtype = valid_df[target_col].dtype

    if dtype.is_numeric():
        return valid_df

    if dtype == pl.Boolean:
        raise ValidationError(
            f"Target column '{target_col}' has Boolean dtype. Cast it to an "
            f"integer 0/1 column before running so regression-vs-classification "
            f"intent is explicit."
        )

    if dtype == pl.Utf8:
        try:
            cast_col = valid_df[target_col].cast(pl.Float64, strict=True)
        except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as e:
            raise ValidationError(
                f"Target column '{target_col}' contains non-numeric values that "
                f"cannot be parsed as numbers: {e}. Clean the target column so "
                f"every value is numeric."
            ) from None
        return valid_df.with_columns(cast_col.alias(target_col))

    raise ValidationError(
        f"Target column '{target_col}' has unsupported dtype {dtype}. "
        f"The target must be a numeric column."
    )


@dataclass
class ValidationResult:
    """
    Result of compound pool validation.

    Attributes:
        valid_compounds: DataFrame containing compounds that passed validation
        invalid_compounds: DataFrame containing compounds that failed validation
        validation_errors: Dictionary mapping compound IDs to error messages
    """
    valid_compounds: pl.DataFrame
    invalid_compounds: pl.DataFrame
    validation_errors: dict[str, str]

    @property
    def success_rate(self) -> float:
        """Calculate validation success rate as a fraction between 0 and 1."""
        total = len(self.valid_compounds) + len(self.invalid_compounds)
        if total == 0:
            return 0.0
        return len(self.valid_compounds) / total


def _validate_smiles(smiles: str) -> tuple[bool, str, str]:
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

    except (ValueError, RuntimeError, TypeError, AttributeError, OSError) as e:
        return False, '', str(e)


def validate_compound_pool(
    compound_pool: pl.DataFrame,
    n_jobs: int = -1,
    progress: bool = True,
    target_col: str | None = None,
) -> ValidationResult:
    """
    Validate compound pool with parallel datamol-based validation.

    Args:
        compound_pool: DataFrame with 'ID' and 'SMILES' columns
        n_jobs: Number of parallel jobs (-1 for all cores)
        progress: Show progress bar
        target_col: If given and present in the pool, the target column dtype is
            validated — it must be numeric (Utf8 is strict-cast to Float64).
            Non-numeric or Boolean targets raise ``ValidationError``.

    Returns:
        ValidationResult with valid/invalid compounds and error messages

    Raises:
        ValidationError: If ``target_col`` is present but not a numeric column.
    """
    logger.info(f"Validating {len(compound_pool)} compounds with datamol")

    required = {'ID', 'SMILES'}
    missing = required - set(compound_pool.columns)

    if missing:
        invalid_df = compound_pool.clone()
        for col in required:
            if col not in invalid_df.columns:
                invalid_df = invalid_df.with_columns(pl.lit(None).alias(col))
        errors = {
            str(row.get('ID', idx)): f"Missing columns: {sorted(missing)}"
            for idx, row in enumerate(invalid_df.iter_rows(named=True))
        }
        return ValidationResult(
            pl.DataFrame(schema={col: pl.Utf8 for col in required}),
            invalid_df,
            errors
        )

    # Check for duplicate IDs
    duplicate_mask = compound_pool.select(
        pl.col('ID').is_duplicated().alias('is_dup')
    )['is_dup']

    if duplicate_mask.any():
        # Separate duplicates from unique compounds
        dup_df = compound_pool.filter(duplicate_mask)
        unique_pool = compound_pool.filter(~duplicate_mask)

        # Build error dict for duplicates
        dup_errors = {
            str(row['ID']): "Duplicate ID"
            for row in dup_df.iter_rows(named=True)
        }

        logger.info(
            f"Found {duplicate_mask.sum()} duplicate IDs, "
            f"continuing validation with {len(unique_pool)} unique compounds"
        )
    else:
        unique_pool = compound_pool
        dup_df = pl.DataFrame(schema=compound_pool.schema)
        dup_errors = {}

    logger.debug("Validating compounds using datamol.sanitize_smiles()")

    smiles_list = unique_pool['SMILES'].to_list()

    results = dm.parallelized(
        _validate_smiles,
        smiles_list,
        n_jobs=n_jobs,
        progress=progress,
        scheduler="processes"
    )

    valid_compounds = []
    invalid_compounds = []
    smiles_errors = {}

    for idx, (compound_row, (is_valid, std_smiles, error_msg)) in enumerate(
        zip(unique_pool.iter_rows(named=True), results)
    ):
        if is_valid:
            valid_compounds.append(compound_row)
        else:
            invalid_compounds.append(compound_row)
            compound_id = str(compound_row['ID']) if compound_row['ID'] is not None else str(idx)
            smiles_errors[compound_id] = error_msg

    valid_df = pl.DataFrame(valid_compounds, schema=compound_pool.schema) if valid_compounds else pl.DataFrame(schema=compound_pool.schema)
    smiles_invalid_df = pl.DataFrame(invalid_compounds, schema=compound_pool.schema) if invalid_compounds else pl.DataFrame(schema=compound_pool.schema)

    if target_col is not None and target_col in valid_df.columns and len(valid_df) > 0:
        valid_df = _validate_target_dtype(valid_df, target_col)

    # Combine duplicate errors with SMILES validation errors
    all_errors = {**dup_errors, **smiles_errors}

    # Combine duplicate compounds with SMILES invalid compounds
    if len(dup_df) > 0 or len(smiles_invalid_df) > 0:
        invalid_df = pl.concat([dup_df, smiles_invalid_df], how="vertical")
    else:
        invalid_df = pl.DataFrame(schema=compound_pool.schema)

    result = ValidationResult(valid_df, invalid_df, all_errors)

    if len(invalid_df) > 0:
        logger.debug(f"Invalid compounds detected: {len(invalid_df)} failed validation")
        if len(dup_df) > 0:
            logger.debug(f"  - {len(dup_df)} duplicate IDs")
        if len(smiles_invalid_df) > 0:
            logger.debug(f"  - {len(smiles_invalid_df)} invalid SMILES")
        compound_ids = invalid_df['ID'].head(5).to_list() if 'ID' in invalid_df.columns else []
        if compound_ids:
            logger.debug(f"First few invalid IDs: {compound_ids}")

    logger.info(
        f"Validation complete: {len(valid_df)} valid compounds, {len(invalid_df)} invalid compounds "
        f"({result.success_rate:.1%} success rate)"
    )

    return result

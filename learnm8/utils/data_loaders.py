"""Data loading utilities for LearnM8 experiments."""

import logging
from pathlib import Path

import polars as pl

from learnm8.utils.file_loaders import load_compound_file

logger = logging.getLogger(__name__)


def validate_csv_columns(df: pl.DataFrame, required_columns: list, file_description: str) -> None:
    """Validate that DataFrame contains required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        available_columns = list(df.columns)
        raise ValueError(
            f"{file_description} missing required columns: {missing_columns}. "
            f"Available columns: {available_columns}"
        )


def load_run_data(compounds_path: str, oracle_path: str) -> pl.DataFrame:
    """
    Load data for run mode with separate compound pool.

    Args:
        compounds_path: Path to compound pool file (CSV/SDF/SMI)
        oracle_path: Path to oracle Python file (for validation only)

    Returns:
        compound_pool_df: DataFrame with compound data
    """
    compounds_path = Path(compounds_path)
    oracle_path = Path(oracle_path)

    if not compounds_path.exists():
        raise FileNotFoundError(f"Compound pool file not found: {compounds_path}")

    if not oracle_path.exists():
        raise FileNotFoundError(f"Oracle file not found: {oracle_path}")

    # Load compound pool using multi-format loader
    compound_pool = load_compound_file(compounds_path, progress=False)

    # Validate required columns
    required_columns = ['ID', 'SMILES']
    validate_csv_columns(compound_pool, required_columns, "Compound pool file")

    # Validate oracle file is Python
    if oracle_path.suffix != '.py':
        logger.warning("Oracle file should have .py extension, got: %s", oracle_path.suffix)

    logger.info("Loaded %d compounds for active learning", len(compound_pool))
    logger.info("Oracle file: %s", oracle_path)

    return compound_pool


def detect_score_direction_from_data(df: pl.DataFrame, target_col: str) -> str:
    """
    Detect score direction from data distribution.

    Args:
        df: DataFrame containing target column
        target_col: Name of target column

    Returns:
        'higher' or 'lower'
    """
    if target_col not in df.columns:
        return 'higher'  # Default

    column_lower = target_col.lower()

    # Patterns for lower-is-better
    lower_patterns = ['dock', 'binding_energy', 'energy', 'rmsd', 'error', 'loss', 'distance']
    for pattern in lower_patterns:
        if pattern in column_lower:
            return 'lower'

    # Patterns for higher-is-better
    higher_patterns = ['activity', 'affinity', 'score', 'rank', 'similarity', 'accuracy']
    for pattern in higher_patterns:
        if pattern in column_lower:
            return 'higher'

    # Check data distribution
    try:
        values = df.get_column(target_col).drop_nulls()
        if len(values) > 0 and (values < 0).mean() > 0.7:
            return 'lower'
    except:
        pass

    # Default to higher-is-better
    return 'higher'

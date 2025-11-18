"""Data loading utilities for LearnM8 experiments."""

import polars as pl
from pathlib import Path
from typing import Tuple


def validate_csv_columns(df: pl.DataFrame, required_columns: list, file_description: str) -> None:
    """Validate that DataFrame contains required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        available_columns = list(df.columns)
        raise ValueError(
            f"{file_description} missing required columns: {missing_columns}. "
            f"Available columns: {available_columns}"
        )


def load_benchmark_data(data_path: str, target_col: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load data for benchmark mode from single CSV file.

    Args:
        data_path: Path to CSV file
        target_col: Name of target column

    Returns:
        Tuple of (compound_pool_df, ground_truth_df)
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load the data
    df = pl.read_csv(data_path)

    # Validate required columns
    required_columns = ['ID', 'SMILES', target_col]
    validate_csv_columns(df, required_columns, "Benchmark data file")

    # Check for optional Activity column
    has_activity = 'Activity' in df.columns
    if has_activity:
        print(f"Found Activity column - enrichment calculations will be performed")
        # Validate Activity column is binary
        unique_values = df.get_column('Activity').drop_nulls().unique().to_list()
        if not set(unique_values).issubset({0, 1}):
            print(f"Warning: Activity column should contain only 0 and 1, found: {unique_values}")
    else:
        print(f"No Activity column found - enrichment calculations will be skipped")

    # For benchmark mode, compound pool and ground truth are the same
    compound_pool = df.select(['ID', 'SMILES'])
    ground_truth = df.clone()

    print(f"Loaded {len(df)} compounds for benchmarking")
    print(f"Target column: {target_col}")

    return compound_pool, ground_truth


def load_run_data(compounds_path: str, oracle_path: str) -> pl.DataFrame:
    """
    Load data for run mode with separate compound pool.

    Args:
        compounds_path: Path to compound pool CSV file
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

    # Load compound pool
    compound_pool = pl.read_csv(compounds_path)

    # Validate required columns
    required_columns = ['ID', 'SMILES']
    validate_csv_columns(compound_pool, required_columns, "Compound pool file")

    # Validate oracle file is Python
    if oracle_path.suffix != '.py':
        print(f"Warning: Oracle file should have .py extension, got: {oracle_path.suffix}")

    print(f"Loaded {len(compound_pool)} compounds for active learning")
    print(f"Oracle file: {oracle_path}")

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
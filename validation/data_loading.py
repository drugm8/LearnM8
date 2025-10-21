import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

from learnm8 import validate_compound_pool
from validation.config import get_dataset_info, get_dataset_path, DEFAULT_DATASET

def load_validation_dataset(
    dataset_name: str = DEFAULT_DATASET,
    subsample_size: Optional[int] = None,
    random_state: int = 42,
    clean_invalid_scores: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a standard validation dataset with consistent preprocessing.

    This function provides a unified interface for loading validation datasets,
    ensuring consistent column naming, data cleaning, and error handling across
    all validation notebooks.

    Parameters
    ----------
    dataset_name : str, default='ampc_30k'
        Name of the dataset to load (e.g., 'ampc_30k', '

ampc_500k')
    subsample_size : int, optional
        If provided, randomly subsample to this many compounds
    random_state : int, default=42
        Random seed for reproducible subsampling
    clean_invalid_scores : bool, default=True
        Remove compounds with invalid target scores (NaN, 'no_score', etc.)

    Returns
    -------
    compound_pool : pd.DataFrame
        DataFrame with standardized columns: ID, SMILES, target_column
    metadata : dict
        Dataset metadata including original size, final size, columns, etc.

    Examples
    --------
    >>> compound_pool, metadata = load_validation_dataset('ampc_30k')
    >>> print(f"Loaded {len(compound_pool)} compounds")
    >>> print(f"Target column: {metadata['target_column']}")

    >>> compound_pool, metadata = load_validation_dataset(
    ...     'ampc_30k', subsample_size=10000, random_state=42
    ... )
    """
    logger = logging.getLogger(__name__)

    dataset_info = get_dataset_info(dataset_name)
    dataset_path = get_dataset_path(dataset_name)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            f"Dataset: {dataset_name}\n"
            f"Expected location: {dataset_path}"
        )

    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"  Path: {dataset_path}")
    logger.info(f"  Description: {dataset_info['description']}")

    df = pd.read_csv(dataset_path)
    original_size = len(df)
    logger.info(f"  Original size: {original_size:,} compounds")

    id_col = dataset_info['id_column']
    smiles_col = dataset_info['smiles_column']
    target_col = dataset_info['target_column']

    required_cols = [id_col, smiles_col, target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(columns={
        id_col: 'ID',
        smiles_col: 'SMILES',
        target_col: target_col
    })

    df = df[['ID', 'SMILES', target_col]]

    if clean_invalid_scores:
        logger.info(f"  Cleaning invalid {target_col} values...")
        initial_count = len(df)

        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        df = df.dropna(subset=[target_col])

        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"  Removed {removed_count:,} compounds with invalid scores")

    if subsample_size is not None and subsample_size < len(df):
        logger.info(f"  Subsampling to {subsample_size:,} compounds (seed={random_state})...")
        np.random.seed(random_state)
        subsample_indices = np.random.choice(len(df), size=subsample_size, replace=False)
        df = df.iloc[subsample_indices].reset_index(drop=True)

    final_size = len(df)
    logger.info(f"  Final size: {final_size:,} compounds")

    if target_col != 'dockscore' and target_col != 'ESSENCE-Dock_Score':
        logger.info(f"  Target column '{target_col}' range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")
    else:
        logger.info(f"  {target_col} range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")
        logger.info(f"  Note: {dataset_info['note']}")

    metadata = {
        'dataset_name': dataset_name,
        'dataset_path': str(dataset_path),
        'description': dataset_info['description'],
        'original_size': original_size,
        'final_size': final_size,
        'subsample_size': subsample_size,
        'target_column': target_col,
        'score_direction': dataset_info['score_direction'],
        'cleaned': clean_invalid_scores,
        'removed_invalid': original_size - final_size if not subsample_size else None
    }

    return df, metadata

def validate_compounds_with_features(
    compounds_df: pd.DataFrame,
    featurizer_type: str = 'morgan',
    cache_dir: Path = Path('.cache')
) -> pd.DataFrame:
    """
    Validate compounds by attempting feature extraction using v1.0.0 API.

    Uses validate_compound_pool from v1.0.0 API which automatically validates
    all compounds and caches features for later use.

    Parameters
    ----------
    compounds_df : pd.DataFrame
        DataFrame with ID and SMILES columns
    featurizer_type : str, default='morgan'
        Type of molecular featurizer to use
    cache_dir : Path, default='.cache'
        Directory for feature caching

    Returns
    -------
    pd.DataFrame
        DataFrame with only valid compounds that have extractable features
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info("Validating compounds using v1.0.0 API...")
        logger.info(f"  Featurizer: {featurizer_type}")
        logger.info(f"  Cache directory: {cache_dir}")

        validation_result = validate_compound_pool(
            compound_pool=compounds_df,
            featurizer_type=featurizer_type,
            cache_dir=cache_dir
        )

        logger.info(f"Validation complete:")
        logger.info(f"  Valid compounds: {len(validation_result.valid_compounds):,}")
        logger.info(f"  Invalid compounds: {len(validation_result.invalid_compounds):,}")
        logger.info(f"  Success rate: {validation_result.success_rate:.1%}")

        if len(validation_result.invalid_compounds) > 0:
            logger.warning(f"  {len(validation_result.invalid_compounds)} compounds failed validation")
            logger.warning("  Check validation_errors dict for details")

        return validation_result.valid_compounds

    except Exception as e:
        logger.error(f"Error during compound validation: {e}")
        raise

print("✅ Data loading module initialized (v1.0.0 API)")
print("   - load_validation_dataset(): Load standard datasets with consistent preprocessing")
print("   - validate_compounds_with_features(): Validate SMILES using v1.0.0 validation API")

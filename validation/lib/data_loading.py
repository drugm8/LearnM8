import logging
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

from learnm8 import validate_compound_pool
from validation.lib.dataset_config import get_dataset_info, get_dataset_path, DEFAULT_DATASET


def load_validation_dataset(
    dataset_name: str = DEFAULT_DATASET,
    subsample_size: Optional[int] = None,
    random_state: int = 42,
    clean_invalid_scores: bool = True
) -> Tuple[pl.DataFrame, Dict]:

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

    df = pl.read_csv(dataset_path)
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

    df = df.rename({
        id_col: 'ID',
        smiles_col: 'SMILES',
        target_col: target_col
    })

    df = df.select(['ID', 'SMILES', target_col])

    if clean_invalid_scores:
        logger.info(f"  Cleaning invalid {target_col} values...")
        initial_count = len(df)

        df = df.with_columns(
            pl.col(target_col).cast(pl.Float64, strict=False)
        )

        df = df.filter(pl.col(target_col).is_not_null())

        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"  Removed {removed_count:,} compounds with invalid scores")

    if subsample_size is not None and subsample_size < len(df):
        logger.info(f"  Subsampling to {subsample_size:,} compounds (seed={random_state})...")
        df = df.sample(n=subsample_size, seed=random_state)

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
    compounds_df: pl.DataFrame,
    featurizer_type: str = 'morgan',
    cache_dir: Path = Path('.cache')
) -> pl.DataFrame:

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

"""
Parallel feature extraction with automatic optimization.

This module provides efficient feature extraction with automatic parallelization
based on dataset size and optional progress tracking.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from learnm8.utils.featurizers import (
    smiles_to_morgan_fingerprint,
    smiles_to_maccs_fingerprint,
    smiles_to_ecfp6_fingerprint,
    smiles_to_morgan_feature_fingerprint,
    _compute_mordred_descriptors
)
from .cache import cache_features

logger = logging.getLogger(__name__)


def _get_optimal_n_jobs(n_compounds: int, n_jobs: int = -1) -> int:
    """
    Determine optimal number of parallel jobs based on dataset size.

    Args:
        n_compounds: Number of compounds to process
        n_jobs: User-specified number of jobs (-1 for auto)

    Returns:
        Optimal number of parallel jobs

    Strategy:
        - < 100: Sequential (overhead > benefit)
        - 100-10k: All available cores
        - > 10k: Cap at 32 cores (diminishing returns)
    """
    if n_jobs == 1:
        return 1

    if n_jobs == -1:
        if n_compounds < 100:
            return 1
        elif n_compounds < 10000:
            return os.cpu_count() or 1
        else:
            return min(os.cpu_count() or 1, 32)

    return max(1, n_jobs)


def _get_feature_dimension(featurizer_type: str) -> int:
    """
    Get the feature dimension for a given featurizer type.

    Args:
        featurizer_type: Type of featurizer

    Returns:
        Feature dimension for the featurizer

    Raises:
        ValueError: If featurizer_type is unknown
    """
    if featurizer_type in ['morgan', 'ecfp6', 'morgan_feat']:
        return 2048
    elif featurizer_type == 'maccs':
        return 167
    elif featurizer_type == 'descriptors':
        return 1613
    else:
        raise ValueError(f"Unknown featurizer type: {featurizer_type}")


def _extract_single_feature(smiles: str, featurizer_type: str) -> np.ndarray:
    """
    Extract feature for a single SMILES string.

    Args:
        smiles: SMILES string
        featurizer_type: Type of featurizer to use

    Returns:
        Feature array for the molecule

    Raises:
        ValueError: If featurizer_type is unknown or SMILES is invalid
    """
    if featurizer_type == 'morgan':
        return smiles_to_morgan_fingerprint(smiles)
    elif featurizer_type == 'maccs':
        return smiles_to_maccs_fingerprint(smiles)
    elif featurizer_type == 'ecfp6':
        return smiles_to_ecfp6_fingerprint(smiles)
    elif featurizer_type == 'morgan_feat':
        return smiles_to_morgan_feature_fingerprint(smiles)
    elif featurizer_type == 'descriptors':
        desc_df = _compute_mordred_descriptors([smiles])
        numeric_cols = [col for col, dtype in desc_df.schema.items()
                        if dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                                    pl.Float64, pl.Float32, pl.UInt64, pl.UInt32,
                                    pl.UInt16, pl.UInt8]]
        desc = desc_df.select([
            pl.when(pl.col(c).is_infinite())
              .then(0.0)
              .otherwise(pl.col(c))
              .fill_null(0.0)
              .alias(c)
            for c in numeric_cols
        ])
        return desc.cast(pl.Float32).row(0, named=False)
    else:
        raise ValueError(f"Unknown featurizer type: {featurizer_type}")


def _extract_features_parallel(
    smiles_list: List[str],
    featurizer_type: str,
    n_jobs: int = -1,
    show_progress: bool = False
) -> np.ndarray:
    """
    Internal function for parallel feature extraction.

    Args:
        smiles_list: List of SMILES strings
        featurizer_type: Type of featurizer to use
        n_jobs: Number of parallel jobs (-1 for auto)
        show_progress: Show progress bar (requires tqdm)

    Returns:
        Array of features with shape (n_compounds, n_features)
    """
    # Handle empty input: return array with proper feature dimension
    if len(smiles_list) == 0:
        feature_dim = _get_feature_dimension(featurizer_type)
        return np.empty((0, feature_dim), dtype=np.float32)

    optimal_n_jobs = _get_optimal_n_jobs(len(smiles_list), n_jobs)
    logger.debug(f"Extracting {featurizer_type} features for {len(smiles_list)} compounds with n_jobs={optimal_n_jobs}")

    if featurizer_type == 'descriptors':
        desc_df = _compute_mordred_descriptors(smiles_list)
        numeric_cols = [col for col, dtype in desc_df.schema.items()
                        if dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                                    pl.Float64, pl.Float32, pl.UInt64, pl.UInt32,
                                    pl.UInt16, pl.UInt8]]
        desc = desc_df.select([
            pl.when(pl.col(c).is_infinite())
              .then(0.0)
              .otherwise(pl.col(c))
              .fill_null(0.0)
              .alias(c)
            for c in numeric_cols
        ])
        return desc.cast(pl.Float32).to_numpy()

    smiles_iterator = smiles_list
    if show_progress:
        try:
            from tqdm import tqdm
            smiles_iterator = tqdm(smiles_list, desc="Extracting features")
        except ImportError:
            logger.debug("tqdm not available, running without progress bar")

    if optimal_n_jobs == 1:
        features = [_extract_single_feature(smiles, featurizer_type) for smiles in smiles_iterator]
    else:
        features = Parallel(n_jobs=optimal_n_jobs)(
            delayed(_extract_single_feature)(smiles, featurizer_type)
            for smiles in smiles_iterator
        )

    features_array = np.array(features)
    logger.debug(f"Feature matrix shape: {features_array.shape}, dtype: {features_array.dtype}")
    return features_array


@cache_features(Path('.cache'))
def extract_features(
    smiles_list: List[str],
    featurizer_type: str,
    cache_dir: Optional[Path] = None,
    n_jobs: int = -1,
    show_progress: bool = False
) -> np.ndarray:
    """
    Extract molecular features with caching and parallel processing.

    This is the main public API for feature extraction. It automatically:
    - Caches features to HDF5 (by SMILES hash)
    - Uses parallel processing for speed
    - Handles errors gracefully

    Args:
        smiles_list: List of SMILES strings
        featurizer_type: Type of featurizer ('morgan', 'maccs', 'ecfp6', 'morgan_feat', 'descriptors')
        cache_dir: Directory for cache files (default: .cache)
        n_jobs: Number of parallel jobs (-1 for auto, 1 for sequential)
        show_progress: Show progress bar for long operations (requires tqdm)

    Returns:
        Array of features with shape (n_compounds, n_features)

    Raises:
        ValueError: If featurizer_type is unknown

    Example:
        >>> features = extract_features(['CCO', 'CCC'], 'morgan')
        >>> features.shape
        (2, 2048)
    """
    return _extract_features_parallel(smiles_list, featurizer_type, n_jobs=n_jobs, show_progress=show_progress)

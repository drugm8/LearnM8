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
        numeric_cols = desc_df.select_dtypes(include=[np.number]).columns
        desc = desc_df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
        return desc.iloc[0].astype(np.float32).values
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
    # Uniform empty input handling: return (0, 0) arrays for all featurizer types
    # This ensures consumers handle empty inputs uniformly without assuming fixed feature dimensions
    if len(smiles_list) == 0:
        return np.empty((0, 0), dtype=np.float32)

    optimal_n_jobs = _get_optimal_n_jobs(len(smiles_list), n_jobs)
    logger.debug(f"Extracting {featurizer_type} features for {len(smiles_list)} compounds with n_jobs={optimal_n_jobs}")

    if featurizer_type == 'descriptors':
        desc_df = _compute_mordred_descriptors(smiles_list)
        numeric_cols = desc_df.select_dtypes(include=[np.number]).columns
        desc = desc_df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
        return desc.astype(np.float32).values

    smiles_iterator = smiles_list
    if show_progress:
        try:
            from tqdm import tqdm
            smiles_iterator = tqdm(smiles_list, desc="Extracting features")
        except ImportError:
            logger.debug("tqdm not available, falling back to no progress bar")

    if optimal_n_jobs == 1:
        features = [_extract_single_feature(smiles, featurizer_type) for smiles in smiles_iterator]
    else:
        features = Parallel(n_jobs=optimal_n_jobs)(
            delayed(_extract_single_feature)(smiles, featurizer_type)
            for smiles in smiles_iterator
        )

    logger.info(f"Extracted {len(features)} {featurizer_type} feature vectors")
    return np.array(features)


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

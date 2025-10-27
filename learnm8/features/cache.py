"""
HDF5-based caching for molecular feature extraction.

This module provides a decorator for caching molecular features to HDF5 files
with SMILES hash-based deduplication and blosc compression.
"""

import hashlib
import logging
from functools import wraps
from pathlib import Path
from typing import Callable, List

import h5py
import hdf5plugin
import numpy as np

logger = logging.getLogger(__name__)


def get_smiles_hash(smiles: str) -> str:
    """
    Generate MD5 hash of SMILES string for use as cache key.

    Args:
        smiles: SMILES string to hash

    Returns:
        32-character hexadecimal hash string

    Note:
        Cache keys are based on SMILES string only, not featurizer parameters.
        Current featurizers (morgan, maccs, ecfp6) use fixed parameters, so this is safe.
        If variable parameters are added (radius, nBits), consider including a config hash
        in the HDF5 file path (e.g., "{featurizer_type}_{config_hash}_features.h5").
    """
    return hashlib.md5(smiles.encode('utf-8')).hexdigest()


def cache_features(default_cache_dir: Path) -> Callable:
    """
    Decorator factory for caching molecular features to HDF5.

    This decorator transparently caches features by SMILES hash, allowing
    efficient deduplication and partial loading for large datasets.

    Args:
        default_cache_dir: Default directory where HDF5 cache files will be stored

    Returns:
        Decorator function that wraps feature extraction functions

    Example:
        >>> @cache_features(Path('.cache'))
        >>> def my_featurizer(smiles_list, featurizer_type):
        >>>     return extract_features_parallel(smiles_list, featurizer_type)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(smiles_list: List[str], featurizer_type: str, *args, **kwargs) -> np.ndarray:
            # Delegate empty input handling to underlying extraction function for consistency
            if len(smiles_list) == 0:
                return func(smiles_list, featurizer_type, *args, **kwargs)

            cache_dir = kwargs.pop('cache_dir', default_cache_dir)
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{featurizer_type}_features.h5"

            cached_features = {}
            uncached_smiles = []
            uncached_indices = []

            try:
                with h5py.File(cache_file, 'a') as h5f:
                    features_group = h5f.require_group('features')

                    for idx, smiles in enumerate(smiles_list):
                        smiles_hash = get_smiles_hash(smiles)

                        try:
                            if smiles_hash in features_group:
                                cached_features[idx] = features_group[smiles_hash][:]
                                logger.debug(f"Cache hit for SMILES hash {smiles_hash[:8]}...")
                            else:
                                uncached_smiles.append(smiles)
                                uncached_indices.append(idx)
                                logger.debug(f"Cache miss for SMILES hash {smiles_hash[:8]}...")
                        except (OSError, IOError, KeyError, RuntimeError) as e:
                            logger.warning(f"Cache read failed for SMILES hash {smiles_hash[:8]}: {e}")
                            uncached_smiles.append(smiles)
                            uncached_indices.append(idx)

                    logger.info(f"Cache: {len(cached_features)} hits, {len(uncached_smiles)} misses")

                if uncached_smiles:
                    logger.debug(f"Computing features for {len(uncached_smiles)} uncached SMILES")
                    new_features = func(uncached_smiles, featurizer_type, *args, **kwargs)

                    try:
                        with h5py.File(cache_file, 'a') as h5f:
                            features_group = h5f.require_group('features')

                            for idx, smiles in enumerate(uncached_smiles):
                                smiles_hash = get_smiles_hash(smiles)
                                try:
                                    ds = features_group.create_dataset(
                                        smiles_hash,
                                        data=new_features[idx],
                                        chunks=True,
                                        compression=32001,  # blosc filter ID
                                        compression_opts=(0, 0, 0, 0, 5, 1, 1)  # blosc:lz4 configuration
                                    )
                                    logger.debug(f"Cached features for SMILES hash {smiles_hash[:8]}...")
                                except (TypeError, ValueError) as e:
                                    logger.warning(f"Existing dataset for {smiles_hash[:8]} has incompatible shape/dtype; skipping cache write: {e}")
                                except (OSError, IOError, RuntimeError) as e:
                                    logger.warning(f"Failed to cache features for SMILES hash {smiles_hash[:8]}: {e}")

                            logger.info(f"Cached {len(uncached_smiles)} new feature vectors")
                    except (OSError, IOError, RuntimeError) as e:
                        logger.warning(f"Cache write failed: {e}; continuing without caching")

                    for i, original_idx in enumerate(uncached_indices):
                        cached_features[original_idx] = new_features[i]

                result = np.array([cached_features[i] for i in range(len(smiles_list))])
                return result

            except (OSError, IOError) as e:
                logger.warning(f"Cache file open failed: {e}. Falling back to direct computation.")

                if cache_file.exists():
                    try:
                        logger.warning(f"Attempting to delete corrupted cache file: {cache_file}")
                        cache_file.unlink()
                    except Exception as unlink_error:
                        logger.warning(f"Failed to delete corrupted cache file: {unlink_error}")

                return func(smiles_list, featurizer_type, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Unexpected cache operation error: {e}. Falling back to direct computation.")
                return func(smiles_list, featurizer_type, *args, **kwargs)

        return wrapper
    return decorator

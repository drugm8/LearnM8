"""
HDF5-based caching for molecular feature extraction.

This module provides a decorator for caching molecular features to HDF5 files
with SMILES hash-based deduplication and blosc compression.

Updated for configuration-aware caching to prevent incorrect cache hits
when featurizer parameters change.
"""

import hashlib
import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _generate_cache_key(smiles: str, featurizer) -> str:
    """Generate cache key including SMILES and featurizer configuration.

    Args:
        smiles: SMILES string
        featurizer: Featurizer instance (to get configuration hash)

    Returns:
        MD5 hash string combining SMILES and featurizer config

    Note:
        OLD (WRONG): MD5(SMILES) - ignores featurizer params
        NEW (CORRECT): MD5(SMILES + featurizer_name + config_hash)

    Example:
        >>> featurizer = MorganFeaturizer(radius=3, fp_size=4096)
        >>> key = _generate_cache_key("CCO", featurizer)
    """
    config_hash = featurizer.get_config_hash()
    featurizer_name = featurizer.get_name()

    cache_string = f"{smiles}_{featurizer_name}_{config_hash}"
    return hashlib.md5(cache_string.encode()).hexdigest()


def get_smiles_hash(smiles: str) -> str:
    """
    Generate MD5 hash of SMILES string for use as cache key.

    Args:
        smiles: SMILES string to hash

    Returns:
        32-character hexadecimal hash string

    Note:
        DEPRECATED: Use _generate_cache_key() with featurizer instance instead.
        This function is kept for backward compatibility with old cache files.
    """
    return hashlib.md5(smiles.encode('utf-8')).hexdigest()


def cache_features(default_cache_dir: Path) -> Callable:
    """
    Decorator factory for caching molecular features to HDF5.

    This decorator transparently caches features with configuration-aware keys,
    allowing efficient deduplication and partial loading for large datasets.

    Args:
        default_cache_dir: Default directory where HDF5 cache files will be stored

    Returns:
        Decorator function that wraps feature extraction functions

    Example:
        >>> @cache_features(Path('.cache'))
        >>> def my_featurizer(smiles_list, featurizer):
        >>>     return extract_features_parallel(smiles_list, featurizer)

    Note:
        Cache keys now include featurizer configuration to prevent
        incorrect cache hits when parameters change.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(smiles_list: list[str], featurizer, *args, **kwargs) -> np.ndarray:
            if len(smiles_list) == 0:
                return func(smiles_list, featurizer, *args, **kwargs)

            cache_dir = kwargs.pop('cache_dir', default_cache_dir)
            if cache_dir is None:
                cache_dir = default_cache_dir
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            featurizer_name = featurizer.get_name()
            cache_file = cache_dir / f"features_{featurizer_name}.h5"

            cached_features = {}
            uncached_smiles = []
            uncached_indices = []

            try:
                with h5py.File(cache_file, 'a', libver='latest') as h5f:
                    features_group = h5f.require_group('features')

                    for idx, smiles in enumerate(smiles_list):
                        cache_key = _generate_cache_key(smiles, featurizer)

                        try:
                            if cache_key in features_group:
                                cached_features[idx] = features_group[cache_key][:]
                            else:
                                uncached_smiles.append(smiles)
                                uncached_indices.append(idx)
                        except (OSError, KeyError, RuntimeError) as e:
                            logger.warning(f"Cache read failed for key {cache_key[:8]}: {e}")
                            uncached_smiles.append(smiles)
                            uncached_indices.append(idx)

                    hit_rate = (len(cached_features)/(len(cached_features)+len(uncached_smiles))*100) if (len(cached_features)+len(uncached_smiles)) > 0 else 0
                    logger.debug(f"Cache statistics: {len(cached_features)} hits, {len(uncached_smiles)} misses ({hit_rate:.1f}% hit rate)")

                if uncached_smiles:
                    logger.debug(f"Computing features for {len(uncached_smiles)} uncached SMILES")
                    new_features = func(uncached_smiles, featurizer, *args, **kwargs)

                    try:
                        with h5py.File(cache_file, 'a', libver='latest') as h5f:
                            features_group = h5f.require_group('features')

                            for idx, smiles in enumerate(uncached_smiles):
                                cache_key = _generate_cache_key(smiles, featurizer)
                                try:
                                    ds = features_group.create_dataset(
                                        cache_key,
                                        data=new_features[idx],
                                        chunks=True,
                                        compression=32001,
                                        compression_opts=(0, 0, 0, 0, 5, 1, 1)
                                    )
                                    logger.debug(f"Cached features for key {cache_key[:8]}...")
                                except (TypeError, ValueError) as e:
                                    logger.warning(f"Existing dataset for {cache_key[:8]} has incompatible shape/dtype; skipping cache write: {e}")
                                except (OSError, RuntimeError) as e:
                                    logger.warning(f"Failed to cache features for key {cache_key[:8]}: {e}")

                            logger.debug(f"Cached {len(uncached_smiles)} new feature vectors to HDF5")
                    except (OSError, RuntimeError) as e:
                        logger.warning(f"Cache write failed: {e}; continuing without caching")

                    for i, original_idx in enumerate(uncached_indices):
                        cached_features[original_idx] = new_features[i]

                result = np.array([cached_features[i] for i in range(len(smiles_list))])
                return result

            except OSError as e:
                logger.warning(f"Cache file open failed: {e}. Falling back to direct computation.")

                if cache_file.exists():
                    try:
                        logger.warning(f"Attempting to delete corrupted cache file: {cache_file}")
                        cache_file.unlink()
                    except OSError as unlink_error:
                        logger.warning(f"Failed to delete corrupted cache file: {unlink_error}")

                return func(smiles_list, featurizer, *args, **kwargs)
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                logger.warning(f"Unexpected cache operation error: {e}. Falling back to direct computation.")
                return func(smiles_list, featurizer, *args, **kwargs)

        return wrapper
    return decorator

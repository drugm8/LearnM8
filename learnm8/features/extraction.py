"""
Parallel feature extraction with automatic optimization.

This module provides efficient feature extraction with automatic parallelization
based on dataset size and optional progress tracking.

Updated to support both string-based and Featurizer instance-based extraction.
"""

import copy
import logging
from pathlib import Path

import numpy as np

from learnm8.core.interfaces import Featurizer
from learnm8.exceptions import FeatureExtractionError

from .base import SkfpFeaturizer
from .cache import cache_features

logger = logging.getLogger(__name__)


@cache_features(Path('.cache'))
def _extract_features_with_featurizer(
    smiles_list: list[str],
    featurizer: Featurizer,
    n_jobs: int = -1,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Internal function for feature extraction using Featurizer instance.

    Args:
        smiles_list: List of SMILES strings
        featurizer: Featurizer instance to use
        n_jobs: Number of parallel jobs (-1 for auto) - may be ignored by featurizer
        show_progress: Show progress bar (requires tqdm) - may be ignored by featurizer

    Returns:
        Array of features with shape (n_compounds, n_features)
    """
    if len(smiles_list) == 0:
        return np.empty((0, featurizer.get_dimension()), dtype=np.float32)

    if len(smiles_list) > 1000:
        logger.info(
            f'Extracting {featurizer.get_name()} features for {len(smiles_list)} compounds...'
        )
    else:
        logger.debug(
            f'Extracting {featurizer.get_name()} features for {len(smiles_list)} compounds'
        )

    # REQ-15/15a (feature 025, Item 6): set scikit-fingerprints' native
    # batch_size so the transform splits into >= n_jobs tasks (Fix A — fills
    # every loky worker) while still capping each worker's working set. It is
    # set on a per-call deepcopy of the featurizer — the shared featurizer
    # instance is used concurrently by other extract_features callers (e.g.
    # acquisition/simulated_annealing.py), so it must never be mutated. The
    # deepcopy (~0.02 ms) is unconditional for skfp-backed featurizers. A custom
    # (non-SkfpFeaturizer) Featurizer has no skfp batch_size and is left as-is.
    transform_featurizer: Featurizer = featurizer
    if isinstance(featurizer, SkfpFeaturizer):
        from joblib import effective_n_jobs

        from learnm8.core.batching import featurization_chunk_size

        featurizer_copy = copy.deepcopy(featurizer)
        try:
            # Size the chunk against the worker count skfp will actually use,
            # so input_len // batch_size >= n_jobs and no worker idles.
            n_workers = effective_n_jobs(featurizer_copy.fingerprint.n_jobs)
            featurizer_copy.fingerprint.set_params(
                batch_size=featurization_chunk_size(len(smiles_list), n_workers)
            )
            transform_featurizer = featurizer_copy
        except (ValueError, TypeError):
            # Fingerprint does not accept a batch_size param — featurize as-is.
            transform_featurizer = featurizer

    try:
        features_array = transform_featurizer.transform(smiles_list)
    except FeatureExtractionError:
        raise
    except (ValueError, RuntimeError, TypeError) as e:
        raise FeatureExtractionError(
            f'Feature extraction failed for {featurizer.get_name()} on '
            f'{len(smiles_list)} compounds: {e}. '
            f"Check SMILES validity with 'learnm8 validate your_file.csv'."
        ) from None

    logger.debug(
        f'Feature matrix shape: {features_array.shape}, dtype: {features_array.dtype}'
    )
    return features_array


def extract_features(
    smiles_list: list[str],
    featurizer: str | Featurizer,
    cache_dir: Path | None = None,
    n_jobs: int = -1,
    show_progress: bool = False,
    preferred_dtype: str = 'float32',
) -> np.ndarray:
    """
    Extract molecular features with caching and parallel processing.

    This is the main public API for feature extraction. It automatically:
    - Caches features to HDF5 (by SMILES hash + configuration)
    - Uses parallel processing for speed
    - Handles errors gracefully

    Args:
        smiles_list: List of SMILES strings
        featurizer: Either string name ('morgan', 'maccs', etc.) or Featurizer instance
        cache_dir: Directory for cache files (default: .cache)
        n_jobs: Number of parallel jobs (-1 for auto, 1 for sequential)
        show_progress: Show progress bar for long operations (requires tqdm)

    Returns:
        Array of features with shape (n_compounds, n_features)

    Raises:
        ValueError: If featurizer string is unknown
        TypeError: If featurizer is neither str nor Featurizer

    Example:
        >>> # String API (simple)
        >>> features = extract_features(['CCO', 'CCC'], 'morgan')
        >>> features.shape
        (2, 2048)

        >>> # Class API (custom parameters)
        >>> from learnm8.features import MorganFeaturizer
        >>> my_featurizer = MorganFeaturizer(radius=3, fp_size=4096)
        >>> features = extract_features(['CCO', 'CCC'], my_featurizer)
        >>> features.shape
        (2, 4096)
    """
    if len(smiles_list) == 0:
        if isinstance(featurizer, str):
            from learnm8.features import FEATURIZER_REGISTRY, create_featurizer

            if featurizer not in FEATURIZER_REGISTRY:
                from learnm8.features import list_available_featurizers

                available = list_available_featurizers()
                all_featurizers = ', '.join(sorted(available['all']))
                raise ValueError(
                    f"Unknown featurizer: '{featurizer}'. "
                    f'Available featurizers: {all_featurizers}. '
                    f"Run 'learnm8 list featurizers' to see all options."
                )
            featurizer_obj = create_featurizer(featurizer, n_jobs=1)
        else:
            featurizer_obj = featurizer
        dim = featurizer_obj.get_dimension()
        return np.empty((0, dim), dtype=np.float32)

    if isinstance(featurizer, str):
        from learnm8.features import FEATURIZER_REGISTRY, create_featurizer

        if featurizer not in FEATURIZER_REGISTRY:
            from learnm8.features import list_available_featurizers

            available = list_available_featurizers()
            all_featurizers = ', '.join(sorted(available['all']))
            raise ValueError(
                f"Unknown featurizer: '{featurizer}'. "
                f'Available featurizers: {all_featurizers}. '
                f"Run 'learnm8 list featurizers' to see all options."
            )

        featurizer_obj = create_featurizer(featurizer, n_jobs=n_jobs)

    elif isinstance(featurizer, Featurizer):
        featurizer_obj = featurizer

    else:
        raise TypeError(
            f"featurizer must be a string name (e.g., 'morgan') or a Featurizer instance, "
            f'got {type(featurizer).__name__}. '
            f'Use a string for built-in featurizers or pass a custom Featurizer object.'
        )

    return _extract_features_with_featurizer(
        smiles_list,
        featurizer_obj,
        n_jobs=n_jobs,
        show_progress=show_progress,
        cache_dir=cache_dir,
        preferred_dtype=preferred_dtype,
    )

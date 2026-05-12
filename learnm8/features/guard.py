"""Large feature guard for preventing accidental descriptor-class runs at scale."""

import logging
import warnings

from learnm8.exceptions import ConfigurationError, LearnM8Warning

logger = logging.getLogger(__name__)

LARGE_FEATURE_POOL_THRESHOLD = 10_000_000
LARGE_FEATURE_PER_MOL_BYTES = 1024


def check_large_feature_guard(
    featurizer,
    pool_size: int,
    large_features_ack: bool,
) -> None:
    feature_type = getattr(featurizer, 'feature_type', 'binary')

    if feature_type != 'continuous':
        return

    n_features = _probe_dimension(featurizer)
    if n_features is None:
        return

    feat_instance = getattr(featurizer, 'fingerprint', featurizer)
    dtype = getattr(feat_instance, 'dtype', None)
    itemsize = dtype.itemsize if dtype is not None and hasattr(dtype, 'itemsize') else 4

    per_mol_bytes = n_features * itemsize

    if per_mol_bytes <= LARGE_FEATURE_PER_MOL_BYTES:
        return

    if pool_size <= LARGE_FEATURE_POOL_THRESHOLD:
        return

    raw_gb = pool_size * per_mol_bytes / (1024**3)
    name = getattr(featurizer, '__class__', type(featurizer)).__name__

    if large_features_ack:
        warnings.warn(
            f'Large-feature run: {name} on N={pool_size:,}, ~{raw_gb:.1f} GB cache',
            LearnM8Warning,
            stacklevel=2,
        )
        return

    raise ConfigurationError(
        f"Descriptor-class featurizer '{name}' on pool N={pool_size:,} "
        f'would consume ~{raw_gb:.1f} GB cache. '
        f'Pass large_features_ack=True (API) or --allow-large-features (CLI).'
    )


def _probe_dimension(featurizer) -> int | None:
    try:
        dim = featurizer.get_dimension()
        if isinstance(dim, int) and dim > 0:
            return dim
    except Exception:
        pass

    try:
        result = featurizer.transform(['CC'])
        if hasattr(result, 'shape') and len(result.shape) >= 2:
            return result.shape[1]
    except Exception:
        pass

    return None

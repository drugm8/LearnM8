"""Memory-aware prediction batching for active learning.

This module provides runtime memory probing and learner-specific memory
profiles to compute optimal prediction batch sizes. It replaces the
hardcoded batch size heuristic with adaptive memory-aware batching.

Public API:
    get_available_memory(device) — query available bytes for CPU or GPU
    estimate_batch_size(learner, ...) — compute optimal batch from memory profile
    featurization_chunk_size(input_len) — fixed-cap chunk size for featurization
    predict_with_batching(pool, learner, ...) — memory-safe batched prediction
"""

from __future__ import annotations

import gc
import logging
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from learnm8.exceptions import LearnerError
from learnm8.features.extraction import extract_features

if TYPE_CHECKING:
    from learnm8.core.interfaces import Learner

logger = logging.getLogger(__name__)

MIN_BATCH_GPU = 256
MIN_BATCH_CPU = 1000
GPU_ALIGNMENT = 32
DEFAULT_SAFETY_FACTOR = 0.7
FALLBACK_MEMORY_BYTES = 2 * 1024**3  # 2 GiB

# Featurization chunk-size bounds (feature 025, Item 6). CAP bounds each loky
# worker's working set; FLOOR prevents pathologically small chunks. Distinct
# from the prediction-time batch size — see featurization_chunk_size().
FEATURIZATION_CHUNK_FLOOR = 1024
FEATURIZATION_CHUNK_CAP = 50_000

_allocator_configured = False

# glibc malloc-arena tuning (R1 — cold-featurization worker RSS). Detected
# once at import; configure_host_allocator() is a no-op off a glibc/Linux host.
_IS_GLIBC_LINUX: bool = sys.platform == 'linux' and platform.libc_ver()[0] == 'glibc'
_host_allocator_configured: bool = False


def _configure_cuda_allocator() -> None:
    """Best-effort lazy config of expandable_segments for CUDA allocator."""
    global _allocator_configured
    if _allocator_configured:
        return
    _allocator_configured = True
    conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    if 'expandable_segments' not in conf:
        new_val = (
            f'{conf},expandable_segments:True' if conf else 'expandable_segments:True'
        )
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = new_val
        logger.debug('Set PYTORCH_CUDA_ALLOC_CONF=%s', new_val)


def configure_host_allocator() -> None:
    """Cap glibc malloc arenas for subsequently-spawned featurization workers.

    glibc creates up to ``8 * n_cpu`` per-thread malloc arenas and retains
    free()'d memory per-arena, so worker RSS plateaus at its high-water mark.
    ``MALLOC_ARENA_MAX`` bounds that fragmentation. glibc reads the variable at
    process init, so this cannot shrink the *current* process — but loky
    featurization workers are spawned later, inherit the environment, and hold
    the bulk of the cross-batch RSS. Idempotent, no-op off glibc, and never
    overrides a value the operator already set.
    """
    global _host_allocator_configured
    if _host_allocator_configured:
        return
    _host_allocator_configured = True
    if not _IS_GLIBC_LINUX:
        return
    if 'MALLOC_ARENA_MAX' not in os.environ:
        os.environ['MALLOC_ARENA_MAX'] = '2'
        logger.debug('Set MALLOC_ARENA_MAX=2 for featurization workers')


def _get_learner_device(learner: Learner) -> str:
    """Detect device from learner via attribute fallback chain.

    Checks .device, ._device, .accelerator in order.
    Maps Lightning semantics ('gpu' -> 'cuda').
    """
    for attr in ('device', '_device', 'accelerator'):
        val = getattr(learner, attr, None)
        if val is not None:
            val_str = str(val)
            if val_str in ('gpu', 'cuda') or val_str.startswith('cuda:'):
                return val_str if val_str != 'gpu' else 'cuda'
            if val_str == 'cpu':
                return 'cpu'
    return 'cpu'


def _read_int_file(path: str) -> int | None:
    """Read a single integer from a sysfs/cgroup file; None on 'max'/error."""
    try:
        with open(path) as f:
            txt = f.read().strip()
        if txt in ('max', ''):
            return None
        return int(txt)
    except (OSError, ValueError):
        return None


def _cgroup_memory_headroom() -> int | None:
    """Bytes still allocatable under the effective cgroup memory limit.

    Walks the cgroup hierarchy (v2 ``memory.max`` / v1 ``memory.limit_in_bytes``)
    for the tightest finite cap and subtracts current usage. Returns None when no
    finite limit applies (unconfined) or the cgroup files are unreadable. Never
    raises — confinement-awareness must never break sizing.
    """
    try:
        rel_paths: list[tuple[str, str]] = []
        with open('/proc/self/cgroup') as f:
            for line in f:
                parts = line.strip().split(':', 2)
                if len(parts) == 3:
                    rel_paths.append((parts[1], parts[2]))

        # cgroup v2: unified hierarchy, '0::<path>'.
        if os.path.exists('/sys/fs/cgroup/cgroup.controllers'):
            rel = (rel_paths[0][1] if rel_paths else '/').lstrip('/')
            cur = Path('/sys/fs/cgroup') / rel
            root = Path('/sys/fs/cgroup')
            limit = None
            current = None
            while True:
                lim = _read_int_file(str(cur / 'memory.max'))
                if lim is not None:
                    limit = lim if limit is None else min(limit, lim)
                if current is None:
                    current = _read_int_file(str(cur / 'memory.current'))
                if cur == root:
                    break
                cur = cur.parent
            if limit is None:
                return None
            return max(0, limit - (current or 0))

        # cgroup v1: per-controller hierarchy under /sys/fs/cgroup/memory.
        mem_rel = None
        for controllers, path in rel_paths:
            if 'memory' in controllers.split(','):
                mem_rel = path
                break
        if mem_rel is not None and os.path.isdir('/sys/fs/cgroup/memory'):
            node = Path('/sys/fs/cgroup/memory') / mem_rel.lstrip('/')
            limit = _read_int_file(str(node / 'memory.limit_in_bytes'))
            if limit is None or limit >= (1 << 62):  # v1 'unlimited' sentinel
                return None
            current = _read_int_file(str(node / 'memory.usage_in_bytes')) or 0
            return max(0, limit - current)
    except Exception:
        return None
    return None


def _clamp_to_confinement(available: int) -> int:
    """Clamp node-reported available bytes to the cgroup memory headroom.

    ``psutil.virtual_memory().available`` reports whole-node free RAM, which on a
    shared HPC node vastly exceeds the cgroup / HTCondor slot the job actually
    runs under. Sizing a prediction chunk against the node total over-allocates
    and gets the job OOM-killed at the cgroup wall. Clamp to the real headroom.
    """
    headroom = _cgroup_memory_headroom()
    if headroom is None or headroom <= 0:
        return available
    if headroom < available:
        logger.debug(
            'Clamping available memory %d -> %d bytes (cgroup headroom)',
            available,
            headroom,
        )
        return headroom
    return available


def get_available_memory(device: str = 'cpu') -> int:
    """Query available memory for the target device.

    Args:
        device: Device string ('cpu', 'cuda', 'cuda:N').

    Returns:
        Available memory in bytes.
    """
    if device.startswith('cuda') or device == 'gpu':
        try:
            import torch.cuda

            if torch.cuda.is_available():
                _configure_cuda_allocator()
                dev_idx = 0
                if ':' in device:
                    dev_idx = int(device.split(':')[1])
                torch.cuda.empty_cache()
                free, _total = torch.cuda.mem_get_info(dev_idx)
                logger.debug('GPU %d available memory: %d bytes', dev_idx, free)
                return free
        except Exception:
            logger.debug('GPU memory query failed, falling back to CPU')

    try:
        import psutil

        available = psutil.virtual_memory().available
        available = _clamp_to_confinement(available)
        logger.debug('CPU available memory: %d bytes', available)
        return available
    except Exception:
        logger.debug(
            'psutil unavailable, using fallback %d bytes', FALLBACK_MEMORY_BYTES
        )
        return FALLBACK_MEMORY_BYTES


def estimate_batch_size(
    learner: Learner,
    n_samples: int,
    n_features: int,
    device: str = 'cpu',
    memory_safety_factor: float = DEFAULT_SAFETY_FACTOR,
    n_jobs: int = 1,
) -> int:
    """Compute optimal batch size from learner memory profile and available memory.

    Args:
        learner: Trained learner instance (provides memory_profile()).
        n_samples: Total samples to predict on.
        n_features: Feature dimension.
        device: Target device for memory query.
        memory_safety_factor: Fraction of available memory to use (0.0, 1.0].
        n_jobs: Number of parallel workers used by feature extraction.

    Returns:
        Optimal batch size (>= min_batch, <= n_samples, GPU-aligned if applicable).

    Raises:
        LearnerError: If memory profile is invalid or memory insufficient.
    """
    profile = learner.memory_profile(n_features)
    bytes_per_sample = profile['bytes_per_sample']
    working_multiplier = profile['working_multiplier']
    fixed_overhead = profile.get('fixed_overhead', 0)

    if bytes_per_sample <= 0 or working_multiplier <= 0:
        raise LearnerError(
            f'Invalid memory profile from {learner.get_name()}: '
            f'bytes_per_sample={bytes_per_sample}, working_multiplier={working_multiplier}. '
            f'Both must be positive. Check memory_profile() implementation.'
        )

    available = get_available_memory(device)
    # n_jobs controls parallel featurization inside one chunk. It should not
    # divide the learner prediction budget: LearnM8 materializes one prediction
    # chunk at a time, not one full chunk per worker.
    usable = available * memory_safety_factor - fixed_overhead
    cost_per_sample = bytes_per_sample * working_multiplier

    if usable < cost_per_sample:
        raise LearnerError(
            f'Insufficient memory for even 1 sample. '
            f'Available: {available} bytes, cost per sample: {cost_per_sample:.0f} bytes, '
            f'safety_factor: {memory_safety_factor}. '
            f'Free up memory or reduce model complexity.'
        )

    batch_size = int(usable / cost_per_sample)

    is_gpu = device.startswith('cuda') or device == 'gpu'
    if is_gpu and batch_size >= GPU_ALIGNMENT:
        batch_size = (batch_size // GPU_ALIGNMENT) * GPU_ALIGNMENT

    min_batch = MIN_BATCH_GPU if is_gpu else MIN_BATCH_CPU
    if n_samples >= min_batch:
        batch_size = max(min_batch, batch_size)

    batch_size = min(batch_size, n_samples)
    return batch_size


def featurization_chunk_size(input_len: int, n_jobs: int = 1) -> int:
    """Return the per-call scikit-fingerprints ``batch_size`` for featurization.

    Heuristic: ``clamp(ceil(input_len / n_jobs), FLOOR, CAP)``.

    scikit-fingerprints splits one ``transform`` call into ``input_len //
    batch_size`` tasks and dispatches them across its loky pool. Sizing the
    chunk at ``ceil(input_len / n_jobs)`` therefore yields ``>= n_jobs`` tasks,
    so every worker gets work — a flat ``CAP`` (the pre-Fix-A behaviour) made
    only ``input_len // CAP`` tasks, which starves the pool whenever
    ``input_len < CAP * n_jobs`` (e.g. a 320k prediction chunk → ~6 tasks).

    ``CAP`` (:data:`FEATURIZATION_CHUNK_CAP`) still bounds each worker's working
    set for very large inputs: when ``input_len`` exceeds ``CAP * n_jobs`` the
    chunk is capped at ``CAP`` and the pool simply receives more than ``n_jobs``
    tasks. ``FLOOR`` (:data:`FEATURIZATION_CHUNK_FLOOR`) prevents pathologically
    small chunks whose per-task overhead would dominate.

    This is deliberately NOT :func:`estimate_batch_size`: that helper sizes a
    learner's prediction batch from a memory profile and a host-memory probe.
    The return value is passed to scikit-fingerprints' own ``batch_size``
    keyword argument; it is a distinct concept from the prediction-time
    ``batch_size`` produced by ``estimate_batch_size``.

    Args:
        input_len: number of SMILES to featurize in this call.
        n_jobs: resolved (positive) worker count skfp will parallelise over.
            Pass a value already run through ``joblib.effective_n_jobs`` so
            ``-1`` / ``None`` are concrete; values < 1 are treated as 1.

    Returns:
        The per-call chunk size, clamped to ``[FLOOR, CAP]``.
    """
    per_worker = math.ceil(input_len / max(1, n_jobs))
    return max(FEATURIZATION_CHUNK_FLOOR, min(per_worker, FEATURIZATION_CHUNK_CAP))


def _chunk_dataframe(df: pl.DataFrame, chunk_size: int):
    """Yield DataFrame chunks for batch processing."""
    for i in range(0, len(df), chunk_size):
        yield df[i : i + chunk_size]


def _predict_chunk(
    chunk_df: pl.DataFrame,
    learner: Learner,
    featurizer: str | None,
    cache_dir: Path,
    show_progress: bool = False,
    n_jobs: int = -1,
    compute_uncertainty: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Generate predictions for a chunk of compounds.

    Returns a 3-tuple ``(predictions, uncertainties, feature_extraction_time)``.
    The third element is the wall-clock seconds spent inside ``extract_features``
    so callers can report it separately from learner-side prediction time.

    ``compute_uncertainty`` (feature 023) is forwarded to ``learner.predict``
    as a keyword-only argument so skip-eligible learners can elide
    uncertainty-specific compute paths.
    """
    chunk_smiles = chunk_df['SMILES'].to_list()
    feature_time = 0.0

    preferred_dtype = learner.preferred_feature_dtype()
    if learner.requires_smiles():
        if featurizer is not None:
            t0 = time.perf_counter()
            chunk_features = extract_features(
                chunk_smiles,
                featurizer,
                cache_dir=cache_dir,
                show_progress=show_progress,
                n_jobs=n_jobs,
                preferred_dtype=preferred_dtype,
            )
            feature_time = time.perf_counter() - t0
        else:
            chunk_features = None
        preds, uncerts = learner.predict(
            features=chunk_features,
            smiles=chunk_smiles,
            compute_uncertainty=compute_uncertainty,
        )
        return preds, uncerts, feature_time
    else:
        if featurizer is None:
            raise ValueError(
                f'featurizer is required for {learner.get_name()} because it is a '
                f"feature-based learner. Specify a featurizer (e.g., featurizer='morgan'). "
                f"SMILES-native learners like 'chemprop' and 'fastprop' can run without "
                f'a featurizer (featurizer=None).'
            )
        t0 = time.perf_counter()
        chunk_features = extract_features(
            chunk_smiles,
            featurizer,
            cache_dir=cache_dir,
            show_progress=show_progress,
            n_jobs=n_jobs,
            preferred_dtype=preferred_dtype,
        )
        feature_time = time.perf_counter() - t0
        preds, uncerts = learner.predict(
            chunk_features, compute_uncertainty=compute_uncertainty
        )
        return preds, uncerts, feature_time


def _is_oom_error(error: BaseException) -> bool:
    return isinstance(error, MemoryError) or (
        isinstance(error, RuntimeError) and 'out of memory' in str(error).lower()
    )


def _release_memory_after_oom() -> None:
    gc.collect()
    try:
        import torch.cuda

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _predict_chunk_with_oom_retry(
    chunk_df: pl.DataFrame,
    learner: Learner,
    featurizer: str | None,
    cache_dir: Path,
    n_jobs: int,
    min_batch: int,
    n_samples: int,
    n_features: int,
    device: str,
    compute_uncertainty: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    try:
        return _predict_chunk(
            chunk_df,
            learner,
            featurizer,
            cache_dir,
            n_jobs=n_jobs,
            compute_uncertainty=compute_uncertainty,
        )
    except (RuntimeError, MemoryError) as e:
        if not _is_oom_error(e):
            raise

    _release_memory_after_oom()

    retry_batch_size = len(chunk_df) // 2
    effective_min = min(min_batch, n_samples)
    if retry_batch_size < effective_min:
        raise LearnerError(
            f'OOM at minimum batch size ({effective_min}). '
            f'Cannot predict with available memory. '
            f'Samples: {n_samples}, features: {n_features}, '
            f'available memory: {get_available_memory(device)} bytes. '
            f'Free GPU/CPU memory or reduce model complexity.'
        )

    logger.warning(
        'OOM during prediction, retrying with batch_size=%d (was %d)',
        retry_batch_size,
        len(chunk_df),
    )

    sub_preds = []
    sub_uncerts = []
    feature_time = 0.0
    for sub_chunk in _chunk_dataframe(chunk_df, retry_batch_size):
        sp, su, sft = _predict_chunk_with_oom_retry(
            sub_chunk,
            learner,
            featurizer,
            cache_dir,
            n_jobs,
            min_batch,
            n_samples,
            n_features,
            device,
            compute_uncertainty=compute_uncertainty,
        )
        feature_time += sft
        sub_preds.append(sp)
        if su is not None:
            sub_uncerts.append(su)

    chunk_preds = np.concatenate(sub_preds)
    chunk_uncerts = np.concatenate(sub_uncerts) if sub_uncerts else None
    return chunk_preds, chunk_uncerts, feature_time


def _get_n_features(featurizer: str | None) -> int:
    """Get feature dimension from featurizer name."""
    if featurizer is None:
        return 2048
    try:
        from learnm8.features import FEATURIZER_REGISTRY, create_featurizer

        if featurizer in FEATURIZER_REGISTRY:
            temp = create_featurizer(featurizer, n_jobs=1)
            return temp.get_dimension()
    except Exception:
        pass
    return 2048


def predict_with_batching(
    prediction_pool: pl.DataFrame,
    learner: Learner,
    featurizer: str | None,
    cache_dir: Path,
    memory_safety_factor: float = DEFAULT_SAFETY_FACTOR,
    n_jobs: int = 1,
    compute_uncertainty: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, list, float]:
    """Memory-safe batched prediction with OOM recovery.

    Args:
        prediction_pool: DataFrame with ID and SMILES columns.
        learner: Trained learner instance.
        featurizer: Featurizer name or None for SMILES-native learners.
        cache_dir: Feature cache directory.
        memory_safety_factor: Fraction of available memory to use.
        n_jobs: Number of parallel workers.
        compute_uncertainty: Forwarded to ``learner.predict`` per chunk
            (feature 023). When False, the post-chunk
            ``supports_uncertainty()``-based stack invariant is also relaxed:
            every chunk must return ``None`` for uncertainty (uniform
            skip-path contract). Default True preserves prior behaviour.

    Returns:
        Tuple of ``(predictions, uncertainties_or_none, valid_ids,
        feature_extraction_time)``. The fourth element is the wall-clock seconds
        spent in ``extract_features`` summed across all chunks (including
        OOM-retry sub-chunks), so the caller can report it independently of
        learner-side prediction time. When ``compute_uncertainty`` is False
        ``uncertainties_or_none`` is always ``None``.

    Raises:
        LearnerError: If OOM persists at minimum batch size.
    """
    n_samples = len(prediction_pool)
    n_features = _get_n_features(featurizer)
    device = _get_learner_device(learner)
    is_gpu = device.startswith('cuda') or device == 'gpu'
    min_batch = MIN_BATCH_GPU if is_gpu else MIN_BATCH_CPU

    batch_size = estimate_batch_size(
        learner, n_samples, n_features, device, memory_safety_factor, n_jobs
    )

    n_batches = math.ceil(n_samples / batch_size)
    available = get_available_memory(device)
    logger.info(
        'Batched prediction: %d samples, %d batches (batch_size=%d), '
        'available=%d bytes, safety_factor=%.2f',
        n_samples,
        n_batches,
        batch_size,
        available,
        memory_safety_factor,
    )

    all_predictions: list[np.ndarray] = []
    all_uncertainties: list[np.ndarray] = []
    # Feature 023: `has_uncertainty` is the cross-chunk invariant — when the
    # cycle asked for uncertainty AND the learner supports it, every chunk
    # MUST return non-None uncertainties. When compute_uncertainty=False the
    # contract is uniform: every chunk returns None.
    expect_uncertainty = learner.supports_uncertainty() and compute_uncertainty
    all_valid_ids: list = []
    feature_extraction_time = 0.0

    for _chunk_idx, chunk_df in enumerate(
        _chunk_dataframe(prediction_pool, batch_size)
    ):
        chunk_preds, chunk_uncerts, chunk_feat_time = _predict_chunk_with_oom_retry(
            chunk_df,
            learner,
            featurizer,
            cache_dir,
            n_jobs,
            min_batch,
            n_samples,
            n_features,
            device,
            compute_uncertainty=compute_uncertainty,
        )
        feature_extraction_time += chunk_feat_time

        all_predictions.append(chunk_preds)
        all_valid_ids.extend(chunk_df['ID'].to_list())
        if expect_uncertainty:
            if chunk_uncerts is None:
                raise LearnerError(
                    f'Chunk {_chunk_idx}: learner declared supports_uncertainty()=True '
                    f'but returned None uncertainties. Every chunk must return non-None '
                    f'uncertainties of equal length to predictions.'
                )
            if len(chunk_uncerts) != len(chunk_preds):
                raise LearnerError(
                    f'Chunk {_chunk_idx}: uncertainty length {len(chunk_uncerts)} '
                    f'mismatches prediction length {len(chunk_preds)}. '
                    f'Both arrays must have the same number of rows.'
                )
            all_uncertainties.append(chunk_uncerts)

    predictions = (
        all_predictions[0]
        if len(all_predictions) == 1
        else np.concatenate(all_predictions)
    )
    uncertainties: np.ndarray | None = None
    if all_uncertainties:
        uncertainties = (
            all_uncertainties[0]
            if len(all_uncertainties) == 1
            else np.concatenate(all_uncertainties)
        )

    # Spec 022 FR-008: single dtype-conversion boundary. predict_with_batching is
    # the sole producer cast site; downstream consumers (persistence, metric
    # aggregators) receive float32 inputs. ``copy=False`` skips the copy if the
    # learner already returned float32 (e.g. torch float32 inference).
    predictions = predictions.astype(np.float32, copy=False)
    if uncertainties is not None:
        uncertainties = uncertainties.astype(np.float32, copy=False)

    return predictions, uncertainties, all_valid_ids, feature_extraction_time

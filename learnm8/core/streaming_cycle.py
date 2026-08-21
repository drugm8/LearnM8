"""Two-pass streaming selection for the active-learning cycle.

This is the memory-efficient alternative to the legacy in-memory selection path
in :func:`learnm8.core.cycle.execute_cycle`. It never materialises the full
``(N,)`` prediction arrays, the ``all_valid_ids`` Python list, the unlabeled⋈
predictions join, or the full ``cycle_predictions`` DataFrame in the parent
heap. At 100M scale these structures dominate the per-cycle peak (~6.6 GB of
Python/numpy allocations measured at 1M).

Design (see the feature plan):

- **Pass 1** (expensive — featurize + predict, chunking already exists): iterate
  the unlabeled pool by row index, predict each chunk, and stream the results
  into ``prediction_cycle_N.parquet`` via :class:`CyclePredictionsWriter` (exact
  1M-row groups). Nothing accumulates across chunks.
- **Stats + pruning threshold**: one lazy aggregation over the parquet for the
  metric reductions; an O(N)-time / O(k)-memory :class:`StreamingTopK` pass finds
  the exact ``n_to_remove`` worst predictions for score-based pruning.
- **Pass 2** (cheap — I/O over the parquet just written): iterate by row group,
  score each chunk pointwise (``acq_func.score_chunk``), mask out pruned rows
  (``-inf``, preserving global indices so position-seeded RNG stays correct),
  and reduce to a bounded shortlist. ``acq_func.finalize`` yields the batch.

Eligibility is decided by :func:`is_streaming_eligible`; the cycle falls back to
the legacy path for non-streamable strategies (e.g. simulated_annealing), when
there is no ``output_dir`` (the parquet is the dataflow medium), or for pruning
strategies other than ``'score'``.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from learnm8.acquisition.streaming import StreamingTopK
from learnm8.core.batching import (
    MIN_BATCH_CPU,
    MIN_BATCH_GPU,
    _get_learner_device,
    _get_n_features,
    _predict_chunk_with_oom_retry,
    estimate_batch_size,
)
from learnm8.core.dataframe_ops import iter_status_chunks, mark_pruned_by_ids
from learnm8.core.persistence import (
    SCORING_CHUNK,
    CyclePredictionsWriter,
    read_cycle_predictions,
)
from learnm8.core.prediction_stats import PredictionStats
from learnm8.exceptions import PruningError
from learnm8.utils.numerical import assert_no_inf_uncertainty, assert_no_nan

if TYPE_CHECKING:
    from learnm8.core.config import CycleConfig
    from learnm8.core.interfaces import Learner, Oracle

logger = logging.getLogger(__name__)


def is_streaming_eligible(
    acq_func: Any,
    config: CycleConfig,
    output_dir: Path | None,
    streaming_mode: str,
) -> bool:
    """Decide whether the streaming path can run this cycle.

    Requires: a streamable acquisition strategy, an ``output_dir`` (the parquet
    is the dataflow medium), and a pruning strategy that the streaming pruner
    supports (``None`` or ``'score'``). ``streaming_mode`` is ``'auto'``,
    ``'always'`` or ``'never'``.
    """
    if streaming_mode == 'never':
        return False
    # Defensive getattr: a non-AcquisitionFunction acq object (e.g. a test
    # double or exotic third-party type) without the streaming tier falls back
    # to the legacy path rather than erroring.
    supports = getattr(acq_func, 'supports_streaming', None)
    if not callable(supports) or not supports():
        return False
    if output_dir is None:
        return False
    return config.pruning_strategy in (None, 'score')


def _resolve_pruning(n_unlabeled: int, config: CycleConfig) -> tuple[int, float]:
    """Return ``(n_to_remove, pruning_fraction)`` matching ScoreBasedPruner.

    Replicates ``ScoreBasedPruner``'s exact-count clamp (keep >= 1) so the
    streaming pruner removes the same number of compounds as the legacy pruner.
    """
    if config.pruning_strategy != 'score':
        return 0, 0.0
    params = config.pruning_params or {}
    frac = float(params.get('pruning_fraction', 0.1))
    if frac <= 0.0 or n_unlabeled <= 1:
        return 0, frac
    n_to_remove = int(n_unlabeled * frac)
    n_to_keep = n_unlabeled - n_to_remove
    if n_to_keep <= 0:
        n_to_remove = n_unlabeled - 1
    return n_to_remove, frac


def _pass1_predict_to_parquet(
    *,
    compounds_df: pl.DataFrame,
    learner: Learner,
    featurizer: str | None,
    cache_dir: Path,
    output_dir: Path,
    cycle: int,
    n_unlabeled: int,
    memory_safety_factor: float,
    n_jobs: int,
    compute_uncertainty: bool,
) -> tuple[Path, bool, float, float]:
    """Predict the unlabeled pool chunk-by-chunk into the cycle parquet.

    Returns ``(parquet_path, with_uncertainty, prediction_time, feature_time)``.
    """
    n_features = _get_n_features(featurizer)
    device = _get_learner_device(learner)
    is_gpu = device.startswith('cuda') or device == 'gpu'
    min_batch = MIN_BATCH_GPU if is_gpu else MIN_BATCH_CPU
    batch_size = estimate_batch_size(
        learner, n_unlabeled, n_features, device, memory_safety_factor, n_jobs
    )
    with_uncertainty = learner.supports_uncertainty() and compute_uncertainty

    feature_time = 0.0
    start = time.perf_counter()
    with CyclePredictionsWriter(
        output_dir, cycle, with_uncertainty=with_uncertainty
    ) as writer:
        for _offset, chunk_df in iter_status_chunks(
            compounds_df, 'unlabeled', batch_size
        ):
            preds, uncs, feat_t = _predict_chunk_with_oom_retry(
                chunk_df,
                learner,
                featurizer,
                cache_dir,
                n_jobs,
                min_batch,
                n_unlabeled,
                n_features,
                device,
                compute_uncertainty=compute_uncertainty,
            )
            feature_time += feat_t
            ids = chunk_df['ID']
            # Per-chunk NaN/Inf guards (FR-005) with better error locality than
            # the legacy single full-array guard.
            assert_no_nan(preds, ids, 'predictions')
            chunk_unc = uncs if with_uncertainty else None
            if chunk_unc is not None:
                assert_no_nan(chunk_unc, ids, 'uncertainties')
                assert_no_inf_uncertainty(chunk_unc, ids)
            writer.write_chunk(ids, preds, chunk_unc)
        parquet_path = writer.path
    prediction_time = max(0.0, (time.perf_counter() - start) - feature_time)
    assert parquet_path is not None
    return parquet_path, with_uncertainty, prediction_time, feature_time


def _find_pruned_indices(
    parquet_path: Path, n_to_remove: int, score_direction: str
) -> np.ndarray:
    """Global indices of the ``n_to_remove`` worst predictions (exact, deterministic).

    'Worst' is lowest prediction when maximising ('higher'), highest when
    minimising. Ties at the boundary resolve to the smallest global index
    (StreamingTopK canonical order).
    """
    import pyarrow.parquet as pq

    # Keep the n_to_remove WORST: for 'higher', worst = lowest pred → keep
    # lowest → descending=False; for 'lower', worst = highest → descending=True.
    prune_topk = StreamingTopK(n_to_remove, descending=(score_direction == 'lower'))
    pf = pq.ParquetFile(str(parquet_path))
    offset = 0
    for rb in pf.iter_batches(batch_size=SCORING_CHUNK, columns=['prediction']):
        preds = rb.column('prediction').to_numpy(zero_copy_only=False)
        prune_topk.update(preds.astype(np.float64), offset)
        offset += len(preds)
    idx, _ = prune_topk.result()
    return idx


def _pass2_select(
    *,
    parquet_path: Path,
    acq_func: Any,
    with_uncertainty: bool,
    n_unlabeled: int,
    batch_size: int,
    pruned_mask: np.ndarray | None,
) -> tuple[pl.DataFrame, pl.Series | None]:
    """Score the parquet by row group and reduce to the final batch.

    Returns ``(selected_df, pruned_ids_or_none)``. ``selected_df`` has the
    finalize contract columns (ID, prediction, [uncertainty], acquisition_score,
    global_index). Pruned rows are masked with ``-inf`` so they never win while
    global indices (and thus position-seeded RNG) stay correct.
    """
    import pyarrow.parquet as pq

    shortlist_k = acq_func.shortlist_size(batch_size, n_unlabeled - _mask_count(pruned_mask))
    shortlist_k = max(1, shortlist_k)
    select_topk = StreamingTopK(shortlist_k, descending=True)

    cols = ['ID', 'prediction'] + (['uncertainty'] if with_uncertainty else [])
    pruned_id_parts: list[pl.Series] = []

    pf = pq.ParquetFile(str(parquet_path))
    offset = 0
    for rb in pf.iter_batches(batch_size=SCORING_CHUNK, columns=cols):
        chunk = pl.from_arrow(rb)
        n = chunk.height
        preds = chunk['prediction'].to_numpy()
        uncs = chunk['uncertainty'].to_numpy() if with_uncertainty else None
        scores = acq_func.score_chunk(
            preds, uncs, global_offset=offset, n_total=n_unlabeled
        )
        if pruned_mask is not None:
            local = pruned_mask[offset : offset + n]
            if local.any():
                pruned_id_parts.append(chunk['ID'].filter(pl.Series(local)))
                scores = scores.astype(np.float64, copy=True)
                scores[local] = -np.inf
        select_topk.update(scores, offset)
        offset += n

    sel_idx, sel_scores = select_topk.result()
    selected_df = _gather_shortlist(parquet_path, sel_idx, sel_scores)
    selected_df = acq_func.finalize(selected_df, batch_size)

    pruned_ids = pl.concat(pruned_id_parts) if pruned_id_parts else None
    return selected_df, pruned_ids


def _mask_count(mask: np.ndarray | None) -> int:
    return 0 if mask is None else int(np.count_nonzero(mask))


def _gather_shortlist(
    parquet_path: Path,
    sel_idx: np.ndarray,
    sel_scores: np.ndarray,
) -> pl.DataFrame:
    """Gather shortlist rows from the parquet by global index, in canonical order."""
    sel_idx_u32 = pl.Series('global_index', sel_idx, dtype=pl.UInt32)
    gathered = (
        pl.scan_parquet(parquet_path)
        .with_row_index('global_index')
        .filter(pl.col('global_index').is_in(sel_idx_u32))
        .collect()
    )
    order_df = pl.DataFrame(
        {
            'global_index': pl.Series(sel_idx, dtype=pl.UInt32),
            'acquisition_score': sel_scores,
            '_ord': np.arange(len(sel_idx), dtype=np.int64),
        }
    )
    out = gathered.join(order_df, on='global_index', how='inner').sort('_ord')
    return out.drop('_ord')


def execute_streaming_selection(
    *,
    compounds_df: pl.DataFrame,
    cycle: int,
    config: CycleConfig,
    learner: Learner,
    oracle: Oracle,
    target_col: str,
    featurizer: str | None,
    cache_dir: Path,
    output_dir: Path,
    original_pool_size: int,
    score_direction: str,
    oracle_type: str,
    memory_safety_factor: float,
    n_jobs: int,
    acq_func: Any,
    compute_uncertainty: bool,
    measure_and_label: Any,
) -> dict[str, Any]:
    """Run the two-pass streaming selection for one cycle.

    ``measure_and_label`` is ``cycle._measure_and_label`` (injected to avoid a
    circular import). Returns a dict the caller merges into the cycle metrics.
    """
    n_unlabeled = int(compounds_df.filter(pl.col('status') == 'unlabeled').height)

    # PASS 1: predict the unlabeled pool into the cycle parquet.
    parquet_path, with_uncertainty, prediction_time, predict_feature_time = (
        _pass1_predict_to_parquet(
            compounds_df=compounds_df,
            learner=learner,
            featurizer=featurizer,
            cache_dir=cache_dir,
            output_dir=output_dir,
            cycle=cycle,
            n_unlabeled=n_unlabeled,
            memory_safety_factor=memory_safety_factor,
            n_jobs=n_jobs,
            compute_uncertainty=compute_uncertainty,
        )
    )

    # Stats from the parquet (no full array in the parent heap).
    pred_stats = PredictionStats.from_lazy(
        pl.scan_parquet(parquet_path), with_uncertainty=with_uncertainty
    )
    logger.info(
        'Prediction complete: %d predictions (min=%.2f, max=%.2f, mean=%.2f) in %.2fs',
        pred_stats.count,
        pred_stats.pred_min if pred_stats.pred_min is not None else float('nan'),
        pred_stats.pred_max if pred_stats.pred_max is not None else float('nan'),
        pred_stats.pred_mean if pred_stats.pred_mean is not None else float('nan'),
        prediction_time,
    )

    # Pruning: find the exact worst n_to_remove and build an exclusion mask.
    pruned_mask: np.ndarray | None = None
    n_to_remove, _frac = _resolve_pruning(n_unlabeled, config)
    if n_to_remove > 0:
        pruned_idx = _find_pruned_indices(parquet_path, n_to_remove, score_direction)
        pruned_mask = np.zeros(n_unlabeled, dtype=bool)
        pruned_mask[pruned_idx] = True

    eligible = n_unlabeled - _mask_count(pruned_mask)
    if eligible == 0:
        raise PruningError(
            f'No compounds remaining after pruning in cycle {cycle}. '
            f'Pruning removed all {n_unlabeled} candidates. Reduce pruning_fraction '
            f'or disable pruning by setting pruning_strategy=None.'
        )

    # Batch size matches the legacy formula (clipped to the post-prune pool).
    batch_size = min(
        max(1, math.ceil(original_pool_size * config.batch_fraction)), eligible
    )

    # PASS 2: score + reduce to the batch; collect pruned IDs for the master update.
    acquisition_start = time.perf_counter()
    selected_df, pruned_ids = _pass2_select(
        parquet_path=parquet_path,
        acq_func=acq_func,
        with_uncertainty=with_uncertainty,
        n_unlabeled=n_unlabeled,
        batch_size=batch_size,
        pruned_mask=pruned_mask,
    )
    acquisition_time = time.perf_counter() - acquisition_start

    selected_ids = selected_df['ID'].to_list()
    # selection_history parity: order by global index (ascending pool order).
    sel_cols = ['ID', 'prediction'] + (['uncertainty'] if with_uncertainty else [])
    selected_predictions = selected_df.sort('global_index').select(sel_cols)

    # Mark pruned compounds in the master DataFrame (join-based, no Python list).
    pruned_count = 0
    if pruned_ids is not None and pruned_ids.len() > 0:
        compounds_df = mark_pruned_by_ids(compounds_df, pruned_ids, cycle)
        pruned_count = int(pruned_ids.len())

    # Measure + label the selected batch (shared tail with the legacy path).
    compounds_df, oracle_time = measure_and_label(
        compounds_df=compounds_df,
        selected_ids=selected_ids,
        cycle=cycle,
        target_col=target_col,
        oracle=oracle,
    )

    # Predictions frame for the metrics eval-join. Run mode: the batch suffices
    # (legacy's labeled⋈cycle_predictions inner join yields exactly this batch).
    # Benchmark mode needs the full pool for ranking metrics (O(N), accepted).
    if oracle_type == 'benchmark':
        metrics_cycle_predictions = read_cycle_predictions(parquet_path)
    else:
        metrics_cycle_predictions = selected_predictions

    return {
        'compounds_df': compounds_df,
        'selected_ids': selected_ids,
        'selected_predictions': selected_predictions,
        'pred_stats': pred_stats,
        'pruned_count': pruned_count,
        'parquet_path': parquet_path,
        'prediction_time': prediction_time,
        'predict_feature_time': predict_feature_time,
        'acquisition_time': acquisition_time,
        'oracle_time': oracle_time,
        'metrics_cycle_predictions': metrics_cycle_predictions,
    }

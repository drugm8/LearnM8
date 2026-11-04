"""One-shot batch suggestion for slow, lab-in-the-loop assays.

Each call reads the batch files in ``batch_dir``, trains on the measured
values, selects the next batch and writes it as a new batch file.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import polars as pl

from learnm8.acquisition import get_acquisition_function
from learnm8.core.batching import configure_host_allocator
from learnm8.core.config import CycleConfig
from learnm8.core.cycle import execute_cycle
from learnm8.core.dataframe_ops import update_status
from learnm8.core.initialization import (
    initialize_master_dataframe_empty,
    select_initial_batch,
)
from learnm8.core.interfaces import Oracle
from learnm8.core.lab_batches import (
    batch_path,
    next_cycle,
    pruned_path,
    read_batches,
    read_pruned,
    write_batch,
    write_pruned,
)
from learnm8.core.resources import validate_device, validate_n_jobs
from learnm8.core.validation import canonicalize_for_cache, validate_compound_pool
from learnm8.exceptions import (
    ConfigurationError,
    LearnM8Error,
    ValidationError,
    _truncate_list,
)
from learnm8.features import create_featurizer
from learnm8.features.worker_pool import shutdown_featurization_pool
from learnm8.utils.file_loaders import load_compound_file
from learnm8.utils.logging import configure_learnm8_logging

logger = logging.getLogger(__name__)


class _PendingOracle(Oracle):
    """Oracle that measures nothing and records which compounds were requested."""

    def __init__(self) -> None:
        self.requested: pl.DataFrame | None = None

    def measure(self, compounds: pl.DataFrame, properties: list[str]) -> pl.DataFrame:
        self.requested = compounds.select('ID', 'SMILES')
        return compounds.select('ID').with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(p) for p in properties]
        )

    def known_ids(self) -> set[str] | None:
        return None


def _strip_zeros(col: str) -> pl.Expr:
    return pl.col(col).str.strip_chars_start('0')


def _recover_leading_zeros(
    ledger: pl.DataFrame, pool: pl.DataFrame, batch_dir: Path
) -> pl.DataFrame:
    """Remap all-digit batch IDs that a spreadsheet stripped of leading zeros.

    A batch ID absent from the pool is remapped to the unique pool ID with the
    same zero-stripped form, provided the batch SMILES is empty or agrees. A
    batch ID present in the pool whose zero-stripped form is shared by another
    pool ID, with an empty SMILES cell, is ambiguous and raises.
    """
    digits = ledger.filter(pl.col('ID').str.contains(r'^[0-9]+$'))
    if digits.height == 0:
        return ledger
    missing = digits.join(pool.select('ID'), on='ID', how='anti')
    if missing.height == 0 and digits['smiles'].null_count() == 0:
        return ledger

    wanted = digits.with_columns(_strip_zeros('ID').alias('_s'))
    cands = (
        pool.select('ID', 'SMILES')
        .with_columns(_strip_zeros('ID').alias('_s'))
        .filter(pl.col('_s').is_in(wanted['_s'].implode()))
    )
    by_stripped: dict[str, list[tuple[str, str]]] = {}
    for pid, psmi, st in zip(cands['ID'], cands['SMILES'], cands['_s'], strict=True):
        by_stripped.setdefault(st, []).append((pid, psmi))

    remaps: dict[tuple[int, str], str] = {}
    ambiguous: list[str] = []
    for bid, cycle, smi, st in zip(
        wanted['ID'], wanted['cycle'], wanted['smiles'], wanted['_s'], strict=True
    ):
        matches = by_stripped.get(st, [])
        name = batch_path(batch_dir, cycle).name
        if any(pid == bid for pid, _ in matches):
            if len(matches) > 1 and smi is None:
                ambiguous.append(
                    f'{bid} ({name}) vs {sorted(pid for pid, _ in matches)}'
                )
        elif len(matches) == 1 and (
            smi is None or canonicalize_for_cache(smi) == matches[0][1]
        ):
            remaps[(cycle, bid)] = matches[0][0]
    if ambiguous:
        raise ValidationError(
            f'Ambiguous IDs in batch files: the pool has several IDs that differ '
            f'only in leading zeros and the SMILES cell is empty: '
            f'{_truncate_list(ambiguous)}'
        )
    if not remaps:
        return ledger

    logger.info(
        'Recovered IDs that lost leading zeros: %s',
        _truncate_list(
            [
                f'{batch_path(batch_dir, c).name}: {old} -> {new}'
                for (c, old), new in remaps.items()
            ]
        ),
    )
    ledger = ledger.with_columns(
        pl.struct('cycle', 'ID')
        .map_elements(
            lambda r: remaps.get((r['cycle'], r['ID']), r['ID']), return_dtype=pl.Utf8
        )
        .alias('ID')
    )
    dup_ids = ledger.filter(pl.col('ID').is_duplicated())['ID'].unique().sort()
    if dup_ids.len():
        where = []
        for i in dup_ids.to_list():
            cycles = ledger.filter(pl.col('ID') == i)['cycle'].sort().to_list()
            files = dict.fromkeys(batch_path(batch_dir, c).name for c in cycles)
            where.append(f'{i} ({", ".join(files)})')
        raise ValidationError(
            f'Duplicate IDs in batch files after recovering leading zeros: '
            f'{_truncate_list(where)}'
        )
    return ledger


def _check_against_pool(
    rows: pl.DataFrame,
    pool: pl.DataFrame,
    file_name: Callable[[int], str],
    label: str,
    plural: str,
    unknown_hint: str,
    mismatch_hint: str,
) -> None:
    """Raise ValidationError for rows (ID, cycle, smiles) missing from or differing from the pool."""
    unknown = rows.filter(~pl.col('ID').is_in(pool['ID'].implode())).sort('cycle', 'ID')
    if unknown.height:
        found = [
            f'{i} ({file_name(c)})'
            for i, c in zip(unknown['ID'], unknown['cycle'], strict=True)
        ]
        raise ValidationError(
            f'{label} IDs not found in the validated compound pool: '
            f'{_truncate_list(found)}. {unknown_hint}'
        )
    mismatch = (
        rows.filter(pl.col('smiles').is_not_null())
        .join(pool.select('ID', pl.col('SMILES').alias('_pool_smiles')), on='ID')
        .filter(
            pl.col('smiles').map_elements(canonicalize_for_cache, return_dtype=pl.Utf8)
            != pl.col('_pool_smiles')
        )
        .sort('cycle', 'ID')
    )
    if mismatch.height:
        found = [
            f'{i} ({file_name(c)})'
            for i, c in zip(mismatch['ID'], mismatch['cycle'], strict=True)
        ]
        raise ValidationError(
            f'SMILES in {plural} differ from the compound pool for: '
            f'{_truncate_list(found)}. {mismatch_hint}'
        )


def suggest_next_batch(
    compound_pool: str | Path,
    batch_dir: str | Path,
    target_col: str,
    learner: str,
    featurizer: str | None = None,
    *,
    batch_size: int = 96,
    strategy: str = 'greedy',
    score_direction: str = 'higher',
    acquisition_params: dict | None = None,
    random_state: int = 42,
    smiles_column: str | None = None,
    id_column: str | None = None,
    cache_dir: str | Path | None = None,
    n_jobs: int = -1,
    device: str = 'auto',
    pruning_fraction: float = 0.0,
    pruning_start: int = 3,
) -> pl.DataFrame:
    """Train on the measured batch results and write the next batch file.

    Cycle 0 (no batch files yet) is a random batch; later cycles train
    ``learner`` on all measured values and select with ``strategy``. Pending
    (empty) and failed (non-numeric) compounds are never selected again.

    With ``pruning_fraction > 0``, rounds from cycle ``pruning_start`` on that
    train a model first remove that fraction of the worst-predicted unlabeled
    compounds for good, recorded in ``batch_dir/pruned_cycle_N.parquet``.

    Returns:
        The new batch with columns ``rank, ID, <target_col>, SMILES`` plus
        ``prediction`` and ``uncertainty`` (when computed) for cycles >= 1.
    """
    # Imported lazily: learnm8/__init__ imports api, which would be circular.
    from learnm8.api import _create_learner

    n_jobs = validate_n_jobs(n_jobs)
    device = validate_device(device)
    if batch_size < 1:
        raise ConfigurationError(f'batch_size must be >= 1, got {batch_size}')
    try:
        get_acquisition_function(strategy)
    except KeyError as exc:
        raise ConfigurationError(str(exc.args[0])) from exc
    if score_direction not in ('higher', 'lower'):
        raise ConfigurationError(
            f"score_direction must be 'higher' or 'lower', got {score_direction!r}"
        )
    if acquisition_params is not None and not isinstance(acquisition_params, dict):
        raise ConfigurationError(
            f'acquisition_params must be a dict, got {type(acquisition_params).__name__}'
        )
    if isinstance(pruning_fraction, bool) or not isinstance(
        pruning_fraction, (int, float)
    ):
        raise ConfigurationError(
            f'pruning_fraction must be a number, got {pruning_fraction!r}'
        )
    if not 0.0 <= pruning_fraction <= 0.9:
        raise ConfigurationError(
            f'pruning_fraction must be between 0.0 and 0.9, got {pruning_fraction}'
        )
    if isinstance(pruning_start, bool) or not isinstance(pruning_start, int):
        raise ConfigurationError(
            f'pruning_start must be an integer, got {pruning_start!r}'
        )
    if pruning_start < 1:
        raise ConfigurationError(f'pruning_start must be >= 1, got {pruning_start}')
    compound_pool = Path(compound_pool)
    if not compound_pool.exists():
        raise ConfigurationError(f'Compound pool file not found: {compound_pool}')
    batch_dir = Path(batch_dir)
    work_dir = batch_dir / '.learnm8'
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_path = work_dir / 'cache' if cache_dir is None else Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    configure_learnm8_logging(output_dir=work_dir)
    configure_host_allocator()

    pool = load_compound_file(
        compound_pool,
        smiles_column,
        id_column,
        n_jobs=n_jobs,
        progress=False,
        id_as_text=True,
    )
    pool = pool.with_columns(pl.col('ID').cast(pl.Utf8))
    pool = validate_compound_pool(
        pool, n_jobs=n_jobs, progress=False, target_col=None
    ).valid_compounds

    ledger = _recover_leading_zeros(
        read_batches(batch_dir, target_col), pool, batch_dir
    )
    _check_against_pool(
        ledger,
        pool,
        lambda c: batch_path(batch_dir, c).name,
        'Batch file',
        'batch files',
        'Either the ID is not in the pool, a leading-zero match was ambiguous, '
        'or its SMILES differ.',
        'Was the pool file re-sorted or changed since the batch was written?',
    )
    unchecked = ledger.filter(pl.col('smiles').is_null()).sort('cycle', 'ID')
    if unchecked.height:
        logger.warning(
            'Batch rows with an empty SMILES cell could not be cross-checked '
            'against the pool (shifted or re-sorted rows go undetected): %s',
            _truncate_list(
                [
                    f'{i} ({batch_path(batch_dir, c).name})'
                    for i, c in zip(unchecked['ID'], unchecked['cycle'], strict=True)
                ]
            ),
        )
    cycle = next_cycle(ledger)

    if ledger.height:
        newest = cycle - 1
        newest_states = ledger.filter(pl.col('cycle') == newest)['state']
        if (newest_states == 'pending').all():
            raise ConfigurationError(
                f'{batch_path(batch_dir, newest).name} has no results entered yet; '
                f'enter the measured values (or delete the file) before suggesting '
                f'the next batch.'
            )

    pruned_rows = read_pruned(batch_dir, cycle)
    _check_against_pool(
        pruned_rows,
        pool,
        lambda c: pruned_path(batch_dir, c).name,
        'Prune file',
        'prune files',
        'Either the ID is not in the pool or its SMILES differ.',
        'Was the pool file re-sorted or changed since the round was pruned?',
    )
    pruned_before = pruned_rows['ID']
    # A measured value wins over an earlier prune record: train on it.
    measured_ids = ledger.filter(pl.col('state') == 'measured')['ID']
    revived = pruned_before.filter(pruned_before.is_in(measured_ids.implode()))
    if revived.len():
        logger.warning(
            'Compounds with a measured value are also in a prune record; their '
            'measurements are used for training: %s',
            _truncate_list(revived.to_list()),
        )
        pruned_before = pruned_before.filter(~pruned_before.is_in(revived.implode()))
    master = initialize_master_dataframe_empty(pool, target_col)
    unselectable = pl.concat(
        [ledger.filter(pl.col('state') != 'measured')['ID'], pruned_before]
    )
    master = master.filter(~pl.col('ID').is_in(unselectable.implode()))
    measured = ledger.filter(pl.col('state') == 'measured')
    for (c,), group in measured.group_by('cycle', maintain_order=True):
        ids = group['ID'].to_list()
        master = update_status(
            master,
            ids,
            'labeled',
            int(c),  # type: ignore[call-overload]
            target_col,
            dict(zip(ids, group['value'].to_list(), strict=True)),
        )
    n_measured = measured.height
    n_unlabeled = master.filter(pl.col('status') == 'unlabeled').height
    if n_unlabeled == 0:
        raise ConfigurationError('no selectable compounds remain in the pool')
    prune = pruning_fraction > 0 and cycle >= pruning_start and n_measured > 0
    if prune:
        # Lab mode only: keep at least batch_size unlabeled compounds so the plate
        # is full. The pruners remove int(n * fraction); the +0.5 keeps float
        # rounding from undercutting the exact count.
        requested = int(n_unlabeled * pruning_fraction)
        n_remove = max(0, min(requested, n_unlabeled - batch_size))
        if n_remove < requested:
            pruning_fraction = (n_remove + 0.5) / n_unlabeled if n_remove else 0.0
            prune = n_remove > 0
            logger.info(
                'Pruning fraction lowered to %.4f in cycle %d (%s) so that at '
                'least batch_size=%d of the %d unlabeled compounds remain',
                pruning_fraction,
                cycle,
                'pruning skipped' if not prune else f'{n_remove} removed',
                batch_size,
                n_unlabeled,
            )

    learner_obj = _create_learner(learner, random_state, n_jobs=n_jobs, device=device)
    if featurizer is None and not learner_obj.requires_smiles():
        raise ConfigurationError(
            f"Learner '{learner}' is feature-based and needs a featurizer "
            f"(e.g. featurizer='morgan')."
        )
    featurizer_obj = (
        create_featurizer(featurizer, n_jobs=n_jobs) if featurizer else None
    )

    oracle = _PendingOracle()
    try:
        if n_measured == 0:
            # Nothing to train on yet: random batch (ledger IDs already excluded).
            select_initial_batch(
                master,
                oracle,
                target_col,
                'random',
                1.0,
                featurizer,  # type: ignore[arg-type]
                cache_path,
                original_pool_size=batch_size,
                random_state=random_state,
                score_direction=score_direction,
                oracle_type='run',
                featurizer_obj=featurizer_obj,
            )
            metrics: dict = {}
        else:
            master, metrics = execute_cycle(
                master,
                cycle,
                CycleConfig(
                    strategy,
                    n_cycles=1,
                    batch_fraction=1.0,
                    acquisition_params=acquisition_params,
                    pruning_strategy='score' if prune else None,
                    pruning_params={'pruning_fraction': pruning_fraction}
                    if prune
                    else None,
                ),
                learner_obj,
                oracle,
                target_col,
                featurizer,
                cache_path,
                original_pool_size=batch_size,
                score_direction=score_direction,
                oracle_type='run',
                output_dir=work_dir,
                featurizer_obj=featurizer_obj,
                random_state=random_state,
                n_jobs=n_jobs,
                feature_type=featurizer_obj.feature_type
                if featurizer_obj
                else 'binary',
            )
    finally:
        shutdown_featurization_pool()

    if oracle.requested is None:
        raise LearnM8Error('Selection did not request any compounds from the oracle')
    batch = oracle.requested.with_row_index('rank', offset=1).with_columns(
        pl.col('rank').cast(pl.Int64), pl.col('ID').cast(pl.Utf8)
    )
    if n_measured > 0:
        batch = batch.join(
            metrics['selected_predictions'].with_columns(pl.col('ID').cast(pl.Utf8)),
            on='ID',
            how='left',
            maintain_order='left',
        )
    # Written before the batch file, so a batch never exists without its record.
    # Not metrics['pruned_ids']: the streaming path leaves it empty.
    pruned_now = master.filter(
        pl.col('status') == 'pruned', pl.col('pruned_cycle') == cycle
    ).select('ID', 'SMILES')
    pruned_file = write_pruned(pruned_now, batch_dir, cycle)
    if prune:
        logger.info(
            'Pruned %d compounds in cycle %d (%d pruned so far)%s',
            pruned_now.height,
            cycle,
            pruned_before.len() + pruned_now.height,
            f'; recorded in {pruned_file}' if pruned_file else '',
        )
    path = write_batch(batch, batch_dir, cycle, target_col)
    counts = ledger['state'].value_counts()
    n = {s: int(k) for s, k in zip(counts['state'], counts['count'], strict=True)}
    logger.info(
        'Wrote %s (cycle %d, %d compounds); ledger: %d measured, %d pending, %d failed',
        path,
        cycle,
        batch.height,
        n.get('measured', 0),
        n.get('pending', 0),
        n.get('failed', 0),
    )

    batch = batch.with_columns(pl.lit(None, dtype=pl.Float64).alias(target_col))
    first = ['rank', 'ID', target_col, 'SMILES']
    return batch.select(first + [c for c in batch.columns if c not in first])

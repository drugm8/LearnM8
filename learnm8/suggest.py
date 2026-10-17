"""One-shot batch suggestion for slow, lab-in-the-loop assays.

Each call reads the batch files in ``batch_dir``, trains on the measured
values, selects the next batch and writes it as a new batch file.
"""

import logging
from pathlib import Path

import polars as pl

from learnm8.core.batching import configure_host_allocator
from learnm8.core.config import CycleConfig
from learnm8.core.cycle import execute_cycle
from learnm8.core.dataframe_ops import update_status
from learnm8.core.initialization import (
    initialize_master_dataframe_empty,
    select_initial_batch,
)
from learnm8.core.interfaces import Oracle
from learnm8.core.lab_batches import batch_path, next_cycle, read_batches, write_batch
from learnm8.core.resources import validate_device, validate_n_jobs
from learnm8.core.validation import validate_compound_pool
from learnm8.exceptions import ConfigurationError, ValidationError, _truncate_list
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
) -> pl.DataFrame:
    """Train on the measured batch results and write the next batch file.

    Cycle 0 (no batch files yet) is a random batch; later cycles train
    ``learner`` on all measured values and select with ``strategy``. Pending
    (empty) and failed (non-numeric) compounds are never selected again.

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
    batch_dir = Path(batch_dir)
    work_dir = batch_dir / '.learnm8'
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_path = work_dir / 'cache' if cache_dir is None else Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    configure_learnm8_logging(output_dir=work_dir)
    configure_host_allocator()

    pool = load_compound_file(
        compound_pool, smiles_column, id_column, n_jobs=n_jobs, progress=False
    )
    pool = pool.with_columns(pl.col('ID').cast(pl.Utf8))
    pool = validate_compound_pool(
        pool, n_jobs=n_jobs, progress=False, target_col=None
    ).valid_compounds

    ledger = read_batches(batch_dir, target_col)
    unknown = (
        ledger.filter(~pl.col('ID').is_in(pool['ID'].implode()))['ID'].unique().sort()
    )
    if unknown.len():
        raise ValidationError(
            f'Batch file IDs not found in the validated compound pool: '
            f'{_truncate_list(unknown.to_list())}. Excel may have changed IDs, '
            f'e.g. dropped leading zeros.'
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
        if (ledger['state'] != 'measured').all():
            raise ConfigurationError(
                f'No measured values yet in {batch_dir}; delete '
                f'{batch_path(batch_dir, 0).name} to draw a new random batch.'
            )

    master = initialize_master_dataframe_empty(pool, target_col)
    unselectable = ledger.filter(pl.col('state') != 'measured')['ID']
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
    if master.filter(pl.col('status') == 'unlabeled').height == 0:
        raise ConfigurationError('no selectable compounds remain in the pool')

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
        if cycle == 0:
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
            _, metrics = execute_cycle(
                master,
                cycle,
                CycleConfig(
                    strategy,
                    n_cycles=1,
                    batch_fraction=1.0,
                    acquisition_params=acquisition_params,
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

    assert oracle.requested is not None
    batch = oracle.requested.with_row_index('rank', offset=1).with_columns(
        pl.col('rank').cast(pl.Int64), pl.col('ID').cast(pl.Utf8)
    )
    if cycle >= 1:
        batch = batch.join(
            metrics['selected_predictions'].with_columns(pl.col('ID').cast(pl.Utf8)),
            on='ID',
            how='left',
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

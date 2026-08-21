"""Shared acquisition-difference analysis for Figures 10 and 11."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from learnm8.evaluation.metrics.similarity import _scaffold_for_smiles

RESULTS_DIR = Path.home() / 'LearnM8_RESULTS_FINAL'
# Greedy is A-02, not L-01. L-01 was a stand-in while A-02 had no init-matched runs
# below a 1% batch; the initmatch_v2 wave supplied them, so the acquisition family is
# used directly and fig03/fig10/fig11 now agree on what "greedy" means.
# A-08 (simulated annealing) is new in the same wave.
FAMILY_STRATEGY = {
    'A-01': 'random',
    'A-02': 'greedy',
    'A-03': 'ucb',
    'A-04': 'ei',
    'A-05': 'thompson',
    'A-08': 'simulated_annealing',
}
STRATEGY_ORDER = list(FAMILY_STRATEGY.values())
EXPECTED_REPLICATES = {1: 42, 2: 142, 3: 242}
TOP_K_VALUES = (25, 50, 100, 250, 500, 1_000)
FOCUS_K = 100
OVERLAP_BATCH_FRACTIONS = (0.1, 1.0)
FINGERPRINT_USED = 'morgan_2_2048_uint8packed'
ECDF_GRID = np.linspace(0.0, 1.0, 51)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class RunContext:
    """Validated metadata and paths for one acquisition run."""

    run_id: str
    family: str
    strategy: str
    replicate: int
    seed: int
    batch_fraction_pct: float
    path: Path
    cycle_metrics: pl.DataFrame


def _manifest_row(manifest: pl.DataFrame, run_id: str) -> dict[str, Any]:
    rows = manifest.filter(pl.col('run_id') == run_id).to_dicts()
    if len(rows) != 1:
        raise ValueError(f'Manifest must contain exactly one row for {run_id}')
    return rows[0]


def _find_run_id(
    manifest: pl.DataFrame,
    *,
    family: str,
    replicate: int,
    batch_fraction: float,
    design: str,
) -> str:
    """Look a run up by its properties rather than rebuilding its name.

    Run directory names encode the seed design, so string interpolation breaks
    whenever the grammar changes and silently picks the wrong seed design at the
    0.1% and 0.5% arms, where both designs exist. The manifest is authoritative.
    """
    seed = (
        pl.col('init_pct') == pl.col('batch_fraction_pct')
        if design == 'matched'
        else pl.col('init_pct') == 1.0
    )
    rows = manifest.filter(
        (pl.col('family') == family)
        & (pl.col('pool') == '1M')
        & (pl.col('rep') == replicate)
        & (pl.col('batch_fraction_pct') == batch_fraction)
        & seed
    )['run_id'].to_list()
    if len(rows) != 1:
        raise ValueError(
            f'expected exactly one {design} run for {family} rep{replicate} at '
            f'batch {batch_fraction}%, found {len(rows)}: {rows}'
        )
    return rows[0]


def discover_runs(
    results_dir: Path = RESULTS_DIR,
    *,
    batch_fractions: tuple[float, ...] = (1.0,),
    design: str = 'matched',
) -> list[RunContext]:
    """Discover and fail-closed validate common RF acquisition runs."""
    manifest_path = results_dir / 'manifest.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Manifest does not exist: {manifest_path}')
    manifest = pl.read_csv(manifest_path, infer_schema_length=None)
    contexts: list[RunContext] = []

    for batch_fraction in batch_fractions:
        for family, strategy in FAMILY_STRATEGY.items():
            for replicate, seed in EXPECTED_REPLICATES.items():
                run_id = _find_run_id(
                    manifest,
                    family=family,
                    replicate=replicate,
                    batch_fraction=batch_fraction,
                    design=design,
                )
                run_dir = results_dir / run_id
                if not run_dir.is_dir():
                    raise FileNotFoundError(
                        f'Required run directory is missing: {run_dir}'
                    )

                row = _manifest_row(manifest, run_id)
                checks = {
                    'family': family,
                    'strategy': strategy,
                    'pool': '1M',
                    'rep': replicate,
                    'n_cycles': 10,
                }
                for key, expected in checks.items():
                    if row.get(key) != expected:
                        raise ValueError(
                            f'{run_id} manifest {key}={row.get(key)!r}, '
                            f'expected {expected!r}'
                        )
                if not np.isclose(float(row.get('batch_fraction_pct')), batch_fraction):
                    raise ValueError(f'{run_id} is not a {batch_fraction:g}% batch run')

                config_path = run_dir / 'config.json'
                if not config_path.is_file():
                    raise FileNotFoundError(
                        f'Missing config for {run_id}: {config_path}'
                    )
                config = json.loads(config_path.read_text())
                config_checks = {
                    'target_col': 'dockscore',
                    'featurizer': 'morgan',
                    'score_direction': 'lower',
                    'oracle_type': 'benchmark',
                    'n_cycles': 10,
                    'random_state': seed,
                }
                for key, expected in config_checks.items():
                    if config.get(key) != expected:
                        raise ValueError(
                            f'{run_id} config {key}={config.get(key)!r}, '
                            f'expected {expected!r}'
                        )

                metrics_path = run_dir / 'cycle_metrics.csv'
                if not metrics_path.is_file():
                    raise FileNotFoundError(f'Missing cycle metrics for {run_id}')
                metrics = pl.read_csv(metrics_path, infer_schema_length=None)
                required = {
                    'cycle',
                    'strategy',
                    'selected_count',
                    'fingerprint_used',
                }
                missing = required - set(metrics.columns)
                if missing:
                    raise ValueError(
                        f'{run_id} cycle metrics missing {sorted(missing)}'
                    )
                if metrics.height != 10 or metrics['cycle'].n_unique() != 10:
                    raise ValueError(f'{run_id} must contain exactly cycles 0-9')
                if metrics['cycle'].sort().to_list() != list(range(10)):
                    raise ValueError(f'{run_id} must contain exactly cycles 0-9')
                acquisition = metrics.filter(pl.col('cycle') > 0)
                if acquisition['strategy'].unique().to_list() != [strategy]:
                    raise ValueError(
                        f'{run_id} acquisition cycles do not use {strategy}'
                    )
                fingerprints = (
                    metrics['fingerprint_used'].drop_nulls().unique().to_list()
                )
                if fingerprints != [FINGERPRINT_USED]:
                    raise ValueError(
                        f'{run_id} fingerprint provenance is {fingerprints}, '
                        f'expected {[FINGERPRINT_USED]}'
                    )

                contexts.append(
                    RunContext(
                        run_id=run_id,
                        family=family,
                        strategy=strategy,
                        replicate=replicate,
                        seed=seed,
                        batch_fraction_pct=batch_fraction,
                        path=run_dir,
                        cycle_metrics=metrics.sort('cycle'),
                    )
                )
    return contexts


def load_selection(context: RunContext) -> pl.DataFrame:
    """Load acquisition-only compounds and validate their saved history."""
    path = context.path / 'selection_history.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing selection history: {path}')
    frame = pl.read_csv(path, comment_prefix='#', infer_schema_length=None)
    required = {'cycle', 'strategy', 'ID', 'SMILES', 'measured_value'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{path} is missing required columns: {sorted(missing)}')
    frame = frame.select(sorted(required)).with_columns(
        pl.col('ID').cast(pl.String),
        pl.col('SMILES').cast(pl.String),
        pl.col('measured_value').cast(pl.Float64, strict=False),
    )
    if frame.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
        raise ValueError(f'{path} contains null required values')
    if frame['ID'].n_unique() != frame.height:
        raise ValueError(f'{path} contains duplicate IDs')
    if not frame['measured_value'].is_finite().all():
        raise ValueError(f'{path} contains non-finite measured values')
    if frame['cycle'].unique().sort().to_list() != list(range(1, 10)):
        raise ValueError(f'{path} must contain acquisition cycles 1-9 only')
    if frame['strategy'].unique().to_list() != [context.strategy]:
        raise ValueError(f'{path} strategy does not match {context.strategy}')

    expected = int(
        context.cycle_metrics.filter(pl.col('cycle') > 0)['selected_count'].sum()
    )
    if frame.height != expected:
        raise ValueError(f'{path} has {frame.height} rows; expected {expected}')
    return frame


def load_seed(context: RunContext) -> pl.DataFrame:
    """Load the initial cycle-0 seed molecules from a final-compounds file."""
    path = context.path / 'compounds_final.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing final compounds: {path}')
    available = pl.read_csv(path, comment_prefix='#', n_rows=0).columns
    required = {'ID', 'SMILES', 'selected_cycle'}
    missing = required - set(available)
    if missing:
        raise ValueError(f'{path} is missing required columns: {sorted(missing)}')
    seed = (
        pl.scan_csv(path, comment_prefix='#', infer_schema_length=10_000)
        .filter(pl.col('selected_cycle') == 0)
        .select('ID', 'SMILES')
        .collect()
        .with_columns(pl.col('ID').cast(pl.String), pl.col('SMILES').cast(pl.String))
        .sort('ID')
    )
    if seed.null_count().sum_horizontal()[0] != 0:
        raise ValueError(f'{path} seed contains null IDs or SMILES')
    if seed['ID'].n_unique() != seed.height:
        raise ValueError(f'{path} seed contains duplicate IDs')
    expected = int(
        context.cycle_metrics.filter(pl.col('cycle') == 0)['selected_count'][0]
    )
    if seed.height != expected:
        raise ValueError(f'{path} has {seed.height} seed rows; expected {expected}')
    return seed


def top_candidates(selection: pl.DataFrame, k: int) -> pl.DataFrame:
    """Return the K lowest-scoring acquisition-only candidates."""
    if k < 1:
        raise ValueError('k must be positive')
    if selection.height < k:
        raise ValueError(f'Selection has {selection.height} rows, fewer than k={k}')
    required = {'ID', 'SMILES', 'measured_value'}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f'Selection is missing required columns: {sorted(missing)}')
    if selection['ID'].n_unique() != selection.height:
        raise ValueError('Selection contains duplicate IDs')
    if (
        selection['measured_value'].null_count()
        or not selection['measured_value'].is_finite().all()
    ):
        raise ValueError('Selection contains invalid measured values')
    return selection.sort('measured_value', 'ID').head(k)


def molecular_descriptors(
    smiles: list[str],
) -> tuple[list[str], list[ExplicitBitVect]]:
    """Return matched Murcko scaffolds and Morgan(2, 2048) fingerprints."""
    scaffolds: list[str] = []
    fingerprints: list[ExplicitBitVect] = []
    for index, smi in enumerate(smiles):
        scaffold = _scaffold_for_smiles(smi)
        molecule = Chem.MolFromSmiles(smi)
        if scaffold is None or molecule is None:
            raise ValueError(f'Invalid molecule at position {index}: {smi!r}')
        scaffolds.append(scaffold)
        fingerprints.append(_MORGAN.GetFingerprint(molecule))
    return scaffolds, fingerprints


def effective_scaffold_count(scaffolds: list[str]) -> float:
    """Return exp(Shannon entropy), in effective-scaffold units."""
    if not scaffolds:
        raise ValueError('At least one scaffold is required')
    counts = np.asarray(list(Counter(scaffolds).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def median_nearest_neighbor(fingerprints: list[ExplicitBitVect]) -> float:
    """Return the median within-set maximum Morgan/Tanimoto similarity."""
    if len(fingerprints) < 2:
        raise ValueError('At least two fingerprints are required')
    maxima = np.zeros(len(fingerprints), dtype=np.float64)
    for index, fingerprint in enumerate(fingerprints[:-1]):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[index + 1 :]),
            dtype=np.float64,
        )
        maxima[index] = max(maxima[index], similarities.max())
        maxima[index + 1 :] = np.maximum(maxima[index + 1 :], similarities)
    return float(np.median(maxima))


def max_similarity_to_reference(
    queries: list[ExplicitBitVect], references: list[ExplicitBitVect]
) -> np.ndarray:
    """Return each query's maximum Morgan/Tanimoto similarity to a reference."""
    if not queries or not references:
        raise ValueError('Query and reference fingerprints must be non-empty')
    return np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(query, references))
            for query in queries
        ],
        dtype=np.float64,
    )


def novel_scaffold_count(
    query_scaffolds: list[str], reference_scaffolds: list[str]
) -> int:
    """Count exact Murcko scaffolds absent from the reference set."""
    return len(set(query_scaffolds) - set(reference_scaffolds))


def empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate an empirical CDF on ``grid`` and return percentages."""
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('CDF values must be non-empty and finite')
    if grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError('CDF grid must be non-empty and finite')
    return np.searchsorted(np.sort(values), grid, side='right') / values.size * 100.0


def exact_overlap_percent(left: set[str], right: set[str]) -> float:
    """Return exact compound overlap for two equal-sized candidate sets."""
    if not left or not right:
        raise ValueError('Candidate sets must be non-empty')
    if len(left) != len(right):
        raise ValueError('Candidate sets must have equal size')
    return len(left & right) / len(left) * 100.0


def scaffold_summary(
    smiles: list[str], cache: dict[str, str | None]
) -> tuple[set[str], int, int]:
    """Return cyclic Murcko scaffolds plus acyclic and failure counts."""
    scaffolds: set[str] = set()
    acyclic_count = 0
    invalid_count = 0
    with rdBase.BlockLogs():
        for smi in smiles:
            if smi not in cache:
                cache[smi] = _scaffold_for_smiles(smi)
            scaffold = cache[smi]
            if scaffold is None:
                invalid_count += 1
            elif scaffold == '':
                acyclic_count += 1
            else:
                scaffolds.add(scaffold)
    return scaffolds, acyclic_count, invalid_count


def scaffold_jaccard_percent(left: set[str], right: set[str]) -> float:
    """Return Jaccard overlap between two non-empty scaffold sets."""
    union = left | right
    if not union:
        raise ValueError('At least one scaffold set must be non-empty')
    return len(left & right) / len(union) * 100.0


def paired_final_deltas(
    frame: pl.DataFrame,
    *,
    strategies: tuple[str, ...] = ('ucb', 'ei', 'thompson'),
    value_col: str = 'top_0_1_pct_discovery',
) -> pl.DataFrame:
    """Return replicate-matched final differences from greedy in percentage points."""
    keys = ['batch_fraction_pct', 'rep']
    last = frame.filter(
        pl.col('cycle')
        == pl.col('cycle').max().over('strategy', 'batch_fraction_pct', 'rep')
    )
    baseline = last.filter(pl.col('strategy') == 'greedy').select(
        *keys, pl.col(value_col).alias('greedy_value')
    )
    candidates = last.filter(pl.col('strategy').is_in(strategies)).select(
        *keys, 'strategy', pl.col(value_col).alias('strategy_value')
    )
    paired = candidates.join(baseline, on=keys, how='inner', validate='m:1')
    if paired.height != candidates.height:
        raise ValueError('Every strategy result must have a matched greedy replicate')
    return paired.with_columns(
        (pl.col('strategy_value') - pl.col('greedy_value')).alias('delta_vs_greedy_pp')
    ).sort('batch_fraction_pct', 'strategy', 'rep')


def overlap_records(
    selections: dict[str, pl.DataFrame],
    *,
    batch_fraction_pct: float,
    replicate: int,
    focus_k: int = FOCUS_K,
    scaffold_cache: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """Build run-level molecule, scaffold, and top-ranked overlap records."""
    if set(selections) == set():
        raise ValueError('At least one strategy selection is required')
    sizes = {strategy: selection.height for strategy, selection in selections.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError('All strategy selections must have equal compound counts')
    if min(sizes.values()) < focus_k:
        raise ValueError(
            f'Every strategy selection must contain at least {focus_k} rows'
        )

    cache = scaffold_cache if scaffold_cache is not None else {}
    all_ids: dict[str, set[str]] = {}
    top_ids: dict[str, set[str]] = {}
    scaffold_sets: dict[str, set[str]] = {}
    acyclic_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}

    for strategy, selection in selections.items():
        required = {'ID', 'SMILES', 'measured_value'}
        missing = required - set(selection.columns)
        if missing:
            raise ValueError(
                f'{strategy} selection is missing required columns: {sorted(missing)}'
            )
        if selection['ID'].n_unique() != selection.height:
            raise ValueError(f'{strategy} selection contains duplicate IDs')
        all_ids[strategy] = set(selection['ID'].cast(pl.String).to_list())
        top_ids[strategy] = set(
            top_candidates(selection, focus_k)['ID'].cast(pl.String).to_list()
        )
        scaffolds, acyclic, invalid = scaffold_summary(
            selection['SMILES'].cast(pl.String).to_list(), cache
        )
        scaffold_sets[strategy] = scaffolds
        acyclic_counts[strategy] = acyclic
        invalid_counts[strategy] = invalid

    records: list[dict[str, Any]] = []
    for left in selections:
        for right in selections:
            shared = {
                'batch_fraction_pct': batch_fraction_pct,
                'replicate': replicate,
                'left_strategy': left,
                'right_strategy': right,
                'focus_k': focus_k,
            }
            records.append(
                {
                    **shared,
                    'overlap_type': 'all_molecules',
                    'overlap_percent': exact_overlap_percent(
                        all_ids[left], all_ids[right]
                    ),
                    'left_set_size': len(all_ids[left]),
                    'right_set_size': len(all_ids[right]),
                    'left_acyclic_count': None,
                    'right_acyclic_count': None,
                    'left_acyclic_percent': None,
                    'right_acyclic_percent': None,
                    'left_scaffold_failure_count': None,
                    'right_scaffold_failure_count': None,
                }
            )
            records.append(
                {
                    **shared,
                    'overlap_type': 'all_scaffolds',
                    'overlap_percent': scaffold_jaccard_percent(
                        scaffold_sets[left], scaffold_sets[right]
                    ),
                    'left_set_size': len(scaffold_sets[left]),
                    'right_set_size': len(scaffold_sets[right]),
                    'left_acyclic_count': acyclic_counts[left],
                    'right_acyclic_count': acyclic_counts[right],
                    'left_acyclic_percent': acyclic_counts[left]
                    / (sizes[left] - invalid_counts[left])
                    * 100.0,
                    'right_acyclic_percent': acyclic_counts[right]
                    / (sizes[right] - invalid_counts[right])
                    * 100.0,
                    'left_scaffold_failure_count': invalid_counts[left],
                    'right_scaffold_failure_count': invalid_counts[right],
                }
            )
            records.append(
                {
                    **shared,
                    'overlap_type': 'top_100_molecules',
                    'overlap_percent': exact_overlap_percent(
                        top_ids[left], top_ids[right]
                    ),
                    'left_set_size': len(top_ids[left]),
                    'right_set_size': len(top_ids[right]),
                    'left_acyclic_count': None,
                    'right_acyclic_count': None,
                    'left_acyclic_percent': None,
                    'right_acyclic_percent': None,
                    'left_scaffold_failure_count': None,
                    'right_scaffold_failure_count': None,
                }
            )
    return records


def split_overlap_matrix(
    overlap: pl.DataFrame,
    *,
    overlap_type: str,
    strategies: tuple[str, ...] = tuple(STRATEGY_ORDER),
    lower_batch: float = 0.1,
    upper_batch: float = 1.0,
) -> np.ndarray:
    """Return a mean-over-replicates matrix with batch size encoded by triangle."""
    means = (
        overlap.filter(pl.col('overlap_type') == overlap_type)
        .group_by('batch_fraction_pct', 'left_strategy', 'right_strategy')
        .agg(pl.col('overlap_percent').mean().alias('mean'))
    )
    lookup = {
        (row['batch_fraction_pct'], row['left_strategy'], row['right_strategy']): row[
            'mean'
        ]
        for row in means.iter_rows(named=True)
    }
    values = np.full((len(strategies), len(strategies)), np.nan, dtype=np.float64)
    for row, left in enumerate(strategies):
        for column, right in enumerate(strategies):
            if row == column:
                continue
            batch = upper_batch if row < column else lower_batch
            key = (batch, left, right)
            if key not in lookup:
                raise ValueError(f'Missing overlap value for {key}')
            values[row, column] = lookup[key]
    return values


def build_topk_table(results_dir: Path = RESULTS_DIR) -> pl.DataFrame:
    """Calculate run-level quality and within-set diversity across top-K."""
    records: list[dict[str, Any]] = []
    for context in discover_runs(results_dir):
        ranked = top_candidates(load_selection(context), max(TOP_K_VALUES))
        scaffolds, fingerprints = molecular_descriptors(ranked['SMILES'].to_list())
        for k in TOP_K_VALUES:
            records.append(
                {
                    'run_id': context.run_id,
                    'family': context.family,
                    'strategy': context.strategy,
                    'replicate': context.replicate,
                    'seed': context.seed,
                    'top_k': k,
                    'kth_score': float(ranked['measured_value'][k - 1]),
                    'effective_scaffolds': effective_scaffold_count(scaffolds[:k]),
                    'median_nearest_neighbor': median_nearest_neighbor(
                        fingerprints[:k]
                    ),
                }
            )
    return pl.from_dicts(records, infer_schema_length=None)


def _validate_matched_seed(
    contexts: list[RunContext], replicate: int
) -> tuple[pl.DataFrame, RunContext]:
    matched = [context for context in contexts if context.replicate == replicate]
    reference_context = matched[0]
    reference = load_seed(reference_context)
    for context in matched[1:]:
        candidate = load_seed(context)
        if not candidate.equals(reference):
            raise ValueError(
                f'Replicate {replicate} seed differs between '
                f'{reference_context.run_id} and {context.run_id}'
            )
    return reference, reference_context


def build_overlap_table(results_dir: Path = RESULTS_DIR) -> pl.DataFrame:
    """Calculate acquisition-only overlap for 0.1% and 1% RF batches."""
    contexts = discover_runs(results_dir, batch_fractions=OVERLAP_BATCH_FRACTIONS)
    records: list[dict[str, Any]] = []
    scaffold_cache: dict[str, str | None] = {}

    for batch_fraction in OVERLAP_BATCH_FRACTIONS:
        batch_contexts = [
            context
            for context in contexts
            if np.isclose(context.batch_fraction_pct, batch_fraction)
        ]
        for replicate in EXPECTED_REPLICATES:
            replicate_contexts = [
                context for context in batch_contexts if context.replicate == replicate
            ]
            if [context.strategy for context in replicate_contexts] != STRATEGY_ORDER:
                raise ValueError(
                    f'Batch {batch_fraction:g}% replicate {replicate} does not '
                    'contain the required strategy order'
                )
            _validate_matched_seed(replicate_contexts, replicate)
            selections = {
                context.strategy: load_selection(context)
                for context in replicate_contexts
            }
            records.extend(
                overlap_records(
                    selections,
                    batch_fraction_pct=batch_fraction,
                    replicate=replicate,
                    scaffold_cache=scaffold_cache,
                )
            )

    return pl.from_dicts(records, infer_schema_length=None).sort(
        'overlap_type',
        'batch_fraction_pct',
        'replicate',
        'left_strategy',
        'right_strategy',
    )


def build_novelty_tables(
    results_dir: Path = RESULTS_DIR,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Calculate scaffold novelty, seed-similarity CDFs, and method overlap."""
    contexts = discover_runs(results_dir)
    novelty_records: list[dict[str, Any]] = []
    cdf_records: list[dict[str, Any]] = []
    overlap_records: list[dict[str, Any]] = []

    for replicate in EXPECTED_REPLICATES:
        seed_frame, _ = _validate_matched_seed(contexts, replicate)
        seed_scaffolds, seed_fingerprints = molecular_descriptors(
            seed_frame['SMILES'].to_list()
        )
        focus_ids: dict[str, set[str]] = {}

        for context in [c for c in contexts if c.replicate == replicate]:
            ranked = top_candidates(load_selection(context), max(TOP_K_VALUES))
            scaffolds, fingerprints = molecular_descriptors(ranked['SMILES'].to_list())
            for k in TOP_K_VALUES:
                novelty_records.append(
                    {
                        'run_id': context.run_id,
                        'strategy': context.strategy,
                        'replicate': replicate,
                        'top_k': k,
                        'novel_scaffolds': novel_scaffold_count(
                            scaffolds[:k], seed_scaffolds
                        ),
                    }
                )

            focus_ids[context.strategy] = set(ranked['ID'][:FOCUS_K].to_list())
            similarities = max_similarity_to_reference(
                fingerprints[:FOCUS_K], seed_fingerprints
            )
            cdf = empirical_cdf(similarities, ECDF_GRID)
            cdf_records.extend(
                {
                    'run_id': context.run_id,
                    'strategy': context.strategy,
                    'replicate': replicate,
                    'similarity': float(threshold),
                    'cumulative_percent': float(percent),
                }
                for threshold, percent in zip(ECDF_GRID, cdf, strict=True)
            )

        for left in STRATEGY_ORDER:
            for right in STRATEGY_ORDER:
                overlap_records.append(
                    {
                        'replicate': replicate,
                        'left_strategy': left,
                        'right_strategy': right,
                        'overlap_percent': exact_overlap_percent(
                            focus_ids[left], focus_ids[right]
                        ),
                    }
                )

    return (
        pl.from_dicts(novelty_records, infer_schema_length=None),
        pl.from_dicts(cdf_records, infer_schema_length=None),
        pl.from_dicts(overlap_records, infer_schema_length=None),
    )

"""Load the publication benchmark runs into one tidy table.

Reading run directories takes minutes; reading the tidy parquet takes
milliseconds, so restyling all six figures is cheap. The parquet is also the
artifact referenced by the paper's Data Availability statement.

Build it once:   python publication/scripts/data.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

RESULTS_DIR = Path.home() / 'LearnM8_RESULTS_FINAL'
ARCHIVE_DIR = RESULTS_DIR / '_archive'
TIDY = Path(__file__).resolve().parents[1] / 'data' / 'benchmark_tidy.parquet'

# The manifest's `learner` column is derived from the directory name and is
# wrong for the GPU rows (L-09 rf_fil reads as "rf", L-10 ridge_cuml as
# "ridge"), so learner identity comes from the family code instead.
FAMILY_LEARNER = {
    'L-01': 'rf',
    'L-03': 'xgb',
    'L-04': 'mc_dropout',
    'L-05': 'chemprop',
    'L-06': 'dt',
    'L-07': 'fastprop',
    'L-08': 'lr',
    'L-09': 'rf_fil',
    'L-10': 'ridge_cuml',
    'L-11': 'svgp',
    # AG-01..AG-08 mirror the A-01..A-08 acquisition sweep on svgp rather than rf,
    # so the uncertainty-driven strategies are measured through a real predictive
    # posterior. They need entries here for the same reason the L-* rows do: the
    # default below is 'rf', so an unmapped family is silently labelled rf.
    'AG-01': 'svgp',
    'AG-02': 'svgp',
    'AG-03': 'svgp',
    'AG-04': 'svgp',
    'AG-05': 'svgp',
    'AG-06': 'svgp',
    'AG-07': 'svgp',
    'AG-08': 'svgp',
}
# The S-04 family screens 96,214,206 compounds; it was previously labelled "100M",
# which overstated the pool by 3.9%. There is no 100M run.
#
# d4-116M is the scored subset of the 138,312,677-row D4 table: 22,071,493 rows
# carry an empty `dockscore` and CSVOracle rejects a selected row with a null
# target, so the publication pool is the 116,241,184 rows that have one.
POOL_SIZE = {
    '1M': 1_000_000,
    '10M': 10_000_000,
    '50M': 50_000_000,
    '96M': 96_214_206,
    'd4-840K': 840_670,
    'd4-116M': 116_241_184,
}

TIME_COLS = [
    'training_time',
    'prediction_time',
    'acquisition_time',
    'feature_extraction_time',
]


def build(results_dir: Path = RESULTS_DIR, out: Path = TIDY) -> pl.DataFrame:
    """Read every publication run's cycle metrics and write the tidy parquet."""
    manifest = pl.read_csv(results_dir / 'manifest.csv', infer_schema_length=None)

    # The manifest is the sole index of publication data. Check the bijection in
    # both directions so a run can never be silently plotted or silently dropped.
    listed = set(manifest['run_id'].to_list())
    on_disk = {
        path.name
        for path in results_dir.glob('lm8_*')
        if path.is_dir() and not path.name.endswith('_FAILED')
    }
    if unlisted := on_disk - listed:
        raise ValueError(
            f'{len(unlisted)} run directories are absent from the manifest: '
            f'{sorted(unlisted)[:5]}'
        )
    if missing := listed - on_disk:
        raise FileNotFoundError(
            f'{len(missing)} manifest runs have no directory: {sorted(missing)[:5]}'
        )

    frames = []
    for run_id in manifest['run_id']:
        path = results_dir / run_id / 'cycle_metrics.csv'
        if not path.exists():
            raise FileNotFoundError(f'{run_id}: no cycle_metrics.csv at {path}')
        frames.append(
            pl.read_csv(path, infer_schema_length=None, null_values=[''])
            # cycle_metrics.strategy is the strategy that ran in THAT cycle
            # (cycle 0 is always "random"); the manifest's `strategy` is the
            # run's label. Renaming keeps the join from silently suffixing one.
            .rename({'strategy': 'cycle_strategy'})
            .with_columns(run_id=pl.lit(run_id))
        )
    # Column sets differ between waves (pre-026 runs lack selection_path), so
    # concatenate diagonally and let absent columns become nulls.
    metrics = pl.concat(frames, how='diagonal_relaxed')

    df = metrics.join(manifest, on='run_id', how='left').with_columns(
        learner=pl.col('family').replace_strict(FAMILY_LEARNER, default='rf'),
        pool_size=pl.col('pool').replace_strict(POOL_SIZE, default=None),
        batch_fraction=pl.col('batch_fraction_pct').cast(pl.Float64),
        prune_fraction=pl.col('prune_pct').cast(pl.Float64).fill_null(0.0),
        compounds_evaluated=pl.col('cumulative_labeled'),
        ml_time=pl.col('training_time') + pl.col('prediction_time'),
        # Seed design is a property of the run, not of where it came from. At a
        # 1.0% batch the two rules coincide (seed = 1% of pool = one batch), so
        # those runs legitimately belong to both sets and are shared between them.
        is_matched=pl.col('init_pct') == pl.col('batch_fraction_pct'),
        is_fixed=pl.col('init_pct') == 1.0,
    )
    df = df.sort('run_id', 'cycle').with_columns(
        [
            pl.col(c).cum_sum().over('run_id').alias(f'cum_{c}')
            for c in [*TIME_COLS, 'ml_time', 'total_time']
        ]
    )

    collisions = [c for c in df.columns if c.endswith('_right')]
    if collisions:
        raise ValueError(
            f'manifest/metrics column collision: {collisions} - rename before joining'
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f'wrote {out} ({df.height} rows, {len(frames)} runs)')
    return df


DESIGNS = ('matched', 'fixed')


def load(design: str | None = None) -> pl.DataFrame:
    """Load the tidy table, optionally restricted to one cycle-0 seed design.

    `matched` seeds cycle 0 with exactly one acquisition batch; `fixed` seeds it
    with 1% of the pool regardless of batch size. Runs at a 1.0% batch satisfy
    both and appear in either set. `None` returns every run.
    """
    if not TIDY.exists():
        raise FileNotFoundError(
            f'{TIDY} missing - run `python {Path(__file__).name}` first'
        )
    df = pl.read_parquet(TIDY)
    if design is None:
        return df
    if design not in DESIGNS:
        raise ValueError(f'design must be one of {DESIGNS}, got {design!r}')
    selected = df.filter(pl.col(f'is_{design}'))
    if selected.is_empty():
        raise ValueError(f'no runs with design={design!r}')
    return selected


def design_suffix(design: str) -> str:
    """Figure-name suffix: matched is the main-text set and owns the plain name."""
    return '' if design == 'matched' else '_fixedinit'


def design_arg(description: str | None = None) -> str:
    """Parse the shared --design flag used by the design-aware figure scripts."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--design', choices=DESIGNS, default='matched')
    return parser.parse_args().design


def replicate_note(curves: pl.DataFrame, group: str) -> str:
    """Describe replicate counts, calling out any group that falls short."""
    counts = curves.group_by(group).agg(pl.col('n_reps').max().alias('n')).sort(group)
    common = counts['n'].mode().max()
    short = counts.filter(pl.col('n') != common)
    note = f'n = {common}'
    if not short.is_empty():
        detail = ', '.join(
            f'{r[group]} n = {r["n"]}' for r in short.iter_rows(named=True)
        )
        note += f' ({detail})'
    return note


def curve(df: pl.DataFrame, group: list[str], x: str, y: str) -> pl.DataFrame:
    """Aggregate replicates per cycle: mean x, and mean/min/max y."""
    return (
        df.filter(pl.col(y).is_not_null())
        .group_by([*group, 'cycle'])
        .agg(
            pl.col(x).mean().alias(x),
            pl.col(y).mean().alias('mean'),
            pl.col(y).min().alias('lo'),
            pl.col(y).max().alias('hi'),
            pl.len().alias('n_reps'),
        )
        .sort([*group, 'cycle'])
    )


def final(df: pl.DataFrame, group: list[str], cols: list[str]) -> pl.DataFrame:
    """Last-cycle value of `cols` per run, aggregated over replicates."""
    last = df.filter(pl.col('cycle') == pl.col('cycle').max().over('run_id'))
    return (
        last.group_by(group)
        .agg(
            [
                expr
                for c in cols
                for expr in (
                    pl.col(c).mean().alias(f'{c}_mean'),
                    pl.col(c).min().alias(f'{c}_lo'),
                    pl.col(c).max().alias(f'{c}_hi'),
                )
            ]
        )
        .sort(group)
    )


def check_fingerprint(df: pl.DataFrame, context: str) -> None:
    """Warn when compared runs used different fingerprints (diversity metrics
    are featurizer-dependent and not comparable across them)."""
    used = df['fingerprint_used'].drop_nulls().unique().to_list()
    if len(used) > 1:
        print(
            f'WARNING [{context}]: mixed fingerprint_used {used} - diversity values not comparable'
        )


if __name__ == '__main__':
    build()

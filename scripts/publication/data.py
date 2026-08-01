"""Load the benchmark runs into one tidy table that every figure script reads.

Reading 170 run directories takes minutes; reading the tidy parquet takes
milliseconds, so restyling all six figures is cheap. The parquet is also the
artifact referenced by the paper's Data Availability statement.

Build it once:   python scripts/publication/data.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

RESULTS_DIR = Path.home() / 'LearnM8_RESULTS_FINAL'
TIDY = Path(__file__).resolve().parent / 'benchmark_tidy.parquet'

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
}
POOL_SIZE = {
    '1M': 1_000_000,
    '10M': 10_000_000,
    '50M': 50_000_000,
    '100M': 100_000_000,
    'd4-840K': 840_000,
}

TIME_COLS = [
    'training_time',
    'prediction_time',
    'acquisition_time',
    'feature_extraction_time',
]


def build(results_dir: Path = RESULTS_DIR, out: Path = TIDY) -> pl.DataFrame:
    """Read every run's cycle_metrics.csv, join the manifest, write the parquet."""
    manifest = pl.read_csv(results_dir / 'manifest.csv', infer_schema_length=None)

    frames = []
    for run_id in manifest['run_id']:
        path = results_dir / run_id / 'cycle_metrics.csv'
        if not path.exists():
            print(f'skip {run_id}: no cycle_metrics.csv')
            continue
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


def load() -> pl.DataFrame:
    if not TIDY.exists():
        raise FileNotFoundError(
            f'{TIDY} missing - run `python {Path(__file__).name}` first'
        )
    return pl.read_parquet(TIDY)


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

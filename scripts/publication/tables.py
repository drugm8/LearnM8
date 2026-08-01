"""T2 - benchmark results summary, plus the full per-run table for the SI.

T1 (the capability matrix of learners, acquisitions and featurizers) is
hand-authored from the registries; it is not derived from run data.
"""

from __future__ import annotations

import data
import polars as pl
import style

Y = 'top_0_1_pct_discovery'
METRICS = [Y, 'selected_percentile', 'cum_ml_time']
TABLE_DIR = style.FIGURE_DIR.parent / 'tables'


def main() -> None:
    df = data.load()
    group = [
        'family',
        'learner',
        'pool',
        'strategy',
        'batch_fraction',
        'prune_fraction',
    ]

    summary = data.final(df, group, METRICS).with_columns(
        n_reps=pl.col('family').count().over(group)
    )
    summary = summary.with_columns(
        [
            pl.format(
                '{} [{}, {}]',
                pl.col(f'{m}_mean').round(2),
                pl.col(f'{m}_lo').round(2),
                pl.col(f'{m}_hi').round(2),
            ).alias(m)
            for m in METRICS
        ]
    ).select([*group, *METRICS])

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.write_csv(TABLE_DIR / 'table2_benchmark_summary.csv')
    (TABLE_DIR / 'table2_benchmark_summary.md').write_text(
        'Values are final-cycle mean [min, max] across replicates.\n\n'
        + summary.to_pandas().to_markdown(index=False)
    )

    df.write_csv(TABLE_DIR / 'si_all_runs_per_cycle.csv')
    print(
        f'wrote {TABLE_DIR}/table2_benchmark_summary.{{csv,md}} ({summary.height} conditions)'
    )
    print(f'wrote {TABLE_DIR}/si_all_runs_per_cycle.csv ({df.height} rows)')


if __name__ == '__main__':
    main()

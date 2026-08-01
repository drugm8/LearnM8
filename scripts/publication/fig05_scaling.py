"""F5 - compute characterization: pool scale and GPU acceleration.

A-C  scaling: 1M -> 10M -> 50M (-> 100M when the S-04 run lands), rf + greedy
     + morgan at a 1.0% batch, one replicate each. Characterization, not
     comparison: no error bars are available and none are implied.
D-E  GPU pairs at 1.0%: rf vs rf_fil (cuML) and lr vs ridge_cuml, 3 replicates.
"""

from __future__ import annotations

import data
import numpy as np
import polars as pl
import style

Y = 'top_0_1_pct_discovery'
Y2 = 'top_100_discovery'
SCALE_FAMILIES = {
    'PILOT-1M': 1_000_000,
    'S-02': 10_000_000,
    'S-03': 50_000_000,
    'S-04': 100_000_000,
}
BREAKDOWN = [
    'cum_feature_extraction_time',
    'cum_training_time',
    'cum_prediction_time',
    'cum_acquisition_time',
]
BREAKDOWN_LABELS = ['featurization', 'training', 'prediction', 'acquisition']
GPU_PAIRS = [('rf', 'rf_fil'), ('lr', 'ridge_cuml')]


def main() -> None:
    style.apply()
    df = data.load()

    scales = df.filter(pl.col('family').is_in(list(SCALE_FAMILIES)))
    present = [f for f in SCALE_FAMILIES if f in scales['family'].unique().to_list()]
    for missing in set(SCALE_FAMILIES) - set(present):
        print(
            f'note: {missing} ({SCALE_FAMILIES[missing]:,} compounds) not on disk - omitted'
        )
    colors = {
        f: style.SEQUENTIAL(v)
        for f, v in zip(present, np.linspace(0.35, 1.0, len(present)), strict=True)
    }

    fig, axd = style.mosaic('ABC\nDDE', panel_h_mm=52)

    totals = data.final(scales, ['family'], ['cum_total_time'])
    totals = totals.with_columns(
        pool_size=pl.col('family').replace_strict(SCALE_FAMILIES, return_dtype=pl.Int64)
    ).sort('pool_size')

    ax = axd['A']
    ax.plot(
        totals['pool_size'],
        totals['cum_total_time_mean'],
        marker='o',
        color=style.PRIMARY,
    )
    ax.set(
        xscale='log',
        yscale='log',
        xlabel='pool size (compounds)',
        ylabel='total wall-clock (s)',
    )
    style.compact_axis(ax)

    ax = axd['B']
    parts = data.final(scales, ['family'], BREAKDOWN)
    parts = parts.with_columns(
        pool_size=pl.col('family').replace_strict(SCALE_FAMILIES, return_dtype=pl.Int64)
    ).sort('pool_size')
    means = np.array([parts[f'{c}_mean'].fill_null(0).to_numpy() for c in BREAKDOWN])
    shares = means / means.sum(axis=0) * 100
    bottom = np.zeros(parts.height)
    scale_labels = [f'{n // 1_000_000}M' for n in parts['pool_size']]
    for share, label, color in zip(
        shares, BREAKDOWN_LABELS, style.CATEGORICAL[: len(BREAKDOWN)], strict=True
    ):
        ax.bar(scale_labels, share, bottom=bottom, label=label, color=color)
        bottom += share
    ax.set_xlabel('pool size')
    ax.set_ylabel('share of total time (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncols=2)

    ax = axd['C']
    for family in present:
        s = scales.filter(pl.col('family') == family).sort('cycle')
        ax.plot(
            s['cycle'],
            s[Y],
            color=colors[family],
            label=f'{SCALE_FAMILIES[family]:,}'.replace(',', ' '),
        )
        ax.plot(s['cycle'], s[Y2], color=colors[family], linestyle='--')
    ax.set(xlabel='cycle', ylabel='discovery (%)', ylim=(0, 100))
    ax.legend(title='pool size', loc='lower right', fontsize=6)
    ax.text(
        0.03,
        0.95,
        'solid: top-0.1%\ndashed: top-100',
        transform=ax.transAxes,
        va='top',
        fontsize=6,
    )

    gpu = df.filter(
        (pl.col('batch_fraction') == 1.0)
        & pl.col('learner').is_in([x for p in GPU_PAIRS for x in p])
    )
    speed = data.final(gpu, ['learner'], ['cum_ml_time', Y]).to_dicts()
    speed = {row['learner']: row for row in speed}

    ax = axd['D']
    x = np.arange(len(GPU_PAIRS))
    for offset, kind in ((-0.18, 0), (0.18, 1)):
        ax.bar(
            x + offset,
            [speed[p[kind]]['cum_ml_time_mean'] for p in GPU_PAIRS],
            width=0.34,
            color=style.PRIMARY if kind == 0 else '#E69F00',
            label='CPU' if kind == 0 else 'GPU',
            yerr=[
                [
                    speed[p[kind]]['cum_ml_time_mean']
                    - speed[p[kind]]['cum_ml_time_lo']
                    for p in GPU_PAIRS
                ],
                [
                    speed[p[kind]]['cum_ml_time_hi']
                    - speed[p[kind]]['cum_ml_time_mean']
                    for p in GPU_PAIRS
                ],
            ],
            error_kw={'elinewidth': 0.6, 'ecolor': style.INK},
        )
    for i, (cpu, gpu_learner) in enumerate(GPU_PAIRS):
        factor = speed[cpu]['cum_ml_time_mean'] / speed[gpu_learner]['cum_ml_time_mean']
        ax.annotate(
            f'{factor:.1f}x',
            (
                i,
                max(
                    speed[cpu]['cum_ml_time_mean'],
                    speed[gpu_learner]['cum_ml_time_mean'],
                ),
            ),
            textcoords='offset points',
            xytext=(0, 4),
            ha='center',
            fontsize=7,
        )
    ax.set_xticks(x, [f'{c} / {g}' for c, g in GPU_PAIRS])
    ax.set_ylabel(style.LABELS['cum_ml_time'])
    ax.legend(loc='upper left')

    ax = axd['E']
    deltas = [speed[g][f'{Y}_mean'] - speed[c][f'{Y}_mean'] for c, g in GPU_PAIRS]
    ax.bar(
        x,
        deltas,
        width=0.5,
        color=[style.PRIMARY if abs(d) < 5 else '#D55E00' for d in deltas],
    )
    ax.axhline(0, color=style.INK, linewidth=0.6)
    ax.set_xticks(x, [g for _, g in GPU_PAIRS])
    ax.set_ylabel(f'Δ final {style.LABELS[Y]}\n(GPU − CPU)')
    limit = max(6.0, 1.2 * max(abs(d) for d in deltas))
    ax.set_ylim(-limit, limit)
    ax.axhspan(-5, 5, color=style.GRID, alpha=0.5, linewidth=0, zorder=0)

    style.label_panels(axd)
    fig.tight_layout()
    style.save(fig, 'fig05_scaling')


if __name__ == '__main__':
    main()

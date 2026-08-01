"""F4 - batch size and pruning frontiers (10M AmpC, rf, greedy, 3 replicates).

A-C  batch family: every condition screens exactly 500,000 compounds, split
     into 1 to 100 batches, so differences are purely retraining frequency
D-E  pruning family: all screen 1,000,000 compounds; C-01 is the 0% baseline
     (B-02 is NOT - it screens 500,000 over 5 cycles)
"""

from __future__ import annotations

import data
import numpy as np
import polars as pl
import style

Y = 'top_0_1_pct_discovery'
X = 'compounds_evaluated'
BATCH_FAMILIES = ['B-01', 'B-02', 'B-03', 'B-04', 'B-05', 'B-06']
PRUNE_FAMILIES = ['C-01', 'P-02', 'P-03', 'P-04', 'P-05']


def batch_colors(fractions: list[float]) -> dict[float, str]:
    """Sequential purple ramp ordered by batch size (log-spaced conditions)."""
    ranks = np.linspace(0.25, 1.0, len(fractions))
    return {
        f: style.SEQUENTIAL(r) for f, r in zip(sorted(fractions), ranks, strict=True)
    }


def main() -> None:
    style.apply()
    df = data.load()
    batch = df.filter(pl.col('family').is_in(BATCH_FAMILIES))
    prune = df.filter(pl.col('family').is_in(PRUNE_FAMILIES))

    fractions = sorted(batch['batch_fraction'].unique().to_list())
    colors = batch_colors(fractions)

    fig, axd = style.mosaic('ABC\nDDE', panel_h_mm=52)

    ax = axd['A']
    curves = data.curve(batch, ['batch_fraction'], X, Y)
    for frac in fractions:
        s = curves.filter(pl.col('batch_fraction') == frac)
        # Conditions run 2-101 cycles; thin the markers so the 101-cycle curve
        # stays a line rather than a solid band of dots.
        ax.plot(
            s[X],
            s['mean'],
            color=colors[frac],
            label=f'{frac}%',
            markevery=max(1, s.height // 12),
        )
        style.band(ax, s[X], s['lo'], s['hi'], colors[frac])
    ax.set(xlabel=style.LABELS[X], ylabel=style.LABELS[Y])
    style.compact_axis(ax)
    ax.legend(
        title='batch fraction', loc='lower right', ncols=2, fontsize=6, title_fontsize=6
    )

    # Cycle 0 is a random seed batch with no model, so retrains = max cycle.
    retrains = batch.group_by('batch_fraction').agg(
        pl.col('cycle').max().alias('retrains')
    )
    summary = data.final(batch, ['batch_fraction'], [Y, 'cum_ml_time']).join(
        retrains, on='batch_fraction'
    )

    summary = summary.sort('retrains')

    ax = axd['B']
    ax.errorbar(
        summary['retrains'],
        summary[f'{Y}_mean'],
        yerr=[
            summary[f'{Y}_mean'] - summary[f'{Y}_lo'],
            summary[f'{Y}_hi'] - summary[f'{Y}_mean'],
        ],
        marker='o',
        color=style.PRIMARY,
        elinewidth=0.6,
    )
    ax.set(xscale='log', xlabel='number of retrains', ylabel=f'final {style.LABELS[Y]}')

    ax = axd['C']
    for row in summary.iter_rows(named=True):
        ax.plot(
            row['cum_ml_time_mean'],
            row[f'{Y}_mean'],
            marker='o',
            color=colors[row['batch_fraction']],
        )
        ax.annotate(
            f'{row["batch_fraction"]}%',
            (row['cum_ml_time_mean'], row[f'{Y}_mean']),
            textcoords='offset points',
            xytext=(4, -6),
            fontsize=6,
        )
    ax.set(
        xscale='log',
        xlabel=style.LABELS['cum_ml_time'],
        ylabel=f'final {style.LABELS[Y]}',
    )
    ax.margins(x=0.18, y=0.12)

    if prune.filter(pl.col('prune_fraction') > 0)['pruned_count'].sum() == 0:
        # The P runs requested pruning but pruned_count is 0 in every cycle:
        # parse_cycle_schedule drops the top-level pruning_strategy whenever an
        # explicit cycles=[...] schedule is given, so pruning silently no-ops.
        # Plotting these would show a flat line that reads as "pruning is free".
        for letter in 'DE':
            axd[letter].text(
                0.5,
                0.5,
                'pruning did not execute in P-02…P-05\n(pruned_count = 0 in all cycles)\npanels pending re-run',
                transform=axd[letter].transAxes,
                ha='center',
                va='center',
                fontsize=7,
                color=style.MUTED,
            )
            axd[letter].set_axis_off()
        print('ERROR: P family contains no pruning - panels D/E suppressed')
        style.label_panels(axd)
        fig.tight_layout()
        style.save(fig, 'fig04_batch_pruning')
        return

    pruned = data.final(prune, ['prune_fraction'], [Y, 'cum_prediction_time']).sort(
        'prune_fraction'
    )
    baseline = pruned.filter(pl.col('prune_fraction') == 0)[f'{Y}_mean'].item()

    ax = axd['D']
    ax.errorbar(
        pruned['prune_fraction'],
        pruned[f'{Y}_mean'],
        yerr=[
            pruned[f'{Y}_mean'] - pruned[f'{Y}_lo'],
            pruned[f'{Y}_hi'] - pruned[f'{Y}_mean'],
        ],
        marker='o',
        color=style.PRIMARY,
        elinewidth=0.6,
    )
    ax.axhspan(0.95 * baseline, baseline, color=style.LIGHT, alpha=0.25, linewidth=0)
    ax.annotate(
        '95% of unpruned',
        (pruned['prune_fraction'].max(), 0.95 * baseline),
        textcoords='offset points',
        xytext=(-4, -10),
        ha='right',
        fontsize=6,
    )
    ax.set(xlabel='pruning fraction per cycle (%)', ylabel=f'final {style.LABELS[Y]}')

    ax = axd['E']
    ax.bar(
        pruned['prune_fraction'].cast(pl.String),
        pruned['cum_prediction_time_mean'],
        yerr=[
            pruned['cum_prediction_time_mean'] - pruned['cum_prediction_time_lo'],
            pruned['cum_prediction_time_hi'] - pruned['cum_prediction_time_mean'],
        ],
        color=[style.SEQUENTIAL(v) for v in np.linspace(0.25, 1.0, pruned.height)],
        error_kw={'elinewidth': 0.6, 'ecolor': style.INK},
    )
    ax.set(xlabel='pruning fraction (%)', ylabel=style.LABELS['cum_prediction_time'])

    style.label_panels(axd)
    fig.tight_layout()
    style.save(fig, 'fig04_batch_pruning')


if __name__ == '__main__':
    main()

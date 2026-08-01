"""F3 - acquisition strategies (1M AmpC, rf, morgan, 3 replicates).

A-C  discovery curves at each batch fraction
D    selected percentile - does informed acquisition actually pick the top?
E    cumulative acquisition time - what that selection costs

`greedy` comes from L-01, which is the same configuration as the A family.
Caveat: A-01 (random) seeds cycle 0 with `batch_fraction` of the pool while
every other family seeds a fixed 10,000, so its 0.1% and 0.5% arms span a
different budget window. The compounds-evaluated x-axis keeps the comparison
honest; only the 1.0% facet has a fully aligned random baseline.
"""

from __future__ import annotations

import data
import polars as pl
import style
from matplotlib.lines import Line2D

Y = 'top_0_1_pct_discovery'
X = 'compounds_evaluated'
FRACTIONS = [0.1, 0.5, 1.0]
FAMILIES = ['A-01', 'A-03', 'A-04', 'A-05', 'L-01']
STRATEGIES = ['random', 'greedy', 'ucb', 'ei', 'thompson']


def main() -> None:
    style.apply()
    df = data.load().filter(pl.col('family').is_in(FAMILIES))

    fig, axd = style.mosaic('ABC\nDDE', panel_h_mm=52)

    for letter, frac in zip('ABC', FRACTIONS, strict=True):
        ax = axd[letter]
        curves = data.curve(
            df.filter(pl.col('batch_fraction') == frac), ['strategy'], X, Y
        )
        for strategy in STRATEGIES:
            s = curves.filter(pl.col('strategy') == strategy)
            color = style.STRATEGY_COLOR[strategy]
            ax.plot(s[X], s['mean'], color=color)
            style.band(ax, s[X], s['lo'], s['hi'], color)
        ax.set(title=f'batch = {frac}%', xlabel=style.LABELS[X], ylim=(0, 100))
        style.compact_axis(ax)
    axd['A'].set_ylabel(style.LABELS[Y])
    for letter in 'BC':
        axd[letter].set_yticklabels([])

    ax = axd['D']
    percentile = data.curve(
        df.filter(pl.col('batch_fraction') == 0.1),
        ['strategy'],
        X,
        'selected_percentile',
    )
    for strategy in STRATEGIES:
        s = percentile.filter(pl.col('strategy') == strategy)
        color = style.STRATEGY_COLOR[strategy]
        ax.plot(s[X], s['mean'], color=color, marker='o')
        style.band(ax, s[X], s['lo'], s['hi'], color)
    ax.axhline(50, color=style.INK, linewidth=0.6, linestyle=':')
    ax.set(
        xlabel=style.LABELS[X],
        ylabel=style.LABELS['selected_percentile'],
        title='batch = 0.1%',
        ylim=(0, 100),
    )
    style.compact_axis(ax)

    ax = axd['E']
    cost = data.final(
        df.filter(pl.col('batch_fraction') == 1.0),
        ['strategy'],
        ['cum_acquisition_time'],
    )
    order = [s for s in STRATEGIES if s in cost['strategy'].to_list()]
    cost = cost.sort(
        pl.col('strategy').replace_strict({s: i for i, s in enumerate(order)})
    )
    ax.bar(
        cost['strategy'],
        cost['cum_acquisition_time_mean'],
        yerr=[
            cost['cum_acquisition_time_mean'] - cost['cum_acquisition_time_lo'],
            cost['cum_acquisition_time_hi'] - cost['cum_acquisition_time_mean'],
        ],
        color=[style.STRATEGY_COLOR[s] for s in cost['strategy']],
        error_kw={'elinewidth': 0.6, 'ecolor': style.INK},
    )
    ax.set_ylabel('cumulative acquisition time (s)\n(1.0% batch)')
    ax.tick_params(axis='x', rotation=45)

    style.label_panels(axd)
    fig.legend(
        handles=[
            Line2D([], [], color=style.STRATEGY_COLOR[s], label=s) for s in STRATEGIES
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.04),
        ncols=5,
    )
    fig.tight_layout()
    style.save(fig, 'fig03_acquisition')


if __name__ == '__main__':
    main()

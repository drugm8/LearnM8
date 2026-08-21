"""F3 - acquisition strategies (1M AmpC, rf, morgan).

A-C  discovery curves at each batch fraction

The complete A-family reruns provide random, greedy, UCB, EI, Thompson, PI and
entropy. Renders one figure per cycle-0 seed design; see `data.load`. Panels follow
the batch fractions present in the data rather than a hardcoded list.
"""

from __future__ import annotations

import data
import polars as pl
import style
from matplotlib.lines import Line2D

Y = 'top_0_1_pct_discovery'
X = 'compounds_evaluated'
# A-08 (simulated annealing) is the initmatch_v2 wave; A-02/A-06/A-07 gained their
# init-matched sub-1% arms in the same wave, which removed the earlier need to source
# greedy from L-01 as a stand-in. Every strategy is now its own family in both designs.
FAMILIES = ['A-01', 'A-02', 'A-03', 'A-04', 'A-05', 'A-06', 'A-07', 'A-08']
STRATEGIES = [
    'random',
    'greedy',
    'ucb',
    'ei',
    'thompson',
    'pi',
    'entropy',
    'simulated_annealing',
]


def main() -> None:
    design = data.design_arg(__doc__)
    style.apply()
    df = data.load(design).filter(pl.col('family').is_in(FAMILIES))

    fractions = sorted(df['batch_fraction'].unique().to_list())
    letters = 'ABCDEF'[: len(fractions)]
    fig, axd = style.mosaic(letters, panel_h_mm=72)

    curve_markers = {
        strategy: style.CURVE_MARKERS[index % len(style.CURVE_MARKERS)]
        for index, strategy in enumerate(STRATEGIES)
    }
    for letter, frac in zip(letters, fractions, strict=True):
        ax = axd[letter]
        curves = data.curve(
            df.filter(pl.col('batch_fraction') == frac), ['strategy'], X, Y
        )
        for strategy in STRATEGIES:
            s = curves.filter(pl.col('strategy') == strategy)
            color = style.STRATEGY_COLOR[strategy]
            ax.plot(
                s[X],
                s['mean'],
                color=color,
                linestyle='-',
                linewidth=style.DATA_LINEWIDTH,
                marker=curve_markers[strategy],
                markevery=max(1, s.height // 10),
            )
            style.band(ax, s[X], s['lo'], s['hi'], color, alpha=0.08)
        ax.set(title=f'batch = {frac}%', xlabel=style.LABELS[X], ylim=(0, 100))
        # Bands are min-max over replicates, so the replicate count must be stated.
        # The legend is figure-level and lists every strategy, so a panel missing
        # some must say so or the legend implies coverage this panel does not have.
        note = data.replicate_note(curves, 'strategy')
        absent = [s for s in STRATEGIES if s not in curves['strategy'].to_list()]
        if absent:
            note += '\nnot run at this batch: ' + ', '.join(absent)
        # Upper left: discovery curves rise left-to-right, so this corner is the
        # one reliably free of data in every panel.
        ax.text(
            0.03,
            0.97,
            note,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=5.5,
            color=style.MUTED,
            bbox={
                'facecolor': style.BACKGROUND,
                'edgecolor': 'none',
                'alpha': 0.85,
                'pad': 1.5,
            },
        )
        style.compact_axis(ax)
        pool_size = int(
            df.filter(pl.col('batch_fraction') == frac)['pool_size'].unique().item()
        )
        style.set_compound_axis(
            ax,
            pool_size,
            ticks=curves[X].unique().sort().to_list(),
            rotation=45,
        )
    axd['A'].set_ylabel(style.LABELS[Y])
    for letter in letters[1:]:
        axd[letter].set_yticklabels([])

    style.label_panels(axd)
    fig.legend(
        handles=[
            Line2D([], [], color=style.STRATEGY_COLOR[s], label=s) for s in STRATEGIES
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.18),
        ncols=4,
    )
    style.save(fig, f'fig03_acquisition{data.design_suffix(design)}')


if __name__ == '__main__':
    main()

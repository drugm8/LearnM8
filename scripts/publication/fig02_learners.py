"""F2 - learner comparison (L family, 1M AmpC, greedy, 3 replicates).

A-C  discovery curves at each batch fraction
D    final discovery vs cumulative ML time (efficiency frontier), one labelled
     trajectory per learner across the three batch fractions
"""

from __future__ import annotations

import data
import polars as pl
import style
from matplotlib.lines import Line2D

Y = 'top_0_1_pct_discovery'
X = 'compounds_evaluated'
FRACTIONS = [0.1, 0.5, 1.0]
MARKERS = {0.1: 'o', 0.5: 's', 1.0: '^'}
LEARNERS = list(style.LEARNER_FAMILY)


def main() -> None:
    style.apply()
    df = data.load().filter(pl.col('family').is_in(list(data.FAMILY_LEARNER)))

    fig, axd = style.mosaic('ABC\nDDD', panel_h_mm=52)

    for letter, frac in zip('ABC', FRACTIONS, strict=True):
        ax = axd[letter]
        curves = data.curve(
            df.filter(pl.col('batch_fraction') == frac), ['learner'], X, Y
        )
        for learner in LEARNERS:
            s = curves.filter(pl.col('learner') == learner)
            color, ls = style.learner_style(learner)
            ax.plot(s[X], s['mean'], color=color, linestyle=ls)
            style.band(ax, s[X], s['lo'], s['hi'], color)
        ax.set_title(f'batch = {frac}%', pad=3)
        ax.set_xlabel(style.LABELS[X])
        ax.set_ylim(0, 100)
        style.compact_axis(ax)
    axd['A'].set_ylabel(style.LABELS[Y])
    for letter in 'BC':
        axd[letter].set_yticklabels([])

    # One trajectory per learner across the three batch fractions, labelled at
    # its most expensive point: colour alone cannot identify ten learners.
    ax = axd['D']
    frontier = data.final(df, ['learner', 'batch_fraction'], [Y, 'cum_ml_time'])
    for index, learner in enumerate(LEARNERS):
        s = frontier.filter(pl.col('learner') == learner).sort('cum_ml_time_mean')
        color, ls = style.learner_style(learner)
        ax.plot(
            s['cum_ml_time_mean'],
            s[f'{Y}_mean'],
            color=color,
            linestyle=ls,
            linewidth=0.8,
            marker='none',
        )
        for row in s.iter_rows(named=True):
            ax.plot(
                row['cum_ml_time_mean'],
                row[f'{Y}_mean'],
                marker=MARKERS[row['batch_fraction']],
                color=color,
                markersize=4,
            )
        # Final points cluster at 93-99%, so stagger the labels vertically.
        ax.annotate(
            learner,
            (s['cum_ml_time_mean'][-1], s[f'{Y}_mean'][-1]),
            textcoords='offset points',
            xytext=(5, (4, -9, 13, -18)[index % 4]),
            fontsize=6,
            color=color,
        )
    ax.set(
        xscale='log',
        xlabel=style.LABELS['cum_ml_time'],
        ylabel=f'final {style.LABELS[Y]}',
    )
    ax.margins(x=0.12)
    ax.legend(
        handles=[
            Line2D([], [], marker=m, color=style.INK, linestyle='none', label=f'{f}%')
            for f, m in MARKERS.items()
        ],
        title='batch fraction',
        loc='center left',
        ncols=1,
    )

    style.label_panels(axd)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=style.learner_style(x)[0],
                linestyle=style.learner_style(x)[1],
                label=x,
            )
            for x in LEARNERS
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.06),
        ncols=5,
    )
    fig.tight_layout()
    style.save(fig, 'fig02_learners')


if __name__ == '__main__':
    main()

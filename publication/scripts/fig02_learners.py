"""F2 - learner comparison (L family, 1M AmpC, greedy).

A-C  discovery curves at each batch fraction

Renders one figure per cycle-0 seed design; see `data.load`. Panels are built from
the batch fractions actually present, so the layout follows the data rather than a
hardcoded list.
"""

from __future__ import annotations

import data
import polars as pl
import style
from matplotlib.lines import Line2D

Y = 'top_0_1_pct_discovery'
X = 'compounds_evaluated'
LEARNERS = list(style.LEARNER_FAMILY)
MODEL_COLORS = dict(
    zip(
        LEARNERS,
        [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ],
        strict=True,
    )
)


def main() -> None:
    design = data.design_arg(__doc__)
    style.apply()
    df = data.load(design).filter(pl.col('family').is_in(list(data.FAMILY_LEARNER)))

    fractions = sorted(df['batch_fraction'].unique().to_list())
    letters = 'ABCDEF'[: len(fractions)]
    fig, axd = style.mosaic(letters, panel_h_mm=72)

    pool_size = int(df['pool_size'].unique().item())
    for letter, frac in zip(letters, fractions, strict=True):
        ax = axd[letter]
        curves = data.curve(
            df.filter(pl.col('batch_fraction') == frac), ['learner'], X, Y
        )
        for learner in LEARNERS:
            s = curves.filter(pl.col('learner') == learner)
            color = MODEL_COLORS[learner]
            ax.plot(
                s[X],
                s['mean'],
                color=color,
                linestyle='-',
                linewidth=style.DATA_LINEWIDTH,
            )
            style.band(ax, s[X], s['lo'], s['hi'], color, alpha=0.06)
        ax.set_title(f'batch = {frac}%')
        ax.set_xlabel(style.LABELS[X])
        ax.set_ylim(0, 100)
        # Bands are min-max over replicates, so the replicate count must be stated.
        # Upper left stays free of data: discovery curves rise left-to-right.
        ax.text(
            0.03,
            0.97,
            data.replicate_note(curves, 'learner'),
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
        style.set_compound_axis(
            ax,
            pool_size,
            ticks=curves[X].unique().sort().to_list(),
            rotation=45,
        )
        tick_labels = ax.get_xticklabels()
        if tick_labels:
            ax.tick_params(axis='x', labelsize=tick_labels[0].get_fontsize() * 0.7)
    axd['A'].set_ylabel(style.LABELS[Y])
    for letter in letters[1:]:
        axd[letter].set_yticklabels([])

    style.label_panels(axd)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=MODEL_COLORS[x],
                linestyle='-',
                label=x,
            )
            for x in LEARNERS
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),
        ncols=5,
    )
    style.save(fig, f'fig02_learners{data.design_suffix(design)}')


if __name__ == '__main__':
    main()

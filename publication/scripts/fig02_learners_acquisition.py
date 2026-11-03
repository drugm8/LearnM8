"""F2 (combined) - learners and acquisition strategies on 1M AmpC.

Row 1 (A-C)  learner comparison, greedy acquisition   (was fig02_learners)
Row 2 (D-F)  acquisition strategies, random forest    (was fig03_acquisition)

Each row is a subfigure with its own legend, because the two rows plot different
series. The learner and strategy lists come from the standalone scripts, so the two
stay in step. Renders one figure per cycle-0 seed design; see `data.load`.
"""

from __future__ import annotations

from string import ascii_uppercase

import data
import matplotlib.pyplot as plt
import polars as pl
import style
from fig02_learners import LEARNERS, MODEL_COLORS
from fig03_acquisition import FAMILIES, STRATEGIES
from matplotlib.lines import Line2D

from learnm8.visualization.style import style_axes

Y = 'top_0_1_pct_discovery'
X = 'compounds_evaluated'
PANEL_H_MM = 36
ROW_EXTRA_MM = 21  # row heading above, legend below


def draw_row(axes, df, fractions, group, order, color, marker, band_alpha) -> None:
    """Draw one row of discovery-curve panels, one panel per batch fraction."""
    for ax, frac in zip(axes, fractions, strict=True):
        sub = df.filter(pl.col('batch_fraction') == frac)
        curves = data.curve(sub, [group], X, Y)
        for name in order:
            s = curves.filter(pl.col(group) == name)
            line = (
                {}
                if marker is None
                else {
                    'marker': marker[name],
                    'markevery': max(1, s.height // 10),
                }
            )
            ax.plot(
                s[X],
                s['mean'],
                color=color[name],
                linestyle='-',
                linewidth=style.DATA_LINEWIDTH,
                **line,
            )
            style.band(ax, s[X], s['lo'], s['hi'], color[name], alpha=band_alpha)
        ax.set(title=f'batch = {frac}%', xlabel=style.LABELS[X], ylim=(0, 100))
        # Bands are min-max over replicates, so the replicate count must be stated.
        # The legend lists every series, so a panel missing some must say so.
        note = data.replicate_note(curves, group)
        absent = [n for n in order if n not in curves[group].to_list()]
        if absent:
            note += '\nnot run at this batch: ' + ', '.join(absent)
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
        style.set_compound_axis(
            ax,
            int(sub['pool_size'].unique().item()),
            ticks=curves[X].unique().sort().to_list(),
            rotation=45,
        )
        ax.tick_params(axis='x', labelsize=ax.get_xticklabels()[0].get_fontsize() * 0.7)
    axes[0].set_ylabel(style.LABELS[Y])
    for ax in axes[1:]:
        ax.set_yticklabels([])


def main() -> None:
    design = data.design_arg(__doc__)
    style.apply()
    df = data.load(design)
    df_learners = df.filter(pl.col('family').is_in(list(data.FAMILY_LEARNER)))
    df_strategies = df.filter(pl.col('family').is_in(FAMILIES))

    fractions = sorted(df_learners['batch_fraction'].unique().to_list())
    if fractions != sorted(df_strategies['batch_fraction'].unique().to_list()):
        raise ValueError('learner and acquisition rows have different batch fractions')
    n = len(fractions)
    top, bottom = ascii_uppercase[:n], ascii_uppercase[n : 2 * n]

    fig = plt.figure(
        figsize=(style.DOUBLE_COL, 2 * (PANEL_H_MM + ROW_EXTRA_MM) * style.MM)
    )
    rows = fig.subfigures(2, 1)
    axd = {}
    for sub, letters in zip(rows, (top, bottom), strict=True):
        axd.update(sub.subplot_mosaic([list(letters)]))
    for ax in axd.values():
        style_axes(ax)

    draw_row(
        [axd[c] for c in top],
        df_learners,
        fractions,
        'learner',
        LEARNERS,
        MODEL_COLORS,
        None,
        0.06,
    )
    markers = {
        s: style.CURVE_MARKERS[i % len(style.CURVE_MARKERS)]
        for i, s in enumerate(STRATEGIES)
    }
    draw_row(
        [axd[c] for c in bottom],
        df_strategies,
        fractions,
        'strategy',
        STRATEGIES,
        style.STRATEGY_COLOR,
        markers,
        0.08,
    )
    style.label_panels(axd)

    rows[0].suptitle(
        'Learners (greedy acquisition)',
        x=0.0,
        ha='left',
        fontsize=8,
        fontweight='semibold',
    )
    rows[1].suptitle(
        'Acquisition strategies (random forest)',
        x=0.0,
        ha='left',
        fontsize=8,
        fontweight='semibold',
    )
    rows[0].legend(
        handles=[
            Line2D([], [], color=MODEL_COLORS[x], linestyle='-', label=x)
            for x in LEARNERS
        ],
        loc='outside lower center',
        ncols=5,
    )
    rows[1].legend(
        handles=[
            Line2D(
                [],
                [],
                color=style.STRATEGY_COLOR[s],
                marker=markers[s],
                linestyle='-',
                label=s,
            )
            for s in STRATEGIES
        ],
        loc='outside lower center',
        ncols=4,
    )
    style.save(fig, f'fig02_learners_acquisition{data.design_suffix(design)}')


if __name__ == '__main__':
    main()

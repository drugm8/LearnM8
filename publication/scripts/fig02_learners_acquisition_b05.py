"""F2 (single column) - learners and acquisition strategies at the 0.5% batch.

Same data and styling as `fig02_learners_acquisition`, reduced to the 0.5% batch
fraction so that the figure fits one column (A learners, B acquisition strategies).
Renders `fig02_learners_acquisition_b05.png`; the three-batch figure is untouched.
"""

from __future__ import annotations

import data
import matplotlib.pyplot as plt
import polars as pl
import style
from fig02_learners import LEARNERS, MODEL_COLORS
from fig02_learners_acquisition import PANEL_H_MM, ROW_EXTRA_MM, draw_row
from fig03_acquisition import FAMILIES as _ALL_FAMILIES
from fig03_acquisition import STRATEGIES as _ALL_STRATEGIES
from matplotlib.lines import Line2D

from learnm8.visualization.style import style_axes

BATCH = 0.5  # percent of the pool per cycle
# topk (A-09) is deliberately not described in the paper (author decision
# 2026-11-02), so the seven described strategies are shown.
STRATEGIES = [s for s in _ALL_STRATEGIES if s != 'topk']
FAMILIES = [f for f in _ALL_FAMILIES if f != 'A-09']


def main() -> None:
    style.apply()
    df = data.load('matched')
    df_learners = df.filter(pl.col('family').is_in(list(data.FAMILY_LEARNER)))
    df_strategies = df.filter(pl.col('family').is_in(FAMILIES))

    fig = plt.figure(
        figsize=(style.SINGLE_COL, 2 * (PANEL_H_MM + ROW_EXTRA_MM + 6) * style.MM)
    )
    rows = fig.subfigures(2, 1)
    axd = {}
    for sub, letter in zip(rows, 'AB', strict=True):
        axd.update(sub.subplot_mosaic([[letter]]))
    for ax in axd.values():
        style_axes(ax)

    draw_row(
        [axd['A']], df_learners, [BATCH], 'learner', LEARNERS, MODEL_COLORS, None, 0.06
    )
    markers = {
        s: style.CURVE_MARKERS[i % len(style.CURVE_MARKERS)]
        for i, s in enumerate(STRATEGIES)
    }
    draw_row(
        [axd['B']],
        df_strategies,
        [BATCH],
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
        ncols=4,
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
    style.save(fig, 'fig02_learners_acquisition_b05')


if __name__ == '__main__':
    main()

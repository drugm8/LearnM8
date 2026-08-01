"""F6 - cycle scheduling, diversity, and generalization to a second target.

A-D  schedules at 10M AmpC (3 replicates): flat greedy, ucb->greedy,
     greedy->simulated annealing
E    AmpC (L-01, 1M) against D4 (D-01, 840K) under an identical configuration

Diversity metrics are featurizer-dependent, so panels B-D are only valid when
every compared run used the same fingerprint; the script checks and warns.
"""

from __future__ import annotations

import data
import polars as pl
import style
from matplotlib.lines import Line2D

Y = 'top_0_1_pct_discovery'
X = 'compounds_evaluated'
SCAFFOLD = 'scaffold_diversity_index_cumulative'
TANIMOTO = 'mean_tanimoto_similarity_sampled_cumulative'
SCHEDULES = {
    'C-01': 'greedy (flat)',
    'C-02': 'ucb → greedy',
    'C-03': 'greedy → annealing',
}
SCHEDULE_COLOR = {'C-01': style.PRIMARY, 'C-02': '#E69F00', 'C-03': '#009E73'}


def main() -> None:
    style.apply()
    df = data.load()
    schedules = df.filter(pl.col('family').is_in(list(SCHEDULES)))
    data.check_fingerprint(schedules, 'F6 schedules')

    fig, axd = style.mosaic('ABC\nDDE', panel_h_mm=52)

    for letter, metric, ylabel in (
        ('A', Y, style.LABELS[Y]),
        ('B', SCAFFOLD, style.LABELS[SCAFFOLD]),
        ('C', TANIMOTO, style.LABELS[TANIMOTO]),
    ):
        ax = axd[letter]
        curves = data.curve(schedules, ['family'], X, metric)
        for family in SCHEDULES:
            s = curves.filter(pl.col('family') == family)
            color = SCHEDULE_COLOR[family]
            ax.plot(s[X], s['mean'], color=color)
            style.band(ax, s[X], s['lo'], s['hi'], color)
        ax.set(xlabel=style.LABELS[X], ylabel=ylabel)
        style.compact_axis(ax)

    ax = axd['D']
    pareto = data.curve(schedules, ['family'], Y, SCAFFOLD)
    for family in SCHEDULES:
        s = pareto.filter(pl.col('family') == family).sort('cycle')
        color = SCHEDULE_COLOR[family]
        ax.plot(s[Y], s['mean'], color=color, marker='o', markersize=2.5)
        for cycle in (1, s['cycle'].max()):
            point = s.filter(pl.col('cycle') == cycle)
            if point.height:
                ax.annotate(
                    f'cycle {cycle}',
                    (point[Y].item(), point['mean'].item()),
                    textcoords='offset points',
                    xytext=(4, -2),
                    fontsize=6,
                    color=color,
                )
    ax.set(xlabel=style.LABELS[Y], ylabel=style.LABELS[SCAFFOLD])

    ax = axd['E']
    targets = {
        'AmpC (1M)': df.filter(
            (pl.col('family') == 'L-01') & (pl.col('batch_fraction') == 1.0)
        ),
        'D4 (840K)': df.filter(pl.col('family') == 'D-01'),
    }
    for (label, subset), color in zip(
        targets.items(), (style.PRIMARY, '#0072B2'), strict=True
    ):
        s = data.curve(subset, ['family'], X, Y)
        ax.plot(s[X], s['mean'], color=color, label=label)
        style.band(ax, s[X], s['lo'], s['hi'], color)
    ax.set(xlabel=style.LABELS[X], ylabel=style.LABELS[Y], ylim=(0, 100))
    style.compact_axis(ax)
    ax.legend(loc='lower right')

    style.label_panels(axd)
    fig.legend(
        handles=[
            Line2D([], [], color=SCHEDULE_COLOR[f], label=lab)
            for f, lab in SCHEDULES.items()
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.04),
        ncols=3,
    )
    fig.tight_layout()
    style.save(fig, 'fig06_diversity')


if __name__ == '__main__':
    main()

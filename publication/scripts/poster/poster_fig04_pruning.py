"""P4 - pruning on 10M AmpC (poster), in two variants.

A  top-0.1% recovered against pruning fraction
B  prediction speedup (cumulative prediction-time ratio)
D  reference hits pruned in each cycle, plus a cumulative Total column

Poster variant of the pruning figure in ``fig04_batch_pruning.py`` (paper Fig. 5),
drawn at two-column width (515 mm) so every panel keeps text >= 20 pt. The paper's
panels C (hits pruned against fraction, now the Total column of D) and E
(predicted percentile of the pruned hits) are left out.

Panel D is written twice, from one data pass:

* ``poster_fig04_pruning``      cells show the number of reference hits pruned
* ``poster_fig04_pruning_pct``  cells show that number as a share of the 10,000
                                reference hits (the top 0.1% of the 10M pool)

Cells with no pruning are marked by why: before a delayed run starts pruning they
are white with a dash, and after a run has stopped (the pool was exhausted) they
are grey with a cross.

Needs the raw 10M AmpC pool, which is deposited on Zenodo rather than tracked;
without it this script exits with ``EXIT_RAW_DATA_MISSING`` and the poster build
keeps the copies already in ``poster/assets``.
"""

from __future__ import annotations  # noqa: I001 - poster_style must import first (sets sys.path)

import copy
import sys

import poster_style as ps  # first: puts the shared scripts on sys.path
import data
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import style
from fig04_batch_pruning import (
    DELAYED_PRUNE_FAMILIES,
    INITIAL_POOL,
    PRUNE_FAMILIES,
    SCHEDULE_STYLE,
    TOP_FRACTION,
    Y,
    _pruned_top_summary,
    _top_status_records,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from learnm8.visualization.style import style_axes

HEIGHT_MM = 172
PAPER_HEIGHT_MM = 105

# Sizes the drawing code uses for text, markers and strokes. The poster values are
# printed at A0; ``build_paper`` swaps in the paper values (the paper theme sets the
# font) and the figure is then drawn at the paper's 170 mm two-column width.
POSTER_SIZES = {
    'text': 20,
    'marker': 12,
    'legend_marker': 13,
    'elinewidth': 2,
    'capsize': 4,
    'box': 1.6,
    'marker_x': -0.06,
    'label_x': -0.085,
    'title': True,
}
PAPER_SIZES = {
    'text': 6.5,
    'marker': 4,
    'legend_marker': 4.5,
    'elinewidth': 0.7,
    'capsize': 1.5,
    'box': 0.6,
    'marker_x': -0.085,
    'label_x': -0.125,
    'title': False,
}
SZ = dict(POSTER_SIZES)


def _off(points: float) -> float:
    """Scale a label offset (given in poster points) to the active size set."""
    return points * SZ['text'] / POSTER_SIZES['text']


SCHEDULE_NAMES = {'immediate': 'immediate', 'delayed': 'delayed (cycle 5)'}
SHORT_NAMES = {'immediate': 'immediate', 'delayed': 'delayed'}
KEYS = ['pruning_schedule', 'prune_fraction']
STOPPED = '#DEDBE5'  # grey fill for cycles that did not happen
STOPPED_MARK = '#8A8794'
WAITING = '#FFFFFF'  # white fill for cycles before a delayed run starts pruning


def reference_top_ids() -> tuple[set[str], int]:
    """Top-0.1% IDs of the starting pool, with a deterministic tie-break.

    ``fig04_batch_pruning._initial_pool_top_ids`` sorts on ``dockscore`` alone, and
    Polars' sort is not stable. Where scores tie at the cutoff, a different tied
    compound can enter the reference set on each run, which moves the pruned counts
    by one. Sorting on (dockscore, ID) makes the poster's numbers reproducible.
    """
    source = pl.scan_csv(INITIAL_POOL, infer_schema_length=10_000)
    n_total = int(source.select(pl.len()).collect(engine='streaming').item())
    top_k = max(1, int(n_total * TOP_FRACTION))
    top = (
        source.select(['ID', 'dockscore'])
        .sort(['dockscore', 'ID'])
        .head(top_k)
        .collect(engine='streaming')
    )
    return set(top['ID'].to_list()), top_k


def mean_range(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Mean and min-max of ``column`` per (schedule, fraction)."""
    return (
        df.group_by(KEYS)
        .agg(
            pl.col(column).mean().alias('mean'),
            pl.col(column).min().alias('lo'),
            pl.col(column).max().alias('hi'),
        )
        .sort(['prune_fraction', 'pruning_schedule'])
    )


def prepare() -> dict:
    """Aggregate the runs into everything the panels draw."""
    prune = (
        data.load('fixed')
        .filter(pl.col('family').is_in(PRUNE_FAMILIES))
        .with_columns(
            pl.when(pl.col('family').is_in(DELAYED_PRUNE_FAMILIES))
            .then(pl.lit('delayed'))
            .otherwise(pl.lit('immediate'))
            .alias('pruning_schedule')
        )
    )
    top_ids, top_k = reference_top_ids()
    top_status = _top_status_records(prune, top_ids, top_k)
    lost = _pruned_top_summary(top_status)

    recovery = data.final(prune, KEYS, [Y]).sort(['prune_fraction', 'pruning_schedule'])
    final_runs = prune.filter(
        pl.col('cycle') == pl.col('cycle').max().over('run_id')
    ).select(['run_id', 'rep', *KEYS, 'cum_prediction_time'])
    baseline = final_runs.filter(
        (pl.col('pruning_schedule') == 'immediate') & (pl.col('prune_fraction') == 0)
    ).select(['rep', pl.col('cum_prediction_time').alias('baseline_time')])
    speedup = mean_range(
        final_runs.join(baseline, on='rep', how='left').with_columns(
            (pl.col('baseline_time') / pl.col('cum_prediction_time')).alias('speedup')
        ),
        'speedup',
    )

    # Hits pruned per cycle, averaged over ALL replicate runs of a condition (a run
    # with no hit lost in a cycle counts as 0), so the cycles add up to the total.
    n_runs = prune.group_by(KEYS).agg(pl.col('run_id').n_unique().alias('n_runs'))
    per_cycle = (
        top_status.filter(
            (pl.col('status') == 'pruned') & (pl.col('prune_fraction') > 0)
        )
        .group_by([*KEYS, 'pruned_cycle'])
        .agg(pl.len().alias('hits'))
        .join(n_runs, on=KEYS)
        .with_columns((pl.col('hits') / pl.col('n_runs')).alias('mean'))
    )
    conditions = sorted(
        per_cycle.select(KEYS).unique().iter_rows(named=True),
        key=lambda c: (c['prune_fraction'], c['pruning_schedule'] == 'delayed'),
    )
    cycles = sorted(per_cycle['pruned_cycle'].unique().to_list())
    # A run prunes in the cycles [first, last]: a delayed run starts late, and a run
    # whose pool is exhausted stops early.
    last = {
        (r['pruning_schedule'], r['prune_fraction']): int(r['last'])
        for r in prune.group_by(KEYS)
        .agg(pl.col('cycle').max().alias('last'))
        .iter_rows(named=True)
    }
    first = {
        (r['pruning_schedule'], r['prune_fraction']): (
            1 if r['pruning_schedule'] == 'immediate' else int(r['first'])
        )
        for r in per_cycle.group_by(KEYS)
        .agg(pl.col('pruned_cycle').min().alias('first'))
        .iter_rows(named=True)
    }
    raw = np.zeros((len(conditions), len(cycles)))
    valid = np.zeros_like(raw, dtype=bool)
    waiting = np.zeros_like(raw, dtype=bool)  # delayed run, not pruning yet
    for i, cond in enumerate(conditions):
        key = (cond['pruning_schedule'], cond['prune_fraction'])
        for j, cycle in enumerate(cycles):
            valid[i, j] = first[key] <= cycle <= last[key]
            waiting[i, j] = cycle < first[key]
            cell = per_cycle.filter(
                (pl.col('pruning_schedule') == cond['pruning_schedule'])
                & (pl.col('prune_fraction') == cond['prune_fraction'])
                & (pl.col('pruned_cycle') == cycle)
            )['mean']
            raw[i, j] = cell[0] if len(cell) else 0.0
    if not np.all(raw[~valid] == 0):
        raise ValueError('hits were pruned in a cycle outside the run')

    totals = raw.sum(axis=1)
    check = mean_range(lost, 'pruned_top_count').filter(pl.col('prune_fraction') > 0)
    for cond, total in zip(conditions, totals, strict=True):
        ref = check.filter(
            (pl.col('pruning_schedule') == cond['pruning_schedule'])
            & (pl.col('prune_fraction') == cond['prune_fraction'])
        )['mean'][0]
        if abs(ref - total) > 1e-6:
            raise ValueError(f'cycle sums {total} disagree with the run total {ref}')
    return {
        'recovery': recovery,
        'speedup': speedup,
        'conditions': conditions,
        'cycles': cycles,
        'raw': raw,
        'valid': valid,
        'waiting': waiting,
        'totals': totals,
        'top_k': top_k,
    }


def schedule_lines(ax, summary, labels, names=SCHEDULE_NAMES) -> None:
    """Immediate-vs-delayed lines with min-max bars; label chosen fractions.

    ``labels`` maps a schedule to ``(fractions to label, offset, alignment, digits)``.
    """
    for schedule, (color, marker) in SCHEDULE_STYLE.items():
        s = summary.filter(pl.col('pruning_schedule') == schedule)
        ax.errorbar(
            s['prune_fraction'],
            s['mean'],
            yerr=[s['mean'] - s['lo'], s['hi'] - s['mean']],
            marker=marker,
            markersize=SZ['marker'],
            color=color,
            elinewidth=SZ['elinewidth'],
            capsize=SZ['capsize'],
            label=names[schedule],
        )
        at, offset, align, digits = labels[schedule]
        for row in s.iter_rows(named=True):
            if row['prune_fraction'] in at:
                ax.annotate(
                    f'{row["mean"]:.{digits}f}',
                    (row['prune_fraction'], row['mean']),
                    textcoords='offset points',
                    xytext=(_off(offset[0]), _off(offset[1])),
                    ha=align,
                    fontsize=SZ['text'],
                    color=color,
                    fontweight='bold',
                )


def draw_recovery(ax, d) -> None:
    """Panel A: recovery against pruning fraction."""
    recovery = d['recovery'].rename(
        {f'{Y}_mean': 'mean', f'{Y}_lo': 'lo', f'{Y}_hi': 'hi'}
    )
    schedule_lines(
        ax,
        recovery,
        {
            'immediate': ({0, 70}, (10, 8), 'left', 1),
            'delayed': ({50, 70}, (10, 8), 'left', 1),
        },
        names=SHORT_NAMES,
    )
    ax.set(xlabel='Pruning (%)', ylabel='Top-0.1%\nrecovered (%)')
    ax.set_xticks(sorted(recovery['prune_fraction'].unique().to_list()))
    ax.set_xlim(-8, 92)
    ax.legend(
        loc='lower left', handlelength=1.6, borderaxespad=0.2, fontsize=SZ['text']
    )


def draw_speedup(ax, d) -> None:
    """Panel B: prediction speedup; colour = pruning fraction, hatch = delayed start."""
    sp = d['speedup']
    fr = sorted(sp['prune_fraction'].unique().to_list())
    colors = {
        f: style.SEQUENTIAL(v)
        for f, v in zip(fr, np.linspace(0.25, 1.0, len(fr)), strict=True)
    }
    width = 0.40
    for schedule, offset, hatch in (
        ('immediate', -width / 2, ''),
        ('delayed', width / 2, '//'),
    ):
        s = sp.filter(pl.col('pruning_schedule') == schedule)
        pos = np.array([fr.index(f) for f in s['prune_fraction']]) + offset
        ax.bar(
            pos,
            s['mean'],
            width=width,
            yerr=[s['mean'] - s['lo'], s['hi'] - s['mean']],
            color=[colors[f] for f in s['prune_fraction']],
            hatch=hatch,
            error_kw={'elinewidth': SZ['elinewidth'], 'ecolor': style.INK},
        )
        for position, row in zip(pos, s.iter_rows(named=True), strict=True):
            if row['prune_fraction'] == 0:  # the baseline is 1x by definition
                continue
            ax.annotate(
                f'{row["mean"]:.1f}×',
                (position, row['hi']),
                textcoords='offset points',
                xytext=(_off(10) if schedule == 'delayed' else 0, _off(4)),
                ha='center',
                va='bottom',
                fontsize=SZ['text'],
                fontweight='bold',
                color=style.INK,
            )
    ax.set_xticks(range(len(fr)), [f'{f:g}' for f in fr])
    ax.set(xlabel='Pruning (%)', ylabel='Prediction\nspeedup (×)')
    ax.legend(
        handles=[
            Patch(facecolor='#CFCAD8', edgecolor=style.INK, label='immediate'),
            Patch(
                facecolor='#CFCAD8', edgecolor=style.INK, hatch='//', label='delayed'
            ),
        ],
        loc='upper left',
        handlelength=1.4,
        borderaxespad=0.2,
        fontsize=SZ['text'],
    )
    ax.margins(y=0.24)


def draw_cycles(fig, ax, d, variant: str) -> None:
    """Panel D: reference hits pruned per cycle, with a cumulative Total column."""
    pct = variant == 'pct'
    scale = 100.0 / d['top_k'] if pct else 1.0
    fmt = '{:.2f}' if pct else '{:.0f}'
    n_cond, n_cyc = d['raw'].shape
    grid = np.where(d['valid'], d['raw'] * scale, np.nan)
    totals = d['totals'] * scale
    vmax = float(np.nanmax(grid))

    cmap = copy.copy(style.SEQUENTIAL)
    cmap.set_bad(STOPPED)
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    ax.grid(False)
    for i in range(n_cond):
        for j in range(n_cyc):
            if d['valid'][i, j]:
                ax.text(
                    j,
                    i,
                    fmt.format(grid[i, j]),
                    ha='center',
                    va='center',
                    fontsize=SZ['text'],
                    color=style.BACKGROUND if grid[i, j] >= 0.55 * vmax else style.INK,
                )
            elif d['waiting'][i, j]:  # delayed run, pruning has not started
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1.0, 1.0, facecolor=WAITING, linewidth=0
                    )
                )
                ax.text(
                    j,
                    i,
                    '–',
                    ha='center',
                    va='center',
                    fontsize=SZ['text'],
                    color=STOPPED_MARK,
                )
            else:  # the pool was exhausted, so the run had stopped
                ax.text(
                    j,
                    i,
                    '×',
                    ha='center',
                    va='center',
                    fontsize=SZ['text'],
                    color=STOPPED_MARK,
                )

    # Cumulative column, set apart from the per-cycle cells by a quarter-cell gap.
    total_x = n_cyc + 0.25
    for i in range(n_cond):
        ax.add_patch(
            Rectangle(
                (total_x - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor='white',
                edgecolor='#B685F6',
                linewidth=SZ['box'],
                clip_on=False,
            )
        )
        ax.text(
            total_x,
            i,
            fmt.format(totals[i]),
            ha='center',
            va='center',
            fontsize=SZ['text'],
            fontweight='bold',
            color=style.INK,
        )
    ax.set_xlim(-0.5, total_x + 0.5)
    ax.set_ylim(n_cond - 0.5, -0.5)

    ax.set_xticks([*range(n_cyc), total_x], [*(f'c{c}' for c in d['cycles']), 'Total'])
    ax.set_yticks(range(n_cond), [f'{c["prune_fraction"]:g}%' for c in d['conditions']])
    ax.set(xlabel='Pruning cycle', ylabel='Pruning')
    ax.yaxis.set_label_coords(SZ['label_x'], 0.5)
    # Schedule: the marker beside each fraction repeats the encoding of panel A.
    for i, cond in enumerate(d['conditions']):
        color, marker = SCHEDULE_STYLE[cond['pruning_schedule']]
        ax.plot(
            [SZ['marker_x']],
            [i],
            marker=marker,
            markersize=SZ['legend_marker'],
            color=color,
            linestyle='',
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linestyle='',
                markersize=SZ['legend_marker'],
                label=SCHEDULE_NAMES[name],
            )
            for name, (color, marker) in SCHEDULE_STYLE.items()
        ]
        + [
            Patch(
                facecolor=WAITING,
                edgecolor=STOPPED_MARK,
                linewidth=0.8,
                label='– not started',
            ),
            Patch(
                facecolor=STOPPED,
                edgecolor=STOPPED_MARK,
                linewidth=0.8,
                label='× pool exhausted',
            ),
        ],
        loc='lower left',
        bbox_to_anchor=(0.0, 1.0),
        ncols=4,
        frameon=False,
        borderaxespad=0.1,
        handletextpad=0.2,
        columnspacing=1.6,
        fontsize=SZ['text'],
    )
    # Say what a cell counts: the hits are the top 0.1% of the 10M pool.
    unit = f'% of {d["top_k"]:,}' if pct else f'of {d["top_k"]:,}'
    if SZ['title']:  # the paper caption says this instead
        ax.set_title(
            f'Hits pruned ({unit} top compounds)',
            loc='right',
            fontsize=SZ['text'],
            fontweight=500,
            color=style.MUTED,
        )


def build(d, variant: str) -> None:
    fig = plt.figure(
        figsize=(ps.mm(ps.SPAN_MM), ps.mm(HEIGHT_MM)), layout='constrained'
    )
    gs = fig.add_gridspec(2, 10, height_ratios=[1, 1.35])
    axes = {
        'A': fig.add_subplot(gs[0, 0:5]),
        'B': fig.add_subplot(gs[0, 5:10]),
        'D': fig.add_subplot(gs[1, 0:10]),
    }
    for letter, ax in axes.items():
        style_axes(ax)
        ps.panel_letter(ax, letter)
    draw_recovery(axes['A'], d)
    draw_speedup(axes['B'], d)
    draw_cycles(fig, axes['D'], d, variant)
    ps.save_poster(
        fig, 'poster_fig04_pruning_pct' if variant == 'pct' else 'poster_fig04_pruning'
    )


def build_paper(d) -> None:
    """Paper variant: the paper theme and font, 170 mm wide, same three panels."""
    SZ.update(PAPER_SIZES)
    style.apply()
    fig = plt.figure(
        figsize=(style.DOUBLE_COL, PAPER_HEIGHT_MM * style.MM), layout='constrained'
    )
    gs = fig.add_gridspec(2, 10, height_ratios=[1, 1.35])
    axes = {
        'A': fig.add_subplot(gs[0, 0:5]),
        'B': fig.add_subplot(gs[0, 5:10]),
        'C': fig.add_subplot(gs[1, 0:10]),
    }
    for ax in axes.values():
        style_axes(ax)
    draw_recovery(axes['A'], d)
    draw_speedup(axes['B'], d)
    draw_cycles(fig, axes['C'], d, 'count')
    style.label_panels(axes)
    style.save(fig, 'fig04_pruning_paper')


def main_paper() -> None:
    """Entry point for ``fig04_pruning_paper.py``, which ``make_all.py`` globs."""
    if not INITIAL_POOL.exists():
        print(
            f'paper pruning figure skipped: raw data {INITIAL_POOL} not found',
            file=sys.stderr,
        )
        sys.exit(style.EXIT_RAW_DATA_MISSING)
    build_paper(prepare())


def main() -> None:
    if not INITIAL_POOL.exists():
        print(
            f'poster pruning figure skipped: raw data {INITIAL_POOL} not found',
            file=sys.stderr,
        )
        sys.exit(style.EXIT_RAW_DATA_MISSING)
    ps.apply()
    d = prepare()
    for variant in ('count', 'pct'):
        build(d, variant)


if __name__ == '__main__':
    main()

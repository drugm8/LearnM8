"""Separate batch-frequency and pruning analyses for the publication.

The pruning loss panel uses the initial 10M AmpC pool as the fixed reference:
"lost" means a compound in the true whole-pool top 0.1% ended with
``status == "pruned"``.  It does not treat a currently unlabeled compound as
lost, because it may still be selected later.
"""

from __future__ import annotations

from pathlib import Path

import data
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import style
from matplotlib.lines import Line2D

from learnm8.visualization.style import style_axes

Y = 'top_0_1_pct_discovery'
X = 'compounds_evaluated'
# One batch ladder per cycle-0 seed design; they are never pooled. Both reach the
# same 500,000-compound budget, but they collide at 0.2% and 0.1% where the two
# designs hold different runs, so a pooled ladder would average two experiments.
# B-02 (1.0% batch, 1.0% seed) satisfies both rules and anchors either ladder.
BATCH_FAMILIES = {
    'fixed': ['B-01', 'B-02', 'B-03', 'B-04', 'B-05', 'B-06'],
    'matched': ['B-11', 'B-02', 'B-12', 'B-13', 'B-14', 'B-15'],
}
PRUNE_FAMILIES = [
    'C-01',
    'P-02',
    'P-03',
    'P-04',
    'P-05',
    'P-14',
    'P-15',
]
DELAYED_PRUNE_FAMILIES = ['P-14', 'P-15']
# One immediate/delayed pair shared by the two line panels (A and C): both encode
# the same schedule contrast, so a reader should not have to relearn it between
# them. Both arms are solid — the delayed arm only spans two x positions, and as a
# dashed segment it read as a stray annotation rather than a second series.
# Identity is carried by hue and marker shape together, not by line style.
SCHEDULE_STYLE = {
    'immediate': (style.PRIMARY, 'o'),
    'delayed': (style.ACCENT_GREEN, 'D'),
}
SCHEDULE_LABELS = {'immediate': 'immediate', 'delayed': 'delayed start (c5)'}
# The theme's 3.5 pt default disappears on a two-point arm; these panels have at
# most five glyphs each, so they can afford to be read as points.
SCHEDULE_MARKERSIZE = 5.0
# Panel A and C value labels sit inside the data area on their own point, so they
# run below the 7 pt used for axis-adjacent annotation.
POINT_LABEL_SIZE = 6.0
INITIAL_POOL = Path.home() / 'LearnM8_DATA' / 'AmpC_screen_10000K.csv'
TOP_FRACTION = 0.001
# In-plot batch-fraction labels, 20% down from the 7 pt used for axis-adjacent
# annotation: they sit inside the data area and must not compete with the curves.
FRACTION_LABEL_SIZE = 5.6
MARKER_CYCLE_STRIDE = 4
# Row identity. `matched` seeds cycle 0 with one acquisition batch, `fixed`
# always with 1% of the pool; the ladders are never pooled into one curve.
DESIGN_TITLES = {
    'matched': 'seed = one acquisition batch',
    'fixed': 'seed = 1% of pool',
}
# Two stacked ladders at 62 mm each. Taller than the 52 mm used when this was
# two separate three-panel figures, and two columns instead of three, so each
# panel gains on both axes while the text on it stays the same size.
PANEL_H_MM = 62


def batch_colors(fractions: list[float]) -> dict[float, str]:
    """Opaque sequential purple ramp ordered by batch size.

    Built over the union of both seed designs' fractions rather than per row:
    the two ladders share one figure, so a 0.1% batch has to read the same in
    either of them. Colour is the only series channel here, so the ramp has to
    stay monotone in batch size for the ordering to survive greyscale.
    """
    ordered = sorted(fractions)
    ranks = np.linspace(0.62, 0.98, len(ordered))
    return {
        fraction: style.SEQUENTIAL(rank)
        for fraction, rank in zip(ordered, ranks, strict=True)
    }


def _initial_pool_top_ids() -> tuple[set[str], int, int]:
    """Return true top-fraction IDs from the immutable starting pool."""
    if not INITIAL_POOL.exists():
        raise FileNotFoundError(
            f'{INITIAL_POOL} is required to calculate pruning losses; '
            'the result artifacts do not retain target values for pruned compounds'
        )

    source = pl.scan_csv(INITIAL_POOL, infer_schema_length=10_000)
    stats = (
        source.select(
            [
                pl.len().alias('n_total'),
                pl.col('ID').n_unique().alias('n_unique_ids'),
                pl.col('dockscore').null_count().alias('n_null_scores'),
            ]
        )
        .collect(engine='streaming')
        .row(0, named=True)
    )
    n_total = int(stats['n_total'])
    if int(stats['n_unique_ids']) != n_total:
        raise ValueError('initial AmpC pool contains duplicate IDs')
    if int(stats['n_null_scores']) != 0:
        raise ValueError('initial AmpC pool contains null dockscore values')

    top_k = max(1, int(n_total * TOP_FRACTION))
    top = (
        source.select(['ID', 'dockscore'])
        .sort('dockscore', nulls_last=True)
        .head(top_k)
        .collect(engine='streaming')
    )
    if top.height != top_k:
        raise ValueError(f'expected {top_k} top compounds, found {top.height}')
    return set(top['ID'].to_list()), top_k, n_total


def _top_status_records(
    df: pl.DataFrame, top_ids: set[str], top_k: int
) -> pl.DataFrame:
    """Return final statuses for the initial-pool top compounds per run."""
    runs = (
        df.filter(pl.col('family').is_in(PRUNE_FAMILIES))
        .select(['run_id', 'family', 'prune_fraction', 'pruning_schedule'])
        .unique()
        .sort(['prune_fraction', 'pruning_schedule', 'run_id'])
    )
    rows = []
    for row in runs.iter_rows(named=True):
        result = data.RESULTS_DIR / row['run_id'] / 'compounds_final.parquet'
        if not result.exists():
            raise FileNotFoundError(f'missing final compound table: {result}')
        top_status = (
            pl.scan_parquet(result)
            .filter(pl.col('ID').is_in(top_ids))
            .select(['ID', 'status', 'pruned_cycle'])
            .collect(engine='streaming')
        )
        if top_status.height != top_k:
            raise ValueError(
                f'{row["run_id"]}: matched {top_status.height}/{top_k} '
                'initial-pool top compounds'
            )
        rows.append(
            top_status.with_columns(
                run_id=pl.lit(row['run_id']),
                family=pl.lit(row['family']),
                prune_fraction=pl.lit(row['prune_fraction']),
                pruning_schedule=pl.lit(row['pruning_schedule']),
            )
        )
    return pl.concat(rows, how='vertical')


def _pruned_top_summary(top_status: pl.DataFrame) -> pl.DataFrame:
    """Summarize how many reference compounds were pruned per run."""
    return (
        top_status.group_by(['run_id', 'family', 'pruning_schedule', 'prune_fraction'])
        .agg(
            (pl.col('status') == 'pruned')
            .sum()
            .cast(pl.Int64)
            .alias('pruned_top_count')
        )
        .sort(['prune_fraction', 'run_id'])
    )


def _prediction_rank_percentiles(
    prediction_path: Path,
    target_ids: list[str],
) -> pl.DataFrame:
    """Rank target IDs in a pre-pruning prediction parquet.

    Dockscore is minimized, so percentile 0 is the best predicted score and
    percentile 100 is the worst. The two-pass implementation keeps only the
    target predictions and a small histogram in memory while scanning the
    potentially multi-million-row prediction file.
    """
    if not target_ids:
        return pl.DataFrame(
            schema={
                'ID': pl.String,
                'prediction': pl.Float64,
                'predicted_rank': pl.Int64,
                'predicted_percentile': pl.Float64,
                'prediction_pool_size': pl.Int64,
            }
        )
    if not prediction_path.exists():
        raise FileNotFoundError(f'missing prediction artifact: {prediction_path}')

    target_predictions = (
        pl.scan_parquet(prediction_path)
        .filter(pl.col('ID').is_in(target_ids))
        .select(['ID', 'prediction'])
        .collect(engine='streaming')
    )
    if target_predictions.height != len(target_ids):
        found = set(target_predictions['ID'].to_list())
        missing = sorted(set(target_ids) - found)
        raise ValueError(
            f'{prediction_path.name}: missing {len(missing)} target predictions; '
            f'first missing IDs: {missing[:3]}'
        )

    predictions = (
        pl.scan_parquet(prediction_path)
        .select('prediction')
        .collect(engine='streaming')['prediction']
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    pool_size = predictions.size
    target_prediction_map = dict(
        zip(
            target_predictions['ID'].to_list(),
            target_predictions['prediction'].cast(pl.Float64).to_list(),
            strict=True,
        )
    )
    unique_target_predictions = np.unique(
        np.asarray(list(target_prediction_map.values()), dtype=np.float64)
    )
    bins = np.searchsorted(unique_target_predictions, predictions, side='right')
    counts = np.bincount(
        bins,
        minlength=unique_target_predictions.size + 1,
    )
    cumulative_counts = np.cumsum(counts)

    rows = []
    for compound_id in target_ids:
        prediction = target_prediction_map[compound_id]
        value_index = int(
            np.searchsorted(unique_target_predictions, prediction, side='left')
        )
        less_than = int(cumulative_counts[value_index])
        rank = less_than + 1
        percentile = 100.0 * (rank - 1) / max(1, pool_size - 1)
        rows.append(
            {
                'ID': compound_id,
                'prediction': prediction,
                'predicted_rank': rank,
                'predicted_percentile': percentile,
                'prediction_pool_size': pool_size,
            }
        )
    return pl.DataFrame(rows)


def _same_cycle_rank_records(
    top_status: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate predicted percentiles for reference compounds lost to pruning."""
    lost = top_status.filter(
        (pl.col('status') == 'pruned') & (pl.col('prune_fraction') > 0)
    )
    rows = []
    groups = lost.group_by(
        ['run_id', 'family', 'pruning_schedule', 'prune_fraction', 'pruned_cycle'],
        maintain_order=True,
    )
    for group, targets in groups:
        run_id, family, schedule, prune_fraction, cycle = group
        prediction_path = (
            data.RESULTS_DIR / run_id / f'prediction_cycle_{cycle}.parquet'
        )
        ranks = _prediction_rank_percentiles(
            prediction_path,
            targets['ID'].to_list(),
        )
        rows.append(
            ranks.with_columns(
                run_id=pl.lit(run_id),
                family=pl.lit(family),
                pruning_schedule=pl.lit(schedule),
                prune_fraction=pl.lit(prune_fraction),
                pruned_cycle=pl.lit(cycle),
            )
        )
    return pl.concat(rows, how='vertical') if rows else pl.DataFrame()


def _marker_indices(n_cycles: int) -> list[int]:
    """Mark every MARKER_CYCLE_STRIDE-th cycle, always including the last.

    A fixed marker count per arm gave a 50-cycle 0.1% arm and a 2-cycle 2.5%
    arm identical glyph density, hiding the one quantity the ladder is about.
    Marking on a cycle stride instead makes the glyph count the retrain count.
    """
    indices = list(range(0, n_cycles, MARKER_CYCLE_STRIDE))
    if indices[-1] != n_cycles - 1:
        indices.append(n_cycles - 1)
    return indices


def _draw_ladder(
    axd: dict,
    letters: str,
    batch: pl.DataFrame,
    colors: dict[float, str],
    title: str,
) -> None:
    """Draw one seed design's two batch-ladder panels into ``letters``."""
    fractions = sorted(batch['batch_fraction'].unique().to_list())
    pool_size = int(batch['pool_size'].unique().item())

    ax = axd[letters[0]]
    curves = data.curve(batch, ['batch_fraction'], X, Y)
    for frac in fractions:
        color = colors[frac]
        s = curves.filter(pl.col('batch_fraction') == frac)
        ax.plot(
            s[X],
            s['mean'],
            color=color,
            marker='o',
            markevery=_marker_indices(s.height),
            markersize=2.4,
            label=f'{frac:g}%',
            linewidth=style.DATA_LINEWIDTH,
        )
        style.band(ax, s[X], s['lo'], s['hi'], color, alpha=0.09)
    ax.set(xlabel=style.LABELS[X], ylabel=style.LABELS[Y])
    ax.set_title(title, loc='left')
    style.compact_axis(ax)
    style.set_compound_axis(
        ax,
        pool_size,
        ticks=curves[X].unique().sort().to_list(),
        max_ticks=5,
    )
    ax.margins(x=0.05, y=0.10)

    # Bands are min-max over replicates, so a single-replicate arm draws no band at
    # all. Without the count stated that reads as a perfectly reproducible arm
    # rather than an arm still waiting on the cluster.
    ax.text(
        0.03,
        0.97,
        data.replicate_note(curves, 'batch_fraction'),
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=6,
        color=style.MUTED,
    )

    summary = data.final(batch, ['batch_fraction'], [Y, 'cum_ml_time']).sort(
        'batch_fraction'
    )

    # Points ascend monotonically to the right, so a label sitting directly on
    # its own point is unambiguous and needs no leader line. The previous fan to
    # a shared label column needed one leader per arm, and those five near-
    # horizontal rules read as data in a panel whose data is five dots.
    ax = axd[letters[1]]
    points = list(summary.iter_rows(named=True))
    for index, row in enumerate(points):
        color = colors[row['batch_fraction']]
        # Min-max over replicates on both axes, matching the band in panel A/C
        # rather than mixing a range with a confidence interval in one figure.
        # A single-replicate arm therefore draws a zero-width cross, not a dot.
        ax.errorbar(
            row['cum_ml_time_mean'],
            row[f'{Y}_mean'],
            xerr=[
                [row['cum_ml_time_mean'] - row['cum_ml_time_lo']],
                [row['cum_ml_time_hi'] - row['cum_ml_time_mean']],
            ],
            yerr=[
                [row[f'{Y}_mean'] - row[f'{Y}_lo']],
                [row[f'{Y}_hi'] - row[f'{Y}_mean']],
            ],
            marker='o',
            markersize=4,
            color=color,
            elinewidth=0.6,
            capsize=1.5,
            capthick=0.6,
        )
        # The plateau arms sit within a point of each other in y and roughly a
        # factor of two apart in x, which is not enough room for same-side
        # labels; alternating sides doubles the horizontal pitch available.
        above = index % 2 == 0
        ax.annotate(
            f'{row["batch_fraction"]:g}%',
            (row['cum_ml_time_mean'], row[f'{Y}_mean']),
            textcoords='offset points',
            xytext=(0, 5 if above else -5),
            ha='center',
            va='bottom' if above else 'top',
            fontsize=FRACTION_LABEL_SIZE,
            color=color,
            fontweight='semibold',
        )
    ax.set(
        xscale='log',
        xlabel=style.LABELS['cum_ml_time'],
        ylabel='Top-0.1% recovered (%)',
    )
    ax.margins(x=0.16, y=0.22)


def make_batch_figure(frames: dict[str, pl.DataFrame]) -> None:
    """Write the batch-frequency figure: one ladder row per cycle-0 seed design.

    The two designs share a figure but never a curve. They hold different runs
    at 0.2% and 0.1%, so pooling them would average two experiments; stacking
    them lets a reader see that the ladder's shape survives either seed rule.
    """
    ladders = {}
    for design, df in frames.items():
        ladder = df.filter(pl.col('family').is_in(BATCH_FAMILIES[design]))
        if ladder.is_empty():
            raise ValueError(f'no batch-ladder runs on disk for design={design!r}')
        ladders[design] = ladder

    colors = batch_colors(
        sorted(
            {
                fraction
                for ladder in ladders.values()
                for fraction in ladder['batch_fraction'].unique().to_list()
            }
        )
    )
    fig, axd = style.mosaic('AB\nCD', panel_h_mm=PANEL_H_MM)
    for design, letters in (('matched', 'AB'), ('fixed', 'CD')):
        _draw_ladder(axd, letters, ladders[design], colors, DESIGN_TITLES[design])

    style.label_panels(axd)
    # All arms finish within a few points of 100%, so end labels in panels A/C
    # would have to fan far wider than their anchors are apart, which detaches
    # every label from its curve. A figure-level key covers no data, and matches
    # how F2 and F3 identify their series. Panels B/D keep end labels: their
    # points are genuinely separated.
    #
    # The key is built from `colors` rather than from one row's handles: neither
    # row holds every fraction, and only the union names the whole colour ramp.
    legend = fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=color,
                marker='o',
                markersize=3.6,
                linewidth=style.DATA_LINEWIDTH,
                label=f'{fraction:g}%',
            )
            for fraction, color in colors.items()
        ],
        # Ascending fractions in one tight row, so the key reads as the sequential
        # ramp it is. The defaults leave enough air between handle, label and
        # column for the eight entries to read as eight unrelated series. The
        # anchor clears panel C's rotated tick block and its axis label, which a
        # legend seated any higher lands on top of.
        title='batch fraction',
        loc='lower center',
        bbox_to_anchor=(0.5, -0.105),
        ncols=len(colors),
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.3,
        borderpad=0.0,
    )
    legend.get_title().set_fontsize(7.5)
    legend.get_title().set_fontweight('semibold')
    for text in legend.get_texts():
        text.set_fontsize(7.5)
    style.save(fig, 'fig04_batch')


def make_pruning_figure(df: pl.DataFrame) -> None:
    """Write the pruning frontier and reference-hit loss analyses."""
    prune = df.filter(pl.col('family').is_in(PRUNE_FAMILIES)).with_columns(
        pl.when(pl.col('family').is_in(DELAYED_PRUNE_FAMILIES))
        .then(pl.lit('delayed'))
        .otherwise(pl.lit('immediate'))
        .alias('pruning_schedule')
    )
    top_ids, top_k, _ = _initial_pool_top_ids()
    top_status = _top_status_records(prune, top_ids, top_k)
    lost = _pruned_top_summary(top_status)
    rank_records = _same_cycle_rank_records(top_status)

    pruned = data.final(prune, ['pruning_schedule', 'prune_fraction'], [Y]).sort(
        ['prune_fraction', 'pruning_schedule']
    )
    final_runs = prune.filter(
        pl.col('cycle') == pl.col('cycle').max().over('run_id')
    ).select(
        ['run_id', 'rep', 'pruning_schedule', 'prune_fraction', 'cum_prediction_time']
    )
    baseline_times = final_runs.filter(
        (pl.col('pruning_schedule') == 'immediate') & (pl.col('prune_fraction') == 0)
    ).select(['rep', pl.col('cum_prediction_time').alias('baseline_prediction_time')])
    speedup_summary = (
        final_runs.join(baseline_times, on='rep', how='left')
        .with_columns(
            (pl.col('baseline_prediction_time') / pl.col('cum_prediction_time')).alias(
                'prediction_speedup'
            )
        )
        .group_by(['pruning_schedule', 'prune_fraction'])
        .agg(
            pl.col('prediction_speedup').mean().alias('mean'),
            pl.col('prediction_speedup').min().alias('lo'),
            pl.col('prediction_speedup').max().alias('hi'),
            pl.len().alias('n_reps'),
        )
        .sort(['prune_fraction', 'pruning_schedule'])
    )
    lost_summary = (
        lost.group_by(['pruning_schedule', 'prune_fraction'])
        .agg(
            pl.col('pruned_top_count').mean().alias('mean'),
            pl.col('pruned_top_count').min().alias('lo'),
            pl.col('pruned_top_count').max().alias('hi'),
            pl.len().alias('n_reps'),
        )
        .sort(['prune_fraction', 'pruning_schedule'])
    )

    cycle_counts = (
        top_status.filter(
            (pl.col('status') == 'pruned') & (pl.col('prune_fraction') > 0)
        )
        .group_by(['run_id', 'pruning_schedule', 'prune_fraction', 'pruned_cycle'])
        .agg(pl.len().alias('count'))
    )
    cycle_summary = (
        cycle_counts.group_by(['pruning_schedule', 'prune_fraction', 'pruned_cycle'])
        .agg(
            pl.col('count').mean().alias('mean'),
            pl.col('count').min().alias('lo'),
            pl.col('count').max().alias('hi'),
        )
        .sort(['prune_fraction', 'pruning_schedule', 'pruned_cycle'])
    )
    timing_conditions = sorted(
        cycle_summary.select(['pruning_schedule', 'prune_fraction'])
        .unique()
        .iter_rows(named=True),
        key=lambda condition: (
            condition['prune_fraction'],
            0 if condition['pruning_schedule'] == 'immediate' else 1,
        ),
    )
    timing_cycles = sorted(cycle_summary['pruned_cycle'].unique().to_list())
    timing_values = np.zeros((len(timing_conditions), len(timing_cycles)))
    for row_index, condition in enumerate(timing_conditions):
        for col_index, cycle in enumerate(timing_cycles):
            value = cycle_summary.filter(
                (pl.col('pruning_schedule') == condition['pruning_schedule'])
                & (pl.col('prune_fraction') == condition['prune_fraction'])
                & (pl.col('pruned_cycle') == cycle)
            )['mean']
            if len(value):
                timing_values[row_index, col_index] = value[0]
    total_lost_by_condition = {
        (row['pruning_schedule'], row['prune_fraction']): row['mean']
        for row in lost_summary.iter_rows(named=True)
        if row['prune_fraction'] > 0
    }

    rank_values = {
        (
            condition['pruning_schedule'],
            condition['prune_fraction'],
        ): rank_records.filter(
            (pl.col('pruning_schedule') == condition['pruning_schedule'])
            & (pl.col('prune_fraction') == condition['prune_fraction'])
        )['predicted_percentile'].to_numpy()
        for condition in timing_conditions
    }

    fig = plt.figure(
        figsize=(style.DOUBLE_COL, 2 * 56 * style.MM),
        constrained_layout=True,
    )
    rows = fig.add_gridspec(2, 1)
    top = rows[0].subgridspec(1, 3)
    bottom = rows[1].subgridspec(1, 3)
    axd = {
        'A': fig.add_subplot(top[0, 0]),
        'B': fig.add_subplot(top[0, 1]),
        'C': fig.add_subplot(top[0, 2]),
        'D': fig.add_subplot(bottom[0, :2]),
        'E': fig.add_subplot(bottom[0, 2]),
    }
    for panel in axd.values():
        style_axes(panel)

    ax = axd['A']
    for schedule, (color, marker) in SCHEDULE_STYLE.items():
        summary = pruned.filter(pl.col('pruning_schedule') == schedule)
        if summary.is_empty():
            continue
        ax.errorbar(
            summary['prune_fraction'],
            summary[f'{Y}_mean'],
            yerr=[
                summary[f'{Y}_mean'] - summary[f'{Y}_lo'],
                summary[f'{Y}_hi'] - summary[f'{Y}_mean'],
            ],
            marker=marker,
            markersize=SCHEDULE_MARKERSIZE,
            linestyle='-',
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            elinewidth=0.6,
            label=SCHEDULE_LABELS[schedule],
        )
    ax.set_xticks(sorted(pruned['prune_fraction'].unique().to_list()))
    ax.set(xlabel='pruning fraction (%)', ylabel='Top-0.1% recovered (%)')
    # Both arms fall to the right, so the wedge above-right of every point is
    # clear of its own line. The recovery axis is already a percentage, so the
    # labels carry no % sign: at 6 pt the extra glyph is what makes the 0% and
    # 10% labels, only ten x-units apart, run into each other.
    for schedule, (color, _) in SCHEDULE_STYLE.items():
        summary = pruned.filter(pl.col('pruning_schedule') == schedule)
        for row in summary.iter_rows(named=True):
            ax.annotate(
                f'{row[f"{Y}_mean"]:.1f}',
                (row['prune_fraction'], row[f'{Y}_mean']),
                textcoords='offset points',
                xytext=(4, 4),
                ha='left',
                va='bottom',
                fontsize=POINT_LABEL_SIZE,
                color=color,
                fontweight='semibold',
            )
    # Room to the right for the 70% labels and above for the 0% one; the limits
    # are taken off the error-bar extremes so they track the data.
    recovery_lo = float(pruned[f'{Y}_lo'].min())
    recovery_hi = float(pruned[f'{Y}_hi'].max())
    recovery_span = recovery_hi - recovery_lo
    ax.set_xlim(-5, 84)
    ax.set_ylim(recovery_lo - 0.06 * recovery_span, recovery_hi + 0.12 * recovery_span)
    ax.legend(loc='lower left', handlelength=2.2, handletextpad=0.4)

    ax = axd['B']
    fractions = sorted(speedup_summary['prune_fraction'].unique().to_list())
    x = np.arange(len(fractions))
    width = 0.34
    fraction_colors = {
        fraction: style.SEQUENTIAL(value)
        for fraction, value in zip(
            fractions,
            np.linspace(0.25, 1.0, len(fractions)),
            strict=True,
        )
    }
    for schedule, offset, hatch, label in (
        ('immediate', -width / 2, '', 'immediate'),
        ('delayed', width / 2, '//', 'delayed start (c5)'),
    ):
        summary = speedup_summary.filter(pl.col('pruning_schedule') == schedule)
        if summary.is_empty():
            continue
        positions = np.array(
            [fractions.index(value) for value in summary['prune_fraction']]
        )
        ax.bar(
            positions + offset,
            summary['mean'],
            width=width,
            yerr=[
                summary['mean'] - summary['lo'],
                summary['hi'] - summary['mean'],
            ],
            color=[fraction_colors[value] for value in summary['prune_fraction']],
            hatch=hatch,
            label=label,
            error_kw={'elinewidth': 0.6, 'ecolor': style.INK},
        )
        for position, row in zip(
            positions + offset, summary.iter_rows(named=True), strict=True
        ):
            ax.annotate(
                f'{row["mean"]:.1f}×',
                (position, row['hi']),
                textcoords='offset points',
                xytext=(0, 5),
                ha='center',
                fontsize=7,
                fontweight='bold',
            )
    ax.set(
        xlabel='pruning fraction (%)',
        ylabel='Prediction speedup (×)',
    )
    ax.set_xticks(x, [f'{value:g}' for value in fractions])
    ax.margins(y=0.15)

    ax = axd['C']
    for schedule, (color, marker) in SCHEDULE_STYLE.items():
        summary = lost_summary.filter(pl.col('pruning_schedule') == schedule)
        if summary.is_empty():
            continue
        ax.errorbar(
            summary['prune_fraction'],
            summary['mean'],
            yerr=[
                summary['mean'] - summary['lo'],
                summary['hi'] - summary['mean'],
            ],
            marker=marker,
            markersize=SCHEDULE_MARKERSIZE,
            linestyle='-',
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            elinewidth=0.6,
        )
    ax.set(
        xlabel='pruning fraction (%)',
        ylabel='Top-0.1% pruned (n)',
    )
    ax.set_xticks(sorted(lost_summary['prune_fraction'].unique().to_list()))
    # Headroom for the two-line label above the 70% immediate point, which is the
    # tallest thing in the panel once labels sit on their own points.
    ax.set_ylim(0, max(lost_summary['mean']) * 1.30)
    ax.set_xlim(-9, 88)
    # Each value sits above its own point instead of fanning to a shared column:
    # the fan needed one leader line per point, and those near-horizontal rules
    # read as data in a panel whose data is two lines. Both curves rise
    # monotonically, so each is clear above one side of its own points. The two
    # arms take opposite sides: the immediate line passes above-left of the
    # delayed points, so those labels go right, and the immediate labels go left
    # into the space the delayed arm never reaches.
    for schedule, (color, _) in SCHEDULE_STYLE.items():
        summary = lost_summary.filter(pl.col('pruning_schedule') == schedule)
        offset, align = (
            ((4, 9), 'left') if schedule == 'delayed' else ((-3, 6), 'right')
        )
        for row in summary.iter_rows(named=True):
            # The 0% control goes unlabelled. Its label would read "0 (0.00%)",
            # which the axes already say, and at that width it collides with the
            # 10% label sitting only 43 hits away.
            if row['mean'] == 0:
                continue
            ax.annotate(
                f'{row["mean"]:.0f}\n({row["mean"] / top_k * 100:.2f}%)',
                (row['prune_fraction'], row['mean']),
                textcoords='offset points',
                xytext=offset,
                ha=align,
                va='bottom',
                fontsize=POINT_LABEL_SIZE,
                color=color,
                fontweight='semibold',
            )

    ax = axd['D']
    image = ax.imshow(
        timing_values,
        cmap=style.SEQUENTIAL,
        vmin=0,
        vmax=max(1.0, float(timing_values.max())),
        aspect='equal',
    )
    ax.grid(False)
    ax.set(
        xticks=np.arange(len(timing_cycles)),
        xticklabels=[f'c{cycle}' for cycle in timing_cycles],
        yticks=np.arange(len(timing_conditions)),
        yticklabels=[
            (
                f'{condition["prune_fraction"]:g}% delayed'
                if condition['pruning_schedule'] == 'delayed'
                else f'{condition["prune_fraction"]:g}% immediate'
            )
            for condition in timing_conditions
        ],
        xlabel='pruning cycle',
        ylabel='pruning schedule',
    )
    ax.tick_params(axis='x', labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    for row_index, condition in enumerate(timing_conditions):
        for col_index, value in enumerate(timing_values[row_index]):
            if value:
                ax.text(
                    col_index,
                    row_index,
                    f'{value:.0f}',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color=style.BACKGROUND
                    if value >= timing_values.max() * 0.55
                    else style.INK,
                )
        ax.annotate(
            f'total {total_lost_by_condition[(condition["pruning_schedule"], condition["prune_fraction"])]:.0f}',
            (1.02, row_index),
            xycoords=('axes fraction', 'data'),
            ha='left',
            va='center',
            fontsize=7,
            color=style.MUTED,
        )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.04)
    colorbar.set_label('mean lost hits (n)')
    colorbar.ax.yaxis.label.set_fontsize(7)

    ax = axd['E']
    timing_fractions = sorted(
        {condition['prune_fraction'] for condition in timing_conditions}
    )
    fraction_positions = {
        fraction: position for position, fraction in enumerate(timing_fractions)
    }
    box_data = []
    box_positions = []
    box_conditions = []
    for condition in timing_conditions:
        key = (condition['pruning_schedule'], condition['prune_fraction'])
        offset = -0.18 if condition['pruning_schedule'] == 'immediate' else 0.18
        box_data.append(rank_values[key])
        box_positions.append(fraction_positions[condition['prune_fraction']] + offset)
        box_conditions.append(condition)
    box = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': style.INK, 'linewidth': 1.2},
        whiskerprops={'color': style.MUTED, 'linewidth': 0.8},
        capprops={'color': style.MUTED, 'linewidth': 0.8},
        boxprops={'edgecolor': style.MUTED, 'linewidth': 0.8},
    )
    for patch, condition in zip(box['boxes'], box_conditions, strict=True):
        fraction = condition['prune_fraction']
        patch.set_facecolor(
            style.SEQUENTIAL(
                0.25
                + 0.75
                * timing_fractions.index(fraction)
                / max(1, len(timing_fractions) - 1)
            )
        )
        patch.set_alpha(0.85)
        if condition['pruning_schedule'] == 'delayed':
            patch.set_hatch('//')
    for condition, position in zip(box_conditions, box_positions, strict=True):
        fraction = condition['prune_fraction']
        cutoff = 100.0 * (1.0 - fraction / 100.0)
        ax.plot(
            [position - 0.15, position + 0.15],
            [cutoff, cutoff],
            color=style.DARK,
            linestyle='-',
            linewidth=style.DATA_LINEWIDTH,
        )
    ax.set(
        xticks=np.arange(len(timing_fractions)),
        xticklabels=[f'{fraction:g}%' for fraction in timing_fractions],
        xlabel='pruning fraction (%)',
        ylabel='predicted percentile\n(0 best; 100 worst)',
        ylim=(0, 100),
    )
    ax.text(
        0.02,
        0.04,
        'line = pruning boundary',
        transform=ax.transAxes,
        fontsize=6.5,
        color=style.MUTED,
    )
    ax.xaxis.label.set_fontsize(7)
    ax.yaxis.label.set_fontsize(7)

    style.label_panels(axd)
    fig.canvas.draw()
    left = axd['A'].get_position().frozen()
    middle = axd['B'].get_position().frozen()
    right = axd['C'].get_position().frozen()
    middle_x = (left.x1 + right.x0 - middle.width) / 2
    fig.set_layout_engine('none')
    axd['B'].set_position([middle_x, middle.y0, middle.width, middle.height])
    style.save(fig, 'fig04_pruning')


def main() -> None:
    style.apply()
    # One row per seed design, each loaded separately and never pooled: `fixed`
    # varies only the acquisition batch off a constant 1% seed, `matched` seeds
    # each arm with its own batch. Loading both at once would merge two different
    # runs into one 0.2% and one 0.1% curve.
    make_batch_figure({design: data.load(design) for design in data.DESIGNS})
    # The pruning families all sit at a 1.0% batch off a 1.0% seed, so they satisfy
    # both designs and one figure covers them.
    make_pruning_figure(data.load('fixed'))


if __name__ == '__main__':
    main()

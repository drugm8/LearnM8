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
INITIAL_POOL = Path.home() / 'LearnM8_DATA' / 'AmpC_screen_10000K.csv'
TOP_FRACTION = 0.001
# In-plot batch-fraction labels, 20% down from the 7 pt used for axis-adjacent
# annotation: they sit inside the data area and must not compete with the curves.
FRACTION_LABEL_SIZE = 5.6


def batch_colors(fractions: list[float]) -> dict[float, str]:
    """Opaque sequential purple ramp ordered by batch size."""
    ranks = np.linspace(0.70, 0.98, len(fractions))
    return {
        f: style.SEQUENTIAL(r) for f, r in zip(sorted(fractions), ranks, strict=True)
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


def make_batch_figure(df: pl.DataFrame, design: str) -> None:
    """Write the batch-frequency figure (three panels) for one seed design."""
    batch = df.filter(pl.col('family').is_in(BATCH_FAMILIES[design]))
    if batch.is_empty():
        raise ValueError(f'no batch-ladder runs on disk for design={design!r}')
    fractions = sorted(batch['batch_fraction'].unique().to_list())
    colors = batch_colors(fractions)
    fig, axd = style.mosaic('ABC', panel_h_mm=52)

    ax = axd['A']
    curves = data.curve(batch, ['batch_fraction'], X, Y)
    for index, frac in enumerate(fractions):
        s = curves.filter(pl.col('batch_fraction') == frac)
        ax.plot(
            s[X],
            s['mean'],
            color=colors[frac],
            linestyle=style.CURVE_LINESTYLES[index],
            marker=style.CURVE_MARKERS[index],
            # Six ladder arms crowd the same corner, so dense 3.5 pt glyphs fuse
            # into what reads as a thicker stroke than the 1.5 pt used elsewhere.
            markevery=max(1, s.height // 6),
            markersize=2.6,
            label=f'{frac:g}%',
            linewidth=style.DATA_LINEWIDTH,
        )
        style.band(ax, s[X], s['lo'], s['hi'], colors[frac], alpha=0.09)
    ax.set(xlabel=style.LABELS[X], ylabel=style.LABELS[Y])
    style.compact_axis(ax)
    style.set_compound_axis(
        ax,
        int(batch['pool_size'].unique().item()),
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

    retrains = batch.group_by('batch_fraction').agg(
        pl.col('cycle').max().alias('retrains')
    )
    summary = (
        data.final(batch, ['batch_fraction'], [Y, 'cum_ml_time'])
        .join(retrains, on='batch_fraction')
        .sort('retrains')
    )

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
    ax.set(xscale='log', xlabel='number of retrains', ylabel='Top-0.1% recovered (%)')

    ax = axd['C']
    points = list(summary.iter_rows(named=True))
    for row in points:
        ax.plot(
            row['cum_ml_time_mean'],
            row[f'{Y}_mean'],
            marker='o',
            color=colors[row['batch_fraction']],
        )
    x_values = [row['cum_ml_time_mean'] for row in points]
    y_values = [row[f'{Y}_mean'] for row in points]
    x_max = max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    ax.set_xlim(left=min(x_values) * 0.82, right=x_max * 1.45)
    ax.set_ylim(y_min - 0.8, y_max + 0.8)
    label_x = x_max * 1.05
    label_ys = np.linspace(y_max + 0.4, y_min - 0.4, len(points))
    for row, label_y in zip(
        sorted(points, key=lambda item: item[f'{Y}_mean'], reverse=True),
        label_ys,
        strict=True,
    ):
        color = colors[row['batch_fraction']]
        ax.annotate(
            f'{row["batch_fraction"]:g}%',
            (row['cum_ml_time_mean'], row[f'{Y}_mean']),
            xytext=(label_x, label_y),
            textcoords='data',
            ha='left',
            va='center',
            fontsize=FRACTION_LABEL_SIZE,
            color=color,
            fontweight='semibold',
            arrowprops={
                'arrowstyle': '-',
                'color': color,
                'linewidth': 0.7,
                'shrinkA': 3,
                'shrinkB': 3,
            },
        )
    ax.set(
        xscale='log',
        xlabel=style.LABELS['cum_ml_time'],
        ylabel='Top-0.1% recovered (%)',
    )
    ax.margins(x=0.18, y=0.12)

    style.label_panels(axd)
    # All arms finish within a few points of 100%, so end labels in panel A
    # would have to fan far wider than their anchors are apart, and in a 28 mm
    # panel that detaches every label from its curve. A figure-level key covers
    # no data, and matches how F2 and F3 identify their series. Panel C keeps
    # end labels: its points are genuinely separated.
    handles, labels = axd['A'].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title='batch fraction',
        loc='lower center',
        bbox_to_anchor=(0.5, -0.16),
        ncols=len(labels),
    )
    style.save(fig, f'fig04_batch{data.design_suffix(design)}')


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
    for schedule, marker, linestyle, color, label in (
        ('immediate', 'o', '-', style.PRIMARY, 'immediate'),
        ('delayed', 'D', '--', style.DARK, 'delayed start (c5)'),
    ):
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
            linestyle=linestyle,
            color=color,
            markerfacecolor=style.BACKGROUND if schedule == 'delayed' else color,
            markeredgecolor=color,
            elinewidth=0.6,
            label=label,
        )
    ax.set_xticks(sorted(pruned['prune_fraction'].unique().to_list()))
    ax.set(xlabel='pruning fraction (%)', ylabel='Top-0.1% recovered (%)')
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
    for schedule, marker, linestyle, color in (
        ('immediate', 'o', '-', style.DARK),
        ('delayed', 'D', '--', style.ACCENT_ORANGE),
    ):
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
            linestyle=linestyle,
            color=color,
            markerfacecolor=style.BACKGROUND if schedule == 'delayed' else color,
            markeredgecolor=color,
            elinewidth=0.6,
        )
    ax.set(
        xlabel='pruning fraction (%)',
        ylabel='Top-0.1% pruned (n)',
        ylim=(0, None),
    )
    ax.set_xticks(sorted(lost_summary['prune_fraction'].unique().to_list()))
    ax.set_ylim(0, max(lost_summary['mean']) * 1.15)
    ax.set_xlim(-5, 100)
    label_x = 76.0
    immediate_lost = lost_summary.filter(pl.col('pruning_schedule') == 'immediate')
    label_ys = np.linspace(110.0, 980.0, immediate_lost.height)
    for row, label_y in zip(
        immediate_lost.iter_rows(named=True), label_ys, strict=True
    ):
        ax.annotate(
            f'{row["mean"]:.0f}\n({row["mean"] / top_k * 100:.2f}%)',
            (row['prune_fraction'], row['mean']),
            textcoords='data',
            xytext=(label_x, label_y),
            ha='left',
            va='center',
            fontsize=7,
            fontweight='semibold',
            arrowprops={
                'arrowstyle': '-',
                'color': style.MUTED,
                'linewidth': 0.6,
            },
        )
    for row in lost_summary.filter(pl.col('pruning_schedule') == 'delayed').iter_rows(
        named=True
    ):
        ax.annotate(
            f'{row["mean"]:.0f}',
            (row['prune_fraction'], row['mean']),
            xytext=(5, -12),
            textcoords='offset points',
            fontsize=7,
            color=style.ACCENT_ORANGE,
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
    # One batch figure per seed design, never a pooled one: `fixed` varies only the
    # acquisition batch off a constant 1% seed, `matched` seeds each arm with its own
    # batch. Loading both at once would merge two different runs into one 0.2% and
    # one 0.1% curve, and filtering to either design alone drops the other ladder.
    for design in data.DESIGNS:
        make_batch_figure(data.load(design), design)
    # The pruning families all sit at a 1.0% batch off a 1.0% seed, so they satisfy
    # both designs and one figure covers them.
    make_pruning_figure(data.load('fixed'))


if __name__ == '__main__':
    main()

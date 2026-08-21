"""F5 - when active learning pays, over the library-size x oracle-cost plane.

Three panels over the same plane.

  A (fig05_scaling)           one oracle evaluation at a time.
  B (fig05b_scaling_parallel) ORACLE_WORKERS_B docked at once, AL on AL_CORES_B
                              cores.
  C (fig05c_breakeven_by_machine) break-even alone, one curve per machine in
                              MACHINE_CONFIGS. A and B fix the machine and vary
                              the speedup; C fixes the speedup at 1x and varies
                              the machine, which is the axis a reader actually
                              controls. Its result is a threshold rather than a
                              trend: above ~750 CPUs a 1 s/compound oracle never
                              repays the pipeline at ANY library size, because
                              break-even has risen past the oracle itself.

Panel A's arithmetic is in serial-equivalent seconds, which is the right unit
for "how much compute does this cost" and the wrong one for "when do I get my
answer". The oracle is embarrassingly parallel and the pipeline is not: the
pipeline term is already a wall-clock measurement of a parallel program on one
node, so it does not divide again. Running the oracle W-way therefore shrinks
only the term AL avoids and shifts every iso-speedup locus up by exactly W. At
W=32 break-even moves from ~1.1 ms to ~36 ms per compound, which is the
difference between "AL always pays" and "AL pays for real docking only".

Panel B is the honest picture for anyone with a docking farm, and it is the
same warning as the small-N end of panel A from the other direction: AL's
margin is eaten by whatever you can throw at the oracle but not at the pipeline.

Fill is the campaign time; contours are the speedup over exhaustively docking
the same library. Both follow from

    AL seconds         = set-up + rate * N + labeled(N) * cost
    exhaustive seconds = N * cost

`rate` is the whole-pipeline marginal cost per compound (featurize + train +
predict + select) and `set-up` is the part of a campaign that does not shrink
with the library: interpreter and RDKit start-up, SMILES validation, the cold
featurization pass, and per-cycle bookkeeping.

Earlier versions set the set-up term to zero and quoted one rate, reasoning
that the four measured pools span 0.44-0.96 ms/compound with no constant trend.
Two things are now known about that reading:

  * Those four pools cross three container images, and the manifest records no
    node, CPU model or core count. The documented machine effect is 1.8x
    (PLAN_RESULTS.md 0.2), larger than the 2.2x spread being read as evidence.
    Their range therefore cannot establish that the rate is scale-free, and the
    two-parameter fit that "returns a negative set-up time" was fitting noise:
    all four sit at or above 1e6, where a ten-second set-up is under 1.5% of
    the total and the intercept is simply not identifiable.
  * A controlled ladder at 1e3 / 1e4 / 1e5 on one machine and one image
    separates the terms directly. The per-compound rate there is not constant:
    it falls ~28x from 1e3 to 1e5, and `total = set-up + marginal * N` fits at
    R^2 >= 0.999 with a marginal of 0.58 ms/compound on 32 cores - which is what
    the 1M cluster pool measures (0.57 ms). The two agree closely once the
    intercept is admitted, which is the reason to trust the decomposition.

A third correction applies to the same four pools: only two of them featurized
their library inside the run. Cycle 1 is the only cycle that can meet an empty
cache, so it must dominate - 8.2x the steady-state cycle at 1M, 12.8x at 10M,
13.5-337x on the local ladder. At 50M and 96M the ratio inverts to 0.42x and
0.48x, which no cold run produces: both were started against a pre-warmed cache
and their production time excludes featurising the pool once, by 2.1 h and 4.0 h.
Every rate here is therefore reported cold-equivalent (see COLD_PASS_RATE), and
PIPELINE_RATE is calibrated against those rather than the raw numbers. Uncorrected,
the largest pool reads 0.963 ms/compound instead of 1.114 - understating the
pipeline most exactly where the pipeline is largest.

The consequence for this figure is that the speedup is NOT scale-free. N no
longer cancels out of the iso-speedup condition, so those loci curve: flat
above ~1e6, where they reproduce the old horizontal lines, and climbing steeply
below ~1e5, where set-up dominates. This matters because POOL_LIMITS starts at
1e3, two decades below the smallest pool ever measured on the cluster.

Set-up is hardware-dependent in a way the marginal rate is not, and it grows
with core count rather than shrinking: the featurization worker pool is
dispatched per call regardless of how little there is to featurize. The shaded
band spans a 2-core and a 32-core node - a factor of two in where break-even
sits at 1e3, and invisible by 1e7. Adding cores is therefore not a free win at
the small end: over the same sweep 1e3 compounds finish in 3.5 s on one core
and 22.0 s on thirty-two, while 1e5 compounds reverse that (226 s against 79 s).

The 1.0% batch ladder is the only one with all four pools, and at a 1.0% batch
the two cycle-0 seed designs are the same runs, so this figure renders once.
"""

from __future__ import annotations

import data
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import style
from matplotlib.colors import BoundaryNorm

from learnm8.visualization.style import style_axes

SCALE_FAMILIES = {
    'PILOT-1M': data.POOL_SIZE['1M'],
    'S-02': data.POOL_SIZE['10M'],
    'S-03': data.POOL_SIZE['50M'],
    'S-04': data.POOL_SIZE['96M'],
}
BATCH_FRACTION = 1.0
SECONDS_PER_HOUR = 3_600
# Marginal whole-pipeline seconds per compound. Round, and just above the
# slowest COLD-EQUIVALENT pool (1.114 ms at 96M), so the plane stays a mild
# over-estimate of AL's own cost everywhere.
#
# This was 1.0e-3 while the 96M pool was read as 0.963 ms/compound. That run met
# a pre-warmed cache: its production time never included featurising 96M
# compounds for the first time, so it was not a whole-pipeline rate at all.
# Adding back the first-pass cost the two cold pools actually measure
# (COLD_PASS_RATE) puts it at 1.114 ms, which 1.0e-3 sits BELOW - the plane was
# understating AL's cost at the top end, and an understated pipeline draws
# break-even too low, i.e. flatters AL exactly where the figure is used to argue
# for it. 1.2e-3 restores the one-sided guarantee and stays a round number.
PIPELINE_RATE = 1.2e-3
# Seconds per compound to featurize a pool for the first time, from the cycle-1
# excess over steady state on the two cold cluster pools: 0.135 ms/compound at
# 1M and 0.168 at 10M. Used to put the two pre-warmed pools on the same footing
# as the cold ones before anything is calibrated against them.
COLD_PASS_RATE = 0.151e-3
# Seconds that do NOT shrink with the library: interpreter and RDKit start-up,
# SMILES validation, the cold featurization pass, and per-cycle bookkeeping.
# Setting this to zero is what made the old plane scale-free. Round, for the
# same reason PIPELINE_RATE is - it is hardware-dependent, and the band below
# is the honest width, not this number.
PIPELINE_SETUP = 15.0
# OLS intercepts of total wall seconds on pool size over a 1e3/1e4/1e5 ladder,
# one 32-core node, rf/morgan/greedy, 1% batch, 10 cycles, cold cache
# (R^2 >= 0.999 at every entry). More cores costs MORE set-up: the featurization
# worker pool is dispatched per call whether it has a thousand compounds to chew
# on or a million. The marginal rate moves the other way over the same sweep
# (1.33 -> 0.58 ms/compound from 2 to 32 cores), and that is the whole trade:
# below ~1e4 compounds fewer cores finish sooner, above ~1e5 more cores do.
#
# n_jobs=1 is deliberately absent. It bypasses the worker pool entirely, so its
# true intercept is ~0, but OLS returns -1.8 s there: with no set-up term to
# absorb it, the mildly superlinear RF cost (prediction grows with the training
# set, which grows with the pool) drags the intercept negative. That is the same
# failure the four cluster pools show, and an inadmissible fit is not a datum.
PIPELINE_SETUP_BY_CORES = {2: 10.9, 32: 21.0}
# Panel B pins both sides of the trade instead of leaving the oracle serial:
# ORACLE_WORKERS_B docking jobs at once, and AL on AL_CORES_B cores. Panel A
# leaves the oracle at one worker, which is what makes its y-axis readable as
# "seconds of docking" but also what flatters AL - the term AL avoids is the
# only one that parallelising shrinks.
ORACLE_WORKERS_B = 32
AL_CORES_B = 32
# Panel C sweeps the machine instead of the speedup: total CPUs available, all
# of them docking (the oracle is embarrassingly parallel) and AL using as many
# as were actually measured.
MACHINE_CONFIGS = (1, 32, 128, 512, 1000)
# How much worse one core is per compound: 2.279 ms against 0.577 ms on 32, both
# from the same local sweep. Expressed as a RATIO rather than the absolute
# 2.279e-3, because PIPELINE_RATE is calibrated on the cluster and pairing a
# local absolute with it would mix two machines' hardware into one curve - and
# in the flattering direction, since the local box is the faster of the two per
# compound. The ratio is the part the controlled sweep actually establishes.
SINGLE_CORE_PENALTY = 2.279 / 0.577
SINGLE_CORE_RATE = PIPELINE_RATE * SINGLE_CORE_PENALTY
# Cycle-1 / steady-state featurization ratio above which a run is taken to have
# met an empty cache. Cold runs measure 8.2x and 12.8x on the cluster and
# 13.5-337x locally; pre-warmed ones measure 0.42x and 0.48x. Nothing observed
# lands near 2, so the threshold is not a close call.
COLD_CACHE_RATIO = 2.0
# Wall-clock per compound for two docking engines, as reference rows rather than
# as measurements of this pipeline.
DOCKING_TIMES = {'Uni-Dock': 1.0, 'Vina': 30.0}
# 0.1 ms covers a surrogate model's own inference; 1000 s covers the slowest
# routine physics-based scoring. Anything outside is not a screening oracle.
COST_LIMITS = (1e-4, 1e3)
POOL_LIMITS = (1e3, None)
SPEEDUP_LEVELS = (1.0, 2.0, 5.0, 9.0)
# Campaign-time bands in hours. Durations a reader can hold in their head beat
# decades of a continuous ramp, which reads as a smear at this dynamic range.
TIME_BANDS = {
    1 / 60: '1 min',
    1: '1 h',
    24: '1 day',
    168: '1 week',
    730: '1 month',
    8_766: '1 year',
    87_660: '10 years',
    876_600: '100 years',
}
GRID_POINTS = 400


def _with_production_time(frame: pl.DataFrame) -> pl.DataFrame:
    """Exclude benchmark-only evaluation and CSV-oracle lookup time."""
    return (
        frame.sort(['run_id', 'cycle'])
        .with_columns(
            production_time=(
                pl.col('total_time').fill_null(0.0)
                - pl.col('evaluation_time').fill_null(0.0)
                - pl.col('oracle_time').fill_null(0.0)
            ),
        )
        .with_columns(
            cum_production_time=pl.col('production_time').cum_sum().over('run_id')
        )
    )


def _measured_scaling(df: pl.DataFrame) -> pl.DataFrame:
    """Return per-pool pipeline seconds and labeled compounds for the 1% ladder."""
    scales = df.filter(
        pl.col('family').is_in(list(SCALE_FAMILIES))
        & (pl.col('batch_fraction') == BATCH_FRACTION)
    )
    present = scales['family'].unique().to_list()
    for missing in SCALE_FAMILIES.keys() - set(present):
        print(
            f'note: {missing} ({SCALE_FAMILIES[missing]:,} compounds) '
            'not on disk - omitted'
        )
    totals = data.final(
        _with_production_time(scales),
        ['family'],
        ['cum_production_time', 'cumulative_labeled'],
    )
    return (
        totals.with_columns(
            pool_size=pl.col('family').replace_strict(
                SCALE_FAMILIES, return_dtype=pl.Int64
            )
        )
        .join(_cold_cache_flag(scales), on='family', how='left')
        .sort('pool_size')
    )


def _cold_cache_flag(scales: pl.DataFrame) -> pl.DataFrame:
    """Flag which pools paid their first-pass featurization inside the run.

    Cycle 1 is the only cycle that can meet an empty cache, so on a cold run it
    featurizes the whole pool while cycles 2+ only read it back and featurize
    the new 1% batch. Cycle 1 must therefore dominate. Measured, it does: 8.2x
    the steady-state cycle at 1M and 12.8x at 10M, and 13.5-337x on the local
    ladder where the cache directory is deleted before every run.

    At 50M and 96M the ratio inverts to 0.42x and 0.48x, which no cold run can
    produce. Those two pools were started against a pre-warmed cache, so their
    `production_time` excludes the first-pass featurization entirely - 2.1 h and
    4.0 h respectively, or +0.15 ms/compound. Quoting them as measured pipeline
    rates understates both, and understates the largest pool the most.
    """
    per_cycle = scales.group_by('family', 'cycle').agg(
        pl.col('feature_extraction_time').mean()
    )
    first = per_cycle.filter(pl.col('cycle') == 1).select(
        'family', cycle1=pl.col('feature_extraction_time')
    )
    steady = (
        per_cycle.filter(pl.col('cycle') >= 2)
        .group_by('family')
        .agg(steady=pl.col('feature_extraction_time').mean())
    )
    return first.join(steady, on='family', how='inner').with_columns(
        cold_cache=pl.col('cycle1') > COLD_CACHE_RATIO * pl.col('steady')
    )


def _labeled_fraction(measured: pl.DataFrame) -> float:
    """Return the single labeled-budget fraction shared by the measured ladder."""
    fractions = (measured['cumulative_labeled_mean'] / measured['pool_size']).to_numpy()
    if not np.allclose(fractions, fractions[0], rtol=1e-6):
        raise ValueError(
            f'the ladder does not share one labeled budget: {fractions.tolist()}'
        )
    return float(fractions[0])


def _measured_rates(measured: pl.DataFrame, cold_equivalent: bool = True) -> np.ndarray:
    """Return whole-pipeline seconds per compound, one per pool.

    By default the two pre-warmed pools are put back on a cold footing by adding
    COLD_PASS_RATE, because their raw production time never included featurising
    the pool once. Reporting them raw understates the pipeline, and understates
    the biggest pool most - 0.963 ms against 1.114 ms cold-equivalent at 96M.

    Pass ``cold_equivalent=False`` for the raw numbers as recorded.
    """
    rates = (measured['cum_production_time_mean'] / measured['pool_size']).to_numpy()
    if not cold_equivalent:
        return rates
    warm = ~measured['cold_cache'].to_numpy()
    return rates + warm * COLD_PASS_RATE


def _campaign_hours(
    pools: np.ndarray,
    costs: np.ndarray,
    labeled_fraction: float,
    setup: float = PIPELINE_SETUP,
    oracle_workers: int = 1,
) -> np.ndarray:
    """Return AL campaign hours on the (cost, pool) grid.

    ``oracle_workers`` divides the oracle term only. The pipeline is already a
    parallel program measured as wall-clock on one node, so it does not divide
    again - which is exactly why parallelising the oracle erodes AL's margin.
    """
    al_seconds = (
        setup
        + (PIPELINE_RATE + labeled_fraction * costs[:, None] / oracle_workers)
        * pools[None, :]
    )
    return al_seconds / SECONDS_PER_HOUR


def _speedup_cost(
    level: float,
    labeled_fraction: float,
    pools: np.ndarray,
    setup: float | None = None,
    oracle_workers: int = 1,
    rate: float | None = None,
) -> np.ndarray:
    """Return the oracle cost at which AL is exactly ``level`` times faster.

    Solving (N*c/W) / (setup + rate*N + fraction*N*c/W) = level for c gives

        c(N) = W * level * (setup/N + rate) / (1 - level * fraction)

    which is the old closed form with `rate` replaced by the EFFECTIVE
    per-compound rate at that pool size, setup/N + rate. The setup term is what
    bends the locus. It vanishes as N grows, so the curves flatten to the old
    horizontal asymptote above ~1e6 and the scale-free reading survives where
    it was actually measured; below ~1e5 it dominates and the curves climb.

    W (``oracle_workers``) enters as a clean multiplier: running the oracle W-way
    parallel shifts every iso-speedup locus up by exactly W, because it divides
    the term AL avoids while leaving the pipeline it adds untouched.

    Drawing these directly rather than contouring the grid keeps them exact and
    sidesteps clabel, which erases half of a sparse contour when asked to place
    a label on it.
    """
    if level * labeled_fraction >= 1.0:
        raise ValueError(
            f'{level}x is at or past the {1 / labeled_fraction:g}x ceiling set '
            f'by the {labeled_fraction:.0%} labeled budget'
        )
    seconds = PIPELINE_SETUP if setup is None else setup
    effective_rate = seconds / pools + (PIPELINE_RATE if rate is None else rate)
    return oracle_workers * level * effective_rate / (1 - level * labeled_fraction)


def _machine_model(cpus: int) -> tuple[float, float]:
    """Return (set-up seconds, marginal seconds per compound) for a `cpus` box.

    AL is held at the largest core count actually measured. Beyond 32 the
    marginal was already flattening - 0.612 ms at 16 cores against 0.577 at 32,
    a 6% gain for twice the cores - so projecting a pipeline rate out to 1000
    would be invention, and it would make AL look better exactly where the
    figure is trying to show it looking worse.

    One CPU is its own regime rather than a point on that trend: n_jobs=1 never
    dispatches the featurization worker pool, so there is no pool set-up to pay
    at all. OLS returns -1.8 s there, which is zero within the scatter that the
    superlinear RF term produces, so 0.0 is used as the physical value and the
    measured single-core marginal comes with it.
    """
    if cpus < 2:
        return 0.0, SINGLE_CORE_RATE
    return PIPELINE_SETUP_BY_CORES[max(PIPELINE_SETUP_BY_CORES)], PIPELINE_RATE


def _pool_label(pool: float) -> str:
    """Label a pool size compactly, with no spurious trailing zero."""
    for divisor, suffix in ((1_000_000, 'M'), (1_000, 'k')):
        if pool >= divisor:
            scaled = pool / divisor
            return (
                f'{scaled:.0f}{suffix}'
                if scaled.is_integer()
                else f'{scaled:.1f}{suffix}'
            )
    return f'{pool:.0f}'


def _render(
    measured: pl.DataFrame,
    # Not `name`: the docking-reference loop below already binds that, and
    # shadowing it silently saved both panels as "Vina".
    output_name: str,
    setup: float = PIPELINE_SETUP,
    oracle_workers: int = 1,
) -> None:
    """Draw one panel of the plane. ``oracle_workers=1`` reproduces panel A."""
    # contourf hatching defaults to a heavy black cross-weave that outweighs the
    # data it annotates; this is the same hairline the axes spines use.
    plt.rcParams.update({'hatch.linewidth': 0.4, 'hatch.color': style.MUTED})
    measured_pools = measured['pool_size'].to_numpy().astype(float)
    labeled_fraction = _labeled_fraction(measured)
    parallel = oracle_workers > 1

    pools = np.logspace(
        np.log10(POOL_LIMITS[0]), np.log10(measured_pools.max()), GRID_POINTS
    )
    costs = np.logspace(np.log10(COST_LIMITS[0]), np.log10(COST_LIMITS[1]), GRID_POINTS)
    hours = _campaign_hours(pools, costs, labeled_fraction, setup, oracle_workers)
    grid_pool, grid_cost = np.meshgrid(pools, costs)

    fig, ax = plt.subplots(figsize=(125 * style.MM, 76 * style.MM))
    boundaries = list(TIME_BANDS)
    bands = ax.contourf(
        grid_pool,
        grid_cost,
        hours,
        levels=boundaries,
        colors=style.SEQUENTIAL(np.linspace(0.10, 0.92, len(boundaries) - 1)),
        norm=BoundaryNorm(boundaries, len(boundaries) - 1),
        extend='both',
    )
    bands.cmap.set_under(style.SEQUENTIAL(0.02))
    bands.cmap.set_over(style.DARK)

    break_even = _speedup_cost(1.0, labeled_fraction, pools, setup, oracle_workers)
    # Label on the flat right-hand half, where the curves are separated and
    # horizontal; the left half is where they crowd together as they climb.
    label_at = int(0.60 * len(pools))
    for level in SPEEDUP_LEVELS:
        cost = _speedup_cost(level, labeled_fraction, pools, setup, oracle_workers)
        ax.plot(
            pools,
            cost,
            color=style.INK,
            linewidth=1.4 if level == 1.0 else 0.7,
            # The theme defaults every line to an 'o' marker, which would land a
            # dot on every one of the 400 grid points.
            marker='none',
        )
        ax.annotate(
            f'{level:g}x break-even' if level == 1.0 else f'{level:g}x',
            (pools[label_at], cost[label_at]),
            ha='center',
            va='center',
            fontsize=6,
            color=style.INK,
            bbox={
                'facecolor': style.BACKGROUND,
                'edgecolor': 'none',
                'alpha': 0.9,
                'pad': 1.2,
            },
        )

    if parallel:
        # Panel B fixes the core count, so there is no core-count band to draw.
        # What matters instead is how far parallelising the oracle moved the
        # line: the serial locus is panel A's, exactly `oracle_workers` lower.
        serial = _speedup_cost(1.0, labeled_fraction, pools, setup)
        ax.plot(
            pools,
            serial,
            color=style.INK,
            linewidth=0.9,
            linestyle=(0, (1, 2)),
            marker='none',
        )
        ax.annotate(
            f'break-even with a serial oracle ({oracle_workers}x lower)',
            (pools[label_at], serial[label_at]),
            xytext=(0, -7),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=5.5,
            color=style.MUTED,
        )
    else:
        # How far the break-even line moves across the measured core counts. The
        # bounds converge at large N and separate below ~1e5, which is the honest
        # statement of what is known: at scale the hardware does not change where
        # AL pays, and at the small end it does.
        low, high = (
            _speedup_cost(1.0, labeled_fraction, pools, setup=core_setup)
            for core_setup in (
                min(PIPELINE_SETUP_BY_CORES.values()),
                max(PIPELINE_SETUP_BY_CORES.values()),
            )
        )
        ax.fill_between(pools, low, high, facecolor=style.INK, alpha=0.12, linewidth=0)

    # Below break-even the pipeline costs more than the oracle calls it avoids,
    # so the campaign is slower than simply docking everything.
    ax.fill_between(
        pools,
        costs[0],
        break_even,
        facecolor='none',
        edgecolor=style.MUTED,
        hatch='///',
        linewidth=0,
    )
    ax.annotate(
        'AL slower than exhaustive',
        (0.02, 0.02),
        xycoords='axes fraction',
        va='bottom',
        fontsize=6,
        color=style.INK,
        fontweight='semibold',
        bbox={
            'facecolor': style.BACKGROUND,
            'edgecolor': 'none',
            'alpha': 0.85,
            'pad': 1.5,
        },
    )
    rates = _measured_rates(measured) * 1e3
    if parallel:
        provenance = (
            f'{oracle_workers} oracle jobs in parallel, AL on {AL_CORES_B} cores\n'
            f'y-axis is still cost per compound on ONE core, so the\n'
            f'wall-clock the campaign pays is that over {oracle_workers}'
        )
    else:
        warm = int((~measured['cold_cache']).sum())
        provenance = (
            f'(cluster pools span {rates.min():.2f}-{rates.max():.2f} ms/compound '
            f'cold-equivalent;\n{warm} ran pre-warmed, + {COLD_PASS_RATE * 1e3:.2f} '
            f'ms/compound added back;\nband = {min(PIPELINE_SETUP_BY_CORES)}-'
            f'{max(PIPELINE_SETUP_BY_CORES)} cores)'
        )
    ax.annotate(
        f'pipeline {setup:g} s set-up + {PIPELINE_RATE * 1e3:g} ms/compound\n'
        f'{provenance}\n'
        f'ceiling {1 / labeled_fraction:.0f}x: '
        f'{labeled_fraction:.0%} of the library is docked',
        (0.025, 0.965),
        xycoords='axes fraction',
        va='top',
        fontsize=6,
        color=style.INK,
        bbox={
            'facecolor': style.BACKGROUND,
            'edgecolor': 'none',
            'alpha': 0.85,
            'pad': 2.0,
        },
    )

    for name, seconds in DOCKING_TIMES.items():
        if not COST_LIMITS[0] <= seconds <= COST_LIMITS[1]:
            continue
        ax.axhline(
            seconds,
            color=style.BACKGROUND,
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            marker='none',
        )
        ax.annotate(
            f'{name} ({seconds:g} s)',
            (pools[-1], seconds),
            xytext=(-4, 3),
            textcoords='offset points',
            ha='right',
            fontsize=6,
            color=style.BACKGROUND,
            fontweight='semibold',
        )

    # The pool sizes that were actually run, as a check on the stated rate.
    # Hollow marks a pre-warmed cache: that pool's rate excludes its first-pass
    # featurization, so it is not evidence about the whole-pipeline cost.
    for pool, cold in zip(
        measured_pools, measured['cold_cache'].to_list(), strict=True
    ):
        ax.plot(
            [pool],
            [costs[0]],
            marker='^',
            markersize=3.2,
            color=style.INK,
            markerfacecolor=style.INK if cold else style.BACKGROUND,
            markeredgewidth=0.7,
            clip_on=False,
        )

    ax.set(
        xscale='log',
        yscale='log',
        xlim=(pools[0], pools[-1]),
        ylim=(costs[0], costs[-1]),
        # Cluster pools only. The 1e3-1e5 ladder that calibrates the set-up term
        # ran on a different machine, and plotting both as one series is the
        # provenance error that PLAN_RESULTS.md 0.2 documents.
        # No hollow-triangle glyph here: U+25B3 is absent from Nimbus Sans and
        # renders as tofu. The marker itself carries fill/hollow; this names it.
        xlabel=(
            'Library size (compounds)   cluster pools: filled = cold cache, '
            'hollow = pre-warmed'
        ),
        ylabel='Oracle cost per compound (s)',
    )
    pool_ticks = [1e3, 1e4, 1e5, 1e6, 1e7, measured_pools.max()]
    ax.set_xticks(pool_ticks, [_pool_label(pool) for pool in pool_ticks])
    ax.set_xticks([], minor=True)
    ax.set_yticks(
        [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        ['0.1 ms', '1 ms', '10 ms', '0.1 s', '1 s', '10 s', '100 s', '1000 s'],
    )
    ax.set_yticks([], minor=True)
    style_axes(ax)
    ax.grid(False)

    colorbar = fig.colorbar(bands, ax=ax, pad=0.02, fraction=0.045, spacing='uniform')
    colorbar.set_ticks(boundaries, labels=list(TIME_BANDS.values()))
    # Same bands in both panels, so the fill is comparable between them by eye.
    colorbar.set_label(
        'AL campaign wall-clock' if parallel else 'Active-learning campaign time'
    )
    colorbar.ax.yaxis.label.set_fontsize(7)
    colorbar.ax.tick_params(labelsize=6.5)
    style.save(fig, output_name)


def _render_breakeven(measured: pl.DataFrame, output_name: str) -> None:
    """Panel C: the break-even locus alone, one curve per machine size.

    Panels A and B fix the machine and vary the speedup; this fixes the speedup
    at 1x and varies the machine. Dropping the campaign-time fill with it is
    deliberate - that fill is a function of the machine too, so a single one
    would be wrong for four of the five curves, and the contours it sat under
    are the thing being replaced.
    """
    plt.rcParams.update({'hatch.linewidth': 0.4, 'hatch.color': style.MUTED})
    measured_pools = measured['pool_size'].to_numpy().astype(float)
    labeled_fraction = _labeled_fraction(measured)

    pools = np.logspace(
        np.log10(POOL_LIMITS[0]), np.log10(measured_pools.max()), GRID_POINTS
    )
    fig, ax = plt.subplots(figsize=(125 * style.MM, 76 * style.MM))
    colors = style.SEQUENTIAL(np.linspace(0.30, 0.95, len(MACHINE_CONFIGS)))

    curves = []
    for index, (color, cpus) in enumerate(zip(colors, MACHINE_CONFIGS, strict=True)):
        setup, rate = _machine_model(cpus)
        curve = _speedup_cost(1.0, labeled_fraction, pools, setup, cpus, rate)
        curves.append(curve)
        ax.plot(pools, curve, color=color, linewidth=1.5, marker='none')
        # The top two configurations sit a factor of two apart, which is 0.3 of
        # a decade on a seven-decade axis - too close for stacked labels.
        # Alternating the x position separates neighbours horizontally instead.
        at = int((0.62 if index % 2 == 0 else 0.87) * len(pools))
        ax.annotate(
            f'{cpus:,} CPU' + ('' if cpus == 1 else 's'),
            (pools[at], curve[at]),
            ha='center',
            va='center',
            fontsize=6,
            color=color,
            fontweight='semibold',
            bbox={
                'facecolor': style.BACKGROUND,
                'edgecolor': 'none',
                'alpha': 0.9,
                'pad': 1.2,
            },
        )

    # Below the cheapest machine's locus no configuration makes AL worth it.
    ax.fill_between(
        pools,
        COST_LIMITS[0],
        curves[0],
        facecolor='none',
        edgecolor=style.MUTED,
        hatch='///',
        linewidth=0,
    )
    ax.annotate(
        'AL never pays, on any machine',
        (0.02, 0.02),
        xycoords='axes fraction',
        va='bottom',
        fontsize=6,
        color=style.INK,
        fontweight='semibold',
        bbox={
            'facecolor': style.BACKGROUND,
            'edgecolor': 'none',
            'alpha': 0.85,
            'pad': 1.5,
        },
    )
    # Above the Vina reference line the plane is empty, which is the only place
    # a multi-line note does not land on a curve or on an engine label.
    ax.annotate(
        'Every CPU added to the oracle raises the bar AL must clear:\n'
        'the oracle parallelises and the pipeline does not, so break-even\n'
        'scales with the machine. Only the 1-CPU curve is flat - it is the\n'
        'one configuration paying no worker-pool set-up.',
        (0.025, 0.97),
        xycoords='axes fraction',
        va='top',
        fontsize=6,
        color=style.INK,
        bbox={
            'facecolor': style.BACKGROUND,
            'edgecolor': 'none',
            'alpha': 0.85,
            'pad': 2.0,
        },
    )

    # Labels go left here; the right margin is taken by the CPU-count labels.
    for engine, seconds in DOCKING_TIMES.items():
        if not COST_LIMITS[0] <= seconds <= COST_LIMITS[1]:
            continue
        ax.axhline(
            seconds,
            color=style.INK,
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            marker='none',
        )
        ax.annotate(
            f'{engine} ({seconds:g} s)',
            (pools[0], seconds),
            xytext=(4, 3),
            textcoords='offset points',
            ha='left',
            fontsize=6,
            color=style.INK,
            fontweight='semibold',
        )

    ax.set(
        xscale='log',
        yscale='log',
        xlim=(pools[0], pools[-1]),
        ylim=COST_LIMITS,
        xlabel='Library size (compounds)',
        ylabel='Break-even oracle cost per compound (s)',
    )
    pool_ticks = [1e3, 1e4, 1e5, 1e6, 1e7, measured_pools.max()]
    ax.set_xticks(pool_ticks, [_pool_label(pool) for pool in pool_ticks])
    ax.set_xticks([], minor=True)
    ax.set_yticks(
        [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        ['0.1 ms', '1 ms', '10 ms', '0.1 s', '1 s', '10 s', '100 s', '1000 s'],
    )
    ax.set_yticks([], minor=True)
    style_axes(ax)
    ax.grid(False)
    style.save(fig, output_name)


def main() -> None:
    style.apply()
    measured = _measured_scaling(data.load())
    _render(measured, 'fig05_scaling')
    _render(
        measured,
        'fig05b_scaling_parallel',
        setup=PIPELINE_SETUP_BY_CORES[AL_CORES_B],
        oracle_workers=ORACLE_WORKERS_B,
    )
    _render_breakeven(measured, 'fig05c_breakeven_by_machine')


if __name__ == '__main__':
    main()

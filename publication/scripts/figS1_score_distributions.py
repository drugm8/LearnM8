"""S1 - the docking-score distributions the experiments were actually run on.

A  AmpC, the 1M / 10M / 50M / 96M pools overlaid
B  D4, the 116M scored pool

Every result in this paper is a statement about recovering the left tail of one of
these two distributions, and nothing in the main text shows what that tail looks
like. This is the reference: score on x, count on a log y, so the tail the
acquisition functions chase is legible instead of being flattened into the axis.

`score_direction` is `lower` for both campaigns, so better molecules are further
LEFT and the interesting region is the left edge, not the mode.

Axes. Nothing is cropped - each panel spans its own full observed range, computed
from the data in `_xlim`. That is why the x axes differ: D4 runs from -75.5 out to
+443.3, some 530 score units, against AmpC's -118.8 to +87.9. Forcing one shared x
range would squeeze AmpC into the left fifth of its panel, so peak positions are
NOT comparable by eye between panels - read them off the tick labels. The y scale
IS shared, so bin heights are directly comparable.

Pools, and why these five. They are exactly the pool tokens in `data.POOL_SIZE`
that appear in the manifest, minus d4-840K, which is dropped by request - so panel
B carries a single curve rather than the two-curve overlay panel A has. The
d4-840K pool is still a real experiment pool (family D-01, and it is in
table2_benchmark_summary), so this figure covers the data behind every run
*except* that one. AmpC_screen_500K.csv and D4_screen_1000k.csv exist on disk but
no publication run used them, so they are not here.

The AmpC subsets are not merely similar, they are the same distribution resampled,
which is the load-bearing claim of panel A and the reason the four curves are
overlaid rather than separated:

    pool     n             mean       std      median    top-1% cut
    1M       1,000,000     -30.7309   9.3269   -30.73    -62.25
    10M      10,000,000    -30.7274   9.3250   -30.72    -62.20
    50M      50,000,000    -30.7235   9.3226   -30.72    -62.17
    96M      96,214,206    -30.7230   9.3207   -30.72    -62.16

Agreement to four significant figures in mean and std, and to 0.09 units in the
top-1% cut across a 96x range in pool size. That matters beyond bookkeeping: the
S-family scaling results and every 1M-pool figure are comparable to the 96M ones
because the pools are distributionally interchangeable, not because anyone assumed
so. What does move with n is the extreme: the minimum runs -87.16 / -90.01 /
-117.95 / -118.83 from 1M to 96M, which is sampling of the tail, not a shift in
it - a larger draw simply reaches further into the same distribution. That is also
why the curves in panel A sit at different heights: on a count axis the vertical
offset between them is exactly the pool-size ratio, so identical shape at
different height is what "same distribution, more molecules" looks like.

The dashed rules mark the top-1% and top-0.1% score cutoffs - the tiers fig02,
fig03 and fig12 report recovery against. They are drawn for the largest pool in
each panel only (96M, 116M); the smaller AmpC pools' cutoffs differ from those by
at most 0.10 units and would render as four coincident lines.

Comparing the two panels explains fig12's inversion, which is the reason this is
worth an SI slot rather than a data statement. AmpC's left tail reaches far
further from its own bulk than D4's does:

    pool       10th-best score   sigma below mean   span of the top 10
    AmpC 96M   -93.86            6.77              -118.84..-93.86 (2.68 sigma)
    D4 116M    -72.22            3.99               -75.50..-72.22 (0.30 sigma)

D4's ten best are a tight clump 0.3 sigma wide sitting just past a densely
populated tail, so a greedy model that gets anywhere near the tail sweeps them up
- fig12 has D4 at 100% of top-10 by cycle 4. AmpC's ten best are isolated
singletons scattered across 25 score units at nearly 7 sigma, in bins holding a
handful of molecules each, so they are not where extrapolation from the bulk
points - fig12 has AmpC stuck at 40% of top-10 from cycle 1 onward. The same
geometry inverts at the broad tier: AmpC's top-0.1% cut (-72.08) sits on a
well-populated shoulder, which is why AmpC reaches 99% there while D4 reaches 91%.
So the inversion is a property of these distributions, not of the method.

Binning. Counts are accumulated at 0.02-unit resolution and re-aggregated to
DISPLAY_BIN for drawing; quantiles are read off the 0.02 bins, so the rules are
accurate to that width. The AmpC quantiles were cross-checked against exact
`Series.quantile` values on the full columns and agree within one bin.

Data provenance. The four AmpC pools are streamed from ~/LearnM8_DATA. The D4
pool is 8.4 GB and lives only on the cluster, so its histogram was reduced there
and only the bin counts were transferred; `D4_HIST` holds that output verbatim and
the command that produced it is in `_read_remote_hist`. Rebuild the cache with
`python figS1_score_distributions.py --rebuild-cache`, which needs the AmpC CSVs
present locally; without the flag the script only reads the committed parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import style
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
CACHE = DATA / 'score_distributions.parquet'
D4_HIST = DATA / 'd4_116M_score_hist.csv'
DATA_DIR = Path.home() / 'LearnM8_DATA'

# Shared with the remote reduction: bin b spans [ORIGIN + WIDTH*b, ORIGIN + WIDTH*(b+1)).
BIN_WIDTH = 0.02
ORIGIN = -100.0
# Coarser bins for drawing; 0.02 is quantile resolution, not a readable line.
DISPLAY_BIN = 0.5
# Padding either side of the full data range, in score units. Nothing is clipped:
# the axis is derived from the observed extremes in `_xlim` so a pool whose tail
# grows cannot silently fall off the figure.
XPAD = 8.0
# Shared by both panels, unlike x. The floor sits below 1 so singleton bins render
# as a mark rather than on the spine; the ceiling clears both peaks (2.5e6 AmpC,
# 2.7e6 D4).
YLIM = (0.6, 6e6)

AMPC_POOLS = [
    ('1M', 'AmpC_screen_1000K.csv'),
    ('10M', 'AmpC_screen_10000K.csv'),
    ('50M', 'AmpC_screen_50000K.csv'),
    ('96M', 'AmpC_screen_96214K.csv'),
]
D4_POOL = 'd4-116M'
# Panel -> (title, pools in draw order, pool whose cutoffs get the rules).
PANELS = {
    'A': ('AmpC', [token for token, _ in AMPC_POOLS], '96M'),
    'B': ('D4', [D4_POOL], D4_POOL),
}
TIERS = [(0.01, 'top-1%'), (0.001, 'top-0.1%')]


def _read_remote_hist() -> pl.DataFrame:
    """Parse the cluster-side reduction of the 116M-row D4 pool.

    Produced on conduit with, at the same ORIGIN/BIN_WIDTH used here:

        mawk -F, 'NR>1 && $3!="" { s=$3+0; b=int((s+100)/0.02); c[b]++ ... }' \\
            ~/opt/learnm8-d4/D4_scored_116241184.csv

    `int()` truncates toward zero rather than flooring, which would disagree with
    the local `.floor()` for negative operands - it does not here, because the D4
    minimum is -71.55 and `s + 100` is therefore always positive.
    """
    rows = [
        line.split(',')
        for line in D4_HIST.read_text().splitlines()
        if line and not line.startswith('#')
    ]
    return pl.DataFrame(
        {
            'bin': [int(b) for b, _ in rows],
            'count': [int(c) for _, c in rows],
        }
    ).sort('bin')


def build() -> pl.DataFrame:
    """Re-derive the histogram cache from the pool files."""
    frames = []
    for token, filename in AMPC_POOLS:
        scores = (
            pl.scan_csv(DATA_DIR / filename)
            .select(pl.col('dockscore').cast(pl.Float64))
            .drop_nulls()
            .collect(engine='streaming')['dockscore']
        )
        hist = (
            ((scores - ORIGIN) / BIN_WIDTH)
            .floor()
            .cast(pl.Int64)
            .value_counts()
            .rename({'dockscore': 'bin'})
            # value_counts returns UInt32; the parsed remote counts are Int64, and
            # concat is strict about it.
            .with_columns(pl.col('count').cast(pl.Int64))
        )
        frames.append(hist.with_columns(pool=pl.lit(token), dataset=pl.lit('AmpC')))
    frames.append(
        _read_remote_hist().with_columns(pool=pl.lit(D4_POOL), dataset=pl.lit('D4'))
    )

    out = (
        pl.concat(frames).select('dataset', 'pool', 'bin', 'count').sort('pool', 'bin')
    )
    out.write_parquet(CACHE)
    print(f'wrote {CACHE} ({out.height} bins, {out["pool"].n_unique()} pools)')
    return out


def load() -> pl.DataFrame:
    if not CACHE.exists():
        raise FileNotFoundError(f'{CACHE} missing - run with --rebuild-cache first')
    return pl.read_parquet(CACHE)


def _cutoff(pool_hist: pl.DataFrame, quantile: float) -> float:
    """Score below which `quantile` of the pool lies, read off the fine bins.

    Scores are better when lower, so the tier boundary is the low-side quantile.
    """
    ordered = pool_hist.sort('bin')
    total = ordered['count'].sum()
    cumulative = ordered['count'].cum_sum()
    # First bin whose cumulative count reaches the target; its left edge is the cut.
    index = int((cumulative < quantile * total).sum())
    return ORIGIN + ordered['bin'][index] * BIN_WIDTH


def _xlim(panel_hist: pl.DataFrame) -> tuple[float, float]:
    """Full observed score range for one panel's pools, padded.

    Derived rather than fixed so nothing is ever cropped: the axis follows the
    data, including D4's sparse tail out past +440 and AmpC's handful of compounds
    below -100. Each panel gets its OWN range rather than a shared one, which is
    forced by not cropping - D4 spans ~530 score units against AmpC's ~220, so one
    shared axis would leave AmpC compressed into the left fifth of its panel. The
    cost is that peak positions are not comparable by eye across panels; the y
    scale is still shared, so heights are.
    """
    score = ORIGIN + pl.col('bin') * BIN_WIDTH
    low = panel_hist.select(score.min()).item()
    high = panel_hist.select(score.max()).item() + BIN_WIDTH
    return low - XPAD, high + XPAD


def _display_curve(pool_hist: pl.DataFrame) -> pl.DataFrame:
    """Re-aggregate the fine bins onto DISPLAY_BIN edges."""
    factor = round(DISPLAY_BIN / BIN_WIDTH)
    return (
        pool_hist.with_columns(
            edge=ORIGIN + (pl.col('bin') // factor) * factor * BIN_WIDTH
        )
        .group_by('edge')
        .agg(pl.col('count').sum())
        .sort('edge')
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-cache', action='store_true')
    args = parser.parse_args()

    df = build() if args.rebuild_cache else load()
    style.apply()

    # Sequential ramp for pool scale, per the house rule; D4 keeps the accent it
    # carries in fig12 so the two figures name the same campaign the same way.
    ampc_colors = {
        token: style.SEQUENTIAL(value)
        for token, value in zip(
            [t for t, _ in AMPC_POOLS], (0.35, 0.55, 0.75, 0.95), strict=True
        )
    }
    colors = {**ampc_colors, D4_POOL: style.ACCENT_ORANGE}

    fig, axd = style.mosaic('AB', panel_h_mm=62)
    for letter, (title, pools, rule_pool) in PANELS.items():
        ax = axd[letter]
        xlim = _xlim(df.filter(pl.col('pool').is_in(pools)))
        for token in pools:
            curve = _display_curve(df.filter(pl.col('pool') == token))
            ax.step(
                curve['edge'],
                curve['count'],
                where='post',
                color=colors[token],
                linewidth=style.DATA_LINEWIDTH,
                # The theme sets lines.marker='o' for the 10-point cycle curves the
                # other figures draw; on a few-hundred-bin histogram it is noise.
                marker='',
            )

        rule_hist = df.filter(pl.col('pool') == rule_pool)
        for quantile, tier_label in TIERS:
            cut = _cutoff(rule_hist, quantile)
            # marker='' for the same reason as the curves: axvline inherits it too.
            ax.axvline(cut, color=style.MUTED, linestyle='--', linewidth=0.7, marker='')
            # The cuts are 6-10 score units apart on axes spanning 220-530, so the
            # labels collide at any single height, and anywhere inside the axes
            # they land on a curve - the rules cross the rising left flank, which
            # is where the data is. Stacked just above the frame instead, one tier
            # per row, clear of both the curves and the panel title at y=1.11.
            ax.text(
                cut,
                1.01 + 0.055 * TIERS.index((quantile, tier_label)),
                tier_label,
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                fontsize=5.5,
                color=style.MUTED,
            )

        ax.set_yscale('log')
        ax.set_xlim(*xlim)
        # Both panels on one count scale, or the two campaigns' peak heights are
        # not comparable and the panels only look like they can be read together.
        ax.set_ylim(*YLIM)
        ax.set_title(title)
        ax.set_xlabel('Docking score (lower is better)')
        ax.legend(
            handles=[
                Line2D([], [], color=colors[token], label=token, marker='')
                for token in pools
            ],
            # Right, not left: the tier rules and their labels now occupy the upper
            # left, and both distributions have decayed by the right edge.
            loc='upper right',
            fontsize=5.5,
            frameon=False,
        )
    axd['A'].set_ylabel('Compounds per 0.5-unit bin')

    style.label_panels(axd)
    style.save(fig, 'figS1_score_distributions')
    plt.close(fig)


if __name__ == '__main__':
    main()

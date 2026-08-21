"""F12 - target generality: the same protocol on AmpC and on D4.

A  AmpC, Top-100     B  D4, Top-100
C  AmpC, Top-0.1%    D  D4, Top-0.1%

Every figure before this one measures a single target. This one asks whether the
recovery behaviour they report is a property of the method or of AmpC, by running
the identical protocol against a second docking campaign.

The two runs are protocol twins, not merely similar. Both were launched from the
same cycle string `random:0.01 greedy:0.01*9` - a 1% random seed followed by nine
1% greedy batches, ten cycles, ending at 10% of the library - with `rf`, `morgan`,
`--score-direction lower` and `--random-state 42`. Neither prunes
(`cumulative_pruned = 0` at every cycle). The pools are the only deliberate
difference: 96,214,206 AmpC (S-04, cluster 178520) against 116,241,184 D4 (D-04,
cluster 181783). D4's pool is the scored subset of the 138,312,677-row table;
22,071,493 rows carry an empty `dockscore` and `CSVOracle` rejects a selected row
with a null target, so those rows can never be a publication pool.

The panels are laid out one curve each, target by column and threshold by row, so
a column reads as one campaign at two resolutions and a row reads as one question
asked of two targets. Axes are shared in both directions to make that second
reading exact: identical y limits, and an x limit spanning the larger of the two
campaigns, so a curve that sits lower on the page really is lower. The x tick
labels carry each run's own pool fraction alongside the absolute count, which is
what keeps the columns comparable despite the 20.0M-compound difference in pool
size - cycle 9 is 10% of the library in both columns and 9.6M against 11.6M
molecules across them.

The result is an inversion, and it is the reason both thresholds are plotted
rather than one. AmpC dominates the broad tier: 99.0% of its top 0.1% by cycle 9
against D4's 91.2%, and it gets there faster, clearing 90% at cycle 3 where D4
needs cycle 9 to reach 91%. D4 dominates the tight tier: 98% of its top 100
against AmpC's 93%, and on top-10 (not plotted, n = 10 is too small to draw)
D4 recovers 100% by cycle 4 while AmpC stalls at 40% from cycle 1 onward and
never finds six of its ten best molecules. So neither target is simply "easier".
A greedy campaign on AmpC maps the good region well and misses the extreme tail;
on D4 it locks onto the tail and leaves more of the good region unscreened.

Caveat, load-bearing for how much weight the difference can carry: n = 1 per
target. These are two single runs, one per campaign, with no replicate spread,
because a 116M-compound campaign costs ~40 h on 64 CPUs and neither pool has been
repeated under a second seed. The gaps above are therefore differences between two
observations, not estimates with an interval attached. The two gaps are not
equally solid, and the tier sizes are why: D4's top-100 tier is 100 molecules, so
the 5-point gap in row 1 is literally five molecules and well inside what a single
seed could move, whereas its top-0.1% tier is 116,241 molecules, so the 7.8-point
gap in row 2 stands on roughly 9,100 of them. Read the direction of the inversion,
not its magnitude, and trust row 2 further than row 1.
"""

from __future__ import annotations

import data
import polars as pl
import style

X = 'compounds_evaluated'

# Family code -> display name and curve colour. Both runs are greedy, so colour
# encodes the target here rather than the strategy it does elsewhere.
TARGETS = {
    'S-04': ('AmpC', style.PRIMARY),
    'D-04': ('D4', style.ACCENT_ORANGE),
}
METRIC_SHORT = {
    'top_100_discovery': 'Top-100',
    'top_0_1_pct_discovery': 'Top-0.1%',
}
# Target by column, threshold by row.
PANELS = {
    'A': ('S-04', 'top_100_discovery'),
    'B': ('D-04', 'top_100_discovery'),
    'C': ('S-04', 'top_0_1_pct_discovery'),
    'D': ('D-04', 'top_0_1_pct_discovery'),
}
LEFT_COLUMN = ('A', 'C')
BOTTOM_ROW = ('C', 'D')


def main() -> None:
    style.apply()
    # No --design flag: both runs seed cycle 0 with 1% and batch at 1%, so they
    # satisfy the matched and fixed designs alike and a second variant would be a
    # byte-identical duplicate. Same reason fig05/fig08/fig09 render once.
    df = data.load().filter(
        pl.col('family').is_in(list(TARGETS)) & (pl.col('batch_fraction') == 1.0)
    )

    runs = {}
    for code in TARGETS:
        sub = df.filter(pl.col('family') == code)
        found = sorted(sub['run_id'].unique().to_list())
        # One run per target is the whole premise of the comparison; two would
        # silently average different experiments into one curve.
        if len(found) != 1:
            raise ValueError(
                f'{code}: expected exactly 1 run at a 1% batch, got {found}'
            )
        runs[code] = sub

    x_max = max(int(sub[X].max()) for sub in runs.values())

    fig, axd = style.mosaic('AB\nCD', panel_h_mm=58)
    for letter, (code, metric) in PANELS.items():
        ax = axd[letter]
        label, color = TARGETS[code]
        curves = data.curve(runs[code], [], X, metric)
        ax.plot(
            curves[X],
            curves['mean'],
            color=color,
            linestyle='-',
            linewidth=style.DATA_LINEWIDTH,
            marker=style.CURVE_MARKERS[0],
            markersize=2.6,
        )
        ax.set_title(f'{label} - {METRIC_SHORT[metric]}')
        # Shared limits in both directions: the comparison lives across panels, so
        # a difference in height has to mean a difference in recovery.
        ax.set_ylim(0, 100)
        ax.set_xlim(0, x_max * 1.03)
        style.compact_axis(ax)
        # set_compound_axis writes its own xlabel, so the top row has to suppress
        # it there rather than just declining to set one.
        bottom = letter in BOTTOM_ROW
        style.set_compound_axis(
            ax,
            int(runs[code]['pool_size'].unique().item()),
            xlabel=style.LABELS[X] if bottom else '',
            ticks=curves[X].to_list(),
            max_ticks=4,
            rotation=45,
        )

        if letter in LEFT_COLUMN:
            ax.set_ylabel(style.LABELS[metric])
        else:
            ax.tick_params(labelleft=False)
        if not bottom:
            # Each column is one run, so the row below carries identical ticks.
            ax.tick_params(labelbottom=False)

    style.label_panels(axd)
    style.save(fig, 'fig12_target_comparison')


if __name__ == '__main__':
    main()

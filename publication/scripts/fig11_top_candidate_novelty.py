"""F11 - molecule, scaffold, and top-hit overlap across acquisition policies."""

from __future__ import annotations

from pathlib import Path

import analysis
import numpy as np
import polars as pl
import style
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.image import AxesImage

TABLE_PATH = (
    Path(__file__).resolve().parents[1] / 'tables' / 'fig11_selection_overlap.csv'
)
# Titles stay on one line so they share a baseline with the panel letter; what
# each overlap measures is spelled out in the caption.
PANELS = (
    ('A', 'all_molecules', 'Molecules (exact)'),
    ('B', 'all_scaffolds', 'Scaffolds (Jaccard)'),
    ('C', 'top_100_molecules', 'Top-100 (exact)'),
)
STRATEGY_LABELS = {
    'random': 'random',
    'greedy': 'greedy',
    'ucb': 'UCB',
    'ei': 'EI',
    'thompson': 'Thompson',
    'simulated_annealing': 'annealing',
}
TICK_LABEL_SIZE = 6.5


def _plot_overlap_panel(
    ax: Axes,
    overlap: pl.DataFrame,
    overlap_type: str,
    title: str,
    upper_cmap: Colormap,
    lower_cmap: Colormap,
) -> tuple[AxesImage, AxesImage]:
    values = analysis.split_overlap_matrix(
        overlap,
        overlap_type=overlap_type,
        strategies=tuple(analysis.STRATEGY_ORDER),
    )
    upper_mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    lower_mask = np.tril(np.ones_like(values, dtype=bool), k=-1)
    upper_image = ax.imshow(
        np.ma.masked_where(~upper_mask, values),
        cmap=upper_cmap,
        vmin=0.0,
        vmax=100.0,
    )
    lower_image = ax.imshow(
        np.ma.masked_where(~lower_mask, values),
        cmap=lower_cmap,
        vmin=0.0,
        vmax=100.0,
    )
    ax.set_facecolor('#F1EFF5')
    ax.grid(False)
    positions = np.arange(len(analysis.STRATEGY_ORDER))
    ax.set_title(title)
    ax.set(
        xticks=positions,
        xticklabels=[STRATEGY_LABELS[item] for item in analysis.STRATEGY_ORDER],
        yticks=positions,
        yticklabels=[STRATEGY_LABELS[item] for item in analysis.STRATEGY_ORDER],
    )
    # Rotating about the label's right end anchors each one under its own tick;
    # centre-anchored rotation lets long labels drift sideways into their
    # neighbours, which is what made these collide.
    ax.tick_params(axis='x', labelrotation=45, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
        label.set_rotation_mode('anchor')
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                ax.text(
                    column,
                    row,
                    '—',
                    ha='center',
                    va='center',
                    fontsize=6,
                    color=style.MUTED,
                )
                continue
            ax.text(
                column,
                row,
                f'{value:.0f}',
                ha='center',
                va='center',
                fontsize=6,
                color=style.BACKGROUND if value >= 55 else style.INK,
            )
    return upper_image, lower_image


def main() -> None:
    style.apply()
    overlap = analysis.build_overlap_table()
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    overlap.write_csv(TABLE_PATH)

    fig, axd = style.mosaic('ABC', panel_h_mm=62)
    upper_cmap = style.SEQUENTIAL.copy()
    lower_cmap = LinearSegmentedColormap.from_list(
        'learnm8-overlap-blue',
        ['#F5F9FC', '#C9E3F3', '#72B5DB', style.ACCENT_BLUE, '#00466F'],
    )
    upper_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    lower_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    images = None
    for letter, overlap_type, title in PANELS:
        images = _plot_overlap_panel(
            axd[letter],
            overlap,
            overlap_type,
            title,
            upper_cmap,
            lower_cmap,
        )
        if letter != 'A':
            axd[letter].set_yticklabels([])

    if images is None:
        raise RuntimeError('No overlap panels were rendered')
    upper_image, lower_image = images
    upper_colorbar = fig.colorbar(
        upper_image,
        ax=list(axd.values()),
        pad=0.025,
        fraction=0.018,
        aspect=24,
    )
    upper_colorbar.set_label('1% overlap (%)')
    lower_colorbar = fig.colorbar(
        lower_image,
        ax=list(axd.values()),
        pad=0.075,
        fraction=0.018,
        aspect=24,
    )
    lower_colorbar.set_label('0.1% overlap (%)')
    style.label_panels(axd)
    style.save(fig, 'fig11_top_candidate_novelty')


if __name__ == '__main__':
    main()

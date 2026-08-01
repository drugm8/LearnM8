"""Shared plot style for the LearnM8 application-note figures.

Palette is anchored on the logo gradient (#8352C3 -> #B685F6) with a neutral
#333333 for ink. Categorical series use Okabe-Ito with the brand purple
substituted for its first entry, so every figure stays colour-vision safe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

PRIMARY = '#8352C3'
LIGHT = '#B685F6'
DARK = '#3D2160'
INK = '#333333'
GRID = '#E6E6E6'
MUTED = '#56565E'

CATEGORICAL = [PRIMARY, '#E69F00', '#0072B2', '#009E73', '#D55E00', '#CC79A7', MUTED]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'learnm8', ['#FFFFFF', LIGHT, PRIMARY, DARK]
)

MM = 1 / 25.4
SINGLE_COL = 85 * MM
DOUBLE_COL = 170 * MM

FIGURE_DIR = Path(__file__).resolve().parents[2] / 'docs' / 'publication' / 'figures'

# Colour encodes model family, line style encodes the member within it: ten
# learners in one panel exceed any safe categorical palette, and this pairing
# survives both greyscale printing and colour-vision deficiency.
LEARNER_FAMILY = {
    'rf': 'Classical',
    'xgb': 'Classical',
    'dt': 'Classical',
    'lr': 'Classical',
    'rf_fil': 'GPU',
    'ridge_cuml': 'GPU',
    'svgp': 'GPU',
    'mc_dropout': 'Neural',
    'fastprop': 'Neural',
    'chemprop': 'Graph',
}
FAMILY_COLOR = {
    'Classical': PRIMARY,
    'GPU': '#E69F00',
    'Neural': '#0072B2',
    'Graph': '#009E73',
}
LINESTYLES = ['-', '--', '-.', ':']

STRATEGY_COLOR = {
    'random': MUTED,
    'greedy': PRIMARY,
    'ucb': '#E69F00',
    'ei': '#0072B2',
    'thompson': '#009E73',
    'ucb-greedy': '#D55E00',
    'greedy-sa': '#CC79A7',
}

LABELS = {
    'top_0_1_pct_discovery': 'Top-0.1% discovery (%)',
    'top_100_discovery': 'Top-100 discovery (%)',
    'selected_percentile': 'Selected percentile',
    'compounds_evaluated': 'Compounds evaluated',
    'cum_ml_time': 'Cumulative ML time (s)',
    'cum_prediction_time': 'Cumulative prediction time (s)',
    'acquisition_time': 'Acquisition time (s)',
    'scaffold_diversity_index_cumulative': 'Scaffold diversity index',
    'mean_tanimoto_similarity_sampled_cumulative': 'Mean Tanimoto similarity',
}


def learner_style(learner: str) -> tuple[str, str]:
    """Return (colour, linestyle) for a learner, keyed on its model family."""
    family = LEARNER_FAMILY[learner]
    members = [k for k, v in LEARNER_FAMILY.items() if v == family]
    return FAMILY_COLOR[family], LINESTYLES[members.index(learner) % len(LINESTYLES)]


def apply() -> None:
    """Install the publication rcParams. Call once at the top of a figure script."""
    mpl.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Nimbus Sans', 'Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'axes.edgecolor': INK,
            'axes.labelcolor': INK,
            'axes.linewidth': 0.6,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.color': GRID,
            'grid.linewidth': 0.5,
            'text.color': INK,
            'xtick.color': INK,
            'ytick.color': INK,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.width': 0.6,
            'ytick.major.width': 0.6,
            'lines.linewidth': 1.2,
            'lines.markersize': 3.5,
            'legend.frameon': False,
            'figure.dpi': 150,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )


def mosaic(layout: str, panel_h_mm: float = 48, width: float = DOUBLE_COL):
    """Build a panel grid from a mosaic string, e.g. 'ABC\\nDDE'.

    Returns (fig, {letter: axes}). Panel letters are drawn by label_panels.
    """
    rows = layout.strip().splitlines()
    fig, axd = plt.subplot_mosaic(
        [list(r.strip()) for r in rows], figsize=(width, len(rows) * panel_h_mm * MM)
    )
    for ax in axd.values():
        ax.grid(axis='x', visible=False)
    return fig, axd


def label_panels(axd: dict) -> None:
    """Bold panel letters at the top-left, outside the axes."""
    for letter, ax in axd.items():
        ax.text(
            -0.16,
            1.06,
            letter,
            transform=ax.transAxes,
            fontsize=9,
            fontweight='bold',
            va='bottom',
            ha='left',
        )


def compact_axis(ax, axis: str = 'x') -> None:
    """Format large counts as 10k / 1.5M, avoiding matplotlib's offset text."""
    fmt = FuncFormatter(
        lambda v, _: (
            f'{v / 1e6:g}M'
            if abs(v) >= 1e6
            else f'{v / 1e3:g}k'
            if abs(v) >= 1e3
            else f'{v:g}'
        )
    )
    (ax.xaxis if axis == 'x' else ax.yaxis).set_major_formatter(fmt)


def band(ax, x, lo, hi, color: str) -> None:
    """Min-max replicate band. No edge, so overlapping bands stay readable."""
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)


def save(fig, name: str) -> None:
    """Write vector PDF (submission) and 600 dpi PNG (docs site)."""
    for fmt, dpi in (('pdf', None), ('png', 600)):
        out = FIGURE_DIR / fmt
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f'{name}.{fmt}', dpi=dpi)
    plt.close(fig)
    print(f'wrote {name}.pdf / {name}.png')


# Machado et al. (2009) severity-1.0 CVD matrices, used only by validate_palette.
_CVD = {
    'deuteranopia': [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ],
    'protanopia': [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ],
}


def validate_palette() -> None:
    """Render the palettes under CVD simulation and greyscale as a visual check."""
    apply()
    swatches = [to_rgb(c) for c in CATEGORICAL]
    ramp = [SEQUENTIAL(v)[:3] for v in np.linspace(0, 1, len(CATEGORICAL))]
    views = {'normal': None, **_CVD, 'greyscale': 'grey'}

    fig, axes = plt.subplots(len(views), 1, figsize=(DOUBLE_COL, len(views) * 14 * MM))
    for ax, (name, transform) in zip(axes, views.items(), strict=True):
        for row, colors in enumerate((swatches, ramp)):
            for i, rgb in enumerate(colors):
                ax.add_patch(Rectangle((i, row), 1, 1, color=_simulate(rgb, transform)))
        ax.set(xlim=(0, len(swatches)), ylim=(0, 2), title=name, xticks=[], yticks=[])
        ax.set_frame_on(False)
    fig.tight_layout()
    save(fig, 'palette_validation')


def _simulate(rgb, transform):
    if transform is None:
        return rgb
    if transform == 'grey':
        return (np.dot(rgb, [0.299, 0.587, 0.114]),) * 3
    return np.clip(np.dot(transform, rgb), 0, 1)


if __name__ == '__main__':
    validate_palette()

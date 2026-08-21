"""Publication-facing compatibility wrapper for the shared LearnM8 theme."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from learnm8.visualization import style as _style

PRIMARY = _style.PRIMARY
LIGHT = _style.PRIMARY_LIGHT
DARK = _style.PRIMARY_DARK
PALE = _style.PRIMARY_PALE
INK = _style.INK
GRID = _style.GRID
MUTED = _style.MUTED
BACKGROUND = _style.BACKGROUND
ACCENT_BLUE = _style.ACCENT_BLUE
ACCENT_GREEN = _style.ACCENT_GREEN
ACCENT_AMBER = _style.ACCENT_AMBER
ACCENT_ORANGE = _style.ACCENT_ORANGE
ACCENT_PINK = _style.ACCENT_PINK
CATEGORICAL = _style.CATEGORICAL
CATEGORICAL_CMAP = _style.CATEGORICAL_CMAP
CURVE_LINESTYLES = _style.CURVE_LINESTYLES
CURVE_MARKERS = _style.CURVE_MARKERS
DATA_LINEWIDTH = _style.DATA_LINEWIDTH
SEQUENTIAL = _style.SEQUENTIAL
MM = _style.MM
SINGLE_COL = _style.SINGLE_COL
DOUBLE_COL = _style.DOUBLE_COL
OUTPUT_DPI = _style.OUTPUT_DPI
format_compound_count = _style.format_compound_count
compound_tick_label = _style.compound_tick_label
set_compound_axis = _style.set_compound_axis

FIGURE_DIR = Path(__file__).resolve().parents[1] / 'figures'

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
    'GPU': ACCENT_AMBER,
    'Neural': ACCENT_BLUE,
    'Graph': ACCENT_GREEN,
}
LINESTYLES = ['-'] * 4

STRATEGY_COLOR = {
    'random': MUTED,
    'greedy': PRIMARY,
    'ucb': ACCENT_AMBER,
    'ei': ACCENT_BLUE,
    'thompson': ACCENT_GREEN,
    'pi': ACCENT_ORANGE,
    'entropy': ACCENT_PINK,
    # Standalone simulated annealing (A-08). Every accent is already taken by the
    # seven single-acquisition strategies, so this reuses the dark primary; markers
    # keep strategy identity from depending on colour alone.
    'simulated_annealing': DARK,
    # Composite schedules (EXP-C), never plotted alongside the standalone set.
    'ucb-greedy': ACCENT_ORANGE,
    'greedy-sa': ACCENT_PINK,
}

LABELS = {
    'top_0_1_pct_discovery': 'Top-0.1% recovered\nfrom initial pool (%)',
    'top_100_discovery': 'Top-100 recovered\nfrom initial pool (%)',
    'selected_percentile': 'Selected percentile',
    'compounds_evaluated': 'Evaluated compounds',
    'cum_ml_time': 'Cumulative ML time (s)',
    'cum_prediction_time': 'Cumulative prediction time (s)',
    'acquisition_time': 'Acquisition time (s)',
    'scaffold_diversity_index_cumulative': 'Scaffold diversity index',
    'mean_tanimoto_similarity_sampled_cumulative': 'Mean Tanimoto similarity',
    'screened_pct': 'Library screened (%)\n(cumulative labeled; seed included)',
    'uncertainty': 'Predicted uncertainty',
    'absolute_error': 'Absolute prediction error',
}


def learner_style(learner: str) -> tuple[str, str]:
    """Return the shared family color and member line style for a learner."""
    family = LEARNER_FAMILY[learner]
    members = [
        name
        for name, member_family in LEARNER_FAMILY.items()
        if member_family == family
    ]
    return FAMILY_COLOR[family], LINESTYLES[members.index(learner) % len(LINESTYLES)]


def apply() -> None:
    """Install the shared LearnM8 theme."""
    _style.apply()


def mosaic(layout: str, panel_h_mm: float = 48, width: float = DOUBLE_COL):
    """Build a publication panel grid using the shared layout treatment."""
    return _style.mosaic(layout, panel_h_mm=panel_h_mm, width=width)


PANEL_LABEL_Y = 1.11
PANEL_TITLE_SIZE = 7.5


def label_panels(axes: dict) -> None:
    """Draw bold panel letters, seating any axes title on the same baseline.

    The theme sets ``axes.titlelocation='left'``, so titles live in the
    left-title slot and default to a 12 pt bold that competes with the panel
    letter for the same corner. Placing both on one baseline at ``PANEL_LABEL_Y``
    makes a panel head read as a single line ("A  batch = 0.1%").
    """
    for letter, ax in axes.items():
        ax.text(
            -0.14,
            PANEL_LABEL_Y,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            color=INK,
            va='baseline',
            ha='left',
        )
        title = ax.get_title(loc='left')
        if title:
            ax.set_title(
                title,
                loc='left',
                y=PANEL_LABEL_Y,
                pad=0,
                fontsize=PANEL_TITLE_SIZE,
                fontweight='semibold',
                color=INK,
            )


def compact_axis(ax, axis: str = 'x') -> None:
    """Format large counts with compact ``k``/``M`` labels."""
    _style.compact_axis(ax, axis=axis)


def band(ax, x, lo, hi, color: str, *, alpha: float = 0.12) -> None:
    """Draw a shared low-alpha interval band."""
    _style.band(ax, x, lo, hi, color, alpha=alpha)


def save(fig, name: str) -> None:
    """Write a PNG publication artifact."""
    output = FIGURE_DIR / 'png' / f'{name}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=OUTPUT_DPI,
        bbox_inches='tight',
        pad_inches=0.08,
        facecolor=_style.BACKGROUND,
    )
    plt.close(fig)


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


def _simulate(rgb, transform):
    if transform is None:
        return rgb
    if transform == 'grey':
        return (np.dot(rgb, [0.299, 0.587, 0.114]),) * 3
    return np.clip(np.dot(transform, rgb), 0, 1)


def validate_palette() -> None:
    """Render the shared palettes under CVD simulation and greyscale."""
    apply()
    swatches = [to_rgb(color) for color in CATEGORICAL]
    ramp = [SEQUENTIAL(value)[:3] for value in np.linspace(0, 1, len(CATEGORICAL))]
    views = {'normal': None, **_CVD, 'greyscale': 'grey'}

    fig, axes = plt.subplots(
        len(views),
        1,
        figsize=(DOUBLE_COL, len(views) * 14 * MM),
        constrained_layout=True,
    )
    for ax, (name, transform) in zip(axes, views.items(), strict=True):
        for row, colors in enumerate((swatches, ramp)):
            for index, rgb in enumerate(colors):
                ax.add_patch(
                    Rectangle((index, row), 1, 1, color=_simulate(rgb, transform))
                )
        ax.set(xlim=(0, len(swatches)), ylim=(0, 2), title=name, xticks=[], yticks=[])
        ax.set_frame_on(False)
    save(fig, 'palette_validation')


if __name__ == '__main__':
    validate_palette()

"""Shared visual language for LearnM8 validation and publication plots.

The theme is intentionally small: it centralizes visual constants and the few
layout operations that must be identical across independent plotting scripts.
It does not change data, statistical calculations, or plot-specific geometry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.ticker import FuncFormatter

PRIMARY = '#8352C3'
PRIMARY_LIGHT = '#B685F6'
PRIMARY_DARK = '#3D2160'
PRIMARY_PALE = '#DCCAF4'
INK = '#252036'
MUTED = '#62606B'
MUTED_LIGHT = '#D9D6DF'
GRID = '#E6E3EB'
BACKGROUND = '#FFFFFF'
ACCENT_BLUE = '#0072B2'
ACCENT_GREEN = '#009E73'
ACCENT_AMBER = '#E69F00'
ACCENT_ORANGE = '#D55E00'
ACCENT_PINK = '#CC79A7'

CATEGORICAL = [
    PRIMARY,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_AMBER,
    ACCENT_ORANGE,
    ACCENT_PINK,
    MUTED,
]
CATEGORICAL_CMAP = ListedColormap(CATEGORICAL, name='learnm8-categorical')
CURVE_LINESTYLES = ['-'] * 6
CURVE_MARKERS = ['o', 's', '^', 'D', 'P', 'X']
DATA_LINEWIDTH = 1.5
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'learnm8-sequential',
    ['#F8F6FB', '#DCCAF4', '#B685F6', PRIMARY, PRIMARY_DARK],
)

MM = 1 / 25.4
SINGLE_COL = 85 * MM
DOUBLE_COL = 170 * MM
OUTPUT_DPI = 600

_FONT_STACK = ['Noto Sans', 'Nimbus Sans', 'DejaVu Sans', 'sans-serif']


def apply() -> None:
    """Install the shared LearnM8 Matplotlib theme in the current process."""
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': _FONT_STACK,
        'font.size': 9,
        'axes.labelsize': 7,
        'axes.labelweight': 'semibold',
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.titlepad': 9,
        'axes.unicode_minus': False,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.title_fontsize': 8,
        'legend.frameon': False,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.alpha': 0.9,
        'grid.linewidth': 0.65,
        'text.color': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'lines.linewidth': DATA_LINEWIDTH,
        'lines.marker': 'o',
        'lines.markersize': 3.5,
        'lines.markeredgewidth': 0,
        'patch.linewidth': 0,
        'figure.dpi': OUTPUT_DPI,
        'savefig.dpi': OUTPUT_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'figure.constrained_layout.use': True,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
    if 'axes.titlelocation' in mpl.rcParams:
        params['axes.titlelocation'] = 'left'
    mpl.rcParams.update(params)


def categorical_color(index: int) -> str:
    """Return a deterministic categorical color, cycling after the safe set."""
    return CATEGORICAL[index % len(CATEGORICAL)]


def categorical_colors(count: int) -> list[str]:
    """Return ``count`` deterministic colors from the shared categorical set."""
    return [categorical_color(index) for index in range(count)]


def format_compound_count(value: float) -> str:
    """Format cumulative compound counts compactly for axis tick labels."""
    if abs(value) >= 1_000_000:
        return f'{value / 1_000_000:.1f}M'
    if abs(value) >= 1_000:
        return f'{value / 1_000:g}k'
    return f'{value:g}'


def compound_tick_label(value: float, pool_size: float) -> str:
    """Show an evaluated count together with its fraction of the pool."""
    percentage = value / pool_size * 100
    percentage_text = (
        f'{percentage:.0f}%' if percentage.is_integer() else f'{percentage:.1f}%'
    )
    return f'{format_compound_count(value)} ({percentage_text})'


def set_compound_axis(
    ax,
    pool_size: float,
    *,
    xlabel: str = 'Evaluated compounds',
    ticks=None,
    max_ticks: int | None = None,
    rotation: float = 45,
):
    """Label an absolute screening axis using actual evaluated counts.

    When ``ticks`` is omitted, x positions are collected from plotted lines.
    If ``max_ticks`` limits labels, all actual positions remain visible as
    minor ticks and only actual positions are chosen for the labelled majors.
    """
    if ticks is None:
        values = []
        for line in ax.get_lines():
            values.extend(np.asarray(line.get_xdata(), dtype=float).ravel())
        ticks = values or ax.get_xticks()
    actual_ticks = np.asarray(
        sorted({float(tick) for tick in ticks if np.isfinite(tick) and tick >= 0}),
        dtype=float,
    )
    if actual_ticks.size == 0:
        return ax

    labelled_ticks = actual_ticks
    if max_ticks is not None and actual_ticks.size > max_ticks:
        indices = np.linspace(0, actual_ticks.size - 1, max_ticks).round().astype(int)
        labelled_ticks = actual_ticks[np.unique(indices)]

    if labelled_ticks.size < actual_ticks.size:
        ax.set_xticks(actual_ticks, minor=True)
        ax.tick_params(axis='x', which='minor', length=2, width=0.6)
    ax.set_xticks(labelled_ticks)
    ax.set_xticklabels(
        [compound_tick_label(tick, pool_size) for tick in labelled_ticks],
        rotation=rotation,
        ha='right' if rotation else 'center',
    )
    ax.set_xlabel(xlabel)
    style_axes(ax)
    ax.xaxis.label.set_fontsize(7)
    ax.tick_params(axis='x', labelsize=7.0 if labelled_ticks.size > 6 else 7.5)
    return ax


def style_axes(ax, *, grid_axis: str = 'y') -> None:
    """Apply shared axis, grid, and spine treatment to one axes object."""
    ax.set_facecolor(BACKGROUND)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(INK)
    ax.spines['bottom'].set_color(INK)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', length=4, width=0.8, colors=INK)
    if grid_axis in {'y', 'both'}:
        ax.grid(axis='y', color=GRID, linewidth=0.65)
    else:
        ax.grid(axis='y', visible=False)
    if grid_axis in {'x', 'both'}:
        ax.grid(axis='x', color=GRID, linewidth=0.65)
    else:
        ax.grid(axis='x', visible=False)

    title = ax.title
    if title.get_text():
        title.set_fontweight('bold')
        title.set_color(INK)
        title.set_ha('left')
        title.set_x(0)
    ax.xaxis.label.set_fontweight('semibold')
    ax.yaxis.label.set_fontweight('semibold')


def style_figure(fig) -> None:
    """Apply shared axis treatment to all non-colorbar axes in a figure."""
    for ax in fig.axes:
        if getattr(ax, '_colorbar', None) is not None:
            continue
        style_axes(ax)


def style_heatmap(ax) -> None:
    """Apply the shared heatmap treatment without Cartesian grid overlays."""
    ax.set_facecolor(BACKGROUND)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', colors=INK)
    if ax.title.get_text():
        ax.title.set_fontweight('bold')
        ax.title.set_color(INK)
        ax.title.set_ha('left')
        ax.title.set_x(0)
    ax.xaxis.label.set_fontweight('semibold')
    ax.yaxis.label.set_fontweight('semibold')


def mosaic(
    layout: str,
    panel_h_mm: float = 48,
    width: float = DOUBLE_COL,
):
    """Build a constrained panel grid from a compact mosaic string."""
    rows = layout.strip().splitlines()
    fig, axes = plt.subplot_mosaic(
        [list(row.strip()) for row in rows],
        figsize=(width, len(rows) * panel_h_mm * MM),
        constrained_layout=True,
    )
    for ax in axes.values():
        style_axes(ax)
    return fig, axes


def label_panels(axes: dict) -> None:
    """Place bold panel letters consistently outside the upper-left corner."""
    for letter, ax in axes.items():
        ax.text(
            -0.14,
            1.11,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            color=INK,
            va='bottom',
            ha='left',
        )


def compact_axis(ax, axis: str = 'x') -> None:
    """Format large counts as compact ``k``/``M`` labels."""
    fmt = FuncFormatter(
        lambda value, _: (
            f'{value / 1e6:g}M'
            if abs(value) >= 1e6
            else f'{value / 1e3:g}k'
            if abs(value) >= 1e3
            else f'{value:g}'
        )
    )
    (ax.xaxis if axis == 'x' else ax.yaxis).set_major_formatter(fmt)


def band(ax, x, lo, hi, color: str, *, alpha: float = 0.12) -> None:
    """Draw a low-alpha interval band without an edge."""
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)


def text_color(color) -> str:
    """Return high-contrast annotation ink for a Matplotlib color."""
    rgb = np.asarray(to_rgb(color))
    luminance = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
    return BACKGROUND if luminance < 0.48 else INK


def save_figure(fig, output_path: Path, *, dpi: int = OUTPUT_DPI) -> None:
    """Save one figure using the shared raster resolution and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.08,
        facecolor=BACKGROUND,
    )
    plt.close(fig)

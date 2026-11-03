"""Poster theme for the LearnM8 application-note figures.

Reuses the shared publication palette and rescales text and strokes for an A0
poster. Figures are drawn at the size they are printed (1 inch = 25.4 mm), so a
font size here is the printed size. Output is SVG with text converted to paths,
plus a PNG fallback, under ``publication/figures/poster/``.

These scripts are deliberately not named ``fig*.py``: ``make_all.py`` globs that
pattern, and the paper's figure build must not pick up poster variants.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The poster scripts live one level below the shared ``data``/``style`` modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.text
import style
from matplotlib.ticker import FuncFormatter

POSTER_DIR = Path(__file__).resolve().parents[2] / 'figures' / 'poster'

# Printed widths, mm. The poster has 15 mm margins, 15 mm gutters and 10 mm of
# padding inside each tinted panel, so one column holds 240 mm of figure and a
# two-column span holds 515 mm. poster/poster.css uses the same numbers.
COLUMN_MM = 240
SPAN_MM = 515
MIN_FONT_PT = 20

# Quicksand (OFL) is the poster's one typeface; the files ship next to this script.
FONT_DIR = Path(__file__).resolve().parent / 'fonts'
FONT_FILES = (
    'Quicksand-Regular.ttf',
    'Quicksand-Medium.ttf',
    'Quicksand-SemiBold.ttf',
    'Quicksand-Bold.ttf',
)


def mm(value: float) -> float:
    """Millimetres to inches, the unit matplotlib sizes figures in."""
    return value * style.MM


def register_fonts() -> None:
    """Make Quicksand (the poster's typeface) available to matplotlib."""
    for name in FONT_FILES:
        path = FONT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f'{path} missing; poster text would not match')
        fm.fontManager.addfont(str(path))
    fm.findfont('Quicksand', fallback_to_default=False)


def apply() -> None:
    """Install the shared theme, then rescale it for A0."""
    style.apply()
    register_fonts()
    mpl.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Quicksand'],
            'font.weight': 500,  # Quicksand Medium: Regular is too light at a distance
            'font.size': 22,
            'axes.labelsize': 22,
            'axes.titlesize': 24,
            'axes.linewidth': 1.6,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'xtick.major.size': 8,
            'ytick.major.size': 8,
            'xtick.major.width': 1.6,
            'ytick.major.width': 1.6,
            'legend.fontsize': 20,
            'legend.title_fontsize': 20,
            'lines.linewidth': 3.6,
            'lines.markersize': 10,
            'grid.linewidth': 1.0,
            'svg.fonttype': 'path',
            # Fixed salt: matplotlib otherwise draws random element ids, so two runs
            # of the same script would write different SVG bytes.
            'svg.hashsalt': 'learnm8-poster',
            # White card behind the whole figure, labels and legends included; the
            # layout pad keeps text off the card's edge.
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'figure.constrained_layout.w_pad': 0.10,
            'figure.constrained_layout.h_pad': 0.10,
        }
    )


def compact_counts(ax, axis: str = 'x') -> None:
    """Label large counts as 250k / 1M so the ticks stay short at 20 pt."""

    def fmt(value: float, _position: int) -> str:
        if value >= 1e6:
            return f'{value / 1e6:g}M'
        if value >= 1e3:
            return f'{value / 1e3:g}k'
        return f'{value:g}'

    target = ax.xaxis if axis == 'x' else ax.yaxis
    target.set_major_formatter(FuncFormatter(fmt))


def panel_letter(ax, letter: str) -> None:
    """Bold panel letter at the top-left corner of an axes."""
    ax.text(
        -0.02,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=26,
        fontweight='bold',
        color=style.INK,
        ha='right',
        va='bottom',
    )


def assert_min_font(fig, minimum: float = MIN_FONT_PT) -> None:
    """Fail if any visible text in ``fig`` prints below ``minimum`` pt."""
    fig.canvas.draw()
    small = sorted(
        {
            (round(text.get_fontsize(), 1), text.get_text())
            for text in fig.findobj(mpl.text.Text)
            if text.get_visible() and text.get_text().strip()
            if text.get_fontsize() < minimum
        }
    )
    if small:
        raise ValueError(f'text below {minimum} pt: {small[:6]}')


def save_poster(fig, name: str) -> None:
    """Check text sizes, then write ``name``.svg and ``name``.png."""
    assert_min_font(fig)
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    # No bbox_inches='tight': the file must keep its figsize so that placing it at
    # that width on the poster prints every font at the size asserted above.
    fig.savefig(POSTER_DIR / f'{name}.svg', format='svg')
    fig.savefig(POSTER_DIR / f'{name}.png', dpi=300)
    plt.close(fig)
    print(f'wrote {POSTER_DIR / name}.svg')

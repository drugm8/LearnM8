"""Shared heatmap utilities used by the validation plotting functions."""

from typing import Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from learnm8.visualization import style


def get_purple_colormap(reverse: bool = False) -> mcolors.LinearSegmentedColormap:
    """Return the shared LearnM8 sequential heatmap colormap.

    Args:
        reverse: If True, reverse the colormap (dark→light for "lower is better" metrics)

    Returns:
        LinearSegmentedColormap with muted purple gradient based on LearnM8 brand color
    """
    return style.SEQUENTIAL.reversed() if reverse else style.SEQUENTIAL


def calculate_luminance(rgb: Tuple[float, float, float]) -> float:
    """
    Calculate relative luminance of an RGB color.

    Uses the standard formula: 0.299*R + 0.587*G + 0.114*B

    Args:
        rgb: Tuple of (r, g, b) values in range [0, 1]

    Returns:
        Luminance value in range [0, 1]
    """
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def calculate_text_color(rgb: Tuple[float, float, float]) -> str:
    """
    Determine optimal text color (black or white) for given background RGB.

    Uses luminance calculation to ensure high contrast.

    Args:
        rgb: Tuple of (r, g, b) values in range [0, 1]

    Returns:
        'white' for dark backgrounds, 'black' for light backgrounds
    """
    luminance = calculate_luminance(rgb)
    return style.BACKGROUND if luminance < 0.5 else style.INK


def get_cell_color_from_value(
    value: float,
    colormap: mcolors.Colormap,
    vmin: float,
    vmax: float
) -> Tuple[float, float, float]:
    """
    Get RGB color for a data value using colormap normalization.

    Args:
        value: Data value to map to color
        colormap: Matplotlib colormap
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization

    Returns:
        RGB tuple in range [0, 1]
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    normalized_value = norm(value)
    rgba = colormap(normalized_value)
    return rgba[:3]  # Return only RGB, not alpha


def auto_scale_range(
    data: np.ndarray,
    percentile: float = 5,
    padding: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate dynamic vmin/vmax from actual data distribution.

    Uses percentile clipping to handle outliers and adds padding for visual clarity.

    Args:
        data: Data array (can contain NaN values)
        percentile: Percentile for clipping (default: 5 = clip at 5th and 95th)
        padding: Fractional padding to add to range (default: 0.05 = 5%)

    Returns:
        Tuple of (vmin, vmax) for heatmap scaling
    """
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        return 0, 1

    if percentile > 0:
        vmin, vmax = np.percentile(valid_data, [percentile, 100 - percentile])
    else:
        vmin, vmax = valid_data.min(), valid_data.max()

    data_range = vmax - vmin
    if data_range < 1e-10:
        return vmin - 0.5, vmax + 0.5

    padding_amount = data_range * padding
    vmin_scaled = max(0, vmin - padding_amount)
    vmax_scaled = vmax + padding_amount

    return vmin_scaled, vmax_scaled


def add_heatmap_annotations(
    ax: plt.Axes,
    data: np.ndarray,
    std_data: Optional[np.ndarray],
    colormap: mcolors.Colormap,
    vmin: float,
    vmax: float,
    format_string: str = '{:.1f}',
    show_std: bool = True,
    fontsize: int = 8
) -> None:
    """
    Add text annotations to heatmap with automatic contrast adjustment.

    Args:
        ax: Matplotlib axes containing the heatmap
        data: 2D array of mean values
        std_data: 2D array of std values (optional)
        colormap: Colormap used for heatmap
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
        format_string: Format string for values (default: '{:.1f}')
        show_std: If True and std_data provided, show mean±std format
        fontsize: Font size for annotations (default: 8)
    """
    style.style_heatmap(ax)
    rows, cols = data.shape

    for i in range(rows):
        for j in range(cols):
            value = data[i, j]

            if np.isnan(value):
                continue

            rgb = get_cell_color_from_value(value, colormap, vmin, vmax)
            text_color = calculate_text_color(rgb)

            if show_std and std_data is not None:
                std_val = std_data[i, j]
                if not np.isnan(std_val):
                    text = f'{format_string}±{format_string}'.format(value, std_val)
                else:
                    text = format_string.format(value)
            else:
                text = format_string.format(value)

            ax.text(
                j + 0.5, i + 0.5, text,
                ha='center', va='center',
                fontsize=fontsize,
                fontweight='bold',
                color=text_color
            )

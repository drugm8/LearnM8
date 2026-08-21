import textwrap

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess

from learnm8.visualization import style

from .uq_metrics import (
    compute_ause,
    compute_calibration_curve,
    compute_ence_decomposition,
    compute_sparsification_data,
    compute_spearman,
)

OKABE_ITO = style.CATEGORICAL


def apply_style():
    """Install the shared LearnM8 plotting theme."""
    style.apply()


def _lighten_color(color, amount=0.35):
    """Return a lighter tint of a hex/named color for box backgrounds."""
    rgb = mcolors.to_rgb(color)
    return tuple(c + (1.0 - c) * (1.0 - amount) for c in rgb)


def _create_subplot_grid(n_models, figsize_per_subplot=(4, 3.5), max_cols=4):
    """Create a subplot grid sized to the number of models.

    Returns (fig, axes_flat) with hidden unused subplots.
    """
    ncols = min(n_models, max_cols)
    nrows = (n_models + ncols - 1) // ncols
    figsize = (figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)
    return fig, axes_flat


def _save_and_close(fig, output_path):
    """Save figure and close if output_path is provided."""
    if output_path:
        for ax in fig.axes:
            if getattr(ax, '_learnm8_heatmap', False):
                style.style_heatmap(ax)
            elif getattr(ax, '_colorbar', None) is None:
                style.style_axes(ax)
        fig.savefig(
            output_path,
            dpi=style.OUTPUT_DPI,
            bbox_inches='tight',
            pad_inches=0.08,
            facecolor=style.BACKGROUND,
        )
        plt.close(fig)


def _curve_kwargs(index: int, n_points: int | None = None) -> dict:
    """Return redundant line encodings for dense multi-model curves."""
    kwargs = {
        'linestyle': style.CURVE_LINESTYLES[index % len(style.CURVE_LINESTYLES)],
        'marker': style.CURVE_MARKERS[index % len(style.CURVE_MARKERS)],
    }
    if n_points:
        kwargs['markevery'] = max(1, n_points // 8)
    return kwargs


def _directional_scores(values, lower_is_better: list[bool]):
    """Normalize heterogeneous metrics so darker always means better."""
    scores = values.copy()
    for column, lower in zip(values.columns, lower_is_better, strict=True):
        series = values[column].astype(float)
        valid = series.dropna()
        if valid.empty:
            continue
        minimum = float(valid.min())
        maximum = float(valid.max())
        if np.isclose(minimum, maximum):
            scores[column] = series.notna().astype(float) * 0.5
            continue
        normalized = (series - minimum) / (maximum - minimum)
        scores[column] = 1.0 - normalized if lower else normalized
    return scores


def _apply_title_layout(fig, title, subtitle=None, **_kwargs):
    """Place figure title and optional subtitle using standard matplotlib spacing."""
    if subtitle:
        title = f'{title}\n{subtitle}'
    fig.suptitle(
        textwrap.fill(title, width=72),
        fontweight='bold',
        fontsize=13,
        y=0.98,
    )


# ── Existing plot functions ──────────────────────────────────────────────────


def plot_calibration_curves(models_data, output_path=None, figsize=(7, 5.5),
                            subtitle=None):
    """Plot calibration curves (reliability diagrams) for multiple models.

    Args:
        models_data: dict mapping model_name -> {
            'y_true': array, 'y_pred': array, 'y_std': array,
            'color': str (optional), 'miscal_area': float (optional)
        }
        output_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], '--', color=style.INK, lw=1, alpha=0.55, label='Perfect calibration')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color=style.MUTED)

    for i, (name, data) in enumerate(models_data.items()):
        color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])
        expected, observed = compute_calibration_curve(
            data['y_true'], data['y_pred'], data['y_std']
        )
        miscal = data.get('miscal_area', None)
        label = f"{name} (MA={miscal:.3f})" if miscal is not None else name
        ax.plot(
            expected,
            observed,
            color=color,
            lw=1.8,
            markersize=4,
            label=label,
            alpha=0.85,
            **_curve_kwargs(i, len(expected)),
        )

    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    _apply_title_layout(
        fig,
        "Are the model's predicted confidence intervals statistically reliable?",
        subtitle=subtitle,
    )
    _save_and_close(fig, output_path)
    return fig


def plot_calibration_before_after(models_data_raw, models_data_calibrated,
                                  output_path=None, figsize=(14, 5.5),
                                  subtitle=None):
    """Plot calibration curves before and after STD scaling side by side."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, data_dict, title in [
        (ax1, models_data_raw, 'Before Calibration (Raw)'),
        (ax2, models_data_calibrated, 'After STD Scaling')
    ]:
        ax.plot([0, 1], [0, 1], '--', color=style.INK, lw=1, alpha=0.55)
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color=style.MUTED)

        for i, (name, data) in enumerate(data_dict.items()):
            color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])
            expected, observed = compute_calibration_curve(
                data['y_true'], data['y_pred'], data['y_std']
            )
            miscal = data.get('miscal_area', None)
            label = f"{name} ({miscal:.3f})" if miscal is not None else name
            ax.plot(
                expected,
                observed,
                color=color,
                lw=1.8,
                markersize=4,
                label=label,
                alpha=0.85,
                **_curve_kwargs(i, len(expected)),
            )

        ax.set_xlabel('Expected Coverage')
        ax.set_ylabel('Observed Coverage')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=6, framealpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    _apply_title_layout(
        fig,
        'Does STD scaling calibration correct systematic over- or under-confidence?',
        subtitle=subtitle,
    )
    _save_and_close(fig, output_path)
    return fig


def plot_summary_heatmap(metrics_df, output_path=None, figsize=(10, 5)):
    """Heatmap of UQ metrics across models."""
    apply_style()
    import pandas as pd

    display_metrics = ['ence', 'nll', 'miscalibration_area', 'ause', 'spearman_rho']
    display_labels = ['ENCE', 'NLL', 'Miscal. Area', 'AUSE', 'Spearman rho']
    lower_is_better = [True, True, True, True, False]

    rows = {}
    for model_name, metrics in metrics_df.items():
        rows[model_name] = [metrics.get(m, np.nan) for m in display_metrics]

    df = pd.DataFrame(rows, index=display_labels).T
    score_df = _directional_scores(df, lower_is_better)

    fig, ax = plt.subplots(figsize=figsize)
    mask = df.isna()
    sns.heatmap(
        score_df,
        ax=ax,
        annot=df.to_numpy(),
        fmt='.3f',
        cmap=style.SEQUENTIAL,
        vmin=0,
        vmax=1,
        linewidths=0.5, mask=mask, annot_kws={'size': 9},
        cbar_kws={'label': 'Relative performance (higher is better)', 'shrink': 0.8}
    )
    ax._learnm8_heatmap = True
    ax.set_title('Which model provides the most reliable and well-ordered uncertainty estimates?',
                 fontweight='bold', fontsize=10)
    ax.annotate('final model, out-of-sample', xy=(0.5, 1.02),
                xycoords='axes fraction', ha='center', fontsize=8,
                style='italic', color=style.MUTED)
    ax.set_ylabel('')

    for j, lib in enumerate(lower_is_better):
        col_vals = df.iloc[:, j].dropna()
        if len(col_vals) > 0:
            best_idx = col_vals.idxmin() if lib else col_vals.idxmax()
            row_pos = list(df.index).index(best_idx)
            ax.add_patch(plt.Rectangle((j, row_pos), 1, 1,
                                        fill=False, edgecolor=style.ACCENT_GREEN, lw=2.0))

    _save_and_close(fig, output_path)
    return fig


def plot_summary_heatmap_with_std(metrics_mean, metrics_std, output_path=None,
                                   figsize=(10, 5)):
    """Heatmap with mean +/- std annotations from multi-seed runs."""
    apply_style()
    import pandas as pd

    display_metrics = ['ence', 'nll', 'miscalibration_area', 'ause', 'spearman_rho']
    display_labels = ['ENCE', 'NLL', 'Miscal. Area', 'AUSE', 'Spearman rho']
    lower_is_better = [True, True, True, True, False]

    rows_mean = {}
    rows_annot = {}
    for model_name in metrics_mean:
        row_mean = []
        row_annot = []
        for m in display_metrics:
            mean_val = metrics_mean[model_name].get(m, np.nan)
            std_val = metrics_std[model_name].get(m, 0.0)
            row_mean.append(mean_val)
            if np.isnan(mean_val):
                row_annot.append('')
            else:
                row_annot.append(f'{mean_val:.3f}\n\u00b1{std_val:.3f}')
        rows_mean[model_name] = row_mean
        rows_annot[model_name] = row_annot

    df_mean = pd.DataFrame(rows_mean, index=display_labels).T
    df_annot = pd.DataFrame(rows_annot, index=display_labels).T
    score_df = _directional_scores(df_mean, lower_is_better)

    fig, ax = plt.subplots(figsize=figsize)
    mask = df_mean.isna()
    sns.heatmap(
        score_df,
        ax=ax,
        annot=df_annot.values,
        fmt='',
        cmap=style.SEQUENTIAL,
        vmin=0,
        vmax=1,
        linewidths=0.5, mask=mask, annot_kws={'size': 8},
        cbar_kws={
            'label': 'Relative performance (higher is better)',
            'shrink': 0.8,
        }
    )
    ax._learnm8_heatmap = True
    ax.set_title('Which model provides the most reliable and well-ordered uncertainty estimates?',
                 fontweight='bold', fontsize=10)
    ax.annotate('final model, out-of-sample', xy=(0.5, 1.02),
                xycoords='axes fraction', ha='center', fontsize=8,
                style='italic', color=style.MUTED)
    ax.set_ylabel('')

    for j, lib in enumerate(lower_is_better):
        col_vals = df_mean.iloc[:, j].dropna()
        if len(col_vals) > 0:
            best_idx = col_vals.idxmin() if lib else col_vals.idxmax()
            row_pos = list(df_mean.index).index(best_idx)
            ax.add_patch(plt.Rectangle((j, row_pos), 1, 1,
                                        fill=False, edgecolor=style.ACCENT_GREEN, lw=2.0))

    _save_and_close(fig, output_path)
    return fig


def plot_ensemble_size_ablation(size_metrics, output_path=None, figsize=(14, 8),
                                xlabel='Ensemble Size',
                                title='RF Ensemble: UQ Metrics vs Ensemble Size',
                                metrics_to_show=None, subtitle=None):
    """Plot UQ metrics vs a swept parameter for ablation study.

    Args:
        metrics_to_show: Optional list of metric keys to show (e.g. ['ause',
            'miscalibration_area', 'mae']). When None, all 6 metrics are shown
            in a 2×3 grid. When specified, only those metrics are shown in a
            1×N grid.
    """
    apply_style()
    sizes = sorted(size_metrics.keys())

    if metrics_to_show is not None:
        _metric_label_map = {
            'ause': 'AUSE ↓',
            'miscalibration_area': 'Miscal. Area ↓',
            'mae': 'MAE ↓',
            'ence': 'ENCE ↓',
            'nll': 'NLL ↓',
            'spearman_rho': 'Spearman ρ ↑',
        }
        n = len(metrics_to_show)
        fig_w = max(4 * n, 10)
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 4))
        if n == 1:
            axes = [axes]
        for i, key in enumerate(metrics_to_show):
            ax = axes[i]
            label = _metric_label_map.get(key, key)
            means = [size_metrics[s]['metrics_mean'].get(key, np.nan) for s in sizes]
            stds = [size_metrics[s]['metrics_std'].get(key, 0.0) for s in sizes]
            ax.errorbar(
                sizes,
                means,
                yerr=stds,
                color=OKABE_ITO[i],
                lw=1.8,
                markersize=6,
                capsize=4,
                capthick=1.2,
                **_curve_kwargs(i, len(sizes)),
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label)
            ax.set_title(label, fontweight='bold')
            ax.set_xticks(sizes)
        _apply_title_layout(fig, title, subtitle=subtitle)
        _save_and_close(fig, output_path)
        return fig

    metric_keys = ['ence', 'nll', 'miscalibration_area', 'ause', 'spearman_rho']
    metric_labels = [
        'ENCE \u2193', 'NLL \u2193', 'Miscal. Area \u2193',
        'AUSE \u2193', 'Spearman rho \u2191',
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = axes.flatten()

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels, strict=True)):
        ax = axes_flat[i]
        means = [size_metrics[s]['metrics_mean'].get(key, np.nan) for s in sizes]
        stds = [size_metrics[s]['metrics_std'].get(key, 0.0) for s in sizes]

        ax.errorbar(
            sizes,
            means,
            yerr=stds,
            color=OKABE_ITO[i],
            lw=1.8,
            markersize=6,
            capsize=4,
            capthick=1.2,
            **_curve_kwargs(i, len(sizes)),
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(sizes)

    ax_mae = axes_flat[5]
    maes = [size_metrics[s]['metrics_mean'].get('mae', np.nan) for s in sizes]
    mae_stds = [size_metrics[s]['metrics_std'].get('mae', 0.0) for s in sizes]
    ax_mae.errorbar(
        sizes,
        maes,
        yerr=mae_stds,
        color=OKABE_ITO[5],
        lw=1.8,
        markersize=6,
        capsize=4,
        capthick=1.2,
        **_curve_kwargs(5, len(sizes)),
    )
    ax_mae.set_xlabel(xlabel)
    ax_mae.set_ylabel('MAE')
    ax_mae.set_title('MAE (Prediction Accuracy)', fontweight='bold')
    ax_mae.set_xticks(sizes)

    _apply_title_layout(fig, title, subtitle=subtitle)
    _save_and_close(fig, output_path)
    return fig


def plot_dataset_comparison(metrics_30k, metrics_500k, output_path=None,
                            figsize=(16, 8), subtitle=None):
    """Grouped bar chart comparing UQ metrics between 30K and 500K datasets."""
    apply_style()

    common_models = [m for m in metrics_30k if m in metrics_500k
                     and metrics_30k[m]['mean'] and metrics_500k[m]['mean']]
    if not common_models:
        return None

    metric_keys = ['ence', 'miscalibration_area', 'ause', 'spearman_rho', 'mae']
    metric_labels = ['ENCE', 'Miscal. Area', 'AUSE', 'Spearman rho', 'MAE']
    lower_is_better = [True, True, True, False, True]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = axes.flatten()

    x = np.arange(len(common_models))
    width = 0.35

    for i, (key, label, lib) in enumerate(
        zip(metric_keys, metric_labels, lower_is_better, strict=True)
    ):
        ax = axes_flat[i]

        vals_30k = [metrics_30k[m]['mean'].get(key, np.nan) for m in common_models]
        errs_30k = [metrics_30k[m]['std'].get(key, 0.0) for m in common_models]
        vals_500k = [metrics_500k[m]['mean'].get(key, np.nan) for m in common_models]
        errs_500k = [metrics_500k[m]['std'].get(key, 0.0) for m in common_models]

        ax.bar(x - width / 2, vals_30k, width, yerr=errs_30k,
               label='30K', color=OKABE_ITO[0], alpha=0.8, capsize=4)
        ax.bar(x + width / 2, vals_500k, width, yerr=errs_500k,
               label='500K', color=OKABE_ITO[1], alpha=0.8, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(common_models, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(label)
        direction = '(lower is better)' if lib else '(higher is better)'
        ax.set_title(f'{label} {direction}', fontweight='bold', fontsize=9)
        ax.legend(fontsize=7)

    ax_nll = axes_flat[5]
    nll_30k = [metrics_30k[m]['mean'].get('nll', np.nan) for m in common_models]
    nll_err_30k = [metrics_30k[m]['std'].get('nll', 0.0) for m in common_models]
    nll_500k = [metrics_500k[m]['mean'].get('nll', np.nan) for m in common_models]
    nll_err_500k = [metrics_500k[m]['std'].get('nll', 0.0) for m in common_models]
    ax_nll.bar(x - width / 2, nll_30k, width, yerr=nll_err_30k,
               label='30K', color=OKABE_ITO[0], alpha=0.8, capsize=4)
    ax_nll.bar(x + width / 2, nll_500k, width, yerr=nll_err_500k,
               label='500K', color=OKABE_ITO[1], alpha=0.8, capsize=4)
    ax_nll.set_xticks(x)
    ax_nll.set_xticklabels(common_models, rotation=30, ha='right', fontsize=7)
    ax_nll.set_ylabel('NLL')
    ax_nll.set_title('NLL (lower is better)', fontweight='bold', fontsize=9)
    ax_nll.legend(fontsize=7)

    _apply_title_layout(
        fig,
        'How does uncertainty quality change when scaling from 30K to 500K compounds?',
        subtitle=subtitle or '30K vs 500K, out-of-sample',
    )
    _save_and_close(fig, output_path)
    return fig


def plot_dataset_calibration_comparison(models_30k, models_500k,
                                         output_path=None, figsize=(14, 5.5),
                                         subtitle=None):
    """Side-by-side calibration curves for 30K vs 500K datasets."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, data_dict, title in [
        (ax1, models_30k, '30K Compounds'),
        (ax2, models_500k, '500K Compounds')
    ]:
        ax.plot([0, 1], [0, 1], '--', color=style.INK, lw=1, alpha=0.55)
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color=style.MUTED)

        for i, (name, data) in enumerate(data_dict.items()):
            color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])
            expected, observed = compute_calibration_curve(
                data['y_true'], data['y_pred'], data['y_std']
            )
            miscal = data.get('miscal_area', None)
            label = f"{name} ({miscal:.3f})" if miscal is not None else name
            ax.plot(
                expected,
                observed,
                color=color,
                lw=1.8,
                markersize=4,
                label=label,
                alpha=0.85,
                **_curve_kwargs(i, len(expected)),
            )

        ax.set_xlabel('Expected Coverage')
        ax.set_ylabel('Observed Coverage')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=6, framealpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    _apply_title_layout(
        fig,
        'Does calibration reliability hold when scaling to ten times more compounds?',
        subtitle=subtitle or '30K vs 500K, out-of-sample',
    )
    _save_and_close(fig, output_path)
    return fig


def plot_cycle_progression(cycle_metrics_by_model, metric_key='ence',
                            metric_label='ENCE', title=None,
                            subtitle='out-of-sample predictions per cycle',
                            output_path=None, figsize=(8, 5)):
    """Line plot of a UQ metric across active learning cycles for each model.

    cycle_metrics_by_model: dict mapping model_name ->
        {cycle_num: scalar_value} or {cycle_num: (mean, std)}
    When (mean, std) tuples are provided a shaded ±1σ band is drawn.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, cycle_data) in enumerate(cycle_metrics_by_model.items()):
        cycles = sorted(cycle_data.keys())
        color = OKABE_ITO[i % len(OKABE_ITO)]
        vals = [cycle_data[c] for c in cycles]
        means = np.array([v[0] if isinstance(v, tuple) else v for v in vals])
        stds = np.array([v[1] if isinstance(v, tuple) else 0.0 for v in vals])
        ax.plot(
            cycles,
            means,
            color=color,
            lw=1.8,
            markersize=5,
            label=name,
            alpha=0.85,
            **_curve_kwargs(i, len(cycles)),
        )
        if np.any(stds > 0):
            ax.fill_between(cycles, means - stds, means + stds,
                            color=color, alpha=0.15)

    ax.set_xlabel('Active Learning Cycle')
    ax.set_ylabel(metric_label)
    ax.legend(loc='best', fontsize=7, framealpha=0.9)

    main_title = title or f'{metric_label} Across Active Learning Cycles'
    _apply_title_layout(fig, main_title, subtitle=subtitle, top=0.87)
    _save_and_close(fig, output_path)
    return fig


# ── New plot functions (subplot grid, one per model) ─────────────────────────


def plot_sparsification(models_data, output_path=None, figsize=None,
                        subtitle=None):
    """Sparsification plot with one subplot per model.

    Each subplot shows the model's MAE curve vs the oracle (best possible)
    and random baseline as compounds are progressively removed by uncertainty.
    """
    apply_style()
    n_models = len(models_data)
    if figsize:
        fig, axes = plt.subplots(
            (n_models + 3) // 4, min(n_models, 4), figsize=figsize, squeeze=False
        )
        axes_flat = axes.flatten()
        for j in range(n_models, len(axes_flat)):
            axes_flat[j].set_visible(False)
    else:
        fig, axes_flat = _create_subplot_grid(n_models)

    for i, (name, data) in enumerate(models_data.items()):
        ax = axes_flat[i]
        color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])

        sp = compute_sparsification_data(
            data['y_true'], data['y_pred'], data['y_std']
        )

        ause = compute_ause(data['y_true'], data['y_pred'], data['y_std'])

        ax.plot(
            sp['fractions'],
            sp['model_maes'],
            color=color,
            lw=1.8,
            label='Model',
            **_curve_kwargs(i, len(sp['fractions'])),
        )
        ax.plot(sp['fractions'], sp['oracle_maes'], '--', color=style.INK, lw=1.2,
                label='Oracle')
        ax.axhline(y=sp['random_mae'], color=style.MUTED, ls=':', lw=1, alpha=0.7,
                   label='Random')
        ax.fill_between(sp['fractions'], sp['oracle_maes'], sp['model_maes'],
                        alpha=0.15, color=color)

        ax.text(0.95, 0.95, f'AUSE={ause:.4f}', transform=ax.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=_lighten_color(color),
                          edgecolor=color, alpha=0.8))

        ax.set_xlabel('Fraction Removed')
        ax.set_ylabel('MAE')
        ax.set_title(name, fontweight='bold', fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc='lower left')

    _apply_title_layout(
        fig,
        'Does removing the most uncertain predictions lower average prediction error?',
        subtitle=subtitle,
    )
    _save_and_close(fig, output_path)
    return fig


def plot_uncertainty_vs_error_scatter(models_data, output_path=None, figsize=None,
                                     subtitle=None):
    """Scatter plot of predicted uncertainty vs absolute error.

    Uses hexbin for large datasets (>5000 points) and scatter with LOWESS
    trend line otherwise.
    """
    apply_style()
    n_models = len(models_data)
    if figsize:
        fig, axes = plt.subplots(
            (n_models + 3) // 4, min(n_models, 4), figsize=figsize, squeeze=False
        )
        axes_flat = axes.flatten()
        for j in range(n_models, len(axes_flat)):
            axes_flat[j].set_visible(False)
    else:
        fig, axes_flat = _create_subplot_grid(n_models)

    for i, (name, data) in enumerate(models_data.items()):
        ax = axes_flat[i]
        color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])

        y_true = np.asarray(data['y_true'])
        y_pred = np.asarray(data['y_pred'])
        y_std = np.asarray(data['y_std'])
        errors = np.abs(y_true - y_pred)
        n = len(errors)

        if n > 5000:
            hb = ax.hexbin(y_std, errors, gridsize=40, cmap=style.SEQUENTIAL,
                           mincnt=1, linewidths=0.2)
            fig.colorbar(hb, ax=ax, shrink=0.7, label='Count')
        else:
            ax.scatter(y_std, errors, alpha=0.3, s=8, color=color, edgecolors='none')

        rng = np.random.default_rng(42)
        n_lowess = min(n, 10000)
        idx = rng.choice(n, size=n_lowess, replace=False) if n > n_lowess else np.arange(n)
        smoothed = lowess(errors[idx], y_std[idx], frac=0.3, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], '-', color=style.ACCENT_ORANGE, lw=2,
                label='LOWESS')

        rho, _ = compute_spearman(y_true, y_pred, y_std)
        ax.text(0.95, 0.95, f'\u03c1={rho:.3f}', transform=ax.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=style.BACKGROUND,
                          edgecolor=style.MUTED, alpha=0.8))

        ax.set_xlabel('Predicted \u03c3')
        ax.set_ylabel('|Error|')
        ax.set_title(name, fontweight='bold', fontsize=9)
        if i == 0:
            ax.legend(fontsize=6, loc='upper left')

    _apply_title_layout(
        fig,
        'Do compounds with higher predicted uncertainty have larger prediction errors?',
        subtitle=subtitle,
    )
    _save_and_close(fig, output_path)
    return fig


def plot_prediction_interval_coverage(models_data, output_path=None, figsize=None,
                                       confidence_levels=None, subtitle=None):
    """Prediction interval coverage bar chart.

    Each subplot shows expected vs observed coverage at several confidence levels.
    """
    apply_style()
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95]

    n_models = len(models_data)
    if figsize:
        fig, axes = plt.subplots(
            (n_models + 3) // 4, min(n_models, 4), figsize=figsize, squeeze=False
        )
        axes_flat = axes.flatten()
        for j in range(n_models, len(axes_flat)):
            axes_flat[j].set_visible(False)
    else:
        fig, axes_flat = _create_subplot_grid(n_models)

    for i, (name, data) in enumerate(models_data.items()):
        ax = axes_flat[i]
        color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])

        y_true = np.asarray(data['y_true'], dtype=float)
        y_pred = np.asarray(data['y_pred'], dtype=float)
        y_std = np.clip(np.asarray(data['y_std'], dtype=float), 1e-10, None)

        observed_coverages = []
        for p in confidence_levels:
            z = norm.ppf((1 + p) / 2)
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std
            empirical = np.mean((y_true >= lower) & (y_true <= upper))
            observed_coverages.append(empirical)

        x = np.arange(len(confidence_levels))
        width = 0.35
        labels = [f'{int(p * 100)}%' for p in confidence_levels]

        ax.bar(x - width / 2, confidence_levels, width, color=style.MUTED_LIGHT,
               edgecolor=style.MUTED, label='Expected')
        bars = ax.bar(x + width / 2, observed_coverages, width, color=color,
                      alpha=0.8, label='Observed')

        for bar, obs, exp in zip(bars, observed_coverages, confidence_levels, strict=True):
            edge = style.ACCENT_GREEN if abs(obs - exp) < 0.05 else style.ACCENT_ORANGE
            bar.set_edgecolor(edge)
            bar.set_linewidth(1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Coverage')
        ax.set_ylim(0, 1.1)
        ax.set_title(name, fontweight='bold', fontsize=9)
        if i == 0:
            ax.legend(fontsize=6)

    _apply_title_layout(
        fig,
        'What fraction of true values fall within each predicted confidence interval?',
        subtitle=subtitle,
    )
    _save_and_close(fig, output_path)
    return fig


def plot_qq_residuals(models_data, output_path=None, figsize=None, subtitle=None):
    """Q-Q plot of normalized residuals vs standard normal.

    Each subplot shows whether prediction errors scaled by uncertainty
    follow the expected Gaussian distribution.
    """
    apply_style()
    n_models = len(models_data)
    if figsize:
        fig, axes = plt.subplots(
            (n_models + 3) // 4, min(n_models, 4), figsize=figsize, squeeze=False
        )
        axes_flat = axes.flatten()
        for j in range(n_models, len(axes_flat)):
            axes_flat[j].set_visible(False)
    else:
        fig, axes_flat = _create_subplot_grid(n_models)

    for i, (name, data) in enumerate(models_data.items()):
        ax = axes_flat[i]
        color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])

        y_true = np.asarray(data['y_true'], dtype=float)
        y_pred = np.asarray(data['y_pred'], dtype=float)
        y_std = np.clip(np.asarray(data['y_std'], dtype=float), 1e-10, None)

        z = (y_true - y_pred) / y_std

        (osm, osr), (_, _, _) = scipy_stats.probplot(z, dist='norm')
        ax.scatter(osm, osr, s=6, alpha=0.5, color=color, edgecolors='none')

        line_range = np.array([osm.min(), osm.max()])
        ax.plot(line_range, line_range, '--', color=style.INK, lw=1, alpha=0.7, label='N(0,1)')

        rng = np.random.default_rng(42)
        z_sample = z if len(z) <= 5000 else z[rng.choice(len(z), 5000, replace=False)]
        sw_stat, sw_p = scipy_stats.shapiro(z_sample)

        ax.text(0.05, 0.95, f'SW={sw_stat:.3f}\np={sw_p:.2e}',
                transform=ax.transAxes, fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=style.BACKGROUND,
                          edgecolor=style.MUTED, alpha=0.8))

        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title(name, fontweight='bold', fontsize=9)
        if i == 0:
            ax.legend(fontsize=6, loc='lower right')

    _apply_title_layout(
        fig,
        'Are normalized prediction residuals consistent with a standard normal distribution?',
        subtitle=subtitle,
    )
    _save_and_close(fig, output_path)
    return fig


def plot_ence_decomposition(models_data, output_path=None, figsize=None, n_bins=10,
                            subtitle=None):
    """ENCE decomposition bar chart showing per-bin RMSE vs RMV.

    Each subplot shows where the model is overconfident (RMSE > RMV, red edge)
    vs underconfident (RMV > RMSE, blue edge).
    """
    apply_style()
    n_models = len(models_data)
    if figsize:
        fig, axes = plt.subplots(
            (n_models + 3) // 4, min(n_models, 4), figsize=figsize, squeeze=False
        )
        axes_flat = axes.flatten()
        for j in range(n_models, len(axes_flat)):
            axes_flat[j].set_visible(False)
    else:
        fig, axes_flat = _create_subplot_grid(n_models)

    for i, (name, data) in enumerate(models_data.items()):
        ax = axes_flat[i]
        color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])

        decomp = compute_ence_decomposition(
            data['y_true'], data['y_pred'], data['y_std'], n_bins=n_bins
        )

        x = np.arange(len(decomp['bin_labels']))
        width = 0.35

        bars_rmse = ax.bar(x - width / 2, decomp['rmse'], width, color=color,
                           alpha=0.8, label='RMSE')
        bars_rmv = ax.bar(x + width / 2, decomp['rmv'], width, color=style.MUTED_LIGHT,
                          edgecolor=style.MUTED, label='RMV')

        for j_bin, overconf in enumerate(decomp['overconfident']):
            edge_color = style.ACCENT_ORANGE if overconf else style.ACCENT_BLUE
            bars_rmse[j_bin].set_edgecolor(edge_color)
            bars_rmse[j_bin].set_linewidth(1.5)
            bars_rmv[j_bin].set_edgecolor(edge_color)
            bars_rmv[j_bin].set_linewidth(1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(decomp['bin_labels'], fontsize=7)
        ax.set_xlabel('Uncertainty Bin (low \u2192 high)')
        ax.set_ylabel('Value')
        ax.set_title(name, fontweight='bold', fontsize=9)
        if i == 0:
            ax.legend(fontsize=6)

    _apply_title_layout(
        fig,
        'Are uncertainty magnitudes proportional to actual errors across bins?',
        subtitle=subtitle,
    )
    _save_and_close(fig, output_path)
    return fig


# ── Publication summary (overlaid panels) ────────────────────────────────────


def create_publication_summary(all_models_data, all_metrics, output_path=None,
                                figsize=(16, 12), subtitle=None):
    """Publication-quality 2x2 summary with overlaid panels.

    Layout:
    - A) Calibration curves (all models overlaid)
    - B) Sparsification curves (all models overlaid)
    - C) Uncertainty vs error scatter (top 4 models, overlaid with LOWESS)
    - D) Summary heatmap
    """
    apply_style()
    import pandas as pd

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.28,
        wspace=0.25,
        top=0.86,
        bottom=0.08,
        left=0.06,
        right=0.96,
    )

    # A) Calibration curves
    ax_cal = fig.add_subplot(gs[0, 0])
    ax_cal.plot([0, 1], [0, 1], '--', color=style.INK, lw=1, alpha=0.55)
    for i, (name, data) in enumerate(all_models_data.items()):
        color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])
        expected, observed = compute_calibration_curve(
            data['y_true'], data['y_pred'], data['y_std']
        )
        miscal = all_metrics.get(name, {}).get('miscalibration_area', None)
        label = f"{name} ({miscal:.3f})" if miscal is not None else name
        ax_cal.plot(
            expected,
            observed,
            color=color,
            lw=1.5,
            markersize=3,
            label=label,
            alpha=0.85,
            **_curve_kwargs(i, len(expected)),
        )
    ax_cal.set_xlabel('Expected Coverage')
    ax_cal.set_ylabel('Observed Coverage')
    ax_cal.set_title('A) Calibration Curves', fontweight='bold')
    ax_cal.legend(loc='upper left', fontsize=5, framealpha=0.9)
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_aspect('equal')

    # B) Sparsification curves (overlaid)
    ax_sp = fig.add_subplot(gs[0, 1])
    oracle_plotted = False
    for i, (name, data) in enumerate(all_models_data.items()):
        color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])
        sp = compute_sparsification_data(
            data['y_true'], data['y_pred'], data['y_std']
        )
        ax_sp.plot(
            sp['fractions'],
            sp['model_maes'],
            color=color,
            lw=1.5,
            label=name,
            alpha=0.85,
            **_curve_kwargs(i, len(sp['fractions'])),
        )
        if not oracle_plotted:
            ax_sp.plot(sp['fractions'], sp['oracle_maes'], '--', color=style.INK,
                       lw=1.2, label='Oracle')
            ax_sp.axhline(y=sp['random_mae'], color=style.MUTED, ls=':', lw=1,
                         alpha=0.7, label='Random')
            oracle_plotted = True
    ax_sp.set_xlabel('Fraction Removed')
    ax_sp.set_ylabel('MAE')
    ax_sp.set_title('B) Sparsification Curves', fontweight='bold')
    ax_sp.legend(fontsize=5, loc='lower left', framealpha=0.9)

    # C) Uncertainty vs error scatter (top 4 models by |Spearman|)
    ax_sc = fig.add_subplot(gs[1, 0])
    sorted_models = sorted(
        all_models_data.keys(),
        key=lambda n: abs(all_metrics.get(n, {}).get('spearman_rho', 0)),
        reverse=True
    )
    top_models = sorted_models[:4]
    for i, name in enumerate(top_models):
        data = all_models_data[name]
        color = data.get('color', OKABE_ITO[i % len(OKABE_ITO)])
        y_true = np.asarray(data['y_true'])
        y_pred = np.asarray(data['y_pred'])
        y_std = np.asarray(data['y_std'])
        errors = np.abs(y_true - y_pred)

        rng = np.random.default_rng(42 + i)
        n = len(errors)
        n_sample = min(n, 2000)
        idx = rng.choice(n, size=n_sample, replace=False)
        ax_sc.scatter(y_std[idx], errors[idx], alpha=0.15, s=4, color=color,
                      edgecolors='none', label=name)

        n_lowess = min(n, 10000)
        idx_l = rng.choice(n, size=n_lowess, replace=False) if n > n_lowess else np.arange(n)
        smoothed = lowess(errors[idx_l], y_std[idx_l], frac=0.3, return_sorted=True)
        ax_sc.plot(smoothed[:, 0], smoothed[:, 1], '-', color=color, lw=2, alpha=0.9)
    ax_sc.set_xlabel('Predicted \u03c3')
    ax_sc.set_ylabel('|Error|')
    ax_sc.set_title('C) Uncertainty vs Error (top 4)', fontweight='bold')
    ax_sc.legend(fontsize=5, framealpha=0.9)

    # D) Summary heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    display_metrics = ['ence', 'miscalibration_area', 'ause', 'spearman_rho']
    display_labels = ['ENCE', 'Miscal.', 'AUSE', '\u03c1']
    rows = {}
    for model_name, metrics in all_metrics.items():
        rows[model_name] = [metrics.get(m, np.nan) for m in display_metrics]
    if rows:
        df = pd.DataFrame(rows, index=display_labels).T
        score_df = _directional_scores(df, [True, True, True, False])
        sns.heatmap(
            score_df,
            ax=ax_heat,
            annot=df.to_numpy(),
            fmt='.3f',
            cmap=style.SEQUENTIAL,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            annot_kws={'size': 8},
            cbar_kws={
                'label': 'Relative performance (higher is better)',
                'shrink': 0.8,
            },
        )
        ax_heat._learnm8_heatmap = True
    ax_heat.set_title('D) UQ Metrics Summary', fontweight='bold')
    ax_heat.set_ylabel('')

    _apply_title_layout(
        fig,
        'How do models compare across calibration and uncertainty ordering dimensions?',
        subtitle=subtitle,
    )
    _save_and_close(fig, output_path)
    return fig


def plot_discovery_curves(named_seed_results, top_k_fraction=0.01,
                           output_path=None, figsize=(9, 5.5), subtitle=None):
    """Cumulative % of top-K% compounds found per cycle, with ±std bands.

    Args:
        named_seed_results: dict mapping display_name -> list of seed result dicts,
            each with 'cycle_metrics' as a list of per-cycle dicts containing
            'top_1_pct_discovery' and 'cumulative_labeled'.
        top_k_fraction: fraction defining 'top K%' (default 0.01 = top 1%).
        output_path: Path to save figure (optional).
        figsize: Figure size tuple.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    metric_key = 'top_1_pct_discovery' if top_k_fraction == 0.01 else 'top_10_pct_discovery'
    label_pct = f'{int(top_k_fraction * 100)}%'

    compounds_evaluated_per_cycle = {}

    for i, (name, seed_results) in enumerate(named_seed_results.items()):
        color = OKABE_ITO[i % len(OKABE_ITO)]

        all_cycle_vals = {}
        for result in seed_results:
            if result is None:
                continue
            metrics_list = result.get('cycle_metrics', [])
            for row in metrics_list:
                cycle = row.get('cycle', None)
                val = row.get(metric_key, None)
                if cycle is None or val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                all_cycle_vals.setdefault(cycle, []).append(val)

                cum_labeled = row.get('cumulative_labeled', None)
                pool_size = row.get('pool_size', None)
                if cum_labeled is not None and pool_size is not None:
                    try:
                        count = float(cum_labeled)
                        pct = count / float(pool_size) * 100
                        compounds_evaluated_per_cycle.setdefault(cycle, []).append(
                            (count, pct)
                        )
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass

        if not all_cycle_vals:
            continue

        cycles = sorted(all_cycle_vals.keys())
        means = np.array([np.mean(all_cycle_vals[c]) for c in cycles])
        stds = np.array([np.std(all_cycle_vals[c]) for c in cycles])
        x_values = np.array(
            [
                np.mean([value[0] for value in compounds_evaluated_per_cycle[c]])
                if c in compounds_evaluated_per_cycle
                else c
                for c in cycles
            ]
        )

        ax.plot(
            x_values,
            means,
            color=color,
            lw=1.8,
            markersize=5,
            label=name,
            alpha=0.9,
            **_curve_kwargs(i, len(cycles)),
        )
        ax.fill_between(x_values, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.set_xlabel(
        'Cumulative compounds evaluated, n (initial-pool percentage in ticks)'
    )
    ax.set_ylabel(f'Top-{label_pct} recovered from initial pool (%)')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    if compounds_evaluated_per_cycle:
        cycles = sorted(compounds_evaluated_per_cycle)
        stats = [compounds_evaluated_per_cycle[c] for c in cycles]
        counts = np.array([np.mean([value[0] for value in row]) for row in stats])
        percentages = np.array([np.mean([value[1] for value in row]) for row in stats])
        ax.set_xticks(counts)
        ax.set_xticklabels(
            [
                f'{int(count):,}\n({percentage:.1f}%)'
                for count, percentage in zip(counts, percentages, strict=True)
            ],
            rotation=45,
            ha='right',
            fontsize=7,
        )

    _apply_title_layout(
        fig,
        f'Does better UQ improve top-{label_pct} compound discovery under UCB acquisition?',
        subtitle=subtitle,
    )
    _save_and_close(fig, output_path)
    return fig


def plot_triple_comparison(models_data, title='', output_path=None, figsize=(15, 4),
                           subtitle=None):
    """Create a 1×3 comparison panel: [Calibration | Sparsification | Scatter].

    Designed for head-to-head comparison of two models side by side.

    Args:
        models_data: Dict of name → {'y_true', 'y_pred', 'y_std', 'miscal_area'}.
        title: Figure suptitle.
        output_path: Path to save the figure.
        figsize: Figure size in inches.
    """
    apply_style()
    model_names = list(models_data.keys())
    colors = OKABE_ITO[:len(model_names)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ── Panel 1: Calibration ──────────────────────────────────────────────
    ax = axes[0]
    for (name, d), color in zip(models_data.items(), colors, strict=True):
        try:
            levels, coverages = compute_calibration_curve(
                d['y_true'], d['y_pred'], d['y_std']
            )
            ma = d.get('miscal_area', np.nan)
            label = f'{name} (MA={ma:.3f})' if np.isfinite(ma) else name
            ax.plot(
                levels,
                coverages,
                color=color,
                lw=2,
                markersize=4,
                label=label,
                **_curve_kwargs(len(ax.lines), len(levels)),
            )
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], '--', color=style.INK, lw=1, alpha=0.55)
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('Calibration', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')

    # ── Panel 2: Sparsification ───────────────────────────────────────────
    ax = axes[1]
    ause_labels = []
    for idx, ((name, d), color) in enumerate(
        zip(models_data.items(), colors, strict=True)
    ):
        try:
            sd = compute_sparsification_data(d['y_true'], d['y_pred'], d['y_std'])
            ause_val = compute_ause(d['y_true'], d['y_pred'], d['y_std'])
            ax.plot(
                sd['fractions'],
                sd['model_maes'],
                color=color,
                lw=2,
                label=name,
                **_curve_kwargs(idx, len(sd['fractions'])),
            )
            ax.plot(sd['fractions'], sd['oracle_maes'], '--', color=color,
                    lw=1, alpha=0.5)
            ax.fill_between(sd['fractions'], sd['oracle_maes'], sd['model_maes'],
                            alpha=0.12, color=color)
            ause_labels.append((name, ause_val, color))
        except Exception:
            pass
    for i, (name, ause_val, color) in enumerate(ause_labels):
        y_pos = 0.95 - i * 0.12
        ax.text(0.95, y_pos, f'{name}: AUSE={ause_val:.4f}',
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=_lighten_color(color),
                          edgecolor=color, alpha=0.8))
    ax.set_xlabel('Fraction Removed')
    ax.set_ylabel('MAE')
    ax.set_title('Sparsification (solid=model, dashed=oracle)', fontweight='bold')
    ax.legend(fontsize=7)

    # ── Panel 3: Uncertainty vs Error scatter ─────────────────────────────
    ax = axes[2]
    for (name, d), color in zip(models_data.items(), colors, strict=True):
        try:
            errors = np.abs(np.asarray(d['y_true']) - np.asarray(d['y_pred']))
            stds = np.asarray(d['y_std'])
            mask = np.isfinite(errors) & np.isfinite(stds) & (stds > 0)
            if mask.sum() < 10:
                continue
            e, s = errors[mask], stds[mask]
            rho, _ = compute_spearman(
                np.asarray(d['y_true'])[mask],
                np.asarray(d['y_pred'])[mask],
                s,
            )
            ax.scatter(s, e, alpha=0.08, s=3, color=color,
                       label=f'{name} (ρ={rho:.2f})')
            idx = np.argsort(s)
            try:
                trend = lowess(e[idx], s[idx], frac=0.3, return_sorted=False)
                ax.plot(s[idx], trend, '-', color=color, lw=2)
            except Exception:
                pass
        except Exception:
            pass
    ax.set_xlabel('Predicted Std')
    ax.set_ylabel('|Error|')
    ax.set_title('Uncertainty vs Error', fontweight='bold')
    ax.legend(fontsize=7)

    _apply_title_layout(fig, title, subtitle=subtitle)
    _save_and_close(fig, output_path)
    return fig


def plot_ensemble_baselines_calibration(models_data, title='', output_path=None,
                                        subtitle=None):
    """Create a 1×N panel showing calibration curves for N ensemble models.

    Args:
        models_data: Dict of name → {'y_true', 'y_pred', 'y_std', 'miscal_area'}.
        title: Figure suptitle.
        output_path: Path to save the figure.
        subtitle: Optional subtitle.
    """
    apply_style()
    model_names = list(models_data.keys())
    n = len(model_names)
    colors = OKABE_ITO[:n]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = np.array([axes])

    for col, (name, d, color) in enumerate(
        zip(model_names, models_data.values(), colors, strict=True)
    ):
        ax = axes[col]
        try:
            levels, coverages = compute_calibration_curve(
                d['y_true'], d['y_pred'], d['y_std']
            )
            ma = d.get('miscal_area', np.nan)
            ax.plot(
                levels,
                coverages,
                color=color,
                lw=2,
                markersize=4,
                **_curve_kwargs(col, len(levels)),
            )
            ax.plot([0, 1], [0, 1], '--', color=style.INK, lw=1, alpha=0.55)
            suffix = f'\nMA={ma:.3f}' if np.isfinite(ma) else ''
            ax.set_title(f'{name}{suffix}', fontweight='bold', fontsize=9)
        except Exception:
            ax.set_title(name, fontweight='bold', fontsize=9)
        ax.set_xlabel('Expected Coverage')
        if col == 0:
            ax.set_ylabel('Observed Coverage')

    _apply_title_layout(fig, title, subtitle=subtitle)
    _save_and_close(fig, output_path)
    return fig


def plot_ensemble_baselines_sparsification(models_data, title='',
                                           output_path=None, subtitle=None):
    """Create a 1×N panel showing sparsification curves for N ensemble models.

    Args:
        models_data: Dict of name → {'y_true', 'y_pred', 'y_std', 'miscal_area'}.
        title: Figure suptitle.
        output_path: Path to save the figure.
        subtitle: Optional subtitle.
    """
    apply_style()
    model_names = list(models_data.keys())
    n = len(model_names)
    colors = OKABE_ITO[:n]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = np.array([axes])

    for col, (name, d, color) in enumerate(
        zip(model_names, models_data.values(), colors, strict=True)
    ):
        ax = axes[col]
        try:
            sd = compute_sparsification_data(d['y_true'], d['y_pred'], d['y_std'])
            ause = compute_ause(d['y_true'], d['y_pred'], d['y_std'])
            ax.plot(
                sd['fractions'],
                sd['model_maes'],
                color=color,
                lw=2,
                label='Model',
                **_curve_kwargs(col, len(sd['fractions'])),
            )
            ax.plot(sd['fractions'], sd['oracle_maes'], '--', color=style.INK,
                    lw=1.2, label='Oracle')
            ax.axhline(y=sd['random_mae'], color=style.MUTED, ls=':', lw=1,
                       alpha=0.7, label='Random')
            ax.fill_between(sd['fractions'], sd['oracle_maes'], sd['model_maes'],
                            alpha=0.15, color=color)
            ax.text(0.95, 0.95, f'AUSE={ause:.4f}', transform=ax.transAxes,
                    fontsize=9, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=_lighten_color(color),
                              edgecolor=color, alpha=0.8))
            if col == 0:
                ax.legend(fontsize=7, loc='lower left')
        except Exception:
            pass
        ax.set_xlabel('Fraction Removed')
        if col == 0:
            ax.set_ylabel('MAE')
        ax.set_title(name, fontweight='bold', fontsize=9)

    _apply_title_layout(fig, title, subtitle=subtitle)
    _save_and_close(fig, output_path)
    return fig

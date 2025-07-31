"""Plotting utilities for A/B testing visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.patches as mpatches


# Color palettes for consistent styling
LEARNER_COLORS = {
    'rf': '#1f77b4',
    'gp': '#ff7f0e', 
    'xgb': '#2ca02c',
    'mlp': '#d62728',
    'mc_dropout': '#9467bd',
    'ensemble': '#8c564b',
    'rf_ensemble': '#e377c2',
    'lr_ensemble': '#7f7f7f',
    'xgb_ensemble': '#bcbd22',
    'dt_ensemble': '#17becf',
    'mixed_ensemble': '#aec7e8'
}

STRATEGY_COLORS = {
    'greedy': '#2E86AB',
    'random': '#F18F01',
    'ucb': '#A23B72',
    'ei': '#F24236',
    'pi': '#8E44AD',
    'thompson': '#E67E22',
    'entropy': '#16A085',
    'simulated_annealing': '#8B4513',
    'pca_dbscan': '#FF6B6B',
    'umap_dbscan': '#4ECDC4',
    'tsne_dbscan': '#45B7D1',
    'bitbirch': '#96CEB4'
}

# Line styles for batch sizes
BATCH_SIZE_STYLES = {
    0.001: '-',
    0.005: '--',
    0.01: '-.',
    0.05: ':',
    0.1: (0, (3, 1, 1, 1))
}


def setup_publication_style():
    """Set up publication-quality matplotlib style."""
    plt.style.use('default')
    
    # Font settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        
        # Line and marker settings
        'lines.linewidth': 2,
        'lines.markersize': 6,
        
        # Axes settings  
        'axes.linewidth': 1.2,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Grid and figure settings
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })


def get_color(item: str, color_type: str = 'learner') -> str:
    """Get consistent color for learner or strategy.
    
    Args:
        item: Learner or strategy name
        color_type: 'learner' or 'strategy'
        
    Returns:
        Hex color code
    """
    if color_type == 'learner':
        return LEARNER_COLORS.get(item, '#333333')
    elif color_type == 'strategy':
        return STRATEGY_COLORS.get(item, '#333333')
    else:
        return '#333333'


def get_line_style(batch_size: float) -> str:
    """Get line style for batch size.
    
    Args:
        batch_size: Batch size fraction
        
    Returns:
        Matplotlib line style
    """
    return BATCH_SIZE_STYLES.get(batch_size, '-')


def plot_trajectory_with_ci(ax, x_data, y_data, y_std, label: str, 
                           color: str, line_style: str = '-', 
                           alpha: float = 0.8, marker: str = 'o'):
    """Plot trajectory line with error bars showing standard deviation.
    
    Args:
        ax: Matplotlib axes
        x_data: X-axis data
        y_data: Y-axis data (mean)
        y_std: Standard deviation for error bars
        label: Line label
        color: Line color
        line_style: Line style
        alpha: Line transparency
        marker: Marker style
    """
    # Main trajectory line with error bars
    ax.errorbar(x_data, y_data, yerr=y_std, label=label, color=color, 
               linestyle=line_style, alpha=alpha, marker=marker, 
               markersize=4, markevery=max(1, len(x_data)//10),
               capsize=3, capthick=1, elinewidth=1)
    
    # Optional: Also add confidence interval fill for visual emphasis
    if y_std is not None and not np.isnan(y_std).all():
        y_lower = y_data - y_std
        y_upper = y_data + y_std
        ax.fill_between(x_data, y_lower, y_upper, 
                       color=color, alpha=0.1)


def add_significance_markers(ax, x_pos: float, y_pos: float, 
                           p_value: float, color: str = 'black'):
    """Add statistical significance markers to plot.
    
    Args:
        ax: Matplotlib axes
        x_pos: X position for marker
        y_pos: Y position for marker
        p_value: P-value for significance test
        color: Marker color
    """
    if p_value < 0.001:
        marker = '***'
    elif p_value < 0.01:
        marker = '**'
    elif p_value < 0.05:
        marker = '*'
    else:
        return  # No significance
    
    ax.text(x_pos, y_pos, marker, fontsize=16, color=color, 
           ha='center', va='bottom', weight='bold')


def create_subplot_grid(n_plots: int, max_cols: int = 3) -> Tuple[plt.Figure, np.ndarray]:
    """Create optimal subplot grid.
    
    Args:
        n_plots: Number of subplots needed
        max_cols: Maximum columns per row
        
    Returns:
        Figure and axes array
    """
    cols = min(n_plots, max_cols)
    rows = (n_plots + cols - 1) // cols
    
    fig_width = 5 * cols
    fig_height = 4 * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Handle single subplot case
    if n_plots == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = np.array(axes) if hasattr(axes, '__len__') else np.array([axes])
    else:
        axes = axes.flatten()
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    return fig, axes


def format_axis_labels(ax, x_label: str, y_label: str, title: str):
    """Format axis labels and title consistently.
    
    Args:
        ax: Matplotlib axes
        x_label: X-axis label
        y_label: Y-axis label  
        title: Plot title
    """
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def create_legend_outside(ax, ncol: int = 1, loc: str = 'upper left'):
    """Create legend outside plot area.
    
    Args:
        ax: Matplotlib axes
        ncol: Number of legend columns
        loc: Legend location
    """
    ax.legend(bbox_to_anchor=(1.05, 1), loc=loc, ncol=ncol,
             frameon=True, fancybox=False, shadow=False, 
             framealpha=0.9, edgecolor='gray')


def save_plot(fig, output_path: str, formats: List[str] = ['png']):
    """Save plot in multiple formats.
    
    Args:
        fig: Matplotlib figure
        output_path: Output file path (without extension)
        formats: List of formats to save ('png', 'pdf', 'svg')
    """
    for fmt in formats:
        fig.savefig(f"{output_path}.{fmt}", dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_heatmap(data, x_labels: List[str], y_labels: List[str], 
                  title: str, cmap: str = 'viridis', 
                  annot: bool = True, fmt: str = '.3f') -> plt.Figure:
    """Create publication-quality heatmap.
    
    Args:
        data: 2D array for heatmap
        x_labels: X-axis labels
        y_labels: Y-axis labels
        title: Plot title
        cmap: Colormap name
        annot: Whether to annotate cells
        fmt: Number format for annotations
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(len(x_labels) * 0.8 + 2, len(y_labels) * 0.6 + 2))
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Performance Score', rotation=270, labelpad=20)
    
    # Add annotations if requested
    if annot:
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, format(data[i, j], fmt), 
                           ha='center', va='center', 
                           color='white' if data[i, j] < np.nanmean(data) else 'black')
    
    ax.set_title(title)
    return fig


def create_pruning_plots(embeddings: np.ndarray, 
                        predictions: Optional[np.ndarray] = None,
                        uncertainties: Optional[np.ndarray] = None,
                        pruned_indices: Optional[List[int]] = None,
                        retained_indices: Optional[List[int]] = None,
                        method_name: str = "Pruning Analysis",
                        output_dir: str = None,
                        compound_ids: Optional[List[str]] = None,
                        save_formats: List[str] = ['png']) -> plt.Figure:
    """Create pruning visualization plots showing compound selection patterns.
    
    This function creates scatter plots colored by predictions and/or uncertainty,
    highlighting compounds selected for pruning vs retained compounds. Follows
    the established visualization patterns in LearnM8.
    
    Args:
        embeddings: 2D embedding coordinates (n_compounds, 2)
        predictions: Model predictions for compounds (optional)
        uncertainties: Model uncertainties for compounds (optional)
        pruned_indices: Indices of compounds removed by pruning
        retained_indices: Indices of compounds retained after pruning
        method_name: Name of the pruning method for plot titles
        output_dir: Directory to save plots (if None, plots not saved)
        compound_ids: List of compound identifiers (optional)
        save_formats: List of formats to save plots in
        
    Returns:
        Matplotlib figure object
    """
    # Set up publication style
    setup_publication_style()
    
    # Determine number of plots based on available data
    n_plots = 1  # Always have pruning status plot
    if predictions is not None:
        n_plots += 1
    if uncertainties is not None:
        n_plots += 1
    
    # Create subplot grid
    fig, axes = create_subplot_grid(n_plots, max_cols=3)
    plot_idx = 0
    
    # Define colors for pruning status
    pruning_colors = {
        'retained': '#2ca02c',    # Green
        'pruned': '#d62728',     # Red
        'unspecified': '#7f7f7f' # Gray
    }
    
    # Plot 1: Pruning Status
    ax = axes[plot_idx]
    
    # Create status labels for all compounds
    n_compounds = len(embeddings)
    status_colors = np.full(n_compounds, pruning_colors['unspecified'])
    status_labels = np.full(n_compounds, 'Unspecified', dtype=object)
    
    if retained_indices is not None:
        status_colors[retained_indices] = pruning_colors['retained']
        status_labels[retained_indices] = 'Retained'
    
    if pruned_indices is not None:
        status_colors[pruned_indices] = pruning_colors['pruned']  
        status_labels[pruned_indices] = 'Pruned'
    
    # Create scatter plot
    unique_labels = np.unique(status_labels)
    for label in unique_labels:
        mask = status_labels == label
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                  c=pruning_colors[label.lower()], 
                  alpha=0.7, s=30, label=label, edgecolors='white', linewidth=0.5)
    
    ax.set_title(f'Pruning Status - {method_name}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics to title
    n_pruned = len(pruned_indices) if pruned_indices is not None else 0
    n_retained = len(retained_indices) if retained_indices is not None else 0
    ax.text(0.02, 0.98, f'Pruned: {n_pruned}\nRetained: {n_retained}', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plot_idx += 1
    
    # Plot 2: Predictions (if available)
    if predictions is not None:
        ax = axes[plot_idx]
        
        # Color by predictions, shape by pruning status
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                           c=predictions, cmap='viridis', alpha=0.7, s=30,
                           edgecolors='white', linewidth=0.5)
        
        # Overlay pruned compounds with different markers
        if pruned_indices is not None:
            ax.scatter(embeddings[pruned_indices, 0], embeddings[pruned_indices, 1], 
                      c=predictions[pruned_indices], cmap='viridis', 
                      marker='x', s=60, alpha=0.9, linewidth=2)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Prediction Value')
        
        ax.set_title(f'Predictions - {method_name}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
        
        # Add legend for markers
        retained_patch = mpatches.Patch(color='gray', label='Retained (circles)')
        pruned_patch = mpatches.Patch(color='gray', label='Pruned (×)')
        ax.legend(handles=[retained_patch, pruned_patch], loc='upper right')
        
        plot_idx += 1
    
    # Plot 3: Uncertainties (if available)
    if uncertainties is not None:
        ax = axes[plot_idx]
        
        # Color by uncertainties, shape by pruning status
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                           c=uncertainties, cmap='plasma', alpha=0.7, s=30,
                           edgecolors='white', linewidth=0.5)
        
        # Overlay pruned compounds with different markers
        if pruned_indices is not None:
            ax.scatter(embeddings[pruned_indices, 0], embeddings[pruned_indices, 1], 
                      c=uncertainties[pruned_indices], cmap='plasma', 
                      marker='x', s=60, alpha=0.9, linewidth=2)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Uncertainty')
        
        ax.set_title(f'Uncertainties - {method_name}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
        
        # Add legend for markers
        retained_patch = mpatches.Patch(color='gray', label='Retained (circles)')
        pruned_patch = mpatches.Patch(color='gray', label='Pruned (×)')
        ax.legend(handles=[retained_patch, pruned_patch], loc='upper right')
    
    # Overall title
    plt.suptitle(f'{method_name} - Compound Analysis ({n_compounds} compounds)', 
                fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save plots if output directory specified
    if output_dir is not None:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_filename = output_path / f"{method_name.lower().replace(' ', '_')}_pruning_analysis"
        save_plot(fig, str(plot_filename), formats=save_formats)
    
    return fig
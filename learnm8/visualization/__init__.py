"""
LearnM8 Visualization Package

Simple, focused visualization tools for active learning A/B testing analysis.
Provides clear, publication-quality plots that isolate single variables to reveal trends.

Example usage:
    >>> from learnm8.visualization import generate_all_ab_plots
    >>> generate_all_ab_plots('parameter_sweep_cycle_by_cycle.csv', 'output_plots/')
"""

from .ab_testing_plots import (
    plot_learner_tournament,
    plot_acquisition_tournament, 
    plot_batch_size_impact,
    plot_custom_cycle_comparison,
    plot_uncertainty_model_tournament,
    plot_initial_strategy_impact,
    plot_performance_matrix,
    plot_diversity_performance_tradeoff,
    generate_all_ab_plots
)

from .data_loader import (
    load_cycle_data,
    filter_data,
    get_available_parameters,
    aggregate_by_cycle,
    get_final_performance
)

from .plot_utils import (
    setup_publication_style,
    get_color,
    plot_trajectory_with_ci,
    create_heatmap,
    create_pruning_plots
)

__version__ = "0.2.0"
__author__ = "LearnM8 Development Team"

__all__ = [
    # Main plotting functions
    "plot_learner_tournament",
    "plot_acquisition_tournament", 
    "plot_batch_size_impact",
    "plot_custom_cycle_comparison",
    "plot_uncertainty_model_tournament",
    "plot_initial_strategy_impact",
    "plot_performance_matrix",
    "plot_diversity_performance_tradeoff",
    "generate_all_ab_plots",
    
    # Data utilities
    "load_cycle_data",
    "filter_data", 
    "get_available_parameters",
    "aggregate_by_cycle",
    "get_final_performance",
    
    # Plot utilities
    "setup_publication_style",
    "get_color",
    "plot_trajectory_with_ci",
    "create_heatmap",
    "create_pruning_plots"
]

# Default configuration for easy customization
DEFAULT_CONFIG = {
    'publication_style': True,
    'dpi': 300,
    'formats': ['png'],
    'colorblind_friendly': True,
    'show_confidence_intervals': True,
    'show_significance': True
}
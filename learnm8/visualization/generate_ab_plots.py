#!/usr/bin/env python3
"""
Example script for generating A/B testing visualizations from parameter sweep results.

This script demonstrates how to use the LearnM8 visualization system to create
focused A/B testing plots that isolate single variables and reveal trends.

Usage:
    python generate_ab_plots.py <data_file> <output_dir> [options]

Example:
    python generate_ab_plots.py parameter_sweep_cycle_by_cycle.csv plots/
    python generate_ab_plots.py results/sweep_results.csv analysis/ --plot learner_tournament
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Generate A/B testing visualizations for LearnM8 parameter sweep results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('data_file', type=Path,
                       help='Path to parameter_sweep_cycle_by_cycle.csv file')
    parser.add_argument('output_dir', type=Path,
                       help='Directory to save generated plots')
    
    # Optional arguments
    parser.add_argument('--plot', type=str, 
                       choices=['all', 'learner_tournament', 'acquisition_tournament',
                               'batch_size_impact', 'custom_cycle_comparison', 
                               'uncertainty_tournament', 'initial_strategy_impact',
                               'performance_matrix', 'diversity_tradeoff'],
                       default='all',
                       help='Specific plot to generate (default: all)')
    
    # Parameter overrides for specific plots
    parser.add_argument('--learner', type=str, default='rf_ensemble',
                       help='Learner type for single-learner plots')
    parser.add_argument('--acquisition', type=str, default='greedy',
                       help='Acquisition strategy for single-strategy plots')
    parser.add_argument('--batch-size', type=float, default=0.01,
                       help='Batch size for single-batch-size plots')
    parser.add_argument('--initial-strategy', type=str, default='random',
                       help='Initial strategy for plots')
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.data_file.exists():
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import visualization functions
    try:
        from learnm8.visualization import (
            generate_all_ab_plots,
            plot_learner_tournament,
            plot_acquisition_tournament, 
            plot_batch_size_impact,
            plot_custom_cycle_comparison,
            plot_uncertainty_model_tournament,
            plot_initial_strategy_impact,
            plot_performance_matrix,  
            plot_diversity_performance_tradeoff
        )
    except ImportError as e:
        print(f"Error importing LearnM8 visualization modules: {e}")
        print("Make sure LearnM8 is properly installed with: pip install -e .")
        sys.exit(1)
    
    print(f"Loading data from: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Generate requested plots
    if args.plot == 'all':
        print("Generating all A/B testing plots...")
        generate_all_ab_plots(args.data_file, args.output_dir)
        
    elif args.plot == 'learner_tournament':
        print("Generating learner tournament plot...")
        plot_learner_tournament(args.data_file, args.output_dir,
                               acquisition=args.acquisition,
                               batch_size=args.batch_size,
                               initial_strategy=args.initial_strategy)
        
    elif args.plot == 'acquisition_tournament':
        print("Generating acquisition tournament plot...")
        plot_acquisition_tournament(args.data_file, args.output_dir,
                                   batch_size=args.batch_size,
                                   initial_strategy=args.initial_strategy)
        
    elif args.plot == 'batch_size_impact':
        print("Generating batch size impact plot...")
        plot_batch_size_impact(args.data_file, args.output_dir,
                              learner=args.learner,
                              acquisition=args.acquisition)
        
    elif args.plot == 'custom_cycle_comparison':
        print("Generating custom cycle comparison plot...")
        plot_custom_cycle_comparison(args.data_file, args.output_dir,
                                   learner=args.learner)
        
    elif args.plot == 'uncertainty_tournament':
        print("Generating uncertainty model tournament plot...")
        plot_uncertainty_model_tournament(args.data_file, args.output_dir,
                                         acquisition='ucb',  # Force UCB for uncertainty
                                         batch_size=args.batch_size)
        
    elif args.plot == 'initial_strategy_impact':
        print("Generating initial strategy impact plot...")
        plot_initial_strategy_impact(args.data_file, args.output_dir,
                                    learner=args.learner,
                                    acquisition=args.acquisition)
        
    elif args.plot == 'performance_matrix':
        print("Generating performance matrix plot...")
        plot_performance_matrix(args.data_file, args.output_dir)
        
    elif args.plot == 'diversity_tradeoff':
        print("Generating diversity-performance trade-off plot...")
        plot_diversity_performance_tradeoff(args.data_file, args.output_dir,
                                           learner=args.learner)
    
    print("Visualization generation complete!")
    print(f"Check output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
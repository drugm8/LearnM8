#!/usr/bin/env python3
"""Command-line interface for LearnM8 active learning experiments."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from experiments.runner import create_experiment_config, run_experiment


def detect_score_direction(ground_truth_path: str, target_column: str) -> str:
    """Auto-detect scoring direction based on column name patterns."""
    import pandas as pd
    
    column_lower = target_column.lower()
    
    # Patterns for lower-is-better
    lower_patterns = ['dock', 'binding_energy', 'energy', 'rmsd', 'error', 'loss', 'distance']
    for pattern in lower_patterns:
        if pattern in column_lower:
            return 'lower'
    
    # Patterns for higher-is-better
    higher_patterns = ['activity', 'affinity', 'score', 'rank', 'similarity', 'accuracy']
    for pattern in higher_patterns:
        if pattern in column_lower:
            return 'higher'
    
    # Check data distribution if no pattern matches
    try:
        df = pd.read_csv(ground_truth_path)
        if target_column in df.columns:
            values = df[target_column].dropna()
            if len(values) > 0 and (values < 0).mean() > 0.7:
                return 'lower'
    except:
        pass
    
    # Default to higher-is-better
    return 'higher'


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LearnM8: Active Learning for Molecular Screening",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'compound_pool',
        help='Path to compound pool CSV (must have ID and SMILES columns)'
    )
    
    parser.add_argument(
        'ground_truth',
        help='Path to ground truth CSV (must have ID and target column)'
    )
    
    parser.add_argument(
        'target_column',
        help='Target column name to learn'
    )
    
    # Optional arguments
    parser.add_argument(
        '-c', '--cycles',
        type=int,
        default=10,
        help='Number of active learning cycles'
    )
    
    parser.add_argument(
        '-b', '--batch-size-fraction',
        type=float,
        default=0.1,
        help='Fraction of compounds to select per cycle'
    )
    
    parser.add_argument(
        '-s', '--strategy',
        choices=['greedy', 'random', 'diverse'],
        default='greedy',
        help='Selection strategy'
    )
    
    parser.add_argument(
        '-i', '--initial-strategy',
        choices=['greedy', 'random', 'diverse'],
        default='random',
        help='Initial selection strategy'
    )
    
    parser.add_argument(
        '-d', '--direction',
        choices=['higher', 'lower', 'auto'],
        default='auto',
        help='Score direction (higher/lower is better)'
    )
    
    parser.add_argument(
        '-l', '--learner',
        choices=['random_forest', 'rf'],
        default='random_forest',
        help='Machine learning model'
    )
    
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=100,
        help='K value for top-k overlap metric'
    )
    
    parser.add_argument(
        '-e', '--enrichment-percentile',
        type=float,
        default=1.0,
        help='Percentile for enrichment factor calculation'
    )
    
    parser.add_argument(
        '-r', '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    compound_pool_path = Path(args.compound_pool)
    ground_truth_path = Path(args.ground_truth)
    
    if not compound_pool_path.exists():
        print(f"Error: Compound pool file not found: {compound_pool_path}", file=sys.stderr)
        sys.exit(1)
    
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file not found: {ground_truth_path}", file=sys.stderr)
        sys.exit(1)
    
    # Auto-detect score direction if needed
    score_direction = args.direction
    if score_direction == 'auto':
        score_direction = detect_score_direction(str(ground_truth_path), args.target_column)
        print(f"Auto-detected scoring direction: {score_direction.upper()} is better")
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"learnm8_results_{timestamp}")
    
    print(f"Results will be saved to: {output_dir}")
    
    # Create experiment configuration
    config = create_experiment_config(
        compound_pool_path=str(compound_pool_path),
        ground_truth_path=str(ground_truth_path),
        target_column=args.target_column,
        n_cycles=args.cycles,
        batch_size_fraction=args.batch_size_fraction,
        selection_strategy=args.strategy,
        initial_strategy=args.initial_strategy,
        score_direction=score_direction,
        learner_type=args.learner,
        random_state=args.random_state,
        top_k=args.top_k,
        enrichment_percentile=args.enrichment_percentile
    )
    
    try:
        # Run experiment
        final_predictions, monitoring_results = run_experiment(config, output_dir)
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
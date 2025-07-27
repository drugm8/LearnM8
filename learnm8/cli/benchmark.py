"""CLI for benchmark mode - single CSV with all data."""

import argparse
import sys
from pathlib import Path
from rich.console import Console

from learnm8.cli.common import add_common_arguments, detect_score_direction, validate_file_exists, setup_output_directory, validate_repeats
from learnm8.experiments.runner import create_experiment_config, run_experiment
from learnm8.utils.logging import setup_logging, log_success, log_error_to_stderr, log_file_operation


def create_benchmark_parser() -> argparse.ArgumentParser:
    """Create argument parser for benchmark mode."""
    parser = argparse.ArgumentParser(
        description="LearnM8 Benchmark: Active Learning evaluation on historical data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'data',
        help='Path to CSV file with columns: ID, SMILES, target_column [, Activity]'
    )
    
    parser.add_argument(
        'target_column',
        help='Target column name to learn'
    )
    
    # Add common arguments
    add_common_arguments(parser)
    
    return parser


def run_benchmark(args) -> None:
    """Run benchmark mode experiment."""
    # Setup logging
    console = Console()
    logger = setup_logging(console=console)
    
    # Validate input file
    data_path = validate_file_exists(args.data, "Data file")
    
    # Validate repeats
    validate_repeats(args.repeats)
    
    # Auto-detect score direction if needed
    score_direction = args.direction
    if score_direction == 'auto':
        score_direction = detect_score_direction(str(data_path), args.target_column)
        logger.info(f"[yellow]Auto-detected scoring direction:[/yellow] [bold]{score_direction.upper()}[/bold] is better")
    
    # Setup output directory
    output_dir = setup_output_directory(args.output)
    log_file_operation(logger, "Results directory", str(output_dir))
    
    # Create experiment configuration
    # In benchmark mode, both compound_pool and ground_truth point to the same file
    config = create_experiment_config(
        compound_pool_path=str(data_path),
        ground_truth_path=str(data_path),
        target_column=args.target_column,
        n_cycles=args.cycles,
        batch_size_fraction=args.batch_size_fraction,
        selection_strategy=args.strategy,
        initial_strategy=args.initial_strategy,
        score_direction=score_direction,
        learner_type=args.learner,
        random_state=args.random_state,
        repeats=args.repeats,
        featurizer=args.featurizer,
    )
    
    try:
        # Run experiment
        _final_predictions, _monitoring_results = run_experiment(config, output_dir)
        
        log_success(logger, "Benchmark completed successfully!")
        log_file_operation(logger, "Results saved to", str(output_dir))
        
    except Exception as e:
        log_error_to_stderr(str(e))
        sys.exit(1)


def main():
    """Main entry point for benchmark mode."""
    parser = create_benchmark_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
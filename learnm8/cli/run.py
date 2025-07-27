"""CLI for run mode - real-world active learning with Python oracle."""

import argparse
import sys
from pathlib import Path
from rich.console import Console

from learnm8.cli.common import add_common_arguments, detect_score_direction, validate_file_exists, setup_output_directory, validate_repeats
from learnm8.experiments.runner import create_experiment_config, run_experiment
from learnm8.utils.logging import setup_logging, log_success, log_error_to_stderr, log_file_operation, log_warning


def create_run_parser() -> argparse.ArgumentParser:
    """Create argument parser for run mode."""
    parser = argparse.ArgumentParser(
        description="LearnM8 Run: Real-world active learning with Python oracle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'compounds',
        help='Path to compound pool CSV with columns: ID, SMILES'
    )
    
    parser.add_argument(
        'oracle',
        help='Path to Python file containing oracle function'
    )
    
    parser.add_argument(
        'target_column',
        help='Target column name that oracle function returns'
    )
    
    # Optional oracle configuration
    parser.add_argument(
        '--oracle-function',
        help='Name of oracle function (auto-detected if not specified)'
    )
    
    # Add common arguments
    add_common_arguments(parser)
    
    return parser


def run_experiment_mode(args) -> None:
    """Run experiment mode with Python oracle."""
    # Setup logging
    console = Console()
    logger = setup_logging(console=console)
    
    # Validate input files
    compounds_path = validate_file_exists(args.compounds, "Compound pool file")
    oracle_path = validate_file_exists(args.oracle, "Oracle file")
    
    # Validate repeats
    validate_repeats(args.repeats)
    
    # Validate that oracle is a Python file
    if oracle_path.suffix != '.py':
        log_warning(logger, f"Oracle file should be a Python file (.py), got: {oracle_path.suffix}")
    
    # Setup output directory
    output_dir = setup_output_directory(args.output)
    log_file_operation(logger, "Results directory", str(output_dir))
    
    # For run mode, we need to create a temporary ground truth file or modify the runner
    # For now, let's use the oracle file path as a placeholder and handle it in the runner
    config = create_experiment_config(
        compound_pool_path=str(compounds_path),
        ground_truth_path=str(oracle_path),  # This will be handled specially in runner
        target_column=args.target_column,
        n_cycles=args.cycles,
        batch_size_fraction=args.batch_size_fraction,
        selection_strategy=args.strategy,
        initial_strategy=args.initial_strategy,
        score_direction=args.direction,  # Don't auto-detect for run mode
        learner_type=args.learner,
        random_state=args.random_state,
        repeats=args.repeats,
        featurizer=args.featurizer,
        # Add oracle-specific configuration
        oracle_type='python',
        oracle_function_name=args.oracle_function
    )
    
    try:
        # Run experiment
        _final_predictions, _monitoring_results = run_experiment(config, output_dir)
        
        log_success(logger, "Experiment completed successfully!")
        log_file_operation(logger, "Results saved to", str(output_dir))
        
    except Exception as e:
        log_error_to_stderr(str(e))
        sys.exit(1)


def main():
    """Main entry point for run mode."""
    parser = create_run_parser()
    args = parser.parse_args()
    run_experiment_mode(args)


if __name__ == "__main__":
    main()
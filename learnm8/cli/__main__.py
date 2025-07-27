"""Main CLI entry point for LearnM8 with subcommands."""

import argparse
import sys
from learnm8.cli.benchmark import run_benchmark, create_benchmark_parser
from learnm8.cli.run import run_experiment_mode, create_run_parser


def main():
    """Main entry point with subcommand support."""
    parser = argparse.ArgumentParser(
        description="LearnM8: Active Learning for Molecular Screening",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )
    
    # Benchmark mode subcommand
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run benchmark mode with single CSV file',
        parents=[create_benchmark_parser()],
        add_help=False
    )
    
    # Run mode subcommand  
    run_parser = subparsers.add_parser(
        'run',
        help='Run active learning with Python oracle',
        parents=[create_run_parser()],
        add_help=False
    )
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Execute appropriate subcommand
    if args.command == 'benchmark':
        run_benchmark(args)
    elif args.command == 'run':
        run_experiment_mode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
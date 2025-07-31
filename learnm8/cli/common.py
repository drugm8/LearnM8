"""Common utilities and argument parsing for LearnM8 CLI."""

import argparse
from pathlib import Path


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between run and benchmark modes."""
    
    # Active learning parameters
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
        choices=['greedy', 'random'],
        default='greedy',
        help='Selection strategy'
    )
    
    parser.add_argument(
        '-i', '--initial-strategy',
        choices=['greedy', 'random'],
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
        choices=['random_forest', 'rf', 'advanced_random_forest', 'arf', 'mlp', 'pytorch_mlp', 'mlp_pytorch', 'linear_regression', 'lr', 'decision_tree', 'dt', 'gradient_boosting', 'gb', 'gaussian_process', 'gp', 'gaussian_pytorch', 'gp_pytorch'],
        default='random_forest',
        help='Machine learning model (rf=random_forest, arf=advanced_random_forest, mlp=multi_layer_perceptron, pytorch_mlp=pytorch_mlp, lr=linear_regression, dt=decision_tree, gb=gradient_boosting, gp=gaussian_process, gp_pytorch=gaussian_pytorch)'
    )
    
    parser.add_argument(
        '--featurizer',
        choices=['morgan', 'maccs', 'ecfp6', 'descriptors'],
        default='morgan',
        help='Type of molecular featurizer (morgan=Morgan ECFP4, maccs=MACCS keys, ecfp6=ECFP6, descriptors=Mordred descriptors)'
    )
    
    # Monitoring parameters
    # (Top-K overlap now calculated automatically for K = 100, 1000, 0.1%, 1%, 10%)
    
    
    # General parameters
    parser.add_argument(
        '-r', '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '-n', '--repeats',
        type=int,
        default=1,
        help='Number of experiment repeats with different seeds'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for results'
    )


def detect_score_direction(data_path: str, target_column: str) -> str:
    """Auto-detect scoring direction based on column name patterns and data."""
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
        df = pd.read_csv(data_path)
        if target_column in df.columns:
            values = df[target_column].dropna()
            if len(values) > 0 and (values < 0).mean() > 0.7:
                return 'lower'
    except:
        pass
    
    # Default to higher-is-better
    return 'higher'


def validate_file_exists(file_path: str, file_description: str) -> Path:
    """Validate that a file exists and return Path object."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_description} not found: {file_path}")
    return path


def setup_output_directory(output_arg: str = None) -> Path:
    """Setup output directory with timestamp if not specified."""
    if output_arg:
        output_dir = Path(output_arg)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"learnm8_results_{timestamp}")
    
    return output_dir


def validate_repeats(repeats: int) -> None:
    """Validate that repeats is >= 1."""
    if repeats < 1:
        raise ValueError(f"Number of repeats must be >= 1, got {repeats}")
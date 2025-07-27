"""Experiment runner for active learning workflows."""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from learnm8.core.active_learning import run_active_learning
from learnm8.core.interfaces import Oracle, Learner
from learnm8.oracles.csv_oracle import CSVOracle
from learnm8.oracles.python_oracle import PythonOracle
from learnm8.learners.random_forest import RandomForestLearner
from learnm8.learners.advanced_random_forest import AdvancedRandomForestLearner
from learnm8.learners.mlp import MLPLearner
from learnm8.learners.linear_regression import LinearRegressionLearner
from learnm8.learners.decision_tree import DecisionTreeLearner
from learnm8.learners.gradient_boosting import XGBoostLearner
from learnm8.learners.gaussian_process import GaussianProcessLearner
from learnm8.learners.gaussian_pytorch import GaussianPyTorchLearner
from learnm8.learners.pytorch_mlp import PyTorchMLPLearner
from learnm8.strategies.greedy import select_greedy
from learnm8.strategies.random import select_random
from learnm8.strategies.diversity import select_diverse
from learnm8.utils.logging import (
    setup_logging, get_logger, log_file_operation, log_success, log_warning
)
from learnm8.utils.featurizers import precompute_representations


def create_experiment_config(
    compound_pool_path: str,
    ground_truth_path: str,
    target_column: str,
    n_cycles: int = 10,
    batch_size_fraction: float = 0.1,
    selection_strategy: str = 'greedy',
    initial_strategy: str = 'random',
    score_direction: str = 'higher',
    learner_type: str = 'random_forest',
    random_state: int = 42,
    repeats: int = 1,
    oracle_type: str = 'csv',
    oracle_function_name: Optional[str] = None,
    featurizer: Optional[str] = None
) -> dict:
    """Create experiment configuration dictionary."""
    return {
        'compound_pool_path': compound_pool_path,
        'ground_truth_path': ground_truth_path,
        'target_column': target_column,
        'n_cycles': n_cycles,
        'batch_size_fraction': batch_size_fraction,
        'selection_strategy': selection_strategy,
        'initial_strategy': initial_strategy,
        'score_direction': score_direction,
        'learner_type': learner_type,
        'featurizer': featurizer,
        'random_state': random_state,
        'repeats': repeats,
        'oracle_type': oracle_type,
        'oracle_function_name': oracle_function_name,
        'monitoring': {
            # Top-K overlaps now calculated automatically for multiple K values
        },
        'timestamp': datetime.now().isoformat()
    }


def get_selection_strategy(name: str):
    """Get selection strategy function by name."""
    strategies = {
        'greedy': select_greedy,
        'random': select_random,
        'diverse': select_diverse,
        'diversity': select_diverse
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name]


def get_learner(name: str, random_state: int, results_dir: Path = None, featurizer_type: str = 'morgan') -> Learner:
    """Get learner instance by name."""
    learners = {
        'random_forest': RandomForestLearner,
        'rf': RandomForestLearner,
        'advanced_random_forest': AdvancedRandomForestLearner,
        'arf': AdvancedRandomForestLearner,
        'mlp': MLPLearner,
        'linear_regression': LinearRegressionLearner,
        'lr': LinearRegressionLearner,
        'decision_tree': DecisionTreeLearner,
        'dt': DecisionTreeLearner,
        'gradient_boosting': XGBoostLearner,
        'gb': XGBoostLearner,
        'gaussian_process': GaussianProcessLearner,
        'gp': GaussianProcessLearner,
        'gaussian_pytorch': GaussianPyTorchLearner,
        'gp_pytorch': GaussianPyTorchLearner,
        'pytorch_mlp': PyTorchMLPLearner,
        'mlp_pytorch': PyTorchMLPLearner
    }
    
    if name not in learners:
        raise ValueError(f"Unknown learner: {name}. Available: {list(learners.keys())}")
    
    return learners[name](random_state=random_state, results_dir=results_dir, featurizer_type=featurizer_type)


def get_oracle(config: dict) -> Oracle:
    """Get oracle instance based on configuration."""
    oracle_type = config.get('oracle_type', 'csv')
    
    if oracle_type == 'csv':
        return CSVOracle(config['ground_truth_path'])
    elif oracle_type == 'python':
        return PythonOracle(
            config['ground_truth_path'],
            config.get('oracle_function_name')
        )
    else:
        raise ValueError(f"Unknown oracle type: {oracle_type}. Available: ['csv', 'python']")


def create_monitoring_summary(all_monitoring_results: list[list[dict]]) -> pd.DataFrame:
    """Create aggregate monitoring summary across repeats."""
    if not all_monitoring_results:
        return pd.DataFrame()
    
    # Convert to DataFrame for easier aggregation
    all_data = []
    for repeat_idx, monitoring_results in enumerate(all_monitoring_results):
        for result in monitoring_results:
            result_copy = result.copy()
            result_copy['repeat'] = repeat_idx
            all_data.append(result_copy)
    
    df = pd.DataFrame(all_data)
    
    # Group by cycle and calculate mean/std
    summary_stats = []
    for cycle in sorted(df['cycle'].unique()):
        cycle_data = df[df['cycle'] == cycle]
        
        stats = {'cycle': cycle}
        for col in df.columns:
            if col in ['cycle', 'repeat']:
                continue
            if pd.api.types.is_numeric_dtype(cycle_data[col]):
                stats[f'{col}_mean'] = cycle_data[col].mean()
                stats[f'{col}_std'] = cycle_data[col].std()
        
        summary_stats.append(stats)
    
    return pd.DataFrame(summary_stats)


def run_single_experiment(config: dict, output_dir: Optional[Path] = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run a single active learning experiment.
    
    Args:
        config: Experiment configuration dictionary
        output_dir: Directory to save results
        
    Returns:
        Tuple of (final predictions, monitoring results)
    """
    # Load compound pool
    compound_pool = pd.read_csv(config['compound_pool_path'])
    
    # Validate required columns
    required_cols = ['ID', 'SMILES']
    missing_cols = [col for col in required_cols if col not in compound_pool.columns]
    if missing_cols:
        raise ValueError(f"Compound pool missing columns: {missing_cols}")
    
    # Setup logging
    console = Console()
    logger = setup_logging(console=console)
    
    # Calculate batch size
    batch_size = int(len(compound_pool) * config['batch_size_fraction'])
    if batch_size < 1:
        raise ValueError(f"Batch size too small: {batch_size}. Increase batch_size_fraction.")
    
    logger.info(f"Batch size: [bold]{batch_size}[/bold] ([dim]{config['batch_size_fraction']*100:.1f}% of {len(compound_pool)} compounds[/dim])")
    
    # Setup output directory first (needed for representation storage)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"learnm8_results_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine featurizer type and precompute representations
    featurizer_type = config.get('featurizer')
    if featurizer_type is None:
        # Default to morgan fingerprints since FastProp is deprecated
        featurizer_type = 'morgan'
    
    logger.info(f"Precomputing {featurizer_type} representations for {len(compound_pool)} compounds")
    precompute_representations(compound_pool, featurizer_type, output_dir, logger)
    
    # Create components
    oracle = get_oracle(config)
    learner = get_learner(config['learner_type'], config['random_state'], output_dir, featurizer_type)
    selection_strategy = get_selection_strategy(config['selection_strategy'])
    initial_strategy = get_selection_strategy(config['initial_strategy'])
    
    # Save config
    config_path = output_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    log_file_operation(logger, "Saved config", str(config_path))
    
    # Run active learning
    final_predictions, monitoring_results = run_active_learning(
        compound_pool=compound_pool,
        oracle=oracle,
        learner=learner,
        target_column=config['target_column'],
        n_cycles=config['n_cycles'],
        batch_size=batch_size,
        selection_strategy=selection_strategy,
        initial_selection_strategy=initial_strategy,
        score_direction=config['score_direction'],
        random_state=config['random_state'],
        output_dir=output_dir,
        monitoring_config=config['monitoring']
    )
    
    return final_predictions, monitoring_results


def run_experiment(config: dict, output_dir: Optional[Path] = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run a complete active learning experiment with optional repeats.
    
    Args:
        config: Experiment configuration dictionary
        output_dir: Directory to save results
        
    Returns:
        Tuple of (final predictions, monitoring results)
        - If repeats=1: returns single experiment results
        - If repeats>1: returns aggregated results across all repeats
    """
    repeats = config.get('repeats', 1)
    base_random_state = config['random_state']
    
    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save aggregate config
        with open(output_dir / 'aggregate_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    if repeats == 1:
        # Single experiment - maintain backward compatibility
        repeat_config = config.copy()
        repeat_output_dir = output_dir
        
        final_predictions, monitoring_results = run_single_experiment(repeat_config, repeat_output_dir)
        
        print_experiment_summary(monitoring_results, config)
        return final_predictions, monitoring_results
    
    else:
        # Multiple repeats
        console = Console()
        logger = setup_logging(console=console)
        
        logger.info(f"[bold blue]Running {repeats} experiment repeats[/bold blue]")
        all_predictions = []
        all_monitoring_results = []
        
        # Run experiment repeats
        for repeat_idx in range(repeats):
            logger.info(f"[yellow]Repeat {repeat_idx + 1}/{repeats}[/yellow]")
            
            # Create repeat-specific config with unique seed
            repeat_config = config.copy()
            repeat_config['random_state'] = base_random_state + repeat_idx
            
            # Create repeat subdirectory
            repeat_output_dir = None
            if output_dir:
                repeat_output_dir = output_dir / f'repeat_{repeat_idx}'
                repeat_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run single experiment
            predictions, monitoring = run_single_experiment(repeat_config, repeat_output_dir)
            
            all_predictions.append(predictions)
            all_monitoring_results.append(monitoring)
        
        # Create aggregate monitoring summary
        if output_dir:
            monitoring_summary = create_monitoring_summary(all_monitoring_results)
            summary_path = output_dir / 'monitoring_summary.csv'
            monitoring_summary.to_csv(summary_path, index=False)
            log_file_operation(logger, "Saved monitoring summary", str(summary_path))
        
        print_aggregate_summary(all_monitoring_results, config)
        
        # Return results from first repeat for compatibility
        return all_predictions[0], all_monitoring_results[0]


def print_experiment_summary(monitoring_results: list[dict], config: dict) -> None:
    """Print summary for a single experiment."""
    console = Console()
    
    if not monitoring_results:
        return
    
    first_result = monitoring_results[0]
    last_result = monitoring_results[-1]
    
    # Create summary table
    table = Table(title="[bold blue]Experiment Summary[/bold blue]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("First Cycle", justify="right")
    table.add_column("Last Cycle", justify="right")
    table.add_column("Change", justify="right")
    
    # Top-k overlap metrics (show multiple top-k values, handle None values)
    # Top-100 overlap
    if ('top_100_overlap' in first_result and first_result['top_100_overlap'] is not None and
        'top_100_overlap' in last_result and last_result['top_100_overlap'] is not None):
        overlap_change = last_result['top_100_overlap'] - first_result['top_100_overlap']
        change_style = "green" if overlap_change > 0 else "red" if overlap_change < 0 else "white"
        table.add_row(
            "Top-100 Overlap (%)",
            f"{first_result['top_100_overlap']:.2f}%",
            f"{last_result['top_100_overlap']:.2f}%",
            f"[{change_style}]{overlap_change:+.2f}%[/{change_style}]"
        )
    elif 'top_100_overlap' in first_result:
        table.add_row(
            "Top-100 Overlap (%)",
            "N/A",
            "N/A", 
            "N/A (run mode)"
        )
    
    # Top-1000 overlap
    if ('top_1000_overlap' in first_result and first_result['top_1000_overlap'] is not None and
        'top_1000_overlap' in last_result and last_result['top_1000_overlap'] is not None):
        overlap_change = last_result['top_1000_overlap'] - first_result['top_1000_overlap']
        change_style = "green" if overlap_change > 0 else "red" if overlap_change < 0 else "white"
        table.add_row(
            "Top-1000 Overlap (%)",
            f"{first_result['top_1000_overlap']:.2f}%",
            f"{last_result['top_1000_overlap']:.2f}%",
            f"[{change_style}]{overlap_change:+.2f}%[/{change_style}]"
        )
    elif 'top_1000_overlap' in first_result:
        table.add_row(
            "Top-1000 Overlap (%)",
            "N/A",
            "N/A",
            "N/A (run mode)"
        )
    
    # Average score
    direction_better = config['score_direction'] == 'higher'
    score_change = last_result['avg_score_selected'] - first_result['avg_score_selected']
    score_improved = (score_change > 0) == direction_better
    trend_style = "green" if score_improved else "red"
    trend_text = "Better" if score_improved else "Worse"
    
    table.add_row(
        "Avg Score Selected",
        f"{first_result['avg_score_selected']:.4f}",
        f"{last_result['avg_score_selected']:.4f}",
        f"[{trend_style}]{trend_text}[/{trend_style}]"
    )
    
    # ML Prediction Enrichment factors
    ef_keys = ['ef_5', 'ef_1', 'ef_0_5', 'ef_0_1']
    ef_labels = ['ML EF 5%', 'ML EF 1%', 'ML EF 0.5%', 'ML EF 0.1%']
    
    for key, label in zip(ef_keys, ef_labels):
        if first_result.get(key) is not None and last_result.get(key) is not None:
            ef_change = last_result[key] - first_result[key]
            ef_style = "green" if ef_change > 0 else "red" if ef_change < 0 else "white"
            table.add_row(
                label,
                f"{first_result[key]:.2f}",
                f"{last_result[key]:.2f}",
                f"[{ef_style}]{ef_change:+.2f}[/{ef_style}]"
            )
    
    # Ground Truth Enrichment factors (constant, so no change)
    gt_ef_keys = ['ground_truth_ef_5', 'ground_truth_ef_1', 'ground_truth_ef_0_5', 'ground_truth_ef_0_1']
    gt_ef_labels = ['GT EF 5%', 'GT EF 1%', 'GT EF 0.5%', 'GT EF 0.1%']
    
    for key, label in zip(gt_ef_keys, gt_ef_labels):
        if first_result.get(key) is not None:
            table.add_row(
                label,
                f"{first_result[key]:.2f}",
                f"{first_result[key]:.2f}",
                "[white]Constant[/white]"
            )
    
    console.print()
    console.print(table)
    
    # Additional info
    cumulative = last_result.get('cumulative_compounds_selected', 'N/A')
    console.print(f"\n[dim]Cumulative compounds selected: {cumulative}[/dim]")


def print_aggregate_summary(all_monitoring_results: list[list[dict]], config: dict) -> None:
    """Print aggregate summary across multiple repeats."""
    console = Console()
    
    if not all_monitoring_results:
        return
    
    # Calculate statistics across repeats for final cycle
    final_cycle_results = [results[-1] for results in all_monitoring_results if results]
    
    if not final_cycle_results:
        return
    
    # Create aggregate summary table
    table = Table(title="[bold green]Aggregate Summary Across Repeats[/bold green]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Range", justify="right")
    
    # Top-k overlap statistics (multiple top-k values, handle None values)
    # Top-100 overlap statistics
    if all('top_100_overlap' in r and r['top_100_overlap'] is not None for r in final_cycle_results):
        top_100_values = [r['top_100_overlap'] for r in final_cycle_results]
        table.add_row(
            "Top-100 Overlap (%)",
            f"{sum(top_100_values)/len(top_100_values):.2f}%",
            f"{pd.Series(top_100_values).std():.2f}%",
            f"{min(top_100_values):.2f}% - {max(top_100_values):.2f}%"
        )
    elif any('top_100_overlap' in r for r in final_cycle_results):
        table.add_row(
            "Top-100 Overlap (%)",
            "N/A",
            "N/A",
            "N/A (run mode)"
        )
    
    # Top-1000 overlap statistics
    if all('top_1000_overlap' in r and r['top_1000_overlap'] is not None for r in final_cycle_results):
        top_1000_values = [r['top_1000_overlap'] for r in final_cycle_results]
        table.add_row(
            "Top-1000 Overlap (%)",
            f"{sum(top_1000_values)/len(top_1000_values):.2f}%",
            f"{pd.Series(top_1000_values).std():.2f}%",
            f"{min(top_1000_values):.2f}% - {max(top_1000_values):.2f}%"
        )
    elif any('top_1000_overlap' in r for r in final_cycle_results):
        table.add_row(
            "Top-1000 Overlap (%)",
            "N/A",
            "N/A",
            "N/A (run mode)"
        )
    
    # Average score statistics
    avg_scores = [r['avg_score_selected'] for r in final_cycle_results]
    table.add_row(
        "Avg Score Selected",
        f"{sum(avg_scores)/len(avg_scores):.4f}",
        f"{pd.Series(avg_scores).std():.4f}",
        f"{min(avg_scores):.4f} - {max(avg_scores):.4f}"
    )
    
    # ML Prediction Enrichment factors
    ef_keys = ['ef_5', 'ef_1', 'ef_0_5', 'ef_0_1']
    ef_labels = ['ML EF 5%', 'ML EF 1%', 'ML EF 0.5%', 'ML EF 0.1%']
    
    for key, label in zip(ef_keys, ef_labels):
        enrichment_values = [r.get(key) for r in final_cycle_results if r.get(key) is not None]
        if enrichment_values:
            table.add_row(
                label,
                f"{sum(enrichment_values)/len(enrichment_values):.2f}",
                f"{pd.Series(enrichment_values).std():.2f}",
                f"{min(enrichment_values):.2f} - {max(enrichment_values):.2f}"
            )
    
    # Ground Truth Enrichment factors (constant across repeats)
    gt_ef_keys = ['ground_truth_ef_5', 'ground_truth_ef_1', 'ground_truth_ef_0_5', 'ground_truth_ef_0_1']
    gt_ef_labels = ['GT EF 5%', 'GT EF 1%', 'GT EF 0.5%', 'GT EF 0.1%']
    
    for key, label in zip(gt_ef_keys, gt_ef_labels):
        enrichment_values = [r.get(key) for r in final_cycle_results if r.get(key) is not None]
        if enrichment_values:
            # Ground truth enrichment factors should be identical across repeats
            avg_value = sum(enrichment_values)/len(enrichment_values)
            table.add_row(
                label,
                f"{avg_value:.2f}",
                "0.00",  # Should be 0 since they're constant
                f"{avg_value:.2f} - {avg_value:.2f}"
            )
    
    # Cumulative compounds selected
    cumulative_values = [r.get('cumulative_compounds_selected') for r in final_cycle_results if r.get('cumulative_compounds_selected') is not None]
    if cumulative_values:
        table.add_row(
            "Cumulative Selected",
            f"{sum(cumulative_values)/len(cumulative_values):.1f}",
            f"{pd.Series(cumulative_values).std():.1f}",
            f"{min(cumulative_values)} - {max(cumulative_values)}"
        )
    
    console.print()
    console.print(table)
    console.print(f"\n[dim]Total repeats completed: {len(all_monitoring_results)}[/dim]")
#!/usr/bin/env python3
"""
Comprehensive validation script for ChempropLearner testing all combinations of:
- Early stopping (enabled/disabled)
- Featurization options (none, morgan, maccs, ecfp6, morgan_feat, descriptors)
- Model configurations (baseline, deep, wide)

Reports both performance metrics and timing data.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from learnm8 import run_active_learning
from learnm8.learners.torch.chemprop_learner import ChempropLearner
from learnm8.oracles.csv_oracle import CSVOracle
from validation.lib import load_validation_dataset, get_dataset_info, get_dataset_path

console = Console()
warnings.filterwarnings('ignore')

from learnm8 import setup_logging
setup_logging(level='INFO')

MODEL_CONFIGS = {
    'baseline': {
        'name': 'baseline',
        'depth': 3,
        'message_hidden_dim': 300,
        'ffn_hidden_dim': 300,
        'ffn_num_layers': 1,
        'dropout': 0.0,
        'batch_norm': False,
        'atom_messages': False,
    },
    'deep': {
        'name': 'deep',
        'depth': 5,
        'message_hidden_dim': 300,
        'ffn_hidden_dim': 300,
        'ffn_num_layers': 1,
        'dropout': 0.0,
        'batch_norm': False,
        'atom_messages': False,
    },
    'wide': {
        'name': 'wide',
        'depth': 3,
        'message_hidden_dim': 500,
        'ffn_hidden_dim': 500,
        'ffn_num_layers': 1,
        'dropout': 0.0,
        'batch_norm': False,
        'atom_messages': False,
    },
}

FEATURIZERS = [
    None,
    'morgan',
    'maccs',
    'ecfp6',
    'morgan_feat',
    'descriptors',
]

EARLY_STOPPING_OPTIONS = [True, False]


def load_dataset(dataset_name: str) -> tuple[pl.DataFrame, str, CSVOracle]:
    """Load dataset using validation library and create oracle."""
    df, metadata = load_validation_dataset(dataset_name, clean_invalid_scores=True)
    target_col = metadata['target_column']

    dataset_path = get_dataset_path(dataset_name)
    oracle = CSVOracle(str(dataset_path), id_column='ID')

    return df, target_col, oracle


def load_existing_results(
    exp_dir: Path,
    config_name: str,
    config: Dict[str, Any],
    featurizer: Optional[str],
    early_stopping: bool,
) -> Dict[str, Any]:
    """Load results from existing experiment directory."""
    featurizer_name = featurizer if featurizer else 'none'

    cycle_metrics_df = pl.read_csv(exp_dir / 'cycle_metrics.csv')
    final_metrics = cycle_metrics_df[-1]

    cycle_metrics = cycle_metrics_df.to_dicts()
    training_times = [m.get('training_time', 0) for m in cycle_metrics[1:]]
    prediction_times = [m.get('prediction_time', 0) for m in cycle_metrics[1:]]

    return {
        'config_name': config_name,
        'featurizer': featurizer_name,
        'early_stopping': early_stopping,
        'depth': config['depth'],
        'message_hidden_dim': config['message_hidden_dim'],
        'top_10_recovery': final_metrics.get('top_10_discovery', 0),
        'top_100_recovery': final_metrics.get('top_100_discovery', 0),
        'final_discovery_rate': final_metrics.get('discovery_rate', 0),
        'avg_training_time': np.mean(training_times) if training_times else 0,
        'avg_prediction_time': np.mean(prediction_times) if prediction_times else 0,
        'total_training_time': np.sum(training_times) if training_times else 0,
        'total_time': 0,
        'final_labeled_count': final_metrics.get('cumulative_labeled', 0),
        'final_discovery_count': final_metrics.get('discovery_count', 0),
        'success': True,
        'error': None,
    }


def run_single_experiment(
    compound_pool: pl.DataFrame,
    oracle: CSVOracle,
    target_col: str,
    config_name: str,
    config: Dict[str, Any],
    featurizer: Optional[str],
    early_stopping: bool,
    n_cycles: int,
    batch_fraction: float,
    random_state: int,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    featurizer_name = featurizer if featurizer else 'none'
    early_stop_str = 'early_stop' if early_stopping else 'no_early_stop'
    exp_name = f"{config_name}_{featurizer_name}_{early_stop_str}"

    exp_output_dir = output_dir / 'data' / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        console.print(f"[cyan]Starting experiment: {exp_name}[/cyan]")

    learner = ChempropLearner(
        depth=config['depth'],
        message_hidden_dim=config['message_hidden_dim'],
        ffn_hidden_dim=config['ffn_hidden_dim'],
        ffn_num_layers=config['ffn_num_layers'],
        dropout=config['dropout'],
        batch_norm=config['batch_norm'],
        atom_messages=config['atom_messages'],
        early_stopping=early_stopping,
        early_stopping_patience=10,
        max_epochs=50,
        random_state=random_state,
    )

    start_time = time.time()

    try:
        results = run_active_learning(
            compound_pool=compound_pool.clone(),
            oracle=oracle,
            learner=learner,
            target_col=target_col,
            featurizer_type=featurizer,
            n_cycles=n_cycles,
            batch_fraction=batch_fraction,
            random_state=random_state,
            output_dir=exp_output_dir,
            mode='benchmark',
        )

        total_time = time.time() - start_time

        compounds_df = results['compounds_df']
        cycle_metrics = results['cycle_metrics']

        compounds_df.write_csv(exp_output_dir / 'compounds_final.csv')
        pl.DataFrame(cycle_metrics).write_csv(exp_output_dir / 'cycle_metrics.csv')

        training_times = [m.get('training_time', 0) for m in cycle_metrics[1:]]
        prediction_times = [m.get('prediction_time', 0) for m in cycle_metrics[1:]]

        final_metrics = cycle_metrics[-1]

        result = {
            'config_name': config_name,
            'featurizer': featurizer_name,
            'early_stopping': early_stopping,
            'depth': config['depth'],
            'message_hidden_dim': config['message_hidden_dim'],
            'top_10_recovery': final_metrics.get('top_10_discovery', 0),
            'top_100_recovery': final_metrics.get('top_100_discovery', 0),
            'final_discovery_rate': final_metrics.get('discovery_rate', 0),
            'avg_training_time': np.mean(training_times) if training_times else 0,
            'avg_prediction_time': np.mean(prediction_times) if prediction_times else 0,
            'total_training_time': np.sum(training_times) if training_times else 0,
            'total_time': total_time,
            'final_labeled_count': final_metrics.get('cumulative_labeled', 0),
            'final_discovery_count': final_metrics.get('discovery_count', 0),
            'success': True,
            'error': None,
        }

        if debug:
            console.print(f"[green]✓ {exp_name}: top_10={result['top_10_recovery']:.3f}, "
                         f"time={total_time:.1f}s[/green]")

        return result

    except Exception as e:
        if debug:
            console.print(f"[red]✗ {exp_name}: {str(e)}[/red]")

        return {
            'config_name': config_name,
            'featurizer': featurizer_name,
            'early_stopping': early_stopping,
            'depth': config['depth'],
            'message_hidden_dim': config['message_hidden_dim'],
            'top_10_recovery': 0,
            'top_100_recovery': 0,
            'final_discovery_rate': 0,
            'avg_training_time': 0,
            'avg_prediction_time': 0,
            'total_training_time': 0,
            'total_time': 0,
            'final_labeled_count': 0,
            'final_discovery_count': 0,
            'success': False,
            'error': str(e),
        }


def create_visualizations(df: pl.DataFrame, output_dir: Path):
    """Create comprehensive visualization plots using Polars."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_success = df.filter(pl.col('success') == True)

    if len(df_success) == 0:
        console.print("[yellow]Warning: No successful experiments to visualize[/yellow]")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Heatmap of Top-10 Recovery by Configuration and Featurizer
    ax = axes[0, 0]
    performance_pivot = df_success.pivot(
        values='top_10_recovery',
        index='config_name',
        columns='featurizer',
        aggregate_function='mean'
    )
    # Convert to pandas for seaborn heatmap compatibility
    performance_data = performance_pivot.to_pandas()
    performance_data.index.name = 'config_name'
    sns.heatmap(performance_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, vmin=0, vmax=1)
    ax.set_title('Top-10 Recovery by Configuration and Featurizer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Featurizer')
    ax.set_ylabel('Model Configuration')

    # Plot 2: Early Stopping Impact on Performance
    ax = axes[0, 1]
    early_stop_comparison = df_success.group_by(['config_name', 'early_stopping']).agg([
        pl.col('top_10_recovery').mean().alias('top_10_recovery'),
        pl.col('top_100_recovery').mean().alias('top_100_recovery'),
    ]).sort(['config_name', 'early_stopping'])

    config_names = early_stop_comparison['config_name'].unique().sort().to_list()
    x_pos = np.arange(len(config_names))
    width = 0.35

    for idx, early_stop in enumerate([False, True]):
        group = early_stop_comparison.filter(pl.col('early_stopping') == early_stop)
        offset = width * (idx - 0.5)
        label = 'Early Stopping' if early_stop else 'No Early Stopping'
        values = group['top_10_recovery'].to_numpy()
        ax.bar(x_pos + offset, values, width, label=label, alpha=0.8)

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Top-10 Recovery')
    ax.set_title('Early Stopping Impact on Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Timing Comparison
    ax = axes[1, 0]
    timing_data = df_success.group_by('config_name').agg([
        pl.col('avg_training_time').mean().alias('avg_training_time'),
        pl.col('avg_prediction_time').mean().alias('avg_prediction_time'),
    ]).sort('config_name')

    x_pos = np.arange(len(timing_data))
    width = 0.35

    ax.bar(x_pos - width/2, timing_data['avg_training_time'].to_numpy(), width, label='Training Time', alpha=0.8)
    ax.bar(x_pos + width/2, timing_data['avg_prediction_time'].to_numpy(), width, label='Prediction Time', alpha=0.8)

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Training vs Prediction Time', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(timing_data['config_name'].to_list())
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Featurizer Performance Comparison
    ax = axes[1, 1]
    featurizer_comparison = df_success.group_by('featurizer').agg([
        pl.col('top_10_recovery').mean().alias('mean_recovery'),
        pl.col('top_10_recovery').std().alias('std_recovery'),
        pl.col('total_time').mean().alias('mean_time'),
    ]).sort('mean_recovery', descending=True)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(featurizer_comparison)))
    bars = ax.barh(
        featurizer_comparison['featurizer'].to_list(),
        featurizer_comparison['mean_recovery'].to_numpy(),
        xerr=featurizer_comparison['std_recovery'].to_numpy(),
        capsize=5, color=colors, alpha=0.8
    )

    ax.set_xlabel('Top-10 Recovery (mean ± std)')
    ax.set_ylabel('Featurizer')
    ax.set_title('Featurizer Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for i, row in enumerate(featurizer_comparison.iter_rows(named=True)):
        ax.text(0.02, i, f'{row["mean_time"]:.1f}s', va='center', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(plots_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved visualizations to {plots_dir}[/green]")


def print_summary_table(df: pl.DataFrame):
    """Print a rich summary table of results."""
    df_success = df.filter(pl.col('success') == True)

    if len(df_success) == 0:
        console.print("[red]No successful experiments to summarize[/red]")
        return

    df_sorted = df_success.sort('top_10_recovery', descending=True).head(10).to_pandas()

    table = Table(title="Top 10 Configurations by Performance", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan", width=6)
    table.add_column("Config", justify="left", style="green", width=10)
    table.add_column("Featurizer", justify="left", style="yellow", width=12)
    table.add_column("Early Stop", justify="center", width=11)
    table.add_column("Top-10", justify="right", style="bold", width=8)
    table.add_column("Top-100", justify="right", width=8)
    table.add_column("Train Time", justify="right", width=11)
    table.add_column("Total Time", justify="right", width=11)

    for idx, row in enumerate(df_sorted.itertuples(), 1):
        early_stop_icon = "✓" if row.early_stopping else "✗"
        table.add_row(
            str(idx),
            row.config_name,
            row.featurizer,
            early_stop_icon,
            f"{row.top_10_recovery:.3f}",
            f"{row.top_100_recovery:.3f}",
            f"{row.avg_training_time:.1f}s",
            f"{row.total_time:.1f}s",
        )

    console.print(table)

    console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")
    console.print(f"  Total experiments: {len(df)}")
    console.print(f"  Successful: {len(df_success)} ({100*len(df_success)/len(df):.1f}%)")
    console.print(f"  Failed: {len(df) - len(df_success)}")
    console.print(f"  Best top-10 recovery: {df_success['top_10_recovery'].max():.3f}")
    console.print(f"  Mean top-10 recovery: {df_success['top_10_recovery'].mean():.3f}")
    console.print(f"  Avg total time per experiment: {df_success['total_time'].mean():.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Chemprop validation testing early stopping, featurizers, and model configs'
    )
    parser.add_argument('--dataset', type=str, default='ampc_30k',
                       help='Dataset to use (default: ampc_30k)')
    parser.add_argument('--n-cycles', type=int, default=10,
                       help='Number of active learning cycles (default: 3)')
    parser.add_argument('--batch-fraction', type=float, default=0.01,
                       help='Batch fraction for sampling (default: 0.01)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip experiments with existing output directories (default: True)')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                       help='Re-run all experiments even if results exist')
    parser.add_argument('--debug', action='store_true', default=True,
                       help='Print detailed debug information (default: True)')
    parser.add_argument('--no-debug', dest='debug', action='store_false',
                       help='Disable debug output')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: validation/reports/chemprop_comprehensive)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(__file__).parent.parent / 'reports' / 'chemprop_comprehensive'

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Chemprop Comprehensive Validation[/bold cyan]")
    console.print(f"Dataset: {args.dataset}")
    console.print(f"Cycles: {args.n_cycles}")
    console.print(f"Batch fraction: {args.batch_fraction}")
    console.print(f"Output: {args.output_dir}")
    console.print()

    compound_pool, target_col, oracle = load_dataset(args.dataset)
    console.print(f"[green]✓ Loaded {len(compound_pool)} compounds[/green]")
    console.print(f"[cyan]Target column: {target_col}[/cyan]")

    experiments = []
    for config_name, config in MODEL_CONFIGS.items():
        for featurizer in FEATURIZERS:
            for early_stopping in EARLY_STOPPING_OPTIONS:
                experiments.append({
                    'config_name': config_name,
                    'config': config,
                    'featurizer': featurizer,
                    'early_stopping': early_stopping,
                })

    console.print(f"[cyan]Total experiments to run: {len(experiments)}[/cyan]\n")

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running experiments...", total=len(experiments))

        for exp in experiments:
            featurizer_name = exp['featurizer'] if exp['featurizer'] else 'none'
            early_stop_str = 'early_stop' if exp['early_stopping'] else 'no_early_stop'
            exp_name = f"{exp['config_name']}_{featurizer_name}_{early_stop_str}"

            exp_dir = args.output_dir / 'data' / exp_name
            if args.skip_existing and exp_dir.exists() and (exp_dir / 'cycle_metrics.csv').exists():
                try:
                    result = load_existing_results(
                        exp_dir=exp_dir,
                        config_name=exp['config_name'],
                        config=exp['config'],
                        featurizer=exp['featurizer'],
                        early_stopping=exp['early_stopping'],
                    )
                    results.append(result)
                    if args.debug:
                        console.print(f"[yellow]⊙ Loaded existing: {exp_name} (top_10={result['top_10_recovery']:.3f})[/yellow]")
                except Exception as e:
                    if args.debug:
                        console.print(f"[red]✗ Failed to load existing {exp_name}: {e}[/red]")
                        console.print(f"[yellow]  Re-running experiment...[/yellow]")
                else:
                    progress.advance(task)
                    continue

            result = run_single_experiment(
                compound_pool=compound_pool,
                oracle=oracle,
                target_col=target_col,
                config_name=exp['config_name'],
                config=exp['config'],
                featurizer=exp['featurizer'],
                early_stopping=exp['early_stopping'],
                n_cycles=args.n_cycles,
                batch_fraction=args.batch_fraction,
                random_state=args.random_state,
                output_dir=args.output_dir,
                debug=args.debug,
            )

            results.append(result)
            progress.advance(task)

    results_df = pl.DataFrame(results)
    summary_path = args.output_dir / 'summary.csv'
    results_df.write_csv(summary_path)
    console.print(f"\n[green]✓ Saved summary to {summary_path}[/green]")

    create_visualizations(results_df, args.output_dir)

    print_summary_table(results_df)

    console.print(f"\n[bold green]✓ Validation complete![/bold green]")
    console.print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

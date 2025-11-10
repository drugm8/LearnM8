#!/usr/bin/env python3
"""
Comprehensive validation script for ChempropLearner testing all combinations of:
- Featurization options (none, morgan, maccs, ecfp6, morgan_feat, descriptors)
- Model configurations (baseline, deep, wide, baseline_ft, baseline_no_early_stopping)

Each model configuration includes all hyperparameters (depth, width, early_stopping, fine_tuning).
Reports both performance metrics and timing data with multi-seed aggregation.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import pandas as pd
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
from validation.lib.seed_aggregation import (
    aggregate_seed_results,
    save_aggregated_results
)

console = Console()
warnings.filterwarnings('ignore')

from learnm8 import setup_logging
setup_logging(level='INFO')

RANDOM_SEEDS = [42, 123, 456]

MODEL_CONFIGS = {
    'baseline': {
        'depth': 3,
        'message_hidden_dim': 300,
        'ffn_hidden_dim': 300,
        'ffn_num_layers': 1,
        'dropout': 0.0,
        'batch_norm': False,
        'atom_messages': False,
        'early_stopping': False,
        'early_stopping_patience': 3,
        'enable_fine_tuning': False,
    },
    'deep': {
        'depth': 5,
        'message_hidden_dim': 300,
        'ffn_hidden_dim': 300,
        'ffn_num_layers': 2,
        'dropout': 0.0,
        'batch_norm': False,
        'atom_messages': False,
        'early_stopping': False,
        'early_stopping_patience': 3,
        'enable_fine_tuning': False,
    },
    'wide': {
        'depth': 3,
        'message_hidden_dim': 500,
        'ffn_hidden_dim': 500,
        'ffn_num_layers': 1,
        'dropout': 0.0,
        'batch_norm': False,
        'atom_messages': False,
        'early_stopping': False,
        'early_stopping_patience': 3,
        'enable_fine_tuning': False,
    },
    'baseline_ft': {
        'depth': 3,
        'message_hidden_dim': 300,
        'ffn_hidden_dim': 300,
        'ffn_num_layers': 1,
        'dropout': 0.0,
        'batch_norm': False,
        'atom_messages': False,
        'early_stopping': False,
        'early_stopping_patience': 3,
        'enable_fine_tuning': True,
    },
    'baseline_early_stopping': {
        'depth': 3,
        'message_hidden_dim': 300,
        'ffn_hidden_dim': 300,
        'ffn_num_layers': 1,
        'dropout': 0.0,
        'batch_norm': False,
        'atom_messages': False,
        'early_stopping': True,
        'early_stopping_patience': 3,
        'enable_fine_tuning': False,
    },
    'baseline_dropout': {
        'depth': 3,
        'message_hidden_dim': 300,
        'ffn_hidden_dim': 300,
        'ffn_num_layers': 1,
        'dropout': 0.1,
        'batch_norm': False,
        'atom_messages': False,
        'early_stopping': False,
        'early_stopping_patience': 3,
        'enable_fine_tuning': False,
    },
    'baseline_atom': {
        'depth': 3,
        'message_hidden_dim': 300,
        'ffn_hidden_dim': 300,
        'ffn_num_layers': 1,
        'dropout': 0.0,
        'batch_norm': False,
        'atom_messages': True,
        'early_stopping': False,
        'early_stopping_patience': 3,
        'enable_fine_tuning': False,
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


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, str, CSVOracle]:
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
) -> Dict[str, Any]:
    """Load results from existing experiment directory with multi-seed support."""
    featurizer_name = featurizer if featurizer else 'none'

    seed_results = {}
    for seed in RANDOM_SEEDS:
        seed_dir = exp_dir / f'seed_{seed}'
        cycle_metrics_path = seed_dir / 'cycle_metrics.csv'

        if not cycle_metrics_path.exists():
            continue

        cycle_metrics_df = pd.read_csv(cycle_metrics_path)
        cycle_metrics = cycle_metrics_df.to_dict('records')

        seed_results[seed] = {
            'cycle_metrics': cycle_metrics,
            'elapsed_time': 0
        }

    if not seed_results:
        raise FileNotFoundError(f"No seed results found in {exp_dir}")

    aggregated = aggregate_seed_results(seed_results)

    final_metrics_mean = aggregated['cycle_metrics_mean'][-1]
    final_metrics_std = aggregated['cycle_metrics_std'][-1]

    all_training_times = []
    all_prediction_times = []
    for seed_data in seed_results.values():
        training_times = [m.get('training_time', 0) for m in seed_data['cycle_metrics'][1:]]
        prediction_times = [m.get('prediction_time', 0) for m in seed_data['cycle_metrics'][1:]]
        all_training_times.extend(training_times)
        all_prediction_times.extend(prediction_times)

    return {
        'config_name': config_name,
        'featurizer': featurizer_name,
        'depth': config['depth'],
        'message_hidden_dim': config['message_hidden_dim'],
        'top_10_recovery': final_metrics_mean.get('top_10_discovery', 0),
        'top_10_recovery_std': final_metrics_std.get('top_10_discovery', 0),
        'top_100_recovery': final_metrics_mean.get('top_100_discovery', 0),
        'top_100_recovery_std': final_metrics_std.get('top_100_discovery', 0),
        'final_discovery_rate': final_metrics_mean.get('discovery_rate', 0),
        'avg_training_time': np.mean(all_training_times) if all_training_times else 0,
        'avg_prediction_time': np.mean(all_prediction_times) if all_prediction_times else 0,
        'total_training_time': np.sum(all_training_times) if all_training_times else 0,
        'total_time': 0,
        'final_labeled_count': final_metrics_mean.get('cumulative_labeled', 0),
        'final_discovery_count': final_metrics_mean.get('discovery_count', 0),
        'success': True,
        'error': None,
    }


def run_single_experiment(
    compound_pool: pd.DataFrame,
    oracle: CSVOracle,
    target_col: str,
    config_name: str,
    config: Dict[str, Any],
    featurizer: Optional[str],
    n_cycles: int,
    batch_fraction: float,
    random_state: int,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run experiment across all random seeds and aggregate results."""
    featurizer_name = featurizer if featurizer else 'none'
    exp_name = f"{config_name}_{featurizer_name}"

    exp_output_dir = output_dir / 'data' / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        console.print(f"[cyan]Starting experiment: {exp_name} (seeds: {RANDOM_SEEDS})[/cyan]")

    seed_results = {}
    all_training_times = []
    all_prediction_times = []
    total_experiment_time = 0

    for seed in RANDOM_SEEDS:
        seed_output_dir = exp_output_dir / f'seed_{seed}'
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = seed_output_dir / '.checkpoints' if config['enable_fine_tuning'] else None

        learner = ChempropLearner(
            depth=config['depth'],
            message_hidden_dim=config['message_hidden_dim'],
            ffn_hidden_dim=config['ffn_hidden_dim'],
            ffn_num_layers=config['ffn_num_layers'],
            dropout=config['dropout'],
            batch_norm=config['batch_norm'],
            atom_messages=config['atom_messages'],
            early_stopping=config['early_stopping'],
            early_stopping_patience=config['early_stopping_patience'],
            max_epochs=50,
            random_state=seed,
            enable_fine_tuning=config['enable_fine_tuning'],
            checkpoint_dir=checkpoint_dir,
        )

        start_time = time.time()

        try:
            results = run_active_learning(
                compound_pool=compound_pool.copy(),
                oracle=oracle,
                learner=learner,
                target_col=target_col,
                featurizer_type=featurizer,
                n_cycles=n_cycles,
                batch_fraction=batch_fraction,
                random_state=seed,
                output_dir=seed_output_dir,
                mode='benchmark',
            )

            elapsed_time = time.time() - start_time
            total_experiment_time += elapsed_time

            cycle_metrics = results['cycle_metrics']

            seed_results[seed] = {
                'cycle_metrics': cycle_metrics,
                'elapsed_time': elapsed_time
            }

            training_times = [m.get('training_time', 0) for m in cycle_metrics[1:]]
            prediction_times = [m.get('prediction_time', 0) for m in cycle_metrics[1:]]
            all_training_times.extend(training_times)
            all_prediction_times.extend(prediction_times)

        except Exception as e:
            if debug:
                console.print(f"[red]✗ Seed {seed} failed: {str(e)}[/red]")
            raise

    aggregated = aggregate_seed_results(seed_results)
    save_aggregated_results(seed_results, aggregated, exp_output_dir)

    final_metrics_mean = aggregated['cycle_metrics_mean'][-1]
    final_metrics_std = aggregated['cycle_metrics_std'][-1]

    result = {
        'config_name': config_name,
        'featurizer': featurizer_name,
        'depth': config['depth'],
        'message_hidden_dim': config['message_hidden_dim'],
        'top_10_recovery': final_metrics_mean.get('top_10_discovery', 0),
        'top_10_recovery_std': final_metrics_std.get('top_10_discovery', 0),
        'top_100_recovery': final_metrics_mean.get('top_100_discovery', 0),
        'top_100_recovery_std': final_metrics_std.get('top_100_discovery', 0),
        'final_discovery_rate': final_metrics_mean.get('discovery_rate', 0),
        'avg_training_time': np.mean(all_training_times) if all_training_times else 0,
        'avg_prediction_time': np.mean(all_prediction_times) if all_prediction_times else 0,
        'total_training_time': np.sum(all_training_times) if all_training_times else 0,
        'total_time': total_experiment_time,
        'final_labeled_count': final_metrics_mean.get('cumulative_labeled', 0),
        'final_discovery_count': final_metrics_mean.get('discovery_count', 0),
        'success': True,
        'error': None,
    }

    if debug:
        console.print(f"[green]✓ {exp_name}: top_10={result['top_10_recovery']:.3f}±{result['top_10_recovery_std']:.3f}, "
                     f"time={total_experiment_time:.1f}s[/green]")

    return result


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualization plots."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_success = df[df['success'] == True].copy()

    if len(df_success) == 0:
        console.print("[yellow]Warning: No successful experiments to visualize[/yellow]")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    performance_data = df_success.pivot_table(
        values='top_10_recovery',
        index='config_name',
        columns='featurizer',
        aggfunc='mean'
    )
    sns.heatmap(performance_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, vmin=0, vmax=1)
    ax.set_title('Top-10 Recovery by Configuration and Featurizer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Featurizer')
    ax.set_ylabel('Model Configuration')

    ax = axes[0, 1]
    config_comparison = df_success.groupby('config_name').agg({
        'top_10_recovery': ['mean', 'std'],
        'top_100_recovery': ['mean', 'std'],
    }).reset_index()

    config_comparison.columns = ['config_name', 'top_10_mean', 'top_10_std', 'top_100_mean', 'top_100_std']
    config_comparison = config_comparison.sort_values('top_10_mean', ascending=False)

    x_pos = np.arange(len(config_comparison))
    width = 0.35

    ax.bar(x_pos - width/2, config_comparison['top_10_mean'], width,
           yerr=config_comparison['top_10_std'], label='Top-10', alpha=0.8, capsize=5)
    ax.bar(x_pos + width/2, config_comparison['top_100_mean'], width,
           yerr=config_comparison['top_100_std'], label='Top-100', alpha=0.8, capsize=5)

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Recovery Rate (mean ± std)')
    ax.set_title('Configuration Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_comparison['config_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    timing_data = df_success.groupby('config_name').agg({
        'avg_training_time': 'mean',
        'avg_prediction_time': 'mean',
    }).reset_index()

    x_pos = np.arange(len(timing_data))
    width = 0.35

    ax.bar(x_pos - width/2, timing_data['avg_training_time'], width, label='Training Time', alpha=0.8)
    ax.bar(x_pos + width/2, timing_data['avg_prediction_time'], width, label='Prediction Time', alpha=0.8)

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Training vs Prediction Time', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(timing_data['config_name'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    featurizer_comparison = df_success.groupby('featurizer').agg({
        'top_10_recovery': ['mean', 'std'],
        'total_time': 'mean',
    }).reset_index()

    featurizer_comparison.columns = ['featurizer', 'mean_recovery', 'std_recovery', 'mean_time']
    featurizer_comparison = featurizer_comparison.sort_values('mean_recovery', ascending=False)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(featurizer_comparison)))
    bars = ax.barh(featurizer_comparison['featurizer'], featurizer_comparison['mean_recovery'],
                   xerr=featurizer_comparison['std_recovery'], capsize=5, color=colors, alpha=0.8)

    ax.set_xlabel('Top-10 Recovery (mean ± std)')
    ax.set_ylabel('Featurizer')
    ax.set_title('Featurizer Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for i, (feat, time_val) in enumerate(zip(featurizer_comparison['featurizer'],
                                              featurizer_comparison['mean_time'])):
        ax.text(0.02, i, f'{time_val:.1f}s', va='center', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(plots_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved visualizations to {plots_dir}[/green]")


def print_summary_table(df: pd.DataFrame):
    """Print a rich summary table of results."""
    df_success = df[df['success'] == True].copy()

    if len(df_success) == 0:
        console.print("[red]No successful experiments to summarize[/red]")
        return

    df_sorted = df_success.sort_values('top_10_recovery', ascending=False).head(10)

    table = Table(title="Top 10 Configurations by Performance", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan", width=6)
    table.add_column("Config", justify="left", style="green", width=20)
    table.add_column("Featurizer", justify="left", style="yellow", width=12)
    table.add_column("Top-10", justify="right", style="bold", width=10)
    table.add_column("Top-100", justify="right", width=10)
    table.add_column("Train Time", justify="right", width=11)
    table.add_column("Total Time", justify="right", width=11)

    for idx, row in enumerate(df_sorted.itertuples(), 1):
        table.add_row(
            str(idx),
            row.config_name,
            row.featurizer,
            f"{row.top_10_recovery:.3f}±{row.top_10_recovery_std:.3f}",
            f"{row.top_100_recovery:.3f}±{row.top_100_recovery_std:.3f}",
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
        description='Comprehensive Chemprop validation testing model configs (baseline, deep, wide, baseline_ft, baseline_no_early_stopping) with featurizers'
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
            experiments.append({
                'config_name': config_name,
                'config': config,
                'featurizer': featurizer,
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
            exp_name = f"{exp['config_name']}_{featurizer_name}"

            exp_dir = args.output_dir / 'data' / exp_name
            if args.skip_existing and exp_dir.exists() and (exp_dir / 'seed_42' / 'cycle_metrics.csv').exists():
                try:
                    result = load_existing_results(
                        exp_dir=exp_dir,
                        config_name=exp['config_name'],
                        config=exp['config'],
                        featurizer=exp['featurizer'],
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
                n_cycles=args.n_cycles,
                batch_fraction=args.batch_fraction,
                random_state=args.random_state,
                output_dir=args.output_dir,
                debug=args.debug,
            )

            results.append(result)
            progress.advance(task)

    results_df = pd.DataFrame(results)
    summary_path = args.output_dir / 'summary.csv'
    results_df.to_csv(summary_path, index=False)
    console.print(f"\n[green]✓ Saved summary to {summary_path}[/green]")

    create_visualizations(results_df, args.output_dir)

    print_summary_table(results_df)

    console.print(f"\n[bold green]✓ Validation complete![/bold green]")
    console.print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

"""Simple CLI for LearnM8 using the API (Alternative 1).

This CLI provides a simple interface to the active learning API
with support for explicit cycle specifications and legacy parameter conversion.
"""

import argparse
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.traceback import install
from learnm8.utils import parse_cycle_spec

# Install rich traceback for better error messages
install()
console = Console()


def main():
    """Minimal CLI entry point using functional API."""
    parser = argparse.ArgumentParser(
        description="LearnM8: Active Learning for Molecular Screening",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('compound_pool', type=Path,
                       help='CSV file containing compound pool (requires ID, SMILES columns)')
    parser.add_argument('oracle', nargs='?',
                       help='Oracle specification: "data.csv" for benchmark or "module.py:function" for custom (optional - defaults to compound_pool for CSV files)')
    parser.add_argument('target_column',
                       help='Name of target property column')
    
    # Component selection
    parser.add_argument('-l', '--learner', default='rf',
                       choices=['rf', 'gp', 'xgb', 'mlp', 'mc_dropout', 'ensemble', 
                               'rf_ensemble', 'lr_ensemble', 'xgb_ensemble', 'dt_ensemble', 'mixed_ensemble'],
                       help='Machine learning model')
    
    # NEW: Explicit cycle specification (Alternative 1)
    parser.add_argument('--cycles-spec', type=str,
                       help='Explicit cycle specification: "random:0.01 greedy:0.005*5 diverse:0.01 bitbirch:0.01" (Alternative 1)')
    parser.add_argument('--schedule', choices=['quick', 'standard', 'intensive', 'diversity'],
                       help='Use predefined schedule (quick=5 cycles, standard=10 cycles, intensive=20 cycles, diversity=mixed diversity methods)')
    
    # Acquisition method selection
    parser.add_argument('-a', '--acquisition', type=str,
                       help='Default acquisition method for cycles (greedy, random, diverse, bitbirch, pca_dbscan, umap_dbscan, tsne_dbscan, etc.)')
    parser.add_argument('--acquisition-params', type=str,
                       help='JSON string of acquisition method parameters (e.g., \'{"diversity_weight": 0.3, "threshold": 0.65}\')')
    
    # LEGACY: Traditional parameters (only used if --cycles-spec and --schedule not provided)
    parser.add_argument('-c', '--cycles', type=int, default=10,
                       help='Number of active learning cycles (legacy mode only)')
    parser.add_argument('-b', '--batch-fraction', type=float, default=0.1,
                       help='Fraction of compounds to select per cycle (legacy mode only)')
    parser.add_argument('--max-batch-size', type=int, default=1000,
                       help='Maximum compounds per batch')
    parser.add_argument('--initial-size', type=int,
                       help='Initial training set size (default: 1%% of pool)')
    
    # Data configuration  
    parser.add_argument('--featurizer', default='morgan',
                       choices=['morgan', 'descriptors', 'maccs', 'ecfp6'],
                       help='Molecular featurizer type')
    parser.add_argument('--score-direction', default='auto',
                       choices=['higher', 'lower', 'auto'],
                       help='Score optimization direction (auto-detect, higher, or lower is better)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--n-workers', type=int,
                       help='Number of parallel workers (default: auto)')
    
    # Pruning configuration
    parser.add_argument('--pruning-strategy', type=str,
                       choices=['probabilistic', 'uncertainty_threshold', 'prediction_threshold', 
                               'confidence_interval', 'cycle_budget', 'performance_based'],
                       help='Pruning strategy to reduce compound pool size (default: None - no pruning)')
    parser.add_argument('--pruning-params', type=str,
                       help='JSON string of pruning strategy parameters (e.g., \'{"threshold": 0.1}\')')
    
    # Evaluation configuration
    parser.add_argument('--disable-evaluation', action='store_true',
                       help='Disable comprehensive evaluation metrics')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable console progress output')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export detailed metrics to CSV file')
    parser.add_argument('--disable-molecular-similarity', action='store_true',
                       help='Skip expensive molecular similarity calculations')
    parser.add_argument('--advanced-metrics', action='store_true',
                       help='Include additional metrics (MAPE, extended uncertainty)')
    
    # Output
    parser.add_argument('-o', '--output', type=Path,
                       help='Output directory for results')
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Import functional API
    try:
        from ..learnm8 import run_active_learning
    except ImportError as e:
        console.print(f"[red]Failed to import LearnM8 functional API: {e}[/red]")
        sys.exit(1)
    
    # Validate input files
    if not args.compound_pool.exists():
        console.print(f"[red]Compound pool file not found: {args.compound_pool}[/red]")
        sys.exit(1)
    
    # Handle oracle auto-detection and validation
    if args.oracle is None:
        if str(args.compound_pool).endswith('.csv'):
            args.oracle = str(args.compound_pool)  # Auto-detect: use compound pool as oracle
            oracle_mode = "benchmark"
            console.print(f"[blue]Mode:[/blue] Benchmark (auto-detected CSV ground truth)")
        else:
            console.print(f"[red]Oracle parameter required when compound_pool is not a CSV file[/red]")
            sys.exit(1)
    else:
        # Oracle was explicitly provided
        oracle_mode = "benchmark" if args.oracle.endswith('.csv') else "run"
        
        if oracle_mode == "benchmark":
            oracle_path = Path(args.oracle)
            if not oracle_path.exists():
                console.print(f"[red]Oracle CSV file not found: {oracle_path}[/red]")
                sys.exit(1)
            console.print(f"[blue]Mode:[/blue] Benchmark (CSV ground truth)")
        else:
            if ':' not in args.oracle:
                console.print(f"[red]Invalid oracle specification: {args.oracle}[/red]")
                console.print("[yellow]Use format: module.py:function_name[/yellow]")
                sys.exit(1)
            module_path, _ = args.oracle.split(':', 1)
            if not Path(module_path).exists():
                console.print(f"[red]Oracle module not found: {module_path}[/red]")
                sys.exit(1)
            console.print(f"[blue]Mode:[/blue] Run (Python oracle)")
    
    # Determine cycle specification (NEW Alternative 1 API)
    cycles_list = None
    
    if args.cycles_spec:
        # Parse explicit cycle specification string
        console.print("[green]Using explicit cycle specification (Alternative 1)[/green]")
        cycles_list = parse_cycle_spec(args.cycles_spec)
        
    elif args.schedule:
        # Use predefined schedule with simple cycles
        console.print(f"[green]Using predefined schedule: {args.schedule}[/green]")
        if args.schedule == 'quick':
            cycles_list = [('greedy', 0.1)] * 5
        elif args.schedule == 'standard':
            cycles_list = [('greedy', 0.1)] * 10
        elif args.schedule == 'intensive':
            cycles_list = [('greedy', 0.05)] * 20
        elif args.schedule == 'diversity':
            # Mixed diversity methods schedule
            cycles_list = [
                ('random', 0.02),      # Initial random exploration
                ('bitbirch', 0.01),    # Molecular diversity (if available)
                ('pca_dbscan', 0.01),  # Fast diversity
                ('umap_dbscan', 0.01), # High-quality diversity
                ('greedy', 0.005),     # Exploitation
                ('umap_kmeans', 0.01), # Alternative diversity method
                ('greedy', 0.005),     # Final exploitation
            ]
    
    # Parse pruning parameters
    pruning_params = None
    if args.pruning_params:
        try:
            pruning_params = json.loads(args.pruning_params)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in --pruning-params: {e}[/red]")
            sys.exit(1)
    
    # Parse acquisition parameters
    acquisition_params = None
    if args.acquisition_params:
        try:
            acquisition_params = json.loads(args.acquisition_params)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in --acquisition-params: {e}[/red]")
            sys.exit(1)
    
    # Handle score direction auto-detection
    if args.score_direction == 'auto':
        # Default to 'higher' for now (auto-detection can be added later)
        score_direction = 'higher'
        console.print("[yellow]Auto-detecting score direction: defaulting to 'higher' (better scores are higher)[/yellow]")
    else:
        score_direction = args.score_direction
    
    # Display experiment info
    console.print(f"[blue]Data:[/blue] {args.compound_pool}")
    console.print(f"[blue]Oracle:[/blue] {args.oracle}")
    console.print(f"[blue]Target:[/blue] {args.target_column}")
    console.print(f"[blue]Score Direction:[/blue] {score_direction}")
    console.print(f"[blue]Learner:[/blue] {args.learner}")
    
    if args.pruning_strategy:
        console.print(f"[blue]Pruning Strategy:[/blue] {args.pruning_strategy}")
        if pruning_params:
            console.print(f"[blue]Pruning Params:[/blue] {pruning_params}")
    else:
        console.print(f"[blue]Pruning:[/blue] Disabled (default)")
    
    if args.acquisition:
        console.print(f"[blue]Default Acquisition:[/blue] {args.acquisition}")
        if acquisition_params:
            console.print(f"[blue]Acquisition Params:[/blue] {acquisition_params}")
    else:
        console.print(f"[blue]Acquisition:[/blue] Determined by cycle specification")
    
    if cycles_list:
        console.print(f"[blue]Cycles:[/blue] {len(cycles_list)} explicit cycles")
        console.print(f"[dim]  Cycle specification: {cycles_list[:3]}{'...' if len(cycles_list) > 3 else ''}[/dim]")
    else:
        console.print(f"[blue]Cycles:[/blue] {args.cycles} (legacy mode)")
        console.print(f"[blue]Batch Fraction:[/blue] {args.batch_fraction}")
    
    # Run experiment
    try:
        console.print("\n[yellow]Starting active learning experiment...[/yellow]")
        
        # Load compound pool CSV file into DataFrame
        import pandas as pd
        console.print(f"[yellow]Loading compound pool: {args.compound_pool}[/yellow]")
        compound_pool_df = pd.read_csv(args.compound_pool)
        
        # Validate required columns
        required_columns = ['ID', 'SMILES']
        missing_columns = [col for col in required_columns if col not in compound_pool_df.columns]
        if missing_columns:
            console.print(f"[red]Missing required columns in compound pool: {missing_columns}[/red]")
            console.print(f"[yellow]Available columns: {list(compound_pool_df.columns)}[/yellow]")
            sys.exit(1)
        
        # Create Oracle instance based on mode
        if oracle_mode == "benchmark":
            from ..oracles.csv_oracle import CSVOracle
            oracle_instance = CSVOracle(args.oracle)
        else:
            from ..oracles.python_oracle import PythonOracle
            module_path, function_name = args.oracle.split(':', 1)
            oracle_instance = PythonOracle(module_path, function_name)
        
        console.print(f"[yellow]Oracle created: {oracle_instance.__class__.__name__}[/yellow]")
        
        # Prepare parameters for the new functional API
        experiment_params = {
            'compound_pool': compound_pool_df,
            'oracle': oracle_instance,
            'target_column': args.target_column,
            'learner': args.learner,
            'initial_size': args.initial_size,
            'output_dir': args.output,
            'random_state': args.random_state,
            'score_direction': score_direction,
            # Pruning parameters
            'pruning_strategy': args.pruning_strategy,
            'pruning_params': pruning_params,
            # Acquisition parameters
            'default_acquisition': args.acquisition,
            'acquisition_params': acquisition_params,
            # Evaluation parameters
            'enable_evaluation': not args.disable_evaluation,
            'console_output': not args.quiet,
            'export_csv': args.export_csv,
        }
        
        if cycles_list:
            # Use NEW Alternative 1 API with explicit cycles
            experiment_params['cycles'] = cycles_list
        else:
            # Use LEGACY parameters (converted to cycles internally)
            experiment_params['n_cycles'] = args.cycles
            experiment_params['batch_fraction'] = args.batch_fraction
        
        results = run_active_learning(**experiment_params)

        # Display results (updated for API)
        console.print(f"\n[green]✓ Active learning completed successfully![/green]")
        console.print(f"[blue]Results saved to:[/blue] {results['output_dir']}")
        console.print(f"[blue]Total cycles:[/blue] {results['total_cycles']}")
        console.print(f"[blue]Compounds labeled:[/blue] {len(results['labeled_data'])}")
        console.print(f"[blue]Remaining compounds:[/blue] {len(results['unlabeled_data'])}")
        
        # Display basic metrics from final cycle
        if results['cycle_metrics']:
            final_metrics = results['cycle_metrics'][-1]
            console.print("\n[bold cyan]Final Cycle Metrics:[/bold cyan]")
            
            # Show core cycle metrics
            console.print(f"[blue]Strategy:[/blue] {final_metrics.get('strategy', 'Unknown')}")
            console.print(f"[blue]Selected compounds:[/blue] {final_metrics.get('selected_count', 0)}")
            console.print(f"[blue]Cumulative labeled:[/blue] {final_metrics.get('cumulative_labeled', 0)}")
            
            # Show prediction statistics if available
            if 'prediction_mean' in final_metrics and final_metrics['prediction_mean'] is not None:
                console.print(f"[blue]Prediction mean:[/blue] {final_metrics['prediction_mean']:.3f}")
            if 'prediction_std' in final_metrics and final_metrics['prediction_std'] is not None:
                console.print(f"[blue]Prediction std:[/blue] {final_metrics['prediction_std']:.3f}")
            
            # Show uncertainty statistics if available
            if 'uncertainty_mean' in final_metrics and final_metrics['uncertainty_mean'] is not None:
                console.print(f"[blue]Uncertainty mean:[/blue] {final_metrics['uncertainty_mean']:.3f}")
            
            # Show measured value statistics
            if 'measured_mean' in final_metrics and final_metrics['measured_mean'] is not None:
                console.print(f"[blue]Measured mean:[/blue] {final_metrics['measured_mean']:.3f}")
            if 'measured_std' in final_metrics and final_metrics['measured_std'] is not None:
                console.print(f"[blue]Measured std:[/blue] {final_metrics['measured_std']:.3f}")
            if 'measured_max' in final_metrics and final_metrics['measured_max'] is not None:
                console.print(f"[blue]Measured max:[/blue] {final_metrics['measured_max']:.3f}")
            
            # Show pruning statistics if available
            if 'pruning_strategy' in final_metrics and final_metrics['pruning_strategy'] is not None:
                console.print(f"[blue]Pruning strategy:[/blue] {final_metrics['pruning_strategy']}")
                if 'pruned_count' in final_metrics and final_metrics['pruned_count'] > 0:
                    console.print(f"[blue]Compounds pruned:[/blue] {final_metrics['pruned_count']} "
                                f"({final_metrics['pruned_count']/final_metrics['original_pool_size']*100:.1f}%)")
                    console.print(f"[blue]Pool size after pruning:[/blue] {final_metrics['pruned_pool_size']}")
            else:
                console.print(f"[blue]Pruning:[/blue] None")
        
        # Display experiment summary
        if 'experiment_summary' in results and results['experiment_summary']:
            summary = results['experiment_summary']
            console.print(f"\n[blue]Total time:[/blue] {summary.get('total_time_seconds', 0):.1f}s")
            console.print(f"[blue]Experiment type:[/blue] {summary.get('experiment_type', 'pure_functional')}")
            console.print(f"[blue]Learner:[/blue] {summary.get('learner_type', 'Unknown')}")
            console.print(f"[blue]Oracle:[/blue] {summary.get('oracle_type', 'Unknown')}")
            console.print(f"[blue]Featurizer:[/blue] {summary.get('featurizer_type', 'Unknown')}")
            
            # Show final performance if available
            if 'final_measured_mean' in summary and summary['final_measured_mean'] is not None:
                console.print(f"[blue]Final measured mean:[/blue] {summary['final_measured_mean']:.3f}")
        
        # Show cycle summary
        if results['cycle_metrics']:
            strategies_used = {}
            total_selected = 0
            for metrics in results['cycle_metrics']:
                strategy = metrics.get('strategy', 'unknown')
                strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
                total_selected += metrics.get('selected_count', 0)
            
            console.print(f"\n[bold cyan]Cycle Summary:[/bold cyan]")
            console.print(f"[blue]Total compounds selected:[/blue] {total_selected}")
            console.print(f"[blue]Strategies used:[/blue]")
            for strategy, count in strategies_used.items():
                console.print(f"  {strategy}: {count} cycles")
        
        # Display CSV export information if files were created
        if args.export_csv and 'csv_files' in results and results['csv_files']:
            console.print(f"\n[bold cyan]CSV Files Exported:[/bold cyan]")
            csv_files = results['csv_files']
            
            if 'cycle_metrics' in csv_files:
                console.print(f"[blue]📊 Cycle metrics:[/blue] {csv_files['cycle_metrics']}")
            
            if 'predictions_by_cycle' in csv_files:
                console.print(f"[blue]🎯 Predictions by cycle:[/blue] {csv_files['predictions_by_cycle']}")
            
            if 'selection_history' in csv_files:
                console.print(f"[blue]📋 Selection history:[/blue] {csv_files['selection_history']}")
            
            if 'best_compounds' in csv_files:
                console.print(f"[blue]🏆 Best compounds:[/blue] {csv_files['best_compounds']}")
            
            # Count and display summary
            total_csv_files = len(csv_files)
            console.print(f"[green]✓ {total_csv_files} CSV files exported successfully[/green]")
        
        console.print(f"\n[green]📁 All results saved to: {results['output_dir']}[/green]")
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Experiment interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Experiment failed:[/red] {e}")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
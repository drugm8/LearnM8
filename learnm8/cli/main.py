"""Comprehensive CLI for LearnM8 active learning.

Provides three subcommands:
- run: Execute active learning experiment
- list: List available components (learners, acquisition, featurizers, schedules)
- validate: Validate compound pool before running

Supports multiple input modes:
- Simple: Basic parameters (n_cycles, batch_fraction)
- Advanced: Cycle specification string ("random:0.02 greedy:0.01*5")
- Config file: YAML or JSON configuration
- Predefined schedules: quick, standard, intensive, diverse

Features:
- Rich output with tables and progress bars
- Auto-detection of oracle and mode
- Clear error messages
- Config file support (YAML/JSON)
"""

import argparse
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install
import polars as pl

import yaml

from learnm8.api import run_active_learning
from learnm8.core.validation import validate_compound_pool
from learnm8.core.config import parse_cycle_spec, CycleConfig
from learnm8.acquisition import list_acquisition_functions
from learnm8.utils.file_loaders import load_compound_file
from learnm8.utils.logging import configure_learnm8_logging

install()
console = Console()


EPILOG_TEXT = """
Examples:
  # Simple benchmark mode (CSV oracle auto-detected from compound_pool)
  learnm8 run compounds.csv --target Activity --featurizer morgan --learner rf

  # With explicit oracle
  learnm8 run compounds.csv oracle.csv --target Activity --featurizer morgan --learner rf

  # Advanced: custom cycle schedule
  learnm8 run compounds.csv --target Activity --featurizer morgan --learner rf \\
    --cycles "random:0.02 greedy:0.01*5 ucb:0.01*3"

  # With pruning
  learnm8 run compounds.csv --target Activity --featurizer morgan --learner rf \\
    --pruning-fraction 0.3

  # From config file
  learnm8 run --config experiment.yaml

  # Predefined schedule
  learnm8 run compounds.csv --target Activity --featurizer morgan --learner rf \\
    --schedule intensive

  # List available components
  learnm8 list learners
  learnm8 list acquisition

  # Validate compounds
  learnm8 validate compounds.csv --featurizer morgan -o validation_results/

Oracle Auto-Detection:
  - When oracle is omitted, the compound_pool CSV file is used as the oracle (benchmark mode)
  - CSV oracles automatically trigger benchmark mode (enables discovery/ranking metrics using ground truth)
  - Python oracles (module.py:function) automatically trigger run mode (basic metrics only)

Note: Both modes predict on unlabeled compounds only for efficiency. Benchmark mode differs
only in metric computation (includes discovery and ranking metrics using ground truth reference).

Config File Precedence:
  - Config files provide default values for all parameters
  - CLI flags override config values when explicitly provided
  - Use --config to load YAML/JSON configuration files

For more information: https://github.com/volkamerlab/LearnM8
"""


def load_config_file(config_path: Path) -> dict:
    """Load configuration from YAML or JSON file.

    Supports .yaml, .yml, and .json files. YAML files require PyYAML to be installed.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ImportError: If YAML file provided but PyYAML not installed
        ValueError: If unsupported file format
        Exception: If file parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    try:
        if suffix in ['.yaml', '.yml']:
            with open(config_path) as f:
                return yaml.safe_load(f)
        elif suffix == '.json':
            with open(config_path) as f:
                return json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to parse config file: {e}") from e


def get_predefined_schedule(name: str) -> list:
    """Return predefined cycle schedules.

    Available schedules:
    - quick: 5 cycles - fast exploration
    - standard: 10 cycles - balanced approach
    - intensive: 20 cycles - thorough screening
    - diverse: 10 cycles - mixed diversity methods

    Args:
        name: Schedule name

    Returns:
        List of CycleConfig objects

    Raises:
        ValueError: If schedule name not recognized
    """
    available_acquisitions = set(list_acquisition_functions())

    schedules = {
        'quick': [
            CycleConfig('random', n_cycles=1, batch_fraction=0.02),
            CycleConfig('greedy', n_cycles=4, batch_fraction=0.01)
        ],
        'standard': [
            CycleConfig('random', n_cycles=1, batch_fraction=0.02),
            CycleConfig('greedy', n_cycles=9, batch_fraction=0.01)
        ],
        'intensive': [
            CycleConfig('random', n_cycles=1, batch_fraction=0.01),
            CycleConfig('greedy', n_cycles=19, batch_fraction=0.005)
        ],
    }

    if 'bitbirch' in available_acquisitions:
        schedules['diverse'] = [
            CycleConfig('random', n_cycles=1, batch_fraction=0.02),
            CycleConfig('bitbirch', n_cycles=2, batch_fraction=0.01),
            CycleConfig('greedy', n_cycles=3, batch_fraction=0.01),
            CycleConfig('ucb', n_cycles=2, batch_fraction=0.01),
            CycleConfig('greedy', n_cycles=2, batch_fraction=0.01)
        ]
    else:
        console.print("[yellow]Warning:[/yellow] BitBIRCH not available; using random sampling in 'diverse' schedule")
        schedules['diverse'] = [
            CycleConfig('random', n_cycles=1, batch_fraction=0.02),
            CycleConfig('random', n_cycles=2, batch_fraction=0.01),
            CycleConfig('greedy', n_cycles=3, batch_fraction=0.01),
            CycleConfig('ucb', n_cycles=2, batch_fraction=0.01),
            CycleConfig('greedy', n_cycles=2, batch_fraction=0.01)
        ]

    if name not in schedules:
        available = ', '.join(sorted(schedules.keys()))
        raise ValueError(
            f"Unknown schedule '{name}'. Available schedules: {available}"
        )

    return schedules[name]


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with three subcommands.

    Subcommands:
    - run: Execute active learning experiment
    - list: List available components
    - validate: Validate compound pool

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog='learnm8',
        description='LearnM8: Active Learning for Molecular Screening',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG_TEXT
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    run_parser = subparsers.add_parser('run', help='Run active learning experiment')

    run_parser.add_argument(
        'compound_pool',
        nargs='?',
        type=Path,
        default=None,
        help='Compound file (CSV/SDF/SMI) with ID and SMILES. Supported: .csv, .sdf, .smi'
    )

    run_parser.add_argument(
        'oracle',
        nargs='?',
        default=None,
        help='Oracle specification (auto-detect if omitted)'
    )

    core_group = run_parser.add_argument_group('Core Configuration')
    core_group.add_argument(
        '--target', '--target-column',
        dest='target_col',
        required=True,
        help='Target property column'
    )
    # Get available featurizers from registry
    from learnm8.features import list_available_featurizers
    available_featurizers = list_available_featurizers()
    featurizer_choices = available_featurizers['all']

    core_group.add_argument(
        '--featurizer',
        required=True,
        choices=featurizer_choices,
        help='Featurizer type (2D or 3D molecular representations)'
    )
    # Get available learners from registry
    from learnm8.api import list_available_learners
    available_learners = list_available_learners()
    core_group.add_argument(
        '--learner',
        default='rf' if 'rf' in available_learners else (available_learners[0] if available_learners else 'rf'),
        choices=available_learners if available_learners else ['rf'],
        help='ML model (default: rf)'
    )
    core_group.add_argument(
        '--score-direction',
        default='higher',
        choices=['higher', 'lower'],
        help='Optimization direction (default: higher)'
    )

    cycle_group = run_parser.add_mutually_exclusive_group()
    cycle_group.add_argument(
        '--cycles',
        type=str,
        help='Cycle specification: string format "random:0.02 greedy:0.01*5" or list of dicts/CycleConfig from config file'
    )
    cycle_group.add_argument(
        '--schedule',
        choices=['quick', 'standard', 'intensive', 'diverse'],
        help='Predefined schedule'
    )
    cycle_group.add_argument(
        '--config',
        type=Path,
        help='YAML/JSON config file'
    )

    simple_group = run_parser.add_argument_group('Simple Cycle Parameters')
    simple_group.add_argument(
        '--n-cycles',
        type=int,
        default=10,
        help='Number of cycles (default: 10)'
    )
    simple_group.add_argument(
        '--batch-fraction',
        type=float,
        default=0.01,
        help='Fraction per cycle (default: 0.01)'
    )
    simple_group.add_argument(
        '--strategy',
        type=str,
        default='greedy',
        help='Acquisition strategy (default: greedy)'
    )
    simple_group.add_argument(
        '--initial-strategy',
        type=str,
        default='random',
        help='First cycle strategy (default: random)'
    )
    pruning_group = run_parser.add_argument_group('Pruning')
    pruning_group.add_argument(
        '--pruning-fraction',
        type=float,
        help='Fraction to prune (0.0-0.9)'
    )
    pruning_group.add_argument(
        '--pruning-strategy',
        type=str,
        default=None,
        help='Pruning strategy (default: None)'
    )

    acquisition_group = run_parser.add_argument_group('Acquisition Parameters')
    acquisition_group.add_argument(
        '--acquisition-params',
        type=str,
        help='JSON string of parameters'
    )

    output_group = run_parser.add_argument_group('Output')
    output_group.add_argument(
        '-o', '--output',
        type=Path,
        help='Output directory'
    )
    output_group.add_argument(
        '--cache-dir',
        type=Path,
        help='Cache directory for feature storage (default: output_dir/.cache)'
    )
    output_group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    advanced_group = run_parser.add_argument_group('Advanced')
    advanced_group.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    advanced_group.add_argument(
        '--mode',
        choices=['run', 'benchmark'],
        help='Execution mode (auto-detected if not specified)'
    )
    advanced_group.add_argument(
        '--prediction-batch-size',
        type=int,
        help='Prediction batch size for memory management (default: auto-calculated)'
    )

    list_parser = subparsers.add_parser('list', help='List available components')
    list_parser.add_argument(
        'component',
        choices=['learners', 'acquisition', 'featurizers', 'schedules'],
        help='Component type'
    )

    validate_parser = subparsers.add_parser('validate', help='Validate compound pool using datamol SMILES validation')
    validate_parser.add_argument(
        'compound_pool',
        type=Path,
        help='Compound file (CSV/SDF/SMI) with ID and SMILES. Supported: .csv, .sdf, .smi'
    )
    validate_parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output directory for validation report'
    )

    return parser


def cmd_run(args: argparse.Namespace):
    """Handle run subcommand.

    Args:
        args: Parsed command-line arguments
    """
    try:
        if args.config:
            console.print(f"[cyan]Loading config from:[/cyan] {args.config}")
            console.print("[dim]Note: Config values are used as defaults; CLI flags override config if provided[/dim]")
            config = load_config_file(args.config)

            CONFIG_PATH_KEYS = {'compound_pool', 'oracle', 'output'}
            for key, value in config.items():
                if key in CONFIG_PATH_KEYS and isinstance(value, str):
                    setattr(args, key, Path(value))
                else:
                    setattr(args, key, value)

        if args.compound_pool is None and not hasattr(args, 'compound_pool'):
            console.print("[red]Error:[/red] compound_pool is required (provide positional file or via config)")
            sys.exit(1)

        if args.compound_pool is None:
            console.print("[red]Error:[/red] compound_pool not found in config")
            sys.exit(1)

        if isinstance(args.compound_pool, str):
            args.compound_pool = Path(args.compound_pool)

        if not args.compound_pool.exists():
            console.print(f"[red]Error:[/red] Compound pool file not found: {args.compound_pool}")
            sys.exit(1)

        console.print(f"[cyan]Compound pool:[/cyan] {args.compound_pool}")

        cycles = None

        if args.cycles:
            if isinstance(args.cycles, str):
                console.print("[cyan]Using custom cycle specification[/cyan]")
                cycle_dicts = parse_cycle_spec(args.cycles)
                cycles = [CycleConfig(**d) for d in cycle_dicts]
            elif isinstance(args.cycles, list):
                if all(isinstance(c, dict) for c in args.cycles):
                    console.print("[cyan]Using cycle configuration from config (list of dicts)[/cyan]")
                    cycles = [CycleConfig(**c) for c in args.cycles]
                elif all(isinstance(c, CycleConfig) for c in args.cycles):
                    console.print("[cyan]Using cycle configuration from config (list of CycleConfig)[/cyan]")
                    cycles = args.cycles
                else:
                    console.print("[red]Error:[/red] Invalid --cycles format. Use string spec or list of dicts/CycleConfig.")
                    sys.exit(1)
            else:
                console.print("[red]Error:[/red] Unsupported cycles type. Must be string spec or list.")
                sys.exit(1)
        elif args.schedule:
            console.print(f"[cyan]Using predefined schedule:[/cyan] {args.schedule}")
            cycles = get_predefined_schedule(args.schedule)
        else:
            console.print(f"[cyan]Using simple mode[/cyan] ({args.n_cycles} cycles)")
            cycles = None

        table = Table(title="Experiment Configuration", show_header=False)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Compound Pool", str(args.compound_pool))
        table.add_row("Oracle", args.oracle or "Auto-detect")
        table.add_row("Target Column", args.target_col)
        table.add_row("Featurizer", args.featurizer)
        table.add_row("Learner", args.learner)
        table.add_row("Score Direction", args.score_direction)
        table.add_row("Total Cycles", str(len(cycles) if cycles else args.n_cycles))
        pruning_text = f"{args.pruning_fraction:.1%} per cycle" if args.pruning_fraction else "Disabled"
        table.add_row("Pruning", pruning_text)
        table.add_row("Mode", args.mode or "Auto-detect")

        console.print(table)

        acquisition_params = None
        if args.acquisition_params:
            try:
                acquisition_params = json.loads(args.acquisition_params)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error:[/red] Invalid JSON in --acquisition-params: {e}")
                sys.exit(1)

        if not args.quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running active learning...", total=None)

                try:
                    results = run_active_learning(
                        compound_pool=args.compound_pool,
                        oracle=args.oracle,
                        learner=args.learner,
                        target_col=args.target_col,
                        featurizer=args.featurizer,
                        cycles=cycles,
                        n_cycles=args.n_cycles,
                        batch_fraction=args.batch_fraction,
                        strategy=args.strategy,
                        initial_strategy=args.initial_strategy,
                        score_direction=args.score_direction,
                        mode=args.mode,
                        output_dir=args.output,
                        cache_dir=getattr(args, 'cache_dir', None),
                        random_state=args.random_state,
                        pruning_fraction=args.pruning_fraction,
                        pruning_strategy=args.pruning_strategy,
                        acquisition_params=acquisition_params,
                        prediction_batch_size=getattr(args, 'prediction_batch_size', None)
                    )
                    progress.update(task, completed=True, description="[green]✓ Complete")
                except KeyboardInterrupt:
                    progress.stop()
                    console.print("\n[yellow]Experiment interrupted by user[/yellow]")
                    sys.exit(1)
        else:
            try:
                results = run_active_learning(
                    compound_pool=args.compound_pool,
                    oracle=args.oracle,
                    learner=args.learner,
                    target_col=args.target_col,
                    featurizer=args.featurizer,
                    cycles=cycles,
                    n_cycles=args.n_cycles,
                    batch_fraction=args.batch_fraction,
                    strategy=args.strategy,
                    initial_strategy=args.initial_strategy,
                    score_direction=args.score_direction,
                    mode=args.mode,
                    output_dir=args.output,
                    cache_dir=getattr(args, 'cache_dir', None),
                    random_state=args.random_state,
                    pruning_fraction=args.pruning_fraction,
                    pruning_strategy=args.pruning_strategy,
                    acquisition_params=acquisition_params,
                    prediction_batch_size=getattr(args, 'prediction_batch_size', None)
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Experiment interrupted by user[/yellow]")
                sys.exit(1)

        result_table = Table(title="Results Summary", show_header=False)
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", style="yellow")

        result_table.add_row("Output Directory", str(results['output_dir']))
        result_table.add_row("Total Cycles", str(len(results['cycle_metrics'])))
        result_table.add_row("Compounds Labeled", str(len(results['labeled_data'])))
        result_table.add_row("Compounds Remaining", str(len(results['unlabeled_data'])))
        result_table.add_row(
            "Valid Compounds",
            f"{len(results['validation_result'].valid_compounds)} "
            f"({results['validation_result'].success_rate:.1%})"
        )
        result_table.add_row(
            "Invalid Compounds",
            str(len(results['validation_result'].invalid_compounds))
        )

        console.print("\n", result_table)

        console.print("\n[bold cyan]Saved Files:[/bold cyan]")
        file_icons = {
            'compounds_final': '📊',
            'cycle_metrics': '📈',
            'selection_history': '📋',
            'validation_report': '⚠️',
            'config': '⚙️'
        }
        for file_type, file_path in results['saved_files'].items():
            icon = file_icons.get(file_type, '📄')
            console.print(f"  {icon} [blue]{file_type}:[/blue] {file_path}")

        console.print(f"\n[green]📁 All results saved to:[/green] {results['output_dir']}")

    except Exception as e:
        console.print_exception()
        sys.exit(1)


def cmd_list(args: argparse.Namespace):
    """Handle list subcommand.

    Args:
        args: Parsed command-line arguments
    """
    if args.component == 'learners':
        from learnm8.api import list_available_learners
        learners = list_available_learners()
        console.print("[bold cyan]Available Learners:[/bold cyan]")
        if not learners:
            console.print("  [yellow]No learners available (check dependencies)[/yellow]")
        for learner in learners:
            console.print(f"  • {learner}")

    elif args.component == 'acquisition':
        strategies = list_acquisition_functions()
        console.print("[bold cyan]Available Acquisition Strategies:[/bold cyan]")
        for strategy in strategies:
            console.print(f"  • {strategy}")

    elif args.component == 'featurizers':
        from learnm8.features import list_available_featurizers, FEATURIZER_REGISTRY
        from rich.table import Table

        available = list_available_featurizers()

        table = Table(title="Available Molecular Featurizers", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="blue", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Dimension", justify="right", style="yellow")
        table.add_column("Description", style="white")

        # Add 2D featurizers
        for name in available['2d']:
            if name in FEATURIZER_REGISTRY:
                featurizer_class = FEATURIZER_REGISTRY[name]
                try:
                    temp_feat = featurizer_class(n_jobs=1) if callable(featurizer_class) else featurizer_class
                    dim = temp_feat.get_dimension()
                    desc = {
                        'morgan': 'Circular fingerprints (ECFP4)',
                        'ecfp': 'Circular fingerprints (ECFP4)',
                        'ecfp6': 'Extended circular fingerprints (radius=3)',
                        'morgan_feat': 'Morgan with feature vectors',
                        'maccs': 'MACCS structural keys',
                        'avalon': 'Avalon fingerprints',
                        'atom_pair': 'Atom pair fingerprints',
                        'topological_torsion': 'Topological torsion fingerprints',
                        'rdkit': 'RDKit fingerprints',
                        'pubchem': 'PubChem fingerprints',
                        'mordred': 'Mordred molecular descriptors',
                        'descriptors': 'Mordred molecular descriptors'
                    }.get(name, '')
                    table.add_row(name, "2D", str(dim), desc)
                except Exception:
                    table.add_row(name, "2D", "N/A", "")

        # Add 3D featurizers
        for name in available['3d']:
            if name in FEATURIZER_REGISTRY:
                featurizer_class = FEATURIZER_REGISTRY[name]
                try:
                    temp_feat = featurizer_class(n_jobs=1) if callable(featurizer_class) else featurizer_class
                    dim = temp_feat.get_dimension()
                    desc = {
                        'whim': 'WHIM 3D molecular descriptors',
                        'usr': 'USR 3D shape descriptors',
                        'e3fp': 'Extended 3D fingerprints'
                    }.get(name, '')
                    table.add_row(name, "3D", str(dim), desc)
                except Exception:
                    table.add_row(name, "3D", "N/A", "")

        console.print(table)

    elif args.component == 'schedules':
        schedules = {
            'quick': '5 cycles - fast exploration',
            'standard': '10 cycles - balanced',
            'intensive': '20 cycles - thorough',
            'diverse': '10 cycles - mixed diversity methods'
        }
        console.print("[bold cyan]Available Schedules:[/bold cyan]")
        for name, desc in schedules.items():
            console.print(f"  • [blue]{name}:[/blue] {desc}")


def cmd_validate(args: argparse.Namespace):
    """Handle validate subcommand.

    Args:
        args: Parsed command-line arguments
    """
    try:
        if not args.compound_pool.exists():
            console.print(f"[red]Error:[/red] Compound pool file not found: {args.compound_pool}")
            sys.exit(1)

        console.print(f"[cyan]Validating:[/cyan] {args.compound_pool}")
        compounds = load_compound_file(args.compound_pool, progress=False)
        console.print(f"Total compounds: {len(compounds)}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Validating compounds...", total=None)
            result = validate_compound_pool(compounds, n_jobs=-1, progress=False)
            progress.update(task, completed=True, description="[green]✓ Complete")

        console.print("\n[green]✓ Validation complete[/green]")
        console.print(
            f"[blue]Valid compounds:[/blue] {len(result.valid_compounds)} "
            f"({result.success_rate:.1%})"
        )
        console.print(f"[blue]Invalid compounds:[/blue] {len(result.invalid_compounds)}")

        if len(result.invalid_compounds) > 0:
            console.print("\n[yellow]Sample errors:[/yellow]")
            for i, (cid, error) in enumerate(list(result.validation_errors.items())[:5]):
                console.print(f"  {cid}: {error}")
            if len(result.validation_errors) > 5:
                console.print(f"  ... and {len(result.validation_errors) - 5} more")

        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            invalid_df = result.invalid_compounds.clone()
            # Map errors robustly: use ID if available, otherwise use index
            if 'ID' in invalid_df.columns and not invalid_df['ID'].is_null().all():
                error_dict = result.validation_errors
                invalid_df = invalid_df.with_columns(
                    pl.col('ID').cast(pl.Utf8).map_elements(
                        lambda x: error_dict.get(x, "Unknown error"), return_dtype=pl.Utf8
                    ).alias('error')
                )
            else:
                # Use index if ID not available - create index column first
                error_dict = result.validation_errors
                invalid_df = invalid_df.with_row_count('__index__').with_columns(
                    pl.col('__index__').cast(pl.Utf8).map_elements(
                        lambda x: error_dict.get(str(x), "Unknown error"), return_dtype=pl.Utf8
                    ).alias('error')
                ).drop('__index__')

            report_path = output_dir / 'validation_report.csv'
            invalid_df.write_csv(report_path)

            console.print(f"\n[green]Report saved to:[/green] {report_path}")

    except Exception as e:
        console.print_exception()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    configure_learnm8_logging(
        console_type='rich',
        level='INFO'
    )

    parser = create_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'validate':
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

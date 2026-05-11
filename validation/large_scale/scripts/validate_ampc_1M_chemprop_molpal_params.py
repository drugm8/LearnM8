#!/usr/bin/env python3
"""
Standalone validation script for AmpC 1M dataset using Chemprop with MolPAL-style hyperparameters.

This script provides a fair comparison by using hyperparameters matching MolPAL's MPNN implementation:
- FFN layers: 2 (MolPAL default) vs 1 (LearnM8 default)
- Batch size: 50 (MolPAL default) vs 32 (LearnM8 default)
- Learning rate: 1e-3 (MolPAL max_lr) vs 1e-4 (LearnM8 default)
- Early stopping: Disabled (MolPAL doesn't have it)
- Fine-tuning: Disabled (MolPAL doesn't have it)
- Validation split: 20% (MolPAL) vs 10% (LearnM8 default)

Strategy:
- Random (1 cycle) → Greedy (15 cycles)
- Cycles: 16 total
- Batch size: 0.1% per cycle (1,000 compounds)

Usage:
    python validation/scripts/validate_ampc_1M_chemprop_molpal_params.py

Output:
    validation/reports/large_scale/ampc_1M_chemprop_molpal_params/
        ├── data/ampc_1M_chemprop_molpal_params_<timestamp>/
        │   ├── compounds_final.csv
        │   ├── cycle_metrics.csv
        │   └── selection_history.csv
        ├── plots/ampc_1M_chemprop_molpal_params.png
        └── summary.txt

Expected Runtime: Variable (depends on GPU availability)
Memory Requirements: ~10-20 GB (GPU recommended, higher than default due to larger batch size)
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import warnings

from learnm8 import run_active_learning, CycleConfig
from learnm8.oracles import CSVOracle
from learnm8.learners.torch.chemprop_learner import ChempropLearner
from validation.lib import (
    load_validation_dataset,
    get_dataset_path,
    get_dataset_info,
    create_comprehensive_validation_plot
)

warnings.filterwarnings('ignore')

from learnm8.utils.logging import configure_learnm8_logging
configure_learnm8_logging(level='INFO')


def print_header():
    print("=" * 80)
    print("  AmpC 1M Chemprop with MolPAL Hyperparameters Validation")
    print("=" * 80)
    print()


def check_dataset_exists(dataset_path):
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print()
        print("Please ensure the AmpC 1M dataset is available at:")
        print(f"  {dataset_path}")
        sys.exit(1)
    print(f"✓ Dataset found: {dataset_path}")


def print_configuration(compounds_count):
    print()
    print("Configuration:")
    print(f"  Dataset: AmpC 1M ({compounds_count:,} compounds)")
    print(f"  Learner: Chemprop with MolPAL hyperparameters")
    print()
    print("  MolPAL-style Hyperparameters:")
    print(f"    • FFN layers: 2 (vs LearnM8 default: 1)")
    print(f"    • Batch size: 50 (vs LearnM8 default: 32)")
    print(f"    • Learning rate: 1e-3 (vs LearnM8 default: 1e-4)")
    print(f"    • Validation split: 20% (vs LearnM8 default: 10%)")
    print(f"    • Early stopping: Disabled (MolPAL doesn't have it)")
    print(f"    • Fine-tuning: Disabled (MolPAL doesn't have it)")
    print()
    print(f"  Strategy: Random (1 cycle) → Greedy (15 cycles)")
    print(f"  Cycles: 16 total")
    print(f"  Batch size: 0.1% (1,000 compounds/cycle)")
    print(f"  Total to label: 16,000 compounds")
    print()


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def save_summary(results, output_dir, elapsed_time, compounds_count):
    summary_path = output_dir / 'summary.txt'

    final_metrics = results['cycle_metrics'][-1]

    with open(summary_path, 'w') as f:
        f.write("AmpC 1M Chemprop with MolPAL Hyperparameters Validation Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Dataset: AmpC 1M ({compounds_count:,} compounds)\n")
        f.write(f"  Strategy: Random (1 cycle) → Greedy (15 cycles)\n")
        f.write(f"  Learner: Chemprop with MolPAL hyperparameters\n")
        f.write(f"  Cycles: {len(results['cycle_metrics'])}\n")
        f.write(f"  Runtime: {format_time(elapsed_time)}\n\n")

        f.write("MolPAL-style Hyperparameters:\n")
        f.write("  • FFN layers: 2 (vs LearnM8 default: 1)\n")
        f.write("  • Batch size: 50 (vs LearnM8 default: 32)\n")
        f.write("  • Learning rate: 1e-3 (vs LearnM8 default: 1e-4)\n")
        f.write("  • Validation split: 20% (vs LearnM8 default: 10%)\n")
        f.write("  • Early stopping: Disabled\n")
        f.write("  • Fine-tuning: Disabled\n\n")

        f.write("Final Performance Metrics:\n")
        f.write(f"  Total Labeled: {final_metrics.get('cumulative_labeled', 'N/A'):,}\n")
        f.write(f"  Top-10 Discovery: {final_metrics.get('top_10_discovery', 0):.2f}%\n")
        f.write(f"  Top-100 Discovery: {final_metrics.get('top_100_discovery', 0):.2f}%\n")
        f.write(f"  Top-1000 Discovery: {final_metrics.get('top_1000_discovery', 0):.2f}%\n")
        f.write(f"  Top-0.1% Discovery: {final_metrics.get('top_0_1_pct_discovery', 0):.2f}%\n")
        f.write(f"  Top-1% Discovery: {final_metrics.get('top_1_pct_discovery', 0):.2f}%\n")
        f.write(f"  Top-10% Discovery: {final_metrics.get('top_10_pct_discovery', 0):.2f}%\n")
        f.write(f"  Cumulative Score Ratio: {final_metrics.get('cumulative_avg_score_ratio', 1.0):.3f}\n")
        f.write(f"  Batch Score Ratio: {final_metrics.get('batch_avg_score_ratio', 1.0):.3f}\n\n")

        f.write("Model Performance:\n")
        if 'unlabeled_spearman_correlation' in final_metrics:
            f.write(f"  Spearman Correlation: {final_metrics['unlabeled_spearman_correlation']:.3f}\n")
        if 'unlabeled_top_100_overlap' in final_metrics:
            f.write(f"  Top-100 Overlap: {final_metrics['unlabeled_top_100_overlap']:.2f}%\n")
        if 'unlabeled_top_1000_overlap' in final_metrics:
            f.write(f"  Top-1000 Overlap: {final_metrics['unlabeled_top_1000_overlap']:.2f}%\n")

        best_value = final_metrics.get('best_so_far', 'N/A')
        f.write(f"\n  Best Value Found: {best_value}\n")

        f.write("\nOutput Files:\n")
        f.write(f"  Data: {results['output_dir']}/\n")
        f.write(f"  Plot: {output_dir}/plots/ampc_1M_chemprop_molpal_params.png\n")
        f.write(f"  Summary: {summary_path}\n")

    return summary_path


def check_existing_results(data_dir):
    """Check if all required result files exist in a directory."""
    required_files = ['compounds_final.csv', 'cycle_metrics.csv', 'selection_history.csv']
    return all((data_dir / f).exists() for f in required_files)


def load_existing_results(data_dir):
    """Load results from existing run."""
    import polars as pl

    compounds_df = pl.read_csv(data_dir / 'compounds_final.csv', comment_prefix='#')
    cycle_metrics_df = pl.read_csv(data_dir / 'cycle_metrics.csv', comment_prefix='#')

    return {
        'compounds_df': compounds_df,
        'cycle_metrics': cycle_metrics_df.to_dicts(),
        'output_dir': str(data_dir)
    }


def main():
    print_header()

    # Configuration
    DATASET_NAME = 'ampc_1000k'
    LEARNER_NAME = 'Chemprop (MolPAL params)'

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path('validation/reports/large_scale/ampc_1M_chemprop_molpal_params')
    data_dir_parent = base_output_dir / 'data'
    plots_dir = base_output_dir / 'plots'

    print(f"Output directory: {base_output_dir.resolve()}")
    print()

    # Check for existing results
    existing_runs = sorted(data_dir_parent.glob('ampc_1M_chemprop_molpal_params_*'))
    skip_run = False
    results = None
    data_output_dir = None

    if existing_runs:
        latest_run = existing_runs[-1]
        if check_existing_results(latest_run):
            print(f"✓ Found existing results: {latest_run.name}")
            print("  Skipping active learning run, will regenerate plots only...")
            print()
            skip_run = True
            results = load_existing_results(latest_run)
            data_output_dir = latest_run
        else:
            print(f"  Found incomplete run: {latest_run.name}")
            print("  Will create new run...")
            print()

    if not skip_run:
        data_output_dir = data_dir_parent / f'ampc_1M_chemprop_molpal_params_{timestamp}'

        # Check dataset exists
        try:
            dataset_path = get_dataset_path(DATASET_NAME)
            check_dataset_exists(dataset_path)
        except Exception as e:
            print(f"❌ ERROR: Failed to locate dataset: {e}")
            sys.exit(1)

        # Get dataset info for ID column
        try:
            dataset_info = get_dataset_info(DATASET_NAME)
            id_column = dataset_info['id_column']
        except Exception as e:
            print(f"❌ ERROR: Failed to get dataset info: {e}")
            sys.exit(1)

        # Load dataset
        print("Loading dataset...")
        try:
            compound_pool, metadata = load_validation_dataset(
                dataset_name=DATASET_NAME,
                clean_invalid_scores=True,
                random_state=42
            )
            print(f"✓ Loaded {len(compound_pool):,} compounds")
            print(f"  Target column: {metadata['target_column']}")
            print(f"  Score direction: {metadata['score_direction']}")
            print(f"  ID column: {id_column} (mapped to 'ID' in DataFrame)")
        except Exception as e:
            print(f"❌ ERROR: Failed to load dataset: {e}")
            sys.exit(1)

        # Print configuration
        print_configuration(len(compound_pool))

        # Setup oracle (CRITICAL: Use original ID column from dataset)
        print("Setting up oracle...")
        try:
            oracle = CSVOracle(str(dataset_path), id_column=id_column)
            print(f"✓ Oracle configured (ID column: '{id_column}')")
        except Exception as e:
            print(f"❌ ERROR: Failed to setup oracle: {e}")
            sys.exit(1)

        # Create learner with MolPAL-style hyperparameters
        print()
        print("Creating Chemprop learner with MolPAL hyperparameters...")
        try:
            learner = ChempropLearner(
                # Architecture (same as MolPAL)
                message_hidden_dim=300,
                depth=3,
                aggregation='mean',
                atom_messages=False,
                message_bias=False,

                # FFN configuration (MolPAL style)
                ffn_hidden_dim=300,
                ffn_num_layers=2,  # MolPAL default (vs LearnM8 default: 1)
                dropout=0.0,

                # Training configuration (MolPAL style)
                max_epochs=50,
                batch_size=50,  # MolPAL default (vs LearnM8 default: 32)
                learning_rate=1e-3,  # MolPAL max_lr (vs LearnM8 default: 1e-4)

                # Validation (MolPAL style)
                early_stopping=False,  # MolPAL doesn't have early stopping
                val_fraction=0.2,  # MolPAL uses 80/20 split (vs LearnM8 default: 0.1)

                # Features not in MolPAL
                enable_fine_tuning=False,  # Disabled for fair comparison
                batch_norm=False,
                random_state=42,
                accelerator='auto',
                enable_aggressive_gc=True  # Keep LearnM8's memory management
            )
            print("✓ Learner created with MolPAL hyperparameters")
            print()
        except Exception as e:
            print(f"❌ ERROR: Failed to create learner: {e}")
            sys.exit(1)

        # Run active learning
        print()
        print("Running active learning validation...")
        print("⏱  Estimated time: Variable (depends on GPU)")
        print("📊 Progress will be shown below...")
        print()

        start_time = time.time()

        try:
            cycles = [
                CycleConfig('random', n_cycles=1, batch_fraction=0.001),
                CycleConfig('greedy', n_cycles=15, batch_fraction=0.001)
            ]

            results = run_active_learning(
                compound_pool=compound_pool,
                oracle=oracle,
                learner=learner,  # Pass configured learner instance
                target_col=metadata['target_column'],
                cycles=cycles,
                score_direction=metadata['score_direction'],
                output_dir=str(data_output_dir),
                mode='benchmark',
                cache_dir=Path('validation/.shared_cache')
            )

            elapsed_time = time.time() - start_time

            print()
            print("=" * 80)
            print(f"✓ Active learning complete in {format_time(elapsed_time)}")
            print("=" * 80)
            print()

        except Exception as e:
            print()
            print(f"❌ ERROR: Active learning failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        elapsed_time = 0
        compound_pool = results['compounds_df']

    # Generate validation plot
    print("Generating validation plot...")
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'ampc_1M_chemprop_molpal_params.png'

        strategy_config = {
            'name': 'Mixed (Random→Greedy) - MolPAL params',
            'param_name': None
        }

        create_comprehensive_validation_plot(
            result=results,
            strategy_config=strategy_config,
            param_value=None,
            output_path=plot_path,
            dpi=300,
            dataset_name='AmpC 1M',
            learner_name=LEARNER_NAME
        )

        print(f"✓ Validation plot saved: {plot_path.resolve()}")
    except Exception as e:
        print(f"⚠  Warning: Failed to generate plot: {e}")

    # Save summary
    print("Generating summary...")
    try:
        summary_path = save_summary(results, base_output_dir, elapsed_time, len(compound_pool))
        print(f"✓ Summary saved: {summary_path.resolve()}")
    except Exception as e:
        print(f"⚠  Warning: Failed to generate summary: {e}")

    # Print final results
    print()
    print("=" * 80)
    print("  Validation Complete!")
    print("=" * 80)
    print()
    print("Results Summary:")

    final_metrics = results['cycle_metrics'][-1]
    print(f"  Total Labeled: {final_metrics.get('cumulative_labeled', 'N/A'):,} / {len(compound_pool):,}")
    print(f"  Top-10 Discovery: {final_metrics.get('top_10_discovery', 0):.2f}%")
    print(f"  Top-100 Discovery: {final_metrics.get('top_100_discovery', 0):.2f}%")
    print(f"  Top-1000 Discovery: {final_metrics.get('top_1000_discovery', 0):.2f}%")
    print(f"  Top-0.1% Discovery: {final_metrics.get('top_0_1_pct_discovery', 0):.2f}%")
    print(f"  Top-1% Discovery: {final_metrics.get('top_1_pct_discovery', 0):.2f}%")
    print(f"  Top-10% Discovery: {final_metrics.get('top_10_pct_discovery', 0):.2f}%")
    print(f"  Cumulative Score Ratio: {final_metrics.get('cumulative_avg_score_ratio', 1.0):.3f}")
    print()

    print("Output Files:")
    print(f"  📁 Data: {data_output_dir.resolve()}/")
    print(f"  📊 Plot: {plot_path.resolve()}")
    print(f"  📄 Summary: {summary_path.resolve()}")
    print()

    print("Comparison Notes:")
    print("  This run uses MolPAL-style hyperparameters for direct comparison:")
    print("  • Deeper FFN (2 layers vs 1)")
    print("  • Larger batches (50 vs 32)")
    print("  • Higher learning rate (1e-3 vs 1e-4)")
    print("  • No early stopping (trains full 50 epochs)")
    print("  • Larger validation split (20% vs 10%)")
    print()


if __name__ == '__main__':
    main()

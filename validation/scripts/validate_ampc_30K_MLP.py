#!/usr/bin/env python3
"""
Standalone validation script for AmpC 30K dataset using UCB acquisition.

This script runs a complete active learning validation on the AmpC 30K dataset:
- Strategy: UCB (Upper Confidence Bound) with beta=1.0
- Learner: RF Ensemble (Random Forest Ensemble)
- Featurizer: Morgan fingerprints
- Cycles: 10
- Batch size: 1% per cycle (5,000 compounds)

Usage:
    python validation/scripts/validate_ampc_30K_ucb.py

Output:
    validation/reports/ampc_30K_ucb_validation/
        ├── data/ampc_30K_ucb_beta_1.0_<timestamp>/
        │   ├── compounds_final.csv
        │   ├── cycle_metrics.csv
        │   └── selection_history.csv
        ├── plots/ampc_30K_ucb_validation.png
        └── summary.txt

Expected Runtime: 40-60 minutes (first run with cache generation)
Memory Requirements: ~8-12 GB
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from learnm8 import run_active_learning, CycleConfig
from learnm8.oracles import CSVOracle
from validation.lib import (
    load_validation_dataset,
    get_dataset_path,
    get_dataset_info,
    create_comprehensive_validation_plot
)

warnings.filterwarnings('ignore')

from learnm8 import setup_logging
setup_logging(level='INFO')


def print_header():
    print("=" * 80)
    print("  AmpC 30K UCB Validation")
    print("=" * 80)
    print()


def check_dataset_exists(dataset_path):
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print()
        print("Please ensure the AmpC 30K dataset is available at:")
        print(f"  {dataset_path}")
        sys.exit(1)
    print(f"✓ Dataset found: {dataset_path}")


def print_configuration(beta, n_cycles, batch_fraction, compounds_count):
    print()
    print("Configuration:")
    print(f"  Dataset: AmpC 30K ({compounds_count:,} compounds)")
    print(f"  Strategy: UCB (beta={beta})")
    print(f"  Learner: RF Ensemble")
    print(f"  Featurizer: Morgan fingerprints")
    print(f"  Cycles: {n_cycles}")
    print(f"  Batch size: {batch_fraction:.1%} ({int(compounds_count * batch_fraction):,} compounds/cycle)")
    print(f"  Total to label: {int(compounds_count * batch_fraction * n_cycles):,} compounds")
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
        f.write("AmpC 30K UCB Validation Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Dataset: AmpC 30K ({compounds_count:,} compounds)\n")
        f.write(f"  Strategy: UCB (beta=1.0)\n")
        f.write(f"  Learner: RF Ensemble\n")
        f.write(f"  Featurizer: Morgan fingerprints\n")
        f.write(f"  Cycles: {len(results['cycle_metrics'])}\n")
        f.write(f"  Runtime: {format_time(elapsed_time)}\n\n")

        f.write("Final Performance Metrics:\n")
        f.write(f"  Total Labeled: {final_metrics.get('cumulative_labeled', 'N/A'):,}\n")
        f.write(f"  Top-10 Discovery: {final_metrics.get('top_10_discovery', 0):.2f}%\n")
        f.write(f"  Top-100 Discovery: {final_metrics.get('top_100_discovery', 0):.2f}%\n")
        f.write(f"  Top-0.1% Discovery: {final_metrics.get('top_0_1_pct_discovery', 0):.2f}%\n")
        f.write(f"  Top-1% Discovery: {final_metrics.get('top_1_pct_discovery', 0):.2f}%\n")
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
        f.write(f"  Plot: {output_dir}/plots/ampc_30K_ucb_validation.png\n")
        f.write(f"  Summary: {summary_path}\n")

    return summary_path


def main():
    print_header()

    # Configuration
    DATASET_NAME = 'ampc_30k'
    BETA = 1.0
    N_CYCLES = 10
    BATCH_FRACTION = 0.01

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path('validation/reports/ampc_30k_MCD_validation')
    data_output_dir = base_output_dir / 'data' / f'ampc_30k_MCD_beta_{BETA}_{timestamp}'
    plots_dir = base_output_dir / 'plots'

    print(f"Output directory: {base_output_dir.resolve()}")
    print()

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
    print_configuration(BETA, N_CYCLES, BATCH_FRACTION, len(compound_pool))

    # Setup oracle (CRITICAL: Use original ID column from dataset)
    print("Setting up oracle...")
    try:
        oracle = CSVOracle(str(dataset_path), id_column=id_column)
        print(f"✓ Oracle configured (ID column: '{id_column}')")
    except Exception as e:
        print(f"❌ ERROR: Failed to setup oracle: {e}")
        sys.exit(1)

    # Run active learning
    print()
    print("Running active learning validation...")
    print("⏱  Estimated time: 40-60 minutes")
    print("📊 Progress will be shown below...")
    print()

    start_time = time.time()

    try:
        cycles = [CycleConfig('random', n_cycles=1, batch_fraction=0.01), CycleConfig('greedy', n_cycles=15, batch_fraction=0.01)]

        results = run_active_learning(
            compound_pool=compound_pool,
            oracle=oracle,
            learner='mixed_ensemble',
            target_col=metadata['target_column'],
            featurizer_type='morgan',
            cycles=cycles,
            score_direction=metadata['score_direction'],
            acquisition_params={'beta': BETA},
            output_dir=str(data_output_dir),
            mode='benchmark',
            cache_dir=Path(base_output_dir/'.ampc30K_cache')
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

    # Generate validation plot
    print("Generating validation plot...")
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'ampc_30K_ucb_validation.png'

        strategy_config = {
            'name': 'UCB',
            'param_name': 'beta'
        }

        create_comprehensive_validation_plot(
            result=results,
            strategy_config=strategy_config,
            param_value=BETA,
            output_path=plot_path,
            dpi=300
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
    print(f"  Cumulative Score Ratio: {final_metrics.get('cumulative_avg_score_ratio', 1.0):.3f}")
    print()

    print("Output Files:")
    print(f"  📁 Data: {data_output_dir.resolve()}/")
    print(f"  📊 Plot: {plot_path.resolve()}")
    print(f"  📄 Summary: {summary_path.resolve()}")
    print()


if __name__ == '__main__':
    main()

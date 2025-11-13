#!/usr/bin/env python3
"""
Validation script to assess uncertainty-error correlations for ensemble learners.

This script runs active learning experiments with different ensemble types on the
AMP 30K dataset and analyzes the correlation between predicted uncertainties and
actual prediction errors. This helps assess the quality of uncalibrated uncertainties.

Ensembles tested:
- RF Ensemble (Random Forest)
- XGB Ensemble (XGBoost)
- LR Ensemble (Linear Regression)
- DT Ensemble (Decision Tree)
- Mixed Ensemble (Multiple model types)

Outputs:
- Correlation plots for each ensemble (all cycles + sample cycles)
- Summary report with correlation statistics
- Raw data for further analysis

Usage:
    python validation/scripts/validate_uncertainty_correlations.py

Expected Runtime: ~3-5 hours (all 5 ensembles with cache)
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from learnm8 import run_active_learning, extract_features
from learnm8.oracles import CSVOracle
from learnm8.learners import (
    RFEnsemble, XGBEnsemble, LREnsemble, DTEnsemble, MixedEnsemble
)
from validation.lib import (
    load_validation_dataset,
    get_dataset_path,
    get_dataset_info
)

warnings.filterwarnings('ignore')
from learnm8 import setup_logging

# Configuration
DATASET_NAME = 'ampc_30k'
N_CYCLES = 10
BATCH_FRACTION = 0.01
RANDOM_STATE = 42

# Ensemble configurations
ENSEMBLES = {
    'rf_ensemble': {
        'name': 'Random Forest Ensemble',
        'class': RFEnsemble,
        'color': '#1f77b4'
    },
    'xgb_ensemble': {
        'name': 'XGBoost Ensemble',
        'class': XGBEnsemble,
        'color': '#ff7f0e'
    },
    'lr_ensemble': {
        'name': 'Linear Regression Ensemble',
        'class': LREnsemble,
        'color': '#2ca02c'
    },
    'dt_ensemble': {
        'name': 'Decision Tree Ensemble',
        'class': DTEnsemble,
        'color': '#d62728'
    },
    'mixed_ensemble': {
        'name': 'Mixed Ensemble',
        'class': MixedEnsemble,
        'color': '#9467bd'
    }
}


def print_header():
    """Print validation header."""
    print("=" * 80)
    print("  Uncertainty-Error Correlation Validation")
    print("=" * 80)
    print()
    print(f"Dataset: AMP 30K")
    print(f"Ensembles: {len(ENSEMBLES)}")
    print(f"Cycles: {N_CYCLES}")
    print(f"Batch Fraction: {BATCH_FRACTION:.1%}")
    print()


def check_existing_results(ensemble_key, output_base):
    """Check if results already exist for this ensemble."""
    output_dir = output_base / ensemble_key

    required_files = [
        'compounds_final.csv',
        'cycle_metrics.csv',
        'selection_history.csv'
    ]

    return all((output_dir / f).exists() for f in required_files)


def load_existing_results(ensemble_key, output_base):
    """Load existing experiment results."""
    output_dir = output_base / ensemble_key

    compounds_df = pl.read_csv(output_dir / 'compounds_final.csv', comment_prefix='#')
    cycle_metrics_df = pl.read_csv(output_dir / 'cycle_metrics.csv', comment_prefix='#')

    return {
        'compounds_df': compounds_df,
        'cycle_metrics': cycle_metrics_df.to_dicts(),
        'output_dir': output_dir
    }


def run_ensemble_experiment(ensemble_key, ensemble_config, compound_pool, oracle,
                           target_col, score_direction, cache_dir, output_base):
    """Run active learning experiment for one ensemble type."""
    print(f"\n{'='*60}")
    print(f"Running: {ensemble_config['name']}")
    print(f"{'='*60}")

    output_dir = output_base / ensemble_key

    start_time = time.time()

    try:
        results = run_active_learning(
            compound_pool=compound_pool.clone(),
            oracle=oracle,
            learner=ensemble_key,
            target_col=target_col,
            featurizer_type='morgan',
            n_cycles=N_CYCLES,
            batch_fraction=BATCH_FRACTION,
            score_direction=score_direction,
            mode='benchmark',
            output_dir=str(output_dir),
            cache_dir=cache_dir,
            random_state=RANDOM_STATE
        )

        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed/60:.1f} minutes")

        return results, elapsed

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_uncertainties_for_labeled(compounds_df, learner, features, target_col):
    """Re-predict on labeled compounds to get uncertainties."""
    labeled_df = compounds_df.filter(pl.col('status') == 'labeled')

    if len(labeled_df) == 0:
        return None, None, None

    # Get labeled compound features
    labeled_smiles = labeled_df['SMILES'].to_list()
    labeled_ids = labeled_df['ID'].to_list()

    # Get features for labeled compounds
    labeled_features = features.filter(pl.col('ID').is_in(labeled_ids))
    X = labeled_features.drop(['ID']).to_numpy()

    # Get predictions and uncertainties
    predictions, uncertainties = learner.predict(X)

    # Get ground truth
    ground_truth = labeled_df[target_col].to_numpy()

    return predictions, uncertainties, ground_truth


def analyze_uncertainty_error_correlation(results, ensemble_key, ensemble_config,
                                         cache_dir, target_col):
    """Analyze uncertainty-error correlation for completed AL experiment."""
    print(f"\nAnalyzing {ensemble_config['name']}...")

    compounds_df = results['compounds_df']
    labeled_df = compounds_df.filter(pl.col('status') == 'labeled')

    if len(labeled_df) == 0:
        print("  ⚠ No labeled compounds found")
        return None

    # Re-create learner and train on all labeled data
    print(f"  Re-training learner on {len(labeled_df)} labeled compounds...")
    learner = ENSEMBLES[ensemble_key]['class'](
        random_states=[RANDOM_STATE, RANDOM_STATE+81, RANDOM_STATE+314]
    )

    # Extract features for labeled compounds
    labeled_smiles = labeled_df['SMILES'].to_list()
    labeled_ids = labeled_df['ID'].to_list()

    from learnm8.utils.featurizers import extract_features
    features = extract_features(
        smiles_list=labeled_smiles,
        featurizer_type='morgan',
        cache_dir=str(cache_dir),
        n_jobs=-1
    )

    X = features
    y = labeled_df[target_col].to_numpy()

    # Train learner
    learner.train(X, y)

    # Get predictions and uncertainties
    print("  Getting predictions and uncertainties...")
    predictions, uncertainties = learner.predict(X)

    # Calculate errors
    errors = np.abs(predictions - y)

    # Add to dataframe for per-cycle analysis
    analysis_df = labeled_df.clone()
    analysis_df = analysis_df.with_columns([
        pl.Series('prediction', predictions),
        pl.Series('uncertainty', uncertainties),
        pl.Series('abs_error', errors)
    ])

    # Overall correlation
    overall_corr, overall_p = spearmanr(uncertainties, errors)
    print(f"  Overall correlation: {overall_corr:.3f} (p={overall_p:.2e})")

    # Per-cycle correlations (calculate for ALL cycles, not just sample)
    cycle_correlations = {}
    max_cycle = int(analysis_df['labeled_cycle'].max())
    for cycle in range(1, max_cycle + 1):
        cycle_df = analysis_df.filter(pl.col('labeled_cycle') == cycle)
        if len(cycle_df) >= 10:  # Need enough points for correlation
            cycle_unc = cycle_df['uncertainty'].to_numpy()
            cycle_err = cycle_df['abs_error'].to_numpy()
            corr, p = spearmanr(cycle_unc, cycle_err)
            cycle_correlations[cycle] = (corr, p)
            if cycle in [1, 3, 5, 10]:  # Only print sample cycles
                print(f"  Cycle {cycle} correlation: {corr:.3f} (n={len(cycle_df)})")

    return {
        'ensemble_key': ensemble_key,
        'ensemble_name': ensemble_config['name'],
        'analysis_df': analysis_df,
        'overall_correlation': overall_corr,
        'overall_p_value': overall_p,
        'cycle_correlations': cycle_correlations,
        'n_labeled': len(analysis_df)
    }


def plot_uncertainty_error_correlation(analysis_result, output_dir):
    """Generate correlation plots for one ensemble."""
    ensemble_name = analysis_result['ensemble_name']
    ensemble_key = analysis_result['ensemble_key']
    df = analysis_result['analysis_df']
    color = ENSEMBLES[ensemble_key]['color']

    # Create figure with 2 rows: top=progression, bottom=sample cycles
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # Top: Correlation progression across cycles (spans all 4 columns)
    ax_all = fig.add_subplot(gs[0, :])

    # Extract cycle-by-cycle correlations
    cycle_correlations = analysis_result['cycle_correlations']
    if cycle_correlations:
        cycles = sorted(cycle_correlations.keys())
        correlations = [cycle_correlations[c][0] for c in cycles]

        # Plot correlation progression
        ax_all.plot(cycles, correlations, 'o-', color=color, linewidth=2,
                   markersize=8, label='Spearman ρ')
        ax_all.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_all.axhline(y=0.7, color='green', linestyle=':', alpha=0.5,
                      label='Strong (ρ>0.7)')
        ax_all.axhline(y=0.4, color='orange', linestyle=':', alpha=0.5,
                      label='Moderate (ρ>0.4)')

        ax_all.set_xlabel('Cycle', fontsize=12)
        ax_all.set_ylabel('Spearman Correlation (ρ)', fontsize=12)
        ax_all.set_title(
            f'{ensemble_name} - Uncertainty-Error Correlation Progression\n'
            f'Average ρ = {np.mean(correlations):.3f}',
            fontsize=14, fontweight='bold'
        )
        ax_all.grid(True, alpha=0.3)
        ax_all.legend(loc='best')
        ax_all.set_ylim(-0.1, 1.0)
    else:
        ax_all.text(0.5, 0.5, 'No correlation data available',
                   ha='center', va='center', transform=ax_all.transAxes,
                   fontsize=14)
        ax_all.set_title(f'{ensemble_name} - Correlation Progression',
                        fontsize=14, fontweight='bold')

    # Bottom: Sample cycles (1, 3, 5, 10) in 4 subplots
    sample_cycles = [1, 3, 5, 10]
    for idx, cycle in enumerate(sample_cycles):
        ax = fig.add_subplot(gs[1, idx])

        cycle_df = df.filter(pl.col('labeled_cycle') == cycle)

        if len(cycle_df) > 0:
            cycle_unc = cycle_df['uncertainty'].to_numpy()
            cycle_err = cycle_df['abs_error'].to_numpy()

            ax.scatter(cycle_unc, cycle_err, alpha=0.4, s=25, color=color)
            ax.set_xlabel('Uncertainty', fontsize=10)
            ax.set_ylabel('Abs Error', fontsize=10)

            # Get correlation for this cycle
            if cycle in analysis_result['cycle_correlations']:
                corr, p = analysis_result['cycle_correlations'][cycle]
                title = f'Cycle {cycle} (n={len(cycle_df)})\nρ = {corr:.3f}'
            else:
                title = f'Cycle {cycle} (n={len(cycle_df)})'

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add regression line if enough points
            if len(cycle_df) >= 10:
                slope, intercept, r_value, p_value, std_err = linregress(cycle_unc, cycle_err)
                x_line = np.linspace(cycle_unc.min(), cycle_unc.max(), 50)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r--', alpha=0.6)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Cycle {cycle}', fontsize=11)

    plt.suptitle(
        f'Uncertainty-Error Correlation Analysis: {ensemble_name}',
        fontsize=16, fontweight='bold', y=0.98
    )

    # Save plot
    plot_path = output_dir / f'{ensemble_key}_uncertainty_correlation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {plot_path}")

    return plot_path


def generate_summary_report(all_results, output_dir):
    """Generate summary report comparing all ensembles."""
    summary_path = output_dir / 'uncertainty_correlation_summary.txt'

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Uncertainty-Error Correlation Validation Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset: AMP 30K\n")
        f.write(f"Cycles: {N_CYCLES}\n")
        f.write(f"Batch Fraction: {BATCH_FRACTION:.1%}\n")
        f.write(f"Ensembles Tested: {len(ENSEMBLES)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Overall Correlation Results\n")
        f.write("-" * 80 + "\n\n")

        correlations = []
        for result in all_results:
            if result is not None:
                f.write(f"{result['ensemble_name']:30s}: ")
                f.write(f"ρ = {result['overall_correlation']:6.3f} ")
                f.write(f"(p={result['overall_p_value']:.2e}, n={result['n_labeled']})\n")
                correlations.append(result['overall_correlation'])

        if correlations:
            f.write(f"\nAverage correlation: {np.mean(correlations):.3f}\n")
            f.write(f"Std deviation:       {np.std(correlations):.3f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Per-Cycle Correlation Results\n")
        f.write("-" * 80 + "\n\n")

        for cycle in [1, 3, 5, 10]:
            f.write(f"\nCycle {cycle}:\n")
            for result in all_results:
                if result is not None and cycle in result['cycle_correlations']:
                    corr, p = result['cycle_correlations'][cycle]
                    f.write(f"  {result['ensemble_name']:30s}: ρ = {corr:6.3f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Interpretation:\n")
        f.write("=" * 80 + "\n\n")
        f.write("- Spearman ρ > 0.7: Strong positive correlation (good uncertainty quality)\n")
        f.write("- Spearman ρ 0.4-0.7: Moderate correlation (acceptable)\n")
        f.write("- Spearman ρ < 0.4: Weak correlation (poor uncertainty quality)\n\n")

        if correlations:
            avg_corr = np.mean(correlations)
            if avg_corr > 0.7:
                assessment = "GOOD - Strong correlation"
            elif avg_corr > 0.4:
                assessment = "MODERATE - Acceptable but can be improved"
            else:
                assessment = "POOR - Uncertainties need calibration"

            f.write(f"Overall Assessment: {assessment}\n")
            f.write(f"(Average correlation: {avg_corr:.3f})\n\n")

        f.write("Recommendations:\n")
        f.write("- If correlations are weak, implement post-hoc calibration\n")
        f.write("- Consider temperature scaling or isotonic regression\n")
        f.write("- Add calibration metrics (ECE, NLL) to evaluation\n")

    print(f"\nSummary report saved: {summary_path}")
    return summary_path


def main():
    """Main execution function."""
    setup_logging(level='INFO')
    print_header()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path('validation/reports/uncertainty_correlation_validation')
    data_dir = output_base / 'data' / timestamp
    plots_dir = output_base / 'plots' / timestamp
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Setup cache
    cache_dir = output_base / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_base}")
    print(f"Plots directory: {plots_dir}")
    print()

    # Load dataset
    print("Loading dataset...")
    try:
        dataset_path = get_dataset_path(DATASET_NAME)
        compound_pool, metadata = load_validation_dataset(
            dataset_name=DATASET_NAME,
            clean_invalid_scores=True,
            random_state=RANDOM_STATE
        )
        target_col = metadata['target_column']
        score_direction = metadata['score_direction']
        dataset_info = get_dataset_info(DATASET_NAME)
        id_column = dataset_info['id_column']

        print(f"✓ Loaded {len(compound_pool):,} compounds")
        print(f"  Target: {target_col}")
        print(f"  Direction: {score_direction}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    # Setup oracle
    oracle = CSVOracle(str(dataset_path), id_column=id_column)

    # Run experiments for each ensemble
    experiment_results = {}
    runtimes = {}
    skipped_count = 0
    completed_count = 0

    total_start = time.time()

    for ensemble_key, ensemble_config in ENSEMBLES.items():
        # Check if results already exist
        if check_existing_results(ensemble_key, data_dir):
            print(f"\n{'='*60}")
            print(f"Skipping: {ensemble_config['name']}")
            print(f"{'='*60}")
            print("Results already exist, loading from disk...")

            try:
                result = load_existing_results(ensemble_key, data_dir)
                experiment_results[ensemble_key] = result
                runtimes[ensemble_key] = None
                skipped_count += 1
                print(f"✓ Loaded existing results")
            except Exception as e:
                print(f"⚠ Warning: Could not load existing results: {e}")
                print(f"Will re-run experiment...")
                result, elapsed = run_ensemble_experiment(
                    ensemble_key, ensemble_config, compound_pool, oracle,
                    target_col, score_direction, cache_dir, data_dir
                )
                if result is not None:
                    experiment_results[ensemble_key] = result
                    runtimes[ensemble_key] = elapsed
                    completed_count += 1
        else:
            result, elapsed = run_ensemble_experiment(
                ensemble_key, ensemble_config, compound_pool, oracle,
                target_col, score_direction, cache_dir, data_dir
            )

            if result is not None:
                experiment_results[ensemble_key] = result
                runtimes[ensemble_key] = elapsed
                completed_count += 1

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"All experiments complete in {total_elapsed/60:.1f} minutes")
    print(f"Experiments run: {completed_count}")
    print(f"Experiments skipped (existing): {skipped_count}")
    print(f"Total results: {len(experiment_results)}")
    print(f"{'='*60}\n")

    # Analyze uncertainty-error correlations
    print("Analyzing uncertainty-error correlations...")
    analysis_results = []

    for ensemble_key, result in experiment_results.items():
        ensemble_config = ENSEMBLES[ensemble_key]
        analysis = analyze_uncertainty_error_correlation(
            result, ensemble_key, ensemble_config, cache_dir, target_col
        )
        if analysis is not None:
            analysis_results.append(analysis)

    # Generate plots
    print("\nGenerating correlation plots...")
    for analysis in analysis_results:
        plot_uncertainty_error_correlation(analysis, plots_dir)

    # Generate summary report
    print("\nGenerating summary report...")
    summary_path = generate_summary_report(analysis_results, output_base)

    # Print final summary
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
    print(f"Experiments completed: {len(experiment_results)}/{len(ENSEMBLES)}")
    print(f"\nOutputs:")
    print(f"  Data: {data_dir}")
    print(f"  Plots: {plots_dir}")
    print(f"  Summary: {summary_path}")
    print(f"\nCorrelation Results:")
    for analysis in analysis_results:
        print(f"  {analysis['ensemble_name']:30s}: ρ = {analysis['overall_correlation']:.3f}")
    print()


if __name__ == '__main__':
    main()

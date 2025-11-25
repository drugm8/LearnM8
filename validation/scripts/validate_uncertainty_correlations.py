#!/usr/bin/env python3
"""
Validation script to assess uncertainty-error correlations for learners with uncertainty support.

This script runs active learning experiments with different learner types on the
AMP 30K dataset and analyzes the correlation between predicted uncertainties and
actual prediction errors. This helps assess the quality of uncalibrated uncertainties.

Learners tested:
- RF Ensemble (Random Forest ensemble) ✓ Scalable
- XGB Ensemble (XGBoost ensemble) ✓ Scalable
- LR Ensemble (Linear Regression ensemble) ✓ Scalable
- DT Ensemble (Decision Tree ensemble) ✓ Scalable
- Mixed Ensemble (Multiple model types) - Fixed at 3 models (RF+LR+XGB)
- GP (Gaussian Process - gold standard for uncertainty) - Single model
- MC Dropout (Monte Carlo Dropout) - Single model
- Fastprop Ensemble (Fastprop ensemble) ✓ Scalable
- Chemprop Ensemble (Chemprop ensemble, if available) ✓ Scalable

Note: Single Fastprop and Chemprop models do NOT provide uncertainty estimates
and are excluded from this validation. Use their ensemble variants instead.

Configuration:
- N_ENSEMBLE_MEMBERS: Number of models in each scalable ensemble (default: 3)
  Increase this to test if larger ensembles improve uncertainty quality.

  This parameter now ACTUALLY controls ensemble sizes for scalable ensembles.
  Each ensemble type creates diversity using sensible parameter ranges:

    ✓ RF Ensemble: Different random states
    ✓ XGB Ensemble: Learning rates (0.01 to 0.3) + random states
    ✓ LR Ensemble: Regularization strengths (0.01 to 100, log scale) + random states
    ✓ DT Ensemble: Max depths (5 to 30) + random states
    ✓ Fastprop Ensemble: Different random states
    ✓ Chemprop Ensemble: Different random states

    Non-scalable learners (not affected by N_ENSEMBLE_MEMBERS):
    - Mixed Ensemble: Always 3 models (RF + LR + XGB)
    - GP: Single Gaussian Process model
    - MC Dropout: Single model with dropout sampling

Outputs:
- Correlation plots for each ensemble (all cycles + sample cycles)
- Summary report with correlation statistics
- Raw data for further analysis

Usage:
    python validation/scripts/validate_uncertainty_correlations.py

Expected Runtime: ~5-8 hours (all 8 learners with cache, N_ENSEMBLE_MEMBERS=10)
                 Runtime scales with N_ENSEMBLE_MEMBERS and number of learners
                 GPU acceleration significantly speeds up torch-based models
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

from learnm8 import run_active_learning
from learnm8.oracles import CSVOracle
from learnm8.learners import (
    GaussianProcessLearner,
    MCDropoutLearner
)
from learnm8.learners.ensemble import (
    RFEnsemble,
    XGBEnsemble,
    LREnsemble,
    DTEnsemble,
    MixedEnsemble,
    FastpropEnsemble
)

# Check for Chemprop availability
try:
    from learnm8.learners.ensemble.chemprop_ensemble import ChempropEnsemble
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False
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
N_ENSEMBLE_MEMBERS = 3  # Number of models in each ensemble

# Ensemble configurations with scalability metadata
ENSEMBLES = {
    'rf_ensemble': {
        'name': 'Random Forest Ensemble',
        'color': '#1f77b4',
        'scalable': True,
        'type': 'ensemble'
    },
    'xgb_ensemble': {
        'name': 'XGBoost Ensemble',
        'color': '#ff7f0e',
        'scalable': True,
        'type': 'ensemble'
    },
    'lr_ensemble': {
        'name': 'Linear Regression Ensemble',
        'color': '#2ca02c',
        'scalable': True,
        'type': 'ensemble'
    },
    'dt_ensemble': {
        'name': 'Decision Tree Ensemble',
        'color': '#d62728',
        'scalable': True,
        'type': 'ensemble'
    },
    'mixed_ensemble': {
        'name': 'Mixed Ensemble',
        'color': '#9467bd',
        'scalable': False,
        'type': 'ensemble'
    },
    'gp': {
        'name': 'Gaussian Process',
        'color': '#8c564b',
        'scalable': False,
        'type': 'single'
    },
    'mc_dropout': {
        'name': 'MC Dropout',
        'color': '#e377c2',
        'scalable': False,
        'type': 'single'
    },
    'fastprop_ensemble': {
        'name': 'Fastprop Ensemble',
        'color': '#bcbd22',
        'scalable': True,
        'type': 'ensemble'
    }
}

# Add Chemprop ensemble if available (single Chemprop does not support uncertainty)
if CHEMPROP_AVAILABLE:
    ENSEMBLES['chemprop_ensemble'] = {
        'name': 'Chemprop Ensemble',
        'color': '#17becf',
        'scalable': True,
        'type': 'ensemble'
    }


def create_configured_ensemble(ensemble_key, n_members, random_state):
    """
    Create ensemble learner with specified number of members.

    This function instantiates ensemble learners with custom configurations
    based on the desired number of ensemble members. Each ensemble type uses
    appropriate diversity strategies:

    - RF Ensemble: Random states for diversity
    - XGB Ensemble: Learning rates (0.01 to 0.3) + random states
    - LR Ensemble: Regularization strengths (0.01 to 100, log scale) + random states
    - DT Ensemble: Max depths (5 to 30) + random states
    - Fastprop/Chemprop: Random states for diversity
    - GP/MC Dropout: Single models (not ensembles)
    - Mixed Ensemble: Not scalable (returns None)

    Args:
        ensemble_key: Ensemble identifier (e.g., 'rf_ensemble')
        n_members: Number of ensemble members to create
        random_state: Base random seed for reproducibility

    Returns:
        Configured ensemble learner instance, or None if not scalable

    Raises:
        ValueError: If ensemble_key is unknown
    """
    random_states = [random_state + i * 137 for i in range(n_members)]

    if ensemble_key == 'rf_ensemble':
        return RFEnsemble(
            n_estimators=100,
            random_states=random_states
        )

    elif ensemble_key == 'xgb_ensemble':
        learning_rates = np.linspace(0.01, 0.3, n_members).tolist()
        return XGBEnsemble(
            learning_rates=learning_rates,
            random_states=random_states
        )

    elif ensemble_key == 'lr_ensemble':
        alphas = np.logspace(-2, 2, n_members).tolist()
        return LREnsemble(
            regularization_strengths=alphas,
            random_states=random_states
        )

    elif ensemble_key == 'dt_ensemble':
        max_depths = np.linspace(5, 30, n_members, dtype=int).tolist()
        return DTEnsemble(
            max_depths=max_depths,
            random_states=random_states
        )

    elif ensemble_key == 'fastprop_ensemble':
        return FastpropEnsemble(
            random_states=random_states
        )

    elif ensemble_key == 'chemprop_ensemble':
        if not CHEMPROP_AVAILABLE:
            return None
        return ChempropEnsemble(
            random_states=random_states
        )

    elif ensemble_key == 'gp':
        return GaussianProcessLearner(
            random_state=random_state
        )

    elif ensemble_key == 'mc_dropout':
        return MCDropoutLearner(
            random_state=random_state,
            n_dropout_samples=50
        )

    elif ensemble_key == 'mixed_ensemble':
        return None

    else:
        raise ValueError(f"Unknown ensemble key: {ensemble_key}")


def print_header():
    """Print validation header."""
    print("=" * 80)
    print("  Uncertainty-Error Correlation Validation")
    print("=" * 80)
    print()
    print(f"Dataset: AMP 30K")
    print(f"Learners: {len(ENSEMBLES)}")
    print(f"Ensemble Members (for ensembles): {N_ENSEMBLE_MEMBERS}")
    print(f"Cycles: {N_CYCLES}")
    print(f"Batch Fraction: {BATCH_FRACTION:.1%}")
    if CHEMPROP_AVAILABLE:
        print(f"Chemprop: Available")
    else:
        print(f"Chemprop: Not available (1 learner will be skipped)")
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
        learner_instance = create_configured_ensemble(
            ensemble_key,
            N_ENSEMBLE_MEMBERS,
            RANDOM_STATE
        )

        if learner_instance is None:
            print(f"⚠ Skipping {ensemble_config['name']} (not scalable with N_ENSEMBLE_MEMBERS)")
            return None, None

        if ensemble_config.get('scalable', False):
            print(f"  Created {ensemble_config['type']} with {N_ENSEMBLE_MEMBERS} members")
        else:
            print(f"  Using single model configuration")

        results = run_active_learning(
            compound_pool=compound_pool.clone(),
            oracle=oracle,
            learner=learner_instance,
            target_col=target_col,
            featurizer='morgan',
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


def analyze_uncertainty_error_correlation(results, ensemble_key, ensemble_config,
                                         cache_dir, target_col):
    """Analyze uncertainty-error correlation using all predictions at each cycle."""
    print(f"\nAnalyzing {ensemble_config['name']}...")

    compounds_df = results['compounds_df']

    # Find all prediction/uncertainty columns (cycle 1 through max cycle)
    pred_cols = [col for col in compounds_df.columns if col.startswith('prediction_cycle_')]
    unc_cols = [col for col in compounds_df.columns if col.startswith('uncertainty_cycle_')]

    if not pred_cols or not unc_cols:
        print("  ⚠ No prediction/uncertainty columns found")
        return None

    # Extract cycle numbers
    cycles = sorted([int(col.split('_')[-1]) for col in pred_cols])
    print(f"  Found predictions for cycles: {cycles}")

    # Collect all predictions/uncertainties across all cycles
    all_predictions = []
    all_uncertainties = []
    all_ground_truth = []
    all_cycle_labels = []

    for cycle in cycles:
        pred_col = f'prediction_cycle_{cycle}'
        unc_col = f'uncertainty_cycle_{cycle}'

        # Filter to compounds with valid predictions/uncertainties for this cycle
        cycle_df = compounds_df.filter(
            (pl.col(pred_col).is_not_null()) &
            (pl.col(unc_col).is_not_null()) &
            (pl.col(target_col).is_not_null())
        )

        if len(cycle_df) > 0:
            preds = cycle_df[pred_col].to_numpy()
            uncs = cycle_df[unc_col].to_numpy()
            truth = cycle_df[target_col].to_numpy()

            all_predictions.extend(preds)
            all_uncertainties.extend(uncs)
            all_ground_truth.extend(truth)
            all_cycle_labels.extend([cycle] * len(preds))

    if len(all_predictions) == 0:
        print("  ⚠ No valid predictions/uncertainties found")
        return None

    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_ground_truth = np.array(all_ground_truth)
    all_cycle_labels = np.array(all_cycle_labels)

    print(f"  Total predictions across all cycles: {len(all_predictions):,}")

    # Calculate errors
    all_errors = np.abs(all_predictions - all_ground_truth)

    # Overall correlation (across all cycles)
    overall_corr, overall_p = spearmanr(all_uncertainties, all_errors)
    print(f"  Overall correlation: {overall_corr:.3f} (p={overall_p:.2e})")

    # Per-cycle correlations
    cycle_correlations = {}
    for cycle in cycles:
        cycle_mask = all_cycle_labels == cycle
        cycle_unc = all_uncertainties[cycle_mask]
        cycle_err = all_errors[cycle_mask]

        if len(cycle_unc) >= 10:  # Need enough points for correlation
            corr, p = spearmanr(cycle_unc, cycle_err)
            cycle_correlations[cycle] = (corr, p, len(cycle_unc))
            if cycle in [1, 3, 5, 10]:  # Only print sample cycles
                print(f"  Cycle {cycle} correlation: {corr:.3f} (n={len(cycle_unc):,})")

    # Create analysis dataframe for plotting
    analysis_df = pl.DataFrame({
        'prediction': all_predictions,
        'uncertainty': all_uncertainties,
        'ground_truth': all_ground_truth,
        'abs_error': all_errors,
        'cycle': all_cycle_labels
    })

    return {
        'ensemble_key': ensemble_key,
        'ensemble_name': ensemble_config['name'],
        'analysis_df': analysis_df,
        'overall_correlation': overall_corr,
        'overall_p_value': overall_p,
        'cycle_correlations': cycle_correlations,
        'n_predictions': len(all_predictions)
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

        cycle_df = df.filter(pl.col('cycle') == cycle)

        if len(cycle_df) > 0:
            cycle_unc = cycle_df['uncertainty'].to_numpy()
            cycle_err = cycle_df['abs_error'].to_numpy()

            ax.scatter(cycle_unc, cycle_err, alpha=0.4, s=25, color=color)
            ax.set_xlabel('Uncertainty', fontsize=10)
            ax.set_ylabel('Abs Error', fontsize=10)

            # Get correlation for this cycle
            if cycle in analysis_result['cycle_correlations']:
                corr, p, n = analysis_result['cycle_correlations'][cycle]
                title = f'Cycle {cycle} (n={n:,})\nρ = {corr:.3f}'
            else:
                title = f'Cycle {cycle} (n={len(cycle_df):,})'

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
        f.write(f"Ensemble Members (for ensembles): {N_ENSEMBLE_MEMBERS}\n")
        f.write(f"Learners Tested: {len(all_results)}\n")
        f.write(f"Total Learners Available: {len(ENSEMBLES)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Overall Correlation Results\n")
        f.write("-" * 80 + "\n\n")

        correlations = []
        for result in all_results:
            if result is not None:
                f.write(f"{result['ensemble_name']:30s}: ")
                f.write(f"ρ = {result['overall_correlation']:6.3f} ")
                f.write(f"(p={result['overall_p_value']:.2e}, n={result['n_predictions']:,})\n")
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
                    corr, p, n = result['cycle_correlations'][cycle]
                    f.write(f"  {result['ensemble_name']:30s}: ρ = {corr:6.3f} (n={n:,})\n")

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
    output_base = Path('validation/reports/uncertainty_correlation_validation')
    data_dir = output_base / 'data'
    plots_dir = output_base / 'plots'
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

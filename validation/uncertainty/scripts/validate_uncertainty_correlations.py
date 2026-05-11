#!/usr/bin/env python3
"""
Uncertainty Quantification Validation for Active Learning.

Evaluates UQ quality across learner types using scientifically rigorous metrics:
- ENCE (Expected Normalized Calibration Error)
- NLL (Gaussian Negative Log-Likelihood)
- Miscalibration Area
- AUSE (Area Under Sparsification Error)
- Spearman rank correlation

Features:
- Multi-seed experiments (3 seeds) for statistical rigor
- Post-hoc STD scaling calibration with before/after comparison
- RF n_estimators ablation study (50, 100, 200, 500, 1000 trees)
- 500K dataset for scalable models
- Publication-quality multi-panel figures
- CSV export of all metrics
- CLI --models flag to run a subset of models (cached data from others included in analysis)

Learners tested:
- RF single (tree-level uncertainty)
- DT single (leaf-level uncertainty)
- LR single (residual-based uncertainty)
- XGB Ensemble (scalable)
- Mixed Ensemble (fixed 3 models)
- GP (Gaussian Process)
- MC Dropout
- Fastprop/Chemprop Ensembles (scalable)

Usage:
    python validation/scripts/validate_uncertainty_correlations.py
    python validation/scripts/validate_uncertainty_correlations.py --models rf_single,mc_dropout

Expected Runtime: ~8-15 hours (all phases, 3 seeds, GPU recommended)
"""

import argparse
import contextlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from learnm8 import CycleConfig, run_active_learning
from learnm8.utils.logging import configure_learnm8_logging
from learnm8.learners import (
    DecisionTreeLearner,
    EnsembleLearner,
    GaussianProcessLearner,
    LinearRegressionLearner,
    MCDropoutLearner,
    RandomForestLearner,
    XGBoostLearner,
)
from learnm8.learners.ensemble import (
    DTEnsemble,
    FastpropEnsemble,
    XGBEnsemble,
)
from learnm8.oracles import CSVOracle

try:
    from learnm8.learners.ensemble.chemprop_ensemble import ChempropEnsemble
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False

from validation.lib import get_dataset_info, get_dataset_path, load_validation_dataset
from validation.lib.uq_metrics import (
    apply_std_scaling,
    calibrate_std_scaling,
    compute_all_uq_metrics,
)
from validation.lib.uq_plots import (
    create_publication_summary,
    plot_calibration_before_after,
    plot_calibration_curves,
    plot_cycle_progression,
    plot_dataset_calibration_comparison,
    plot_dataset_comparison,
    plot_discovery_curves,
    plot_ensemble_baselines_calibration,
    plot_ensemble_baselines_sparsification,
    plot_ensemble_size_ablation,
    plot_qq_residuals,
    plot_sparsification,
    plot_triple_comparison,
    plot_uncertainty_vs_error_scatter,
)

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────

N_CYCLES = 10
BATCH_FRACTION = 0.01
N_ENSEMBLE_MEMBERS = 3
RANDOM_SEEDS = [42, 123, 456]
RF_ABLATION_N_ESTIMATORS = [10, 50, 100, 200, 500, 1000]
RF_ABLATION_MEMBER_SIZES = [2, 3, 5, 10]
XGB_ABLATION_SIZES = [2, 3, 5, 10]

DATASETS = {
    'ampc_30k': {'all_models': True},
    'ampc_500k': {'scalable_only': True},
}

SCALABLE_MODELS = {
    'lr_single', 'lr_ols', 'lr_ensemble', 'xgb_ensemble', 'dt_single', 'dt_ensemble',
    # 'fastprop_ensemble',  # disabled: see uncertainty_improvement_recommendations.md
    'rf_single', 'rf_ensemble',
}

ENSEMBLES = {
    'xgb_ensemble': {
        'name': 'XGB Ensemble', 'scalable': True, 'type': 'ensemble'
    },
    'lr_single': {
        'name': 'Ridge (alpha=0.1)', 'scalable': True, 'type': 'single'
    },
    'lr_ols': {
        'name': 'Linear Regression (OLS)', 'scalable': True, 'type': 'single'
    },
    'lr_ensemble': {
        'name': 'LR Ensemble (old UQ)', 'scalable': True, 'type': 'ensemble'
    },
    'dt_single': {
        'name': 'Decision Tree (single)', 'scalable': True, 'type': 'single'
    },
    'dt_ensemble': {
        'name': 'DT Ensemble', 'scalable': True, 'type': 'ensemble'
    },
    'mixed_ensemble': {
        'name': 'Mixed Ensemble', 'scalable': False, 'type': 'ensemble'
    },
    'gp': {
        'name': 'GP (Tanimoto, a=1e-3)', 'scalable': False, 'type': 'single'
    },
    'gp_rbf': {
        'name': 'GP (RBF, a=1e-10)', 'scalable': False, 'type': 'single'
    },
    'gp_tanimoto_old': {
        'name': 'GP (Tanimoto, a=1e-10)', 'scalable': False, 'type': 'single'
    },
    'gp_rbf_fair': {
        'name': 'GP (RBF, a=1e-3)', 'scalable': False, 'type': 'single'
    },
    'rf_single': {
        'name': 'Random Forest (single)', 'scalable': True, 'type': 'single'
    },
    'rf_ensemble': {
        'name': 'RF Ensemble (old UQ)', 'scalable': True, 'type': 'ensemble'
    },
    'mc_dropout': {
        'name': 'MC Dropout', 'scalable': False, 'type': 'single'
    },
    'rf_1000': {
        'name': 'RF (1000 trees)', 'scalable': False, 'type': 'single'
    },
    'xgb_large': {
        'name': 'XGB Ensemble (Large)', 'scalable': False, 'type': 'ensemble'
    },
    # 'fastprop_ensemble': {  # disabled: see uncertainty_improvement_recommendations.md
    #     'name': 'Fastprop Ensemble', 'scalable': True, 'type': 'ensemble'
    # },
}

if CHEMPROP_AVAILABLE:
    ENSEMBLES['chemprop_ensemble'] = {
        'name': 'Chemprop Ensemble', 'scalable': True, 'type': 'ensemble'
    }


# ── Learner Factory ──────────────────────────────────────────────────────────

def create_configured_learner(learner_key, n_members, random_state):
    random_states = [random_state + i * 137 for i in range(n_members)]

    if learner_key == 'rf_single':
        return RandomForestLearner(n_estimators=100, random_state=random_state)
    elif learner_key == 'xgb_ensemble':
        learning_rates = np.linspace(0.01, 0.3, n_members).tolist()
        return XGBEnsemble(learning_rates=learning_rates, random_states=random_states)
    elif learner_key == 'lr_single':
        return LinearRegressionLearner(random_state=random_state)
    elif learner_key == 'lr_ols':
        return LinearRegressionLearner(alpha=None, random_state=random_state)
    elif learner_key == 'dt_single':
        return DecisionTreeLearner(random_state=random_state)
    elif learner_key == 'dt_ensemble':
        return DTEnsemble(random_states=random_states[:3])
    elif learner_key == 'fastprop_ensemble':
        return FastpropEnsemble(random_states=random_states)
    elif learner_key == 'chemprop_ensemble':
        if not CHEMPROP_AVAILABLE:
            return None
        return ChempropEnsemble(random_states=random_states)
    elif learner_key == 'gp':
        return GaussianProcessLearner(random_state=random_state)
    elif learner_key == 'gp_rbf':
        return GaussianProcessLearner(kernel='rbf', alpha=1e-10, random_state=random_state)
    elif learner_key == 'gp_tanimoto_old':
        return GaussianProcessLearner(alpha=1e-10, random_state=random_state)
    elif learner_key == 'gp_rbf_fair':
        return GaussianProcessLearner(kernel='rbf', random_state=random_state)
    elif learner_key == 'rf_ensemble':
        rf_learners = [
            RandomForestLearner(n_estimators=100, random_state=random_states[i])
            for i in range(min(3, n_members))
        ]
        return EnsembleLearner(rf_learners, aggregation_method='mean', uncertainty_method='std')
    elif learner_key == 'lr_ensemble':
        alphas = np.logspace(-2, 1, min(3, n_members)).tolist()
        lr_learners = [
            LinearRegressionLearner(alpha=a, random_state=random_states[i])
            for i, a in enumerate(alphas)
        ]
        return EnsembleLearner(lr_learners, aggregation_method='mean', uncertainty_method='std')
    elif learner_key == 'mc_dropout':
        return MCDropoutLearner(random_state=random_state, n_dropout_samples=50)
    elif learner_key == 'rf_1000':
        return RandomForestLearner(n_estimators=1000, random_state=random_state)
    elif learner_key == 'xgb_large':
        learning_rates = np.linspace(0.01, 0.3, n_members).tolist()
        learners = [
            XGBoostLearner(n_estimators=500, learning_rate=lr, random_state=rs)
            for lr, rs in zip(learning_rates, random_states, strict=False)
        ]
        return EnsembleLearner(learners, aggregation_method='mean', uncertainty_method='std')
    elif learner_key == 'mixed_ensemble':
        return None
    elif learner_key == 'xgb_ablation':
        return XGBEnsemble(
            learning_rates=[0.1] * n_members,
            random_states=random_states,
        )
    elif learner_key == 'rf_ablation':
        rf_learners = [
            RandomForestLearner(n_estimators=100, random_state=random_states[i])
            for i in range(n_members)
        ]
        return EnsembleLearner(rf_learners, aggregation_method='mean', uncertainty_method='std')
    elif learner_key == 'rf_ensemble_ucb':
        rf_learners = [
            RandomForestLearner(n_estimators=100, random_state=random_states[i])
            for i in range(min(3, n_members))
        ]
        return EnsembleLearner(rf_learners, aggregation_method='mean', uncertainty_method='std')
    else:
        raise ValueError(f"Unknown learner key: {learner_key}")


# ── Experiment Running ───────────────────────────────────────────────────────

def check_existing_results(output_dir):
    required = ['compounds_final.csv', 'cycle_metrics.csv', 'selection_history.csv']
    return all((output_dir / f).exists() for f in required)


def load_existing_results(output_dir):
    compounds_df = pl.read_csv(output_dir / 'compounds_final.csv', comment_prefix='#')
    cycle_metrics_df = pl.read_csv(output_dir / 'cycle_metrics.csv', comment_prefix='#')
    return {
        'compounds_df': compounds_df,
        'cycle_metrics': cycle_metrics_df.to_dicts(),
        'output_dir': output_dir
    }


def run_single_experiment(learner_key, n_members, random_state,
                          compound_pool, oracle, target_col, score_direction,
                          cache_dir, output_dir):
    learner = create_configured_learner(learner_key, n_members, random_state)
    if learner is None:
        return None

    try:
        results = run_active_learning(
            compound_pool=compound_pool.clone(),
            oracle=oracle,
            learner=learner,
            target_col=target_col,
            featurizer='morgan',
            n_cycles=N_CYCLES,
            batch_fraction=BATCH_FRACTION,
            score_direction=score_direction,
            mode='benchmark',
            output_dir=str(output_dir),
            cache_dir=cache_dir,
            random_state=random_state,
            device='auto',
        )
        return results
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_or_load_experiment(learner_key, n_members, random_state,
                           compound_pool, oracle, target_col, score_direction,
                           cache_dir, output_dir):
    if check_existing_results(output_dir):
        try:
            return load_existing_results(output_dir), True
        except Exception:
            pass

    start = time.time()
    result = run_single_experiment(
        learner_key, n_members, random_state,
        compound_pool, oracle, target_col, score_direction,
        cache_dir, output_dir
    )
    elapsed = time.time() - start
    if result is not None:
        print(f"    Completed in {elapsed/60:.1f} min")
    return result, False


def load_cached_results_for_models(data_dir, models_dict, models_to_skip):
    """Load cached experiment results for models not in the current run set."""
    cached = {}
    for model_key in models_dict:
        if model_key in models_to_skip:
            continue
        seed_results = []
        has_any = False
        for seed in RANDOM_SEEDS:
            exp_dir = data_dir / f'{model_key}__seed{seed}'
            if check_existing_results(exp_dir):
                try:
                    result = load_existing_results(exp_dir)
                    seed_results.append(result)
                    has_any = True
                    continue
                except Exception:
                    pass
            seed_results.append(None)
        if has_any:
            cached[model_key] = seed_results
    return cached


# ── Analysis ─────────────────────────────────────────────────────────────────

def extract_predictions_uncertainties(results, target_col):
    """Extract all predictions and uncertainties from experiment results.

    Predictions live in per-cycle parquet files under ``results['output_dir']``;
    we lazy-scan one cycle at a time and join with the master DataFrame to get
    target values and selected_cycle, so we never materialise all predictions
    at once.
    """
    compounds_df = results['compounds_df']
    output_dir = results.get('output_dir')
    if output_dir is None:
        return None
    output_dir = Path(output_dir)

    parquet_paths = sorted(
        output_dir.glob('prediction_cycle_*.parquet'),
        key=lambda p: int(p.stem.split('_')[-1]),
    )
    if not parquet_paths:
        return None

    per_cycle = {}

    for parquet_path in parquet_paths:
        cycle = int(parquet_path.stem.split('_')[-1])

        cycle_lf = pl.scan_parquet(parquet_path)
        if 'uncertainty' not in cycle_lf.collect_schema().names():
            continue

        # Out-of-sample compounds at cycle N: selected_cycle is null or >= cycle.
        oos_compounds = compounds_df.filter(
            pl.col(target_col).is_not_null()
            & (pl.col('selected_cycle').is_null() | (pl.col('selected_cycle') >= cycle))
        ).select(['ID', target_col])

        cycle_df = (
            cycle_lf.collect()
            .filter(pl.col('prediction').is_not_null() & pl.col('uncertainty').is_not_null())
            .join(oos_compounds, on='ID', how='inner')
        )

        if len(cycle_df) == 0:
            continue

        preds = cycle_df['prediction'].to_numpy().astype(float)
        uncs = cycle_df['uncertainty'].to_numpy().astype(float)
        truth = cycle_df[target_col].to_numpy().astype(float)

        mask = np.isfinite(preds) & np.isfinite(uncs) & np.isfinite(truth) & (uncs > 0)
        preds, uncs, truth = preds[mask], uncs[mask], truth[mask]

        if len(preds) == 0:
            continue

        # Skip degenerate cycles where uncertainty is nearly constant (e.g. collapsed GP)
        unc_cv = np.std(uncs) / (np.mean(uncs) + 1e-10)
        if unc_cv < 1e-4:
            continue

        per_cycle[cycle] = {
            'y_true': truth, 'y_pred': preds, 'y_std': uncs
        }

    if not per_cycle:
        return None

    last_cycle = max(per_cycle.keys())
    return {
        'per_cycle': per_cycle,
        'overall': per_cycle[last_cycle],
        'cycles': sorted(per_cycle.keys())
    }


def compute_metrics_for_experiment(extracted):
    """Compute all UQ metrics for one experiment."""
    if extracted is None:
        return None

    d = extracted['overall']
    overall = compute_all_uq_metrics(d['y_true'], d['y_pred'], d['y_std'])

    per_cycle = {}
    for cycle, cd in extracted['per_cycle'].items():
        per_cycle[cycle] = compute_all_uq_metrics(cd['y_true'], cd['y_pred'], cd['y_std'])

    return {'overall': overall, 'per_cycle': per_cycle}


def compute_calibrated_metrics(extracted):
    """Compute metrics after STD scaling calibration.

    Uses first 30% of data for calibration, rest for evaluation.
    """
    if extracted is None:
        return None, None

    d = extracted['overall']
    n = len(d['y_true'])
    n_cal = max(int(n * 0.3), 100)

    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]

    scale_factor = calibrate_std_scaling(
        d['y_true'][cal_idx], d['y_pred'][cal_idx], d['y_std'][cal_idx]
    )

    calibrated_std = apply_std_scaling(d['y_std'][eval_idx], scale_factor)
    calibrated_metrics = compute_all_uq_metrics(
        d['y_true'][eval_idx], d['y_pred'][eval_idx], calibrated_std
    )

    raw_metrics = compute_all_uq_metrics(
        d['y_true'][eval_idx], d['y_pred'][eval_idx], d['y_std'][eval_idx]
    )

    return {
        'raw': raw_metrics,
        'calibrated': calibrated_metrics,
        'scale_factor': scale_factor,
        'calibrated_data': {
            'y_true': d['y_true'][eval_idx],
            'y_pred': d['y_pred'][eval_idx],
            'y_std': calibrated_std
        },
        'raw_data': {
            'y_true': d['y_true'][eval_idx],
            'y_pred': d['y_pred'][eval_idx],
            'y_std': d['y_std'][eval_idx]
        }
    }, scale_factor


def aggregate_per_cycle_metrics(per_seed_per_cycle):
    """Aggregate per-cycle metrics across seeds.

    Returns (means_by_cycle, stds_by_cycle) where each is
    {cycle_num: {metric_key: float}}.
    """
    all_cycles = set()
    for seed_pc in per_seed_per_cycle:
        if seed_pc is not None:
            all_cycles.update(seed_pc.keys())

    means_by_cycle, stds_by_cycle = {}, {}
    for cycle in sorted(all_cycles):
        cycle_vals = {}
        for seed_pc in per_seed_per_cycle:
            if seed_pc is not None and cycle in seed_pc:
                for metric, val in seed_pc[cycle].items():
                    if np.isfinite(val):
                        cycle_vals.setdefault(metric, []).append(val)
        means_by_cycle[cycle] = {m: np.mean(v) for m, v in cycle_vals.items()}
        stds_by_cycle[cycle] = {m: np.std(v) for m, v in cycle_vals.items()}
    return means_by_cycle, stds_by_cycle


def aggregate_seed_metrics(seed_metrics_list):
    """Aggregate metrics across seeds (mean +/- std)."""
    if not seed_metrics_list:
        return {}, {}

    all_keys = set()
    for m in seed_metrics_list:
        if m is not None:
            all_keys.update(m.keys())

    means = {}
    stds = {}
    for key in all_keys:
        values = [m[key] for m in seed_metrics_list
                  if m is not None and key in m and np.isfinite(m[key])]
        if values:
            means[key] = np.mean(values)
            stds[key] = np.std(values)
        else:
            means[key] = np.nan
            stds[key] = np.nan

    return means, stds


# ── CSV Export ───────────────────────────────────────────────────────────────

def export_metrics_csv(all_metrics, output_path, dataset_name):
    """Export all metrics to CSV."""
    rows = []
    for model_name, seed_data in all_metrics.items():
        for seed_idx, metrics in enumerate(seed_data.get('per_seed_overall', [])):
            if metrics is None:
                continue
            row = {
                'dataset': dataset_name,
                'model': model_name,
                'seed': RANDOM_SEEDS[seed_idx],
            }
            row.update(metrics)
            rows.append(row)

    if rows:
        df = pl.DataFrame(rows)
        df.write_csv(output_path)
        print(f"  Metrics CSV saved: {output_path}")


def export_calibration_csv(calibration_data, output_path):
    """Export calibration comparison to CSV."""
    rows = []
    for model_name, cal_data in calibration_data.items():
        if cal_data is None:
            continue
        for metric_type in ['raw', 'calibrated']:
            metrics = cal_data.get(metric_type, {})
            row = {
                'model': model_name,
                'calibration': metric_type,
                'scale_factor': cal_data.get('scale_factor', np.nan),
            }
            row.update(metrics)
            rows.append(row)

    if rows:
        df = pl.DataFrame(rows)
        df.write_csv(output_path)
        print(f"  Calibration CSV saved: {output_path}")


def export_ablation_csv(ablation_metrics, output_path):
    """Export RF n_estimators ablation metrics to CSV."""
    rows = []
    for size, seed_data in ablation_metrics.items():
        for seed_idx, metrics in enumerate(seed_data.get('per_seed', [])):
            if metrics is None:
                continue
            row = {
                'n_estimators': size,
                'seed': RANDOM_SEEDS[seed_idx],
            }
            row.update(metrics)
            rows.append(row)

    if rows:
        df = pl.DataFrame(rows)
        df.write_csv(output_path)
        print(f"  Ablation CSV saved: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Uncertainty Quantification Validation for Active Learning'
    )
    parser.add_argument(
        '--models',
        type=str,
        default=None,
        help='Comma-separated model keys to run (e.g., rf_single,mc_dropout). '
             'Default: run all models. Cached data from other models is still '
             'included in analysis and plots.'
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    configure_learnm8_logging(level='INFO')

    models_to_run = None
    if args.models:
        models_to_run = set(args.models.split(','))
        unknown = models_to_run - set(ENSEMBLES.keys())
        if unknown:
            print(f"WARNING: Unknown model keys: {unknown}")
            print(f"  Available: {sorted(ENSEMBLES.keys())}")
            models_to_run -= unknown

    print("=" * 80)
    print("  Uncertainty Quantification Validation")
    print("=" * 80)
    print()
    print(f"  Learners:           {len(ENSEMBLES)}")
    print(f"  Ensemble Members:   {N_ENSEMBLE_MEMBERS}")
    print(f"  Seeds:              {RANDOM_SEEDS}")
    print(f"  Cycles:             {N_CYCLES}")
    print(f"  Batch Fraction:     {BATCH_FRACTION:.1%}")
    print(f"  Datasets:           {list(DATASETS.keys())}")
    print(f"  RF Ablation Trees:  {RF_ABLATION_N_ESTIMATORS}")
    print(f"  Chemprop:           {'Available' if CHEMPROP_AVAILABLE else 'Not available'}")
    if models_to_run:
        print(f"  Running only:       {sorted(models_to_run)}")
    print()

    output_base = Path('validation/reports/uncertainty/uncertainty_correlation_validation')
    plots_dir = output_base / 'plots'
    plots_30k = plots_dir / '30k'
    plots_500k = plots_dir / '500k'
    plots_comparison = plots_dir / 'comparison'
    plots_ablation = plots_dir / 'ablation'
    plots_presentation = plots_dir / 'presentation'
    plots_ucb = plots_dir / 'ucb_comparison'
    metrics_dir = output_base / 'metrics'
    cache_dir = Path('validation/.shared_cache')

    # Story-structured presentation folders
    story1a = plots_presentation / 'story1a_ensemble_baselines'
    story1b = plots_presentation / 'story1b_ensemble_size'
    story2a = plots_presentation / 'story2a_rf_treelevel'
    story2b = plots_presentation / 'story2b_gp_kernel'
    story2c = plots_presentation / 'story2c_lr_analytical'
    story2d = plots_presentation / 'story2d_dt_leaf'
    story3  = plots_presentation / 'story3_al_impact'

    for d in [plots_30k, plots_500k, plots_comparison, plots_ablation,
              plots_ucb, metrics_dir, cache_dir,
              story1a, story1b, story2a, story2b, story2c, story2d, story3]:
        d.mkdir(parents=True, exist_ok=True)

    old_files_to_remove = [
        'calibration_curves_raw.png', 'calibration_curves_500k.png',
        'calibration_before_after.png', 'calibration_30k_vs_500k.png',
        'error_by_uncertainty_quantile.png', 'error_by_uncertainty_quantile_500k.png',
        'publication_summary.png', 'publication_summary_500k.png',
        'uq_metrics_heatmap.png', 'uq_metrics_heatmap_500k.png',
        'metrics_30k_vs_500k.png', 'rf_ntrees_ablation.png',
        'rf_ensemble_size_ablation.png',
        'chemprop_ensemble_uncertainty_correlation.png',
        'dt_ensemble_uncertainty_correlation.png',
        'fastprop_ensemble_uncertainty_correlation.png',
        'gp_uncertainty_correlation.png',
        'lr_ensemble_uncertainty_correlation.png',
        'mc_dropout_uncertainty_correlation.png',
        'rf_ensemble_uncertainty_correlation.png',
        'xgb_ensemble_uncertainty_correlation.png',
    ]
    for fname in old_files_to_remove:
        old_path = plots_dir / fname
        if old_path.exists():
            old_path.unlink()

    total_start = time.time()

    # ── Load datasets ────────────────────────────────────────────────────
    loaded_datasets = {}
    for ds_name in DATASETS:
        try:
            ds_path = get_dataset_path(ds_name)
            pool, meta = load_validation_dataset(
                dataset_name=ds_name,
                clean_invalid_scores=True,
                random_state=42
            )
            ds_info = get_dataset_info(ds_name)
            oracle = CSVOracle(str(ds_path), id_column=ds_info['id_column'])
            loaded_datasets[ds_name] = {
                'pool': pool, 'oracle': oracle,
                'target_col': meta['target_column'],
                'score_direction': meta['score_direction'],
            }
            print(f"Loaded {ds_name}: {len(pool):,} compounds")
        except Exception as e:
            print(f"Could not load {ds_name}: {e}")

    if 'ampc_30k' not in loaded_datasets:
        print("FATAL: ampc_30k dataset required")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Model comparison on 30K (selected models x 3 seeds)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PHASE 1: Model Comparison on ampc_30k")
    print("=" * 80)

    ds = loaded_datasets['ampc_30k']
    all_30k_results = {}

    models_for_phase1 = ENSEMBLES if models_to_run is None else {
        k: v for k, v in ENSEMBLES.items() if k in models_to_run
    }

    for ens_key, ens_cfg in models_for_phase1.items():
        print(f"\n  [{ens_key}] {ens_cfg['name']}")
        seed_results = []

        for seed in RANDOM_SEEDS:
            data_dir = output_base / 'data' / f'{ens_key}__seed{seed}'
            data_dir.mkdir(parents=True, exist_ok=True)

            print(f"    seed={seed}...", end=" ", flush=True)
            result, cached = run_or_load_experiment(
                ens_key, N_ENSEMBLE_MEMBERS, seed,
                ds['pool'], ds['oracle'], ds['target_col'],
                ds['score_direction'], cache_dir, data_dir
            )
            if cached:
                print("(cached)")
            seed_results.append(result)

        all_30k_results[ens_key] = seed_results

    # Load cached results for models not in current run
    if models_to_run is not None:
        print("\n  Loading cached results for other models...")
        cached_30k = load_cached_results_for_models(
            output_base / 'data', ENSEMBLES, set(all_30k_results.keys())
        )
        for model_key, seed_results in cached_30k.items():
            print(f"    Loaded cached: {ENSEMBLES[model_key]['name']}")
            all_30k_results[model_key] = seed_results

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Scalable models on 500K (if available)
    # ═══════════════════════════════════════════════════════════════════════
    all_500k_results = {}
    if 'ampc_500k' in loaded_datasets:
        print("\n" + "=" * 80)
        print("  PHASE 2: Scalable Models on ampc_500k")
        print("=" * 80)

        ds500 = loaded_datasets['ampc_500k']

        scalable_to_run = SCALABLE_MODELS if models_to_run is None else (
            SCALABLE_MODELS & models_to_run
        )

        for ens_key in scalable_to_run:
            if ens_key not in ENSEMBLES:
                continue
            ens_cfg = ENSEMBLES[ens_key]
            print(f"\n  [{ens_key}] {ens_cfg['name']}")
            seed_results = []

            for seed in RANDOM_SEEDS:
                data_dir = output_base / 'data_500k' / f'{ens_key}__seed{seed}'
                data_dir.mkdir(parents=True, exist_ok=True)

                print(f"    seed={seed}...", end=" ", flush=True)
                result, cached = run_or_load_experiment(
                    ens_key, N_ENSEMBLE_MEMBERS, seed,
                    ds500['pool'], ds500['oracle'], ds500['target_col'],
                    ds500['score_direction'], cache_dir, data_dir
                )
                if cached:
                    print("(cached)")
                seed_results.append(result)

            all_500k_results[ens_key] = seed_results

        # Load cached 500K results for models not in current run
        if models_to_run is not None:
            cached_500k = load_cached_results_for_models(
                output_base / 'data_500k', ENSEMBLES, set(all_500k_results.keys())
            )
            for model_key, seed_results in cached_500k.items():
                print(f"    Loaded cached 500K: {ENSEMBLES[model_key]['name']}")
                all_500k_results[model_key] = seed_results
    else:
        print("\n  Skipping PHASE 2: ampc_500k not available")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: RF n_estimators ablation on 500K (falls back to 30K)
    # ═══════════════════════════════════════════════════════════════════════
    run_ablation = models_to_run is None or 'rf_single' in models_to_run
    ablation_results = {}
    ds_rf_ablation = loaded_datasets.get('ampc_500k', ds)

    if run_ablation:
        print("\n" + "=" * 80)
        _ablation_ds_label = '500K' if 'ampc_500k' in loaded_datasets else '30K'
        print(f"  PHASE 3: RF n_estimators Ablation ({_ablation_ds_label})")
        print("=" * 80)

        for n_trees in RF_ABLATION_N_ESTIMATORS:
            print(f"\n  RF n_estimators={n_trees}")
            seed_results = []

            for seed in RANDOM_SEEDS:
                data_dir = (output_base / 'data_ablation_rf_ntrees_500k'
                            / f'rf_ntrees{n_trees}__seed{seed}')
                data_dir.mkdir(parents=True, exist_ok=True)

                if check_existing_results(data_dir):
                    try:
                        result = load_existing_results(data_dir)
                        print(f"    seed={seed}... (cached)")
                        seed_results.append(result)
                        continue
                    except Exception:
                        pass

                print(f"    seed={seed}...", end=" ", flush=True)
                learner = RandomForestLearner(n_estimators=n_trees, random_state=seed)
                start = time.time()
                try:
                    result = run_active_learning(
                        compound_pool=ds_rf_ablation['pool'].clone(),
                        oracle=ds_rf_ablation['oracle'],
                        learner=learner,
                        target_col=ds_rf_ablation['target_col'],
                        featurizer='morgan',
                        n_cycles=N_CYCLES,
                        batch_fraction=BATCH_FRACTION,
                        score_direction=ds_rf_ablation['score_direction'],
                        mode='benchmark',
                        output_dir=str(data_dir),
                        cache_dir=cache_dir,
                        random_state=seed,
                        device='auto',
                    )
                    elapsed = time.time() - start
                    print(f"    Completed in {elapsed/60:.1f} min")
                    seed_results.append(result)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    seed_results.append(None)

            ablation_results[n_trees] = seed_results
    else:
        # Try loading cached ablation results
        ablation_dir = output_base / 'data_ablation_rf_ntrees_500k'
        if ablation_dir.exists():
            print("\n  Loading cached RF ablation results (500K)...")
            for n_trees in RF_ABLATION_N_ESTIMATORS:
                seed_results = []
                has_any = False
                for seed in RANDOM_SEEDS:
                    data_dir = ablation_dir / f'rf_ntrees{n_trees}__seed{seed}'
                    if check_existing_results(data_dir):
                        try:
                            result = load_existing_results(data_dir)
                            seed_results.append(result)
                            has_any = True
                            continue
                        except Exception:
                            pass
                    seed_results.append(None)
                if has_any:
                    ablation_results[n_trees] = seed_results

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: XGB Ensemble Member Count Ablation on 500K (falls back to 30K)
    # ═══════════════════════════════════════════════════════════════════════
    run_xgb_ablation = models_to_run is None or 'xgb_ensemble' in models_to_run
    xgb_ablation_results = {}
    ds_xgb_ablation = loaded_datasets.get('ampc_500k', ds)

    if run_xgb_ablation:
        print("\n" + "=" * 80)
        _xgb_ds_label = '500K' if 'ampc_500k' in loaded_datasets else '30K'
        print(f"  PHASE 4: XGB Ensemble Member Count Ablation ({_xgb_ds_label})")
        print("=" * 80)

        for n_xgb_members in XGB_ABLATION_SIZES:
            print(f"\n  XGB n_members={n_xgb_members}")
            seed_results = []

            for seed in RANDOM_SEEDS:
                data_dir = (output_base / 'data_ablation_xgb_members_500k'
                            / f'xgb_{n_xgb_members}members__seed{seed}')
                data_dir.mkdir(parents=True, exist_ok=True)

                if check_existing_results(data_dir):
                    try:
                        result = load_existing_results(data_dir)
                        print(f"    seed={seed}... (cached)")
                        seed_results.append(result)
                        continue
                    except Exception:
                        pass

                print(f"    seed={seed}...", end=" ", flush=True)
                learner = create_configured_learner('xgb_ablation', n_xgb_members, seed)
                start = time.time()
                try:
                    result = run_active_learning(
                        compound_pool=ds_xgb_ablation['pool'].clone(),
                        oracle=ds_xgb_ablation['oracle'],
                        learner=learner,
                        target_col=ds_xgb_ablation['target_col'],
                        featurizer='morgan',
                        n_cycles=N_CYCLES,
                        batch_fraction=BATCH_FRACTION,
                        score_direction=ds_xgb_ablation['score_direction'],
                        mode='benchmark',
                        output_dir=str(data_dir),
                        cache_dir=cache_dir,
                        random_state=seed,
                        device='auto',
                    )
                    elapsed = time.time() - start
                    print(f"Completed in {elapsed/60:.1f} min")
                    seed_results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    seed_results.append(None)

            xgb_ablation_results[n_xgb_members] = seed_results
    else:
        xgb_ablation_dir = output_base / 'data_ablation_xgb_members_500k'
        if xgb_ablation_dir.exists():
            print("\n  Loading cached XGB member ablation results (500K)...")
            for n_xgb_members in XGB_ABLATION_SIZES:
                seed_results = []
                has_any = False
                for seed in RANDOM_SEEDS:
                    data_dir = xgb_ablation_dir / f'xgb_{n_xgb_members}members__seed{seed}'
                    if check_existing_results(data_dir):
                        try:
                            result = load_existing_results(data_dir)
                            seed_results.append(result)
                            has_any = True
                            continue
                        except Exception:
                            pass
                    seed_results.append(None)
                if has_any:
                    xgb_ablation_results[n_xgb_members] = seed_results

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4b: RF Ensemble Member Count Ablation on 500K (falls back to 30K)
    # ═══════════════════════════════════════════════════════════════════════
    run_rf_member_ablation = models_to_run is None or 'rf_ensemble' in models_to_run
    rf_member_ablation_results = {}
    ds_rf_member = loaded_datasets.get('ampc_500k', ds)

    if run_rf_member_ablation:
        print("\n" + "=" * 80)
        _rf_mem_ds_label = '500K' if 'ampc_500k' in loaded_datasets else '30K'
        print(f"  PHASE 4b: RF Ensemble Member Count Ablation ({_rf_mem_ds_label})")
        print("=" * 80)

        for n_rf_members in RF_ABLATION_MEMBER_SIZES:
            print(f"\n  RF n_members={n_rf_members}")
            seed_results = []

            for seed in RANDOM_SEEDS:
                data_dir = (output_base / 'data_ablation_rf_members_500k'
                            / f'rf_{n_rf_members}members__seed{seed}')
                data_dir.mkdir(parents=True, exist_ok=True)

                if check_existing_results(data_dir):
                    try:
                        result = load_existing_results(data_dir)
                        print(f"    seed={seed}... (cached)")
                        seed_results.append(result)
                        continue
                    except Exception:
                        pass

                print(f"    seed={seed}...", end=" ", flush=True)
                learner = create_configured_learner('rf_ablation', n_rf_members, seed)
                start = time.time()
                try:
                    result = run_active_learning(
                        compound_pool=ds_rf_member['pool'].clone(),
                        oracle=ds_rf_member['oracle'],
                        learner=learner,
                        target_col=ds_rf_member['target_col'],
                        featurizer='morgan',
                        n_cycles=N_CYCLES,
                        batch_fraction=BATCH_FRACTION,
                        score_direction=ds_rf_member['score_direction'],
                        mode='benchmark',
                        output_dir=str(data_dir),
                        cache_dir=cache_dir,
                        random_state=seed,
                        device='auto',
                    )
                    elapsed = time.time() - start
                    print(f"Completed in {elapsed/60:.1f} min")
                    seed_results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    seed_results.append(None)

            rf_member_ablation_results[n_rf_members] = seed_results
    else:
        rf_member_dir = output_base / 'data_ablation_rf_members_500k'
        if rf_member_dir.exists():
            print("\n  Loading cached RF member ablation results (500K)...")
            for n_rf_members in RF_ABLATION_MEMBER_SIZES:
                seed_results = []
                has_any = False
                for seed in RANDOM_SEEDS:
                    data_dir = rf_member_dir / f'rf_{n_rf_members}members__seed{seed}'
                    if check_existing_results(data_dir):
                        try:
                            result = load_existing_results(data_dir)
                            seed_results.append(result)
                            has_any = True
                            continue
                        except Exception:
                            pass
                    seed_results.append(None)
                if has_any:
                    rf_member_ablation_results[n_rf_members] = seed_results

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: UCB Comparison (old RF Ensemble vs new RF tree-level)
    # ═══════════════════════════════════════════════════════════════════════
    run_ucb = ('ampc_500k' in loaded_datasets and
               (models_to_run is None or 'rf_single' in models_to_run))
    ucb_comparison_results = {}

    if run_ucb:
        print("\n" + "=" * 80)
        print("  PHASE 5: UCB Comparison on ampc_500k (RF Ensemble vs RF Tree-level)")
        print("=" * 80)

        ds500_ucb = loaded_datasets['ampc_500k']

        def _ucb_cycles(beta):
            return [
                CycleConfig('random', n_cycles=1, batch_fraction=BATCH_FRACTION),
                CycleConfig('ucb', n_cycles=N_CYCLES - 1, batch_fraction=BATCH_FRACTION,
                            acquisition_params={'beta': beta}),
            ]

        greedy_acquisition_cycles = [
            CycleConfig('random', n_cycles=1, batch_fraction=BATCH_FRACTION),
            CycleConfig('greedy', n_cycles=N_CYCLES - 1, batch_fraction=BATCH_FRACTION),
        ]
        UCB_BETA_LOW  = 0.5
        UCB_BETA_HIGH = 4.0
        ucb_configs = {
            'rf_ensemble_ucb_low':  ('rf_ensemble_ucb', N_ENSEMBLE_MEMBERS, _ucb_cycles(UCB_BETA_LOW)),
            'rf_ensemble_ucb_high': ('rf_ensemble_ucb', N_ENSEMBLE_MEMBERS, _ucb_cycles(UCB_BETA_HIGH)),
            'rf_ensemble_greedy':   ('rf_ensemble_ucb', N_ENSEMBLE_MEMBERS, greedy_acquisition_cycles),
            'rf_single_ucb_low':    ('rf_single',        1,                  _ucb_cycles(UCB_BETA_LOW)),
            'rf_single_ucb_high':   ('rf_single',        1,                  _ucb_cycles(UCB_BETA_HIGH)),
            'rf_single_greedy':     ('rf_single',        1,                  greedy_acquisition_cycles),
        }

        for ucb_key, (learner_key, n_mem, cycles_cfg) in ucb_configs.items():
            print(f"\n  [{ucb_key}]")
            seed_results = []

            for seed in RANDOM_SEEDS:
                data_dir = output_base / 'data_ucb_comparison_500k' / f'{ucb_key}__seed{seed}'
                data_dir.mkdir(parents=True, exist_ok=True)

                if check_existing_results(data_dir):
                    try:
                        result = load_existing_results(data_dir)
                        print(f"    seed={seed}... (cached)")
                        seed_results.append(result)
                        continue
                    except Exception:
                        pass

                print(f"    seed={seed}...", end=" ", flush=True)
                learner = create_configured_learner(learner_key, n_mem, seed)
                if learner is None:
                    seed_results.append(None)
                    continue

                start = time.time()
                try:
                    result = run_active_learning(
                        compound_pool=ds500_ucb['pool'].clone(),
                        oracle=ds500_ucb['oracle'],
                        learner=learner,
                        target_col=ds500_ucb['target_col'],
                        featurizer='morgan',
                        cycles=cycles_cfg,
                        score_direction=ds500_ucb['score_direction'],
                        mode='benchmark',
                        output_dir=str(data_dir),
                        cache_dir=cache_dir,
                        random_state=seed,
                        device='auto',
                    )
                    elapsed = time.time() - start
                    print(f"Completed in {elapsed/60:.1f} min")
                    seed_results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    seed_results.append(None)

            ucb_comparison_results[ucb_key] = seed_results
    else:
        ucb_dir = output_base / 'data_ucb_comparison_500k'
        if ucb_dir.exists():
            print("\n  Loading cached UCB comparison results (500K)...")
            for ucb_key in [
                'rf_ensemble_ucb_low', 'rf_ensemble_ucb_high', 'rf_ensemble_greedy',
                'rf_single_ucb_low', 'rf_single_ucb_high', 'rf_single_greedy',
            ]:
                seed_results = []
                has_any = False
                for seed in RANDOM_SEEDS:
                    data_dir = ucb_dir / f'{ucb_key}__seed{seed}'
                    if check_existing_results(data_dir):
                        try:
                            result = load_existing_results(data_dir)
                            seed_results.append(result)
                            has_any = True
                            continue
                        except Exception:
                            pass
                    seed_results.append(None)
                if has_any:
                    ucb_comparison_results[ucb_key] = seed_results

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ANALYSIS: Computing UQ Metrics")
    print("=" * 80)

    target_col = ds['target_col']

    # ── 30K metrics ──────────────────────────────────────────────────────
    metrics_30k = {}
    extracted_30k = {}

    for ens_key, seed_results in all_30k_results.items():
        print(f"\n  Analyzing {ENSEMBLES[ens_key]['name']}...")
        per_seed_overall = []
        per_seed_extracted = []
        per_seed_per_cycle = []

        for _i, result in enumerate(seed_results):
            if result is None:
                per_seed_overall.append(None)
                per_seed_extracted.append(None)
                per_seed_per_cycle.append(None)
                continue
            ext = extract_predictions_uncertainties(result, target_col)
            per_seed_extracted.append(ext)
            m = compute_metrics_for_experiment(ext)
            per_seed_overall.append(m['overall'] if m else None)
            per_seed_per_cycle.append(m['per_cycle'] if m else None)

        mean_m, std_m = aggregate_seed_metrics(per_seed_overall)
        pc_mean, pc_std = aggregate_per_cycle_metrics(per_seed_per_cycle)
        metrics_30k[ens_key] = {
            'mean': mean_m, 'std': std_m,
            'per_seed_overall': per_seed_overall,
            'per_cycle_mean': pc_mean,
            'per_cycle_std': pc_std,
        }
        extracted_30k[ens_key] = per_seed_extracted

        if mean_m:
            print(f"    ENCE={mean_m.get('ence', np.nan):.3f}±{std_m.get('ence', 0):.3f}  "
                  f"NLL={mean_m.get('nll', np.nan):.3f}±{std_m.get('nll', 0):.3f}  "
                  f"AUSE={mean_m.get('ause', np.nan):.4f}±{std_m.get('ause', 0):.4f}  "
                  f"rho={mean_m.get('spearman_rho', np.nan):.3f}±{std_m.get('spearman_rho', 0):.3f}")

    # ── STD scaling calibration ──────────────────────────────────────────
    print("\n  Computing STD scaling calibration...")
    calibration_30k = {}

    for ens_key, extracted_list in extracted_30k.items():
        ext = next((e for e in extracted_list if e is not None), None)
        if ext is None:
            calibration_30k[ens_key] = None
            continue

        cal_result, scale_factor = compute_calibrated_metrics(ext)
        if cal_result:
            calibration_30k[ens_key] = cal_result
            print(f"    {ENSEMBLES[ens_key]['name']}: k={scale_factor:.3f}  "
                  f"ENCE {cal_result['raw']['ence']:.3f} -> {cal_result['calibrated']['ence']:.3f}  "
                  f"MA {cal_result['raw']['miscalibration_area']:.3f} -> "
                  f"{cal_result['calibrated']['miscalibration_area']:.3f}")
        else:
            calibration_30k[ens_key] = None

    # ── 500K metrics ─────────────────────────────────────────────────────
    metrics_500k = {}
    extracted_500k = {}
    if all_500k_results:
        print("\n  Analyzing 500K results...")
        for ens_key, seed_results in all_500k_results.items():
            per_seed_overall = []
            per_seed_extracted = []
            per_seed_per_cycle = []
            for result in seed_results:
                if result is None:
                    per_seed_overall.append(None)
                    per_seed_extracted.append(None)
                    per_seed_per_cycle.append(None)
                    continue
                ext = extract_predictions_uncertainties(
                    result, loaded_datasets['ampc_500k']['target_col']
                )
                per_seed_extracted.append(ext)
                m = compute_metrics_for_experiment(ext)
                per_seed_overall.append(m['overall'] if m else None)
                per_seed_per_cycle.append(m['per_cycle'] if m else None)

            mean_m, std_m = aggregate_seed_metrics(per_seed_overall)
            pc_mean, pc_std = aggregate_per_cycle_metrics(per_seed_per_cycle)
            metrics_500k[ens_key] = {
                'mean': mean_m, 'std': std_m,
                'per_seed_overall': per_seed_overall,
                'per_cycle_mean': pc_mean,
                'per_cycle_std': pc_std,
            }
            extracted_500k[ens_key] = per_seed_extracted

            if mean_m:
                print(f"    {ENSEMBLES[ens_key]['name']}: "
                      f"ENCE={mean_m.get('ence', np.nan):.3f}  "
                      f"rho={mean_m.get('spearman_rho', np.nan):.3f}")

    # ── Ablation metrics ─────────────────────────────────────────────────
    ablation_metrics = {}
    if ablation_results:
        print("\n  Analyzing RF n_estimators ablation results...")
        for n_trees, seed_results in ablation_results.items():
            per_seed = []
            for result in seed_results:
                if result is None:
                    per_seed.append(None)
                    continue
                ext = extract_predictions_uncertainties(result, target_col)
                m = compute_metrics_for_experiment(ext)
                per_seed.append(m['overall'] if m else None)

            mean_m, std_m = aggregate_seed_metrics(per_seed)
            ablation_metrics[n_trees] = {
                'metrics_mean': mean_m, 'metrics_std': std_m, 'per_seed': per_seed
            }

            if mean_m:
                print(f"    n_estimators={n_trees:4d}: ENCE={mean_m.get('ence', np.nan):.3f}  "
                      f"rho={mean_m.get('spearman_rho', np.nan):.3f}  "
                      f"MAE={mean_m.get('mae', np.nan):.4f}")

    # ── XGB member ablation metrics ───────────────────────────────────────
    xgb_ablation_metrics = {}
    if xgb_ablation_results:
        print("\n  Analyzing XGB member count ablation results...")
        for n_xgb_members, seed_results in xgb_ablation_results.items():
            per_seed = []
            for result in seed_results:
                if result is None:
                    per_seed.append(None)
                    continue
                ext = extract_predictions_uncertainties(result, target_col)
                m = compute_metrics_for_experiment(ext)
                per_seed.append(m['overall'] if m else None)

            mean_m, std_m = aggregate_seed_metrics(per_seed)
            xgb_ablation_metrics[n_xgb_members] = {
                'metrics_mean': mean_m, 'metrics_std': std_m, 'per_seed': per_seed
            }

            if mean_m:
                print(f"    n_members={n_xgb_members:2d}: ENCE={mean_m.get('ence', np.nan):.3f}  "
                      f"rho={mean_m.get('spearman_rho', np.nan):.3f}  "
                      f"MAE={mean_m.get('mae', np.nan):.4f}")

    # ── RF member count ablation metrics ──────────────────────────────────
    rf_member_ablation_metrics = {}
    _rf_member_target = ds_rf_member['target_col']
    if rf_member_ablation_results:
        print("\n  Analyzing RF member count ablation results...")
        for n_rf_members, seed_results in rf_member_ablation_results.items():
            per_seed = []
            for result in seed_results:
                if result is None:
                    per_seed.append(None)
                    continue
                ext = extract_predictions_uncertainties(result, _rf_member_target)
                m = compute_metrics_for_experiment(ext)
                per_seed.append(m['overall'] if m else None)

            mean_m, std_m = aggregate_seed_metrics(per_seed)
            rf_member_ablation_metrics[n_rf_members] = {
                'metrics_mean': mean_m, 'metrics_std': std_m, 'per_seed': per_seed
            }

            if mean_m:
                print(f"    n_members={n_rf_members:2d}: ENCE={mean_m.get('ence', np.nan):.3f}  "
                      f"rho={mean_m.get('spearman_rho', np.nan):.3f}  "
                      f"MAE={mean_m.get('mae', np.nan):.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PLOTS: Generating Publication Figures")
    print("=" * 80)

    # ── Build model data dicts (first non-None seed) ────────────────────
    print("\n  Building model data for plots...")
    cal_models_raw = {}
    cal_models_calibrated = {}

    for ens_key, extracted_list in extracted_30k.items():
        ext = next((e for e in extracted_list if e is not None), None)
        if ext is None:
            continue
        name = ENSEMBLES[ens_key]['name']
        d = ext['overall']
        cal_result = calibration_30k.get(ens_key)

        raw_ma = metrics_30k[ens_key]['mean'].get('miscalibration_area', np.nan)
        cal_models_raw[name] = {
            'y_true': d['y_true'], 'y_pred': d['y_pred'], 'y_std': d['y_std'],
            'miscal_area': raw_ma
        }

        if cal_result and 'calibrated_data' in cal_result:
            cd = cal_result['calibrated_data']
            cal_models_calibrated[name] = {
                'y_true': cd['y_true'], 'y_pred': cd['y_pred'], 'y_std': cd['y_std'],
                'miscal_area': cal_result['calibrated'].get('miscalibration_area', np.nan)
            }

    pub_metrics = {ENSEMBLES[k]['name']: v['mean'] for k, v in metrics_30k.items()
                   if v['mean']}

    # ── 30K plots ────────────────────────────────────────────────────────
    print("  Generating 30K plots...")

    plot_calibration_curves(
        cal_models_raw,
        output_path=plots_30k / 'calibration_curves.png',
        subtitle='30K dataset, out-of-sample',
    )
    print("    Saved 30k/calibration_curves.png")

    if cal_models_calibrated:
        raw_for_comparison = {}
        for ens_key, cal_result in calibration_30k.items():
            if cal_result and 'raw_data' in cal_result:
                name = ENSEMBLES[ens_key]['name']
                rd = cal_result['raw_data']
                raw_for_comparison[name] = {
                    'y_true': rd['y_true'], 'y_pred': rd['y_pred'], 'y_std': rd['y_std'],
                    'miscal_area': cal_result['raw'].get('miscalibration_area', np.nan)
                }
        plot_calibration_before_after(
            raw_for_comparison, cal_models_calibrated,
            output_path=plots_30k / 'calibration_before_after.png',
            subtitle='30K dataset, out-of-sample',
        )
        print("    Saved 30k/calibration_before_after.png")

    plot_sparsification(
        cal_models_raw,
        output_path=plots_30k / 'sparsification.png',
        subtitle='30K dataset, out-of-sample',
    )
    print("    Saved 30k/sparsification.png")

    plot_uncertainty_vs_error_scatter(
        cal_models_raw,
        output_path=plots_30k / 'scatter_uncertainty_vs_error.png',
        subtitle='30K dataset, out-of-sample',
    )
    print("    Saved 30k/scatter_uncertainty_vs_error.png")

    plot_qq_residuals(
        cal_models_raw,
        output_path=plots_30k / 'qq_residuals.png',
        subtitle='30K dataset, out-of-sample',
    )
    print("    Saved 30k/qq_residuals.png")

    create_publication_summary(
        cal_models_raw, pub_metrics,
        output_path=plots_30k / 'publication_summary.png',
        subtitle='30K dataset, out-of-sample',
    )
    print("    Saved 30k/publication_summary.png")

    # ── 30K trajectory plots ──────────────────────────────────────────────
    ause_traj_30k, miscal_traj_30k = {}, {}
    for ens_key, m_data in metrics_30k.items():
        name = ENSEMBLES[ens_key]['name']
        pc_m = m_data.get('per_cycle_mean', {})
        pc_s = m_data.get('per_cycle_std', {})
        if pc_m:
            ause_traj_30k[name] = {
                c: (pc_m[c].get('ause', np.nan), pc_s[c].get('ause', 0.0))
                for c in pc_m
            }
            miscal_traj_30k[name] = {
                c: (pc_m[c].get('miscalibration_area', np.nan),
                    pc_s[c].get('miscalibration_area', 0.0))
                for c in pc_m
            }

    if ause_traj_30k:
        plot_cycle_progression(
            ause_traj_30k, metric_key='ause', metric_label='AUSE',
            title="Does uncertainty ordering quality improve across the active learning campaign?",
            subtitle='30K dataset, out-of-sample per cycle',
            output_path=plots_30k / 'trajectory_ause.png',
        )
        print("    Saved 30k/trajectory_ause.png")

    if miscal_traj_30k:
        plot_cycle_progression(
            miscal_traj_30k, metric_key='miscalibration_area',
            metric_label='Miscalibration Area',
            title="Does calibration quality improve as more labeled data is acquired?",
            subtitle='30K dataset, out-of-sample per cycle',
            output_path=plots_30k / 'trajectory_miscalibration.png',
        )
        print("    Saved 30k/trajectory_miscalibration.png")

    # ── Ablation ─────────────────────────────────────────────────────────
    if ablation_metrics:
        print("  Generating RF ablation plot...")
        plot_ensemble_size_ablation(
            ablation_metrics,
            output_path=plots_ablation / 'rf_ntrees_ablation.png',
            xlabel='Number of Trees (n_estimators)',
            title='RF: UQ Metrics vs Number of Trees',
            subtitle='30K dataset, out-of-sample',
        )
        print("    Saved ablation/rf_ntrees_ablation.png")

    if xgb_ablation_metrics:
        print("  Generating XGB member ablation plot...")
        plot_ensemble_size_ablation(
            xgb_ablation_metrics,
            output_path=plots_ablation / 'xgb_members_ablation.png',
            xlabel='XGB Ensemble Members',
            title='XGB Ensemble: UQ Metrics vs Number of Members',
            subtitle='30K dataset, out-of-sample',
        )
        print("    Saved ablation/xgb_members_ablation.png")

    # ── UCB comparison plots ──────────────────────────────────────────────
    if ucb_comparison_results:
        print("  Generating UCB comparison plots...")
        ucb_display = {
            'rf_ensemble_ucb_low':  'RF Ensemble + UCB (β=0.5)',
            'rf_ensemble_ucb_high': 'RF Ensemble + UCB (β=4.0)',
            'rf_ensemble_greedy':   'RF Ensemble + Greedy',
            'rf_single_ucb_low':    'RF Tree-level + UCB (β=0.5)',
            'rf_single_ucb_high':   'RF Tree-level + UCB (β=4.0)',
            'rf_single_greedy':     'RF Tree-level + Greedy',
        }

        ucb_named = {ucb_display[k]: v
                     for k, v in ucb_comparison_results.items()
                     if k in ucb_display}
        if ucb_named:
            plot_discovery_curves(
                ucb_named,
                output_path=plots_ucb / 'discovery_curves_ucb.png',
                subtitle='500K dataset, mean \u00b1 std across 3 seeds',
            )
            print("    Saved ucb_comparison/discovery_curves_ucb.png")

            ause_ucb_traj = {}
            for ucb_key, seed_results in ucb_comparison_results.items():
                per_seed_per_cycle = []
                for result in seed_results:
                    if result is None:
                        per_seed_per_cycle.append(None)
                        continue
                    ext = extract_predictions_uncertainties(result, target_col)
                    m = compute_metrics_for_experiment(ext)
                    per_seed_per_cycle.append(m['per_cycle'] if m else None)
                pc_mean, pc_std = aggregate_per_cycle_metrics(per_seed_per_cycle)
                name = ucb_display.get(ucb_key, ucb_key)
                if pc_mean:
                    ause_ucb_traj[name] = {
                        c: (pc_mean[c].get('ause', np.nan), pc_std[c].get('ause', 0.0))
                        for c in pc_mean
                    }
            if ause_ucb_traj:
                plot_cycle_progression(
                    ause_ucb_traj, metric_key='ause', metric_label='AUSE',
                    title='Does UQ quality during UCB campaign differ between approaches?',
                    subtitle='500K dataset, out-of-sample per cycle',
                    output_path=plots_ucb / 'trajectory_ause_ucb.png',
                )
                print("    Saved ucb_comparison/trajectory_ause_ucb.png")

    # ── 500K plots ───────────────────────────────────────────────────────
    if metrics_500k and extracted_500k:
        print("  Generating 500K plots...")

        cal_models_500k = {}
        for ens_key, extracted_list in extracted_500k.items():
            ext = next((e for e in extracted_list if e is not None), None)
            if ext is None:
                continue
            name = ENSEMBLES[ens_key]['name']
            d = ext['overall']
            ma = metrics_500k[ens_key]['mean'].get('miscalibration_area', np.nan)
            cal_models_500k[name] = {
                'y_true': d['y_true'], 'y_pred': d['y_pred'], 'y_std': d['y_std'],
                'miscal_area': ma
            }

        plot_calibration_curves(
            cal_models_500k,
            output_path=plots_500k / 'calibration_curves.png',
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved 500k/calibration_curves.png")

        plot_sparsification(
            cal_models_500k,
            output_path=plots_500k / 'sparsification.png',
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved 500k/sparsification.png")

        plot_uncertainty_vs_error_scatter(
            cal_models_500k,
            output_path=plots_500k / 'scatter_uncertainty_vs_error.png',
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved 500k/scatter_uncertainty_vs_error.png")

        plot_qq_residuals(
            cal_models_500k,
            output_path=plots_500k / 'qq_residuals.png',
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved 500k/qq_residuals.png")

        pub_metrics_500k = {ENSEMBLES[k]['name']: v['mean']
                            for k, v in metrics_500k.items() if v['mean']}
        create_publication_summary(
            cal_models_500k, pub_metrics_500k,
            output_path=plots_500k / 'publication_summary.png',
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved 500k/publication_summary.png")

        # ── 500K trajectory plots ─────────────────────────────────────────
        ause_traj_500k, miscal_traj_500k = {}, {}
        for ens_key, m_data in metrics_500k.items():
            name = ENSEMBLES[ens_key]['name']
            pc_m = m_data.get('per_cycle_mean', {})
            pc_s = m_data.get('per_cycle_std', {})
            if pc_m:
                ause_traj_500k[name] = {
                    c: (pc_m[c].get('ause', np.nan), pc_s[c].get('ause', 0.0))
                    for c in pc_m
                }
                miscal_traj_500k[name] = {
                    c: (pc_m[c].get('miscalibration_area', np.nan),
                        pc_s[c].get('miscalibration_area', 0.0))
                    for c in pc_m
                }

        if ause_traj_500k:
            plot_cycle_progression(
                ause_traj_500k, metric_key='ause', metric_label='AUSE',
                title="Does uncertainty ordering quality improve across the active learning campaign?",
                subtitle='500K dataset, out-of-sample per cycle',
                output_path=plots_500k / 'trajectory_ause.png',
            )
            print("    Saved 500k/trajectory_ause.png")

        if miscal_traj_500k:
            plot_cycle_progression(
                miscal_traj_500k, metric_key='miscalibration_area',
                metric_label='Miscalibration Area',
                title="Does calibration quality improve as more labeled data is acquired?",
                subtitle='500K dataset, out-of-sample per cycle',
                output_path=plots_500k / 'trajectory_miscalibration.png',
            )
            print("    Saved 500k/trajectory_miscalibration.png")

        # ── Comparison plots ─────────────────────────────────────────────
        common_30k = {k: v for k, v in cal_models_raw.items() if k in cal_models_500k}
        common_500k = {k: v for k, v in cal_models_500k.items() if k in common_30k}
        if common_30k:
            plot_dataset_calibration_comparison(
                common_30k, common_500k,
                output_path=plots_comparison / 'calibration_30k_vs_500k.png'
            )
            print("    Saved comparison/calibration_30k_vs_500k.png")

        comparison_30k = {ENSEMBLES[k]['name']: v for k, v in metrics_30k.items()
                          if k in metrics_500k}
        comparison_500k = {ENSEMBLES[k]['name']: v for k, v in metrics_500k.items()}
        if comparison_30k:
            plot_dataset_comparison(
                comparison_30k, comparison_500k,
                output_path=plots_comparison / 'metrics_30k_vs_500k.png'
            )
            print("    Saved comparison/metrics_30k_vs_500k.png")
    else:
        print("  Skipping 500K plots (no data)")

    # ═══════════════════════════════════════════════════════════════════════
    # STORY-STRUCTURED PRESENTATION PLOTS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PRESENTATION: Story-Structured Figures")
    print("=" * 80)

    def _get_model_data(key, extracted_src, metrics_src):
        """Extract plot data for a single model from extracted results."""
        if key not in extracted_src:
            return None
        ext = next((e for e in extracted_src[key] if e is not None), None)
        if ext is None:
            return None
        d = ext['overall']
        ma = metrics_src[key]['mean'].get('miscalibration_area', np.nan)
        return {'y_true': d['y_true'], 'y_pred': d['y_pred'], 'y_std': d['y_std'],
                'miscal_area': ma}

    def _pick_src(key):
        """Return (extracted, metrics) preferring 500K if model is scalable."""
        if key in SCALABLE_MODELS and key in extracted_500k:
            return extracted_500k, metrics_500k
        return extracted_30k, metrics_30k

    # ── STORY 1a: Ensemble baselines on 500K ─────────────────────────────
    print("\n  Story 1a: Ensemble baselines (500K)")
    s1a_keys = {
        'XGB Ensemble':      'xgb_ensemble',
        'RF Ensemble':       'rf_ensemble',
        'LR Ensemble':       'lr_ensemble',
    }
    s1a_data = {}
    for label, key in s1a_keys.items():
        src_ext, src_met = _pick_src(key)
        d = _get_model_data(key, src_ext, src_met)
        if d is not None:
            s1a_data[label] = d

    if s1a_data:
        plot_ensemble_baselines_calibration(
            s1a_data,
            title='1a. Ensemble Calibration (500K Dataset)',
            output_path=story1a / 'ensemble_calibration_500k.png',
            subtitle='500K dataset, out-of-sample',
        )
        plot_ensemble_baselines_sparsification(
            s1a_data,
            title='1a. Ensemble Sparsification (500K Dataset)',
            output_path=story1a / 'ensemble_sparsification_500k.png',
            subtitle='500K dataset, out-of-sample',
        )
        print(f"    Saved story1a/ensemble_calibration_500k.png ({len(s1a_data)} models)")
        print(f"    Saved story1a/ensemble_sparsification_500k.png ({len(s1a_data)} models)")

    # ── STORY 1b: Ensemble size ablation ─────────────────────────────────
    print("\n  Story 1b: Ensemble size ablation")
    _3metrics = ['ause', 'miscalibration_area', 'mae']

    if xgb_ablation_metrics:
        plot_ensemble_size_ablation(
            xgb_ablation_metrics,
            output_path=story1b / 'xgb_member_ablation.png',
            xlabel='Ensemble Members',
            title='1b. XGB: Does More Members Improve UQ? (500K)',
            metrics_to_show=_3metrics,
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved story1b/xgb_member_ablation.png")

    if rf_member_ablation_metrics:
        plot_ensemble_size_ablation(
            rf_member_ablation_metrics,
            output_path=story1b / 'rf_member_ablation.png',
            xlabel='Ensemble Members',
            title='1b. RF: Does More Members Improve UQ? (500K)',
            metrics_to_show=_3metrics,
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved story1b/rf_member_ablation.png")

    # ── STORY 2a: RF tree-level vs RF Ensemble ────────────────────────────
    print("\n  Story 2a: RF tree-level vs RF Ensemble")
    s2a_data = {}
    for label, key in (
        ('RF Ensemble (old UQ)', 'rf_ensemble'),
        ('RF Tree-level (new UQ)', 'rf_single'),
    ):
        src_ext, src_met = _pick_src(key)
        d = _get_model_data(key, src_ext, src_met)
        if d is not None:
            s2a_data[label] = d

    if s2a_data:
        plot_triple_comparison(
            s2a_data,
            title='2a. RF Tree-level vs RF Ensemble UQ (500K)',
            output_path=story2a / 'rf_comparison.png',
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved story2a/rf_comparison.png")

    if ablation_metrics:
        plot_ensemble_size_ablation(
            ablation_metrics,
            output_path=story2a / 'rf_tree_count_ablation.png',
            xlabel='Number of Trees (n_estimators)',
            title='2a-i. RF: How Many Trees Do We Need for Good UQ? (500K)',
            metrics_to_show=_3metrics,
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved story2a/rf_tree_count_ablation.png")

    # ── STORY 2b: GP kernel 2x2 ablation ─────────────────────────────────
    print("\n  Story 2b: GP kernel 2x2 ablation (30K)")
    s2b_keys = ['gp', 'gp_tanimoto_old', 'gp_rbf_fair', 'gp_rbf']
    s2b_data = {}
    for key in s2b_keys:
        d = _get_model_data(key, extracted_30k, metrics_30k)
        if d is not None:
            s2b_data[ENSEMBLES[key]['name']] = d

    if s2b_data:
        plot_calibration_curves(
            s2b_data,
            output_path=story2b / 'gp_2x2_calibration.png',
            subtitle='30K dataset, out-of-sample',
        )
        plot_sparsification(
            s2b_data,
            output_path=story2b / 'gp_2x2_sparsification.png',
            subtitle='30K dataset, out-of-sample',
        )
        plot_uncertainty_vs_error_scatter(
            s2b_data,
            output_path=story2b / 'gp_2x2_scatter.png',
            subtitle='30K dataset, out-of-sample',
        )
        print(f"    Saved story2b/gp_2x2_{{calibration,sparsification,scatter}}.png ({len(s2b_data)} models)")

    # ── STORY 2c: LR analytical vs LR Ensemble ────────────────────────────
    print("\n  Story 2c: LR analytical vs LR Ensemble")
    s2c_data = {}
    for label, key in (
        ('LR Ensemble (old UQ)', 'lr_ensemble'),
        ('Ridge Analytical (new UQ)', 'lr_single'),
    ):
        src_ext, src_met = _pick_src(key)
        d = _get_model_data(key, src_ext, src_met)
        if d is not None:
            s2c_data[label] = d

    if s2c_data:
        plot_triple_comparison(
            s2c_data,
            title='2c. LR Analytical UQ vs LR Ensemble (500K)',
            output_path=story2c / 'lr_comparison.png',
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved story2c/lr_comparison.png")

    # ── STORY 2d: DT leaf impurity vs DT Ensemble ─────────────────────────
    print("\n  Story 2d: DT leaf impurity vs DT Ensemble")
    s2d_data = {}
    for label, key in (
        ('DT Ensemble (old UQ)', 'dt_ensemble'),
        ('DT Leaf Impurity (new UQ)', 'dt_single'),
    ):
        src_ext, src_met = _pick_src(key)
        d = _get_model_data(key, src_ext, src_met)
        if d is not None:
            s2d_data[label] = d

    if s2d_data:
        plot_triple_comparison(
            s2d_data,
            title='2d. DT Leaf Impurity vs DT Ensemble UQ (500K)',
            output_path=story2d / 'dt_comparison.png',
            subtitle='500K dataset, out-of-sample',
        )
        print("    Saved story2d/dt_comparison.png")

    # ── STORY 3: AL impact ────────────────────────────────────────────────
    print("\n  Story 3: Does better UQ improve the AL process?")
    if ucb_comparison_results:
        story3_display = {
            'rf_ensemble_ucb_low':  'RF Ensemble + UCB (β=0.5)',
            'rf_ensemble_ucb_high': 'RF Ensemble + UCB (β=4.0)',
            'rf_ensemble_greedy':   'RF Ensemble + Greedy',
            'rf_single_ucb_low':    'RF Tree-level + UCB (β=0.5)',
            'rf_single_ucb_high':   'RF Tree-level + UCB (β=4.0)',
            'rf_single_greedy':     'RF Tree-level + Greedy',
        }
        story3_named = {story3_display[k]: v
                        for k, v in ucb_comparison_results.items()
                        if k in story3_display}
        if story3_named:
            plot_discovery_curves(
                story3_named,
                output_path=story3 / 'discovery_curves.png',
                subtitle='500K dataset, mean \u00b1 std across 3 seeds',
            )
            print("    Saved story3/discovery_curves.png")

        ause_traj_story3 = {}
        for ucb_key, seed_results in ucb_comparison_results.items():
            if ucb_key not in story3_display:
                continue
            per_seed_per_cycle = []
            for result in seed_results:
                if result is None:
                    per_seed_per_cycle.append(None)
                    continue
                ext = extract_predictions_uncertainties(result, target_col)
                m = compute_metrics_for_experiment(ext)
                per_seed_per_cycle.append(m['per_cycle'] if m else None)
            pc_mean, pc_std = aggregate_per_cycle_metrics(per_seed_per_cycle)
            name = story3_display[ucb_key]
            if pc_mean:
                ause_traj_story3[name] = {
                    c: (pc_mean[c].get('ause', np.nan), pc_std[c].get('ause', 0.0))
                    for c in pc_mean
                }
        if ause_traj_story3:
            plot_cycle_progression(
                ause_traj_story3, metric_key='ause', metric_label='AUSE',
                title='3. Does UQ quality during the AL campaign differ?',
                subtitle='500K dataset, out-of-sample per cycle',
                output_path=story3 / 'trajectory_ause.png',
            )
            print("    Saved story3/trajectory_ause.png")

            spearman_traj_story3 = {}
            for model_name in ause_traj_story3:
                ucb_key = next(
                    (k for k, v in story3_display.items() if v == model_name), None
                )
                if ucb_key is None:
                    continue
                seed_results = ucb_comparison_results[ucb_key]
                per_seed_per_cycle = []
                for result in seed_results:
                    if result is None:
                        per_seed_per_cycle.append(None)
                        continue
                    ext = extract_predictions_uncertainties(result, target_col)
                    m = compute_metrics_for_experiment(ext)
                    per_seed_per_cycle.append(m['per_cycle'] if m else None)
                pc_mean, pc_std = aggregate_per_cycle_metrics(per_seed_per_cycle)
                if pc_mean:
                    spearman_traj_story3[model_name] = {
                        c: (pc_mean[c].get('spearman_rho', np.nan),
                            pc_std[c].get('spearman_rho', 0.0))
                        for c in pc_mean
                    }
            if spearman_traj_story3:
                plot_cycle_progression(
                    spearman_traj_story3, metric_key='spearman_rho',
                    metric_label='Spearman \u03c1',
                    title='3. How does uncertainty-error correlation evolve during AL?',
                    subtitle='500K dataset, out-of-sample per cycle',
                    output_path=story3 / 'trajectory_spearman.png',
                )
                print("    Saved story3/trajectory_spearman.png")

        best_score_traj_story3 = {}
        for ucb_key, seed_results in ucb_comparison_results.items():
            if ucb_key not in story3_display:
                continue
            name = story3_display[ucb_key]
            per_seed_best = []
            for result in seed_results:
                if result is None:
                    continue
                cycle_best = {}
                for row in result.get('cycle_metrics', []):
                    cycle = row.get('cycle', None)
                    best = row.get('best_so_far', None)
                    if cycle is not None and best is not None:
                        with contextlib.suppress(TypeError, ValueError):
                            cycle_best[int(cycle)] = float(best)
                if cycle_best:
                    per_seed_best.append(cycle_best)
            if per_seed_best:
                all_cycles = sorted(set().union(*per_seed_best))
                best_score_traj_story3[name] = {
                    c: (np.mean([s[c] for s in per_seed_best if c in s]),
                        np.std([s[c] for s in per_seed_best if c in s]))
                    for c in all_cycles
                }
        if best_score_traj_story3:
            plot_cycle_progression(
                best_score_traj_story3, metric_key='best_so_far',
                metric_label='Best Score Found',
                title='3. Does better UQ find better compounds faster?',
                subtitle='500K dataset, mean ± std across 3 seeds',
                output_path=story3 / 'trajectory_best_score.png',
            )
            print("    Saved story3/trajectory_best_score.png")

        unlabeled_spearman_traj = {}
        for ucb_key, seed_results in ucb_comparison_results.items():
            if ucb_key not in story3_display:
                continue
            name = story3_display[ucb_key]
            per_seed_corr = []
            for result in seed_results:
                if result is None:
                    continue
                cycle_corr = {}
                for row in result.get('cycle_metrics', []):
                    cycle = row.get('cycle', None)
                    corr = row.get('unlabeled_spearman_correlation', None)
                    if cycle is not None and corr is not None and corr != '':
                        with contextlib.suppress(TypeError, ValueError):
                            cycle_corr[int(cycle)] = float(corr)
                if cycle_corr:
                    per_seed_corr.append(cycle_corr)
            if per_seed_corr:
                all_cycles = sorted(set().union(*per_seed_corr))
                unlabeled_spearman_traj[name] = {
                    c: (np.mean([s[c] for s in per_seed_corr if c in s]),
                        np.std([s[c] for s in per_seed_corr if c in s]))
                    for c in all_cycles
                }
        if unlabeled_spearman_traj:
            plot_cycle_progression(
                unlabeled_spearman_traj, metric_key='unlabeled_spearman',
                metric_label='Spearman \u03c1 (predicted vs true, unlabeled)',
                title='3. How well do predictions correlate with ground truth on unseen data?',
                subtitle='500K dataset, mean \u00b1 std across 3 seeds',
                output_path=story3 / 'trajectory_unlabeled_spearman.png',
            )
            print("    Saved story3/trajectory_unlabeled_spearman.png")

    # ═══════════════════════════════════════════════════════════════════════
    # CSV EXPORT
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  CSV EXPORT")
    print("=" * 80)

    export_metrics_csv(
        {ENSEMBLES[k]['name']: v for k, v in metrics_30k.items()},
        metrics_dir / 'uq_metrics_30k.csv', 'ampc_30k'
    )

    if metrics_500k:
        export_metrics_csv(
            {ENSEMBLES[k]['name']: v for k, v in metrics_500k.items()},
            metrics_dir / 'uq_metrics_500k.csv', 'ampc_500k'
        )

    export_calibration_csv(
        {ENSEMBLES[k]['name']: v for k, v in calibration_30k.items()},
        metrics_dir / 'calibration_comparison.csv'
    )

    if ablation_metrics:
        export_ablation_csv(ablation_metrics, metrics_dir / 'ablation_rf_ntrees_500k.csv')

    if rf_member_ablation_metrics:
        export_ablation_csv(rf_member_ablation_metrics,
                            metrics_dir / 'ablation_rf_members_500k.csv')

    if xgb_ablation_metrics:
        export_ablation_csv(xgb_ablation_metrics,
                            metrics_dir / 'ablation_xgb_members_500k.csv')

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("  VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\n  Total runtime: {total_elapsed/60:.1f} minutes")
    print("\n  Outputs:")
    print(f"    Plots:       {plots_dir}")
    print(f"    Metrics:     {metrics_dir}")
    print(f"    Story plots: {plots_presentation}")
    print(f"      1a: {story1a}")
    print(f"      1b: {story1b}")
    print(f"      2a: {story2a}")
    print(f"      2b: {story2b}")
    print(f"      2c: {story2c}")
    print(f"      2d: {story2d}")
    print(f"      3:  {story3}")
    print(f"\n  30K Results (mean +/- std across {len(RANDOM_SEEDS)} seeds):")
    print(f"  {'Model':25s} {'ENCE':>12s} {'NLL':>12s} {'Miscal':>12s} {'AUSE':>12s} {'Spearman':>12s}")
    print(f"  {'-'*85}")

    for ens_key in metrics_30k:
        m = metrics_30k[ens_key]['mean']
        s = metrics_30k[ens_key]['std']
        if not m:
            continue
        name = ENSEMBLES[ens_key]['name']
        print(f"  {name:25s} "
              f"{m.get('ence', np.nan):5.3f}+/-{s.get('ence', 0):.3f} "
              f"{m.get('nll', np.nan):5.2f}+/-{s.get('nll', 0):.2f} "
              f"{m.get('miscalibration_area', np.nan):5.3f}+/-{s.get('miscalibration_area', 0):.3f} "
              f"{m.get('ause', np.nan):6.4f}+/-{s.get('ause', 0):.4f} "
              f"{m.get('spearman_rho', np.nan):5.3f}+/-{s.get('spearman_rho', 0):.3f}")

    if metrics_500k:
        print(f"\n  500K Results (mean +/- std across {len(RANDOM_SEEDS)} seeds):")
        print(f"  {'Model':25s} {'ENCE':>12s} {'NLL':>12s} {'Miscal':>12s} {'AUSE':>12s} {'Spearman':>12s}")
        print(f"  {'-'*85}")
        for ens_key in metrics_500k:
            m = metrics_500k[ens_key]['mean']
            s = metrics_500k[ens_key]['std']
            if not m:
                continue
            name = ENSEMBLES[ens_key]['name']
            print(f"  {name:25s} "
                  f"{m.get('ence', np.nan):5.3f}+/-{s.get('ence', 0):.3f} "
                  f"{m.get('nll', np.nan):5.2f}+/-{s.get('nll', 0):.2f} "
                  f"{m.get('miscalibration_area', np.nan):5.3f}+/-{s.get('miscalibration_area', 0):.3f} "
                  f"{m.get('ause', np.nan):6.4f}+/-{s.get('ause', 0):.4f} "
                  f"{m.get('spearman_rho', np.nan):5.3f}+/-{s.get('spearman_rho', 0):.3f}")

    if ablation_metrics:
        print("\n  RF Tree Count Ablation (500K):")
        print(f"  {'Trees':>6s} {'AUSE':>12s} {'Miscal':>12s} {'MAE':>12s}")
        print(f"  {'-'*45}")
        for n_trees in sorted(ablation_metrics.keys()):
            m = ablation_metrics[n_trees]['metrics_mean']
            s = ablation_metrics[n_trees]['metrics_std']
            print(f"  {n_trees:6d} "
                  f"{m.get('ause', np.nan):5.4f}+/-{s.get('ause', 0):.4f} "
                  f"{m.get('miscalibration_area', np.nan):5.3f}+/-{s.get('miscalibration_area', 0):.3f} "
                  f"{m.get('mae', np.nan):6.4f}+/-{s.get('mae', 0):.4f}")

    if rf_member_ablation_metrics:
        print("\n  RF Member Ablation (500K):")
        print(f"  {'Members':>7s} {'AUSE':>12s} {'Miscal':>12s} {'MAE':>12s}")
        print(f"  {'-'*45}")
        for n_mem in sorted(rf_member_ablation_metrics.keys()):
            m = rf_member_ablation_metrics[n_mem]['metrics_mean']
            s = rf_member_ablation_metrics[n_mem]['metrics_std']
            print(f"  {n_mem:7d} "
                  f"{m.get('ause', np.nan):5.4f}+/-{s.get('ause', 0):.4f} "
                  f"{m.get('miscalibration_area', np.nan):5.3f}+/-{s.get('miscalibration_area', 0):.3f} "
                  f"{m.get('mae', np.nan):6.4f}+/-{s.get('mae', 0):.4f}")

    print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from learnm8 import run_active_learning
from learnm8.api import LEARNER_REGISTRY, _create_learner
from learnm8.oracles import CSVOracle
from learnm8.features import FEATURIZER_REGISTRY, FEATURIZERS_2D, FEATURIZERS_3D
from validation.lib import (
    load_validation_dataset,
    get_dataset_path,
    get_dataset_info
)
from validation.lib.featurizer_visualizations import (
    generate_comprehensive_visualizations,
    create_all_cost_performance_plots
)
from validation.lib.seed_aggregation import aggregate_seed_results

from learnm8 import setup_logging
setup_logging(level='INFO')


DATASET_NAME = 'ampc_30k'
N_CYCLES = 10
BATCH_FRACTION = 0.01
ACQUISITION = 'greedy'
RANDOM_SEED = 42
OUTPUT_BASE = Path('validation/reports/learner_featurizer_matrix_single')

ENSEMBLE_LEARNERS = ['ensemble', 'rf_ensemble', 'lr_ensemble', 'xgb_ensemble',
                     'dt_ensemble', 'mixed_ensemble', 'fastprop_ensemble',
                     'chemprop_ensemble']

MATRIX_FEATURIZERS = sorted(FEATURIZERS_2D+FEATURIZERS_3D)


def print_header():
    print("=" * 80)
    print("LEARNER-FEATURIZER MATRIX VALIDATION (Single Seed)")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Cycles: {N_CYCLES}")
    print(f"Batch Fraction: {BATCH_FRACTION}")
    print(f"Acquisition: {ACQUISITION}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Featurizers: {MATRIX_FEATURIZERS}")
    print("=" * 80)
    print()


def get_compatible_combinations() -> List[Tuple[str, Optional[str], str]]:
    """Returns (learner_name, featurizer, display_name) tuples."""
    combinations = []

    for learner_name in sorted(LEARNER_REGISTRY.keys()):
        if learner_name in ENSEMBLE_LEARNERS:
            continue
        if learner_name == 'gp':
            continue

        try:
            learner = _create_learner(learner_name, RANDOM_SEED)

            if learner.requires_smiles():
                combinations.append((learner_name, None, learner_name))
                for featurizer_name in MATRIX_FEATURIZERS:
                    combinations.append((learner_name, featurizer_name, learner_name))
            else:
                for featurizer_name in MATRIX_FEATURIZERS:
                    combinations.append((learner_name, featurizer_name, learner_name))
        except Exception as e:
            print(f"Warning: Could not create learner {learner_name}: {e}")
            continue

    return combinations


def check_existing_results(learner: str, featurizer: Optional[str]) -> bool:
    featurizer_str = featurizer if featurizer is not None else 'none'
    output_dir = OUTPUT_BASE / 'data' / f'{learner}_{featurizer_str}'

    required_files = [
        'compounds_final.csv',
        'cycle_metrics.csv',
        'selection_history.csv'
    ]

    return all((output_dir / f).exists() for f in required_files)


def load_existing_results(learner: str, featurizer: Optional[str]) -> Dict:
    import polars as pl

    featurizer_str = featurizer if featurizer is not None else 'none'
    output_dir = OUTPUT_BASE / 'data' / f'{learner}_{featurizer_str}'

    compounds_df = pl.read_csv(output_dir / 'compounds_final.csv', comment_prefix='#')
    cycle_metrics_df = pl.read_csv(output_dir / 'cycle_metrics.csv', comment_prefix='#')

    cycle_metrics = cycle_metrics_df.to_dicts()

    return {
        'compounds_df': compounds_df,
        'cycle_metrics': cycle_metrics,
        'output_dir': output_dir,
        'elapsed_time': None
    }


def wrap_single_seed_result(result_data: Dict, output_dir: Path) -> Dict:
    """Wrap single-seed result into multi-seed format for visualization compatibility."""
    seed_results = {RANDOM_SEED: result_data}
    aggregated = aggregate_seed_results(seed_results)
    return {
        'seed_results': seed_results,
        'aggregated': aggregated,
        'cycle_metrics': aggregated['cycle_metrics_mean'],
        'compounds_df': result_data['compounds_df'],
        'output_dir': output_dir
    }


def run_single_experiment(
    learner_name: str,
    featurizer_name: Optional[str],
    compound_pool,
    oracle,
    target_col: str,
    score_direction: str,
    cache_dir: Path
) -> Dict:
    featurizer_str = featurizer_name if featurizer_name is not None else 'none'
    output_dir = OUTPUT_BASE / 'data' / f'{learner_name}_{featurizer_str}'

    start_time = time.time()

    try:
        results = run_active_learning(
            compound_pool=compound_pool.clone(),
            oracle=oracle,
            learner=learner_name,
            target_col=target_col,
            featurizer=featurizer_name,
            n_cycles=N_CYCLES,
            batch_fraction=BATCH_FRACTION,
            strategy=ACQUISITION,
            score_direction=score_direction,
            random_state=RANDOM_SEED,
            cache_dir=cache_dir,
            output_dir=output_dir,
            mode='benchmark'
        )

        elapsed_time = time.time() - start_time

        return {
            'compounds_df': results['compounds_df'],
            'cycle_metrics': results['cycle_metrics'],
            'output_dir': output_dir,
            'elapsed_time': elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        raise


def run_all_experiments() -> Dict[Tuple[str, Optional[str]], Dict]:
    print_header()

    print("Loading dataset...")
    compound_pool, metadata = load_validation_dataset(
        dataset_name=DATASET_NAME,
        clean_invalid_scores=True,
        random_state=RANDOM_SEED
    )
    target_col = metadata['target_column']
    score_direction = metadata['score_direction']
    n_compounds = len(compound_pool)
    print(f"Loaded {n_compounds:,} compounds")
    print(f"  Score direction: {score_direction}")
    print()

    print("Creating oracle...")
    dataset_path = get_dataset_path(DATASET_NAME)
    oracle = CSVOracle(str(dataset_path), id_column='ID')
    print(f"Oracle created from {dataset_path}")
    print()

    print("Setting up cache directory...")
    cache_dir = OUTPUT_BASE / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    print()

    print("Identifying compatible learner-featurizer combinations...")
    combinations = get_compatible_combinations()
    print(f"Found {len(combinations)} compatible combinations")
    print()

    print("Compatible learners:")
    learners_seen = set()
    for learner_spec, featurizer, display_name in combinations:
        if display_name not in learners_seen:
            learners_seen.add(display_name)
            print(f"  - {display_name}")
    print()

    all_results = {}
    completed = 0
    skipped = 0
    failed = 0

    print(f"Starting experiments ({len(combinations)} total)...")
    print("=" * 80)
    print()

    pbar = tqdm(combinations, desc="Experiments", unit="exp", smoothing=0)

    for learner_spec, featurizer_name, display_name in pbar:
        learner_name = display_name if display_name else learner_spec
        featurizer_display = featurizer_name if featurizer_name is not None else 'none'
        pbar.set_description(f"{learner_name} + {featurizer_display}")

        featurizer_str = featurizer_name if featurizer_name is not None else 'none'
        output_dir = OUTPUT_BASE / 'data' / f'{learner_name}_{featurizer_str}'

        if check_existing_results(learner_name, featurizer_name):
            try:
                result_data = load_existing_results(learner_name, featurizer_name)
                all_results[(learner_name, featurizer_name)] = wrap_single_seed_result(
                    result_data, output_dir
                )
                skipped += 1
                continue
            except Exception as e:
                pbar.write(f"Warning: Could not load existing results: {e}")
                pbar.write(f"Will re-run experiment...")

        try:
            result_data = run_single_experiment(
                learner_name,
                featurizer_name,
                compound_pool,
                oracle,
                target_col,
                score_direction,
                cache_dir
            )
            all_results[(learner_name, featurizer_name)] = wrap_single_seed_result(
                result_data, output_dir
            )
            completed += 1

        except Exception as e:
            pbar.write(f"FAILED {learner_name} + {featurizer_display}: {e}")
            failed += 1

    pbar.close()
    print()

    print("=" * 80)
    print(f"Experiments completed: {completed}")
    print(f"Experiments skipped (existing): {skipped}")
    print(f"Experiments failed: {failed}")
    print(f"Total combinations with results: {len(all_results)}")
    print()

    return all_results, metadata, n_compounds


def main():
    try:
        start_time = time.time()

        all_results, metadata, n_compounds = run_all_experiments()

        if not all_results:
            print("ERROR: No results to visualize!")
            sys.exit(1)

        print("Generating visualizations...")
        print("=" * 80)
        print()

        config = {
            'dataset_name': DATASET_NAME,
            'n_compounds': n_compounds,
            'n_cycles': N_CYCLES,
            'batch_fraction': BATCH_FRACTION,
            'batch_size': int(n_compounds * BATCH_FRACTION),
            'acquisition': ACQUISITION,
            'random_seeds': [RANDOM_SEED]
        }

        viz_paths = generate_comprehensive_visualizations(
            all_results,
            OUTPUT_BASE,
            config
        )

        cost_perf_paths = create_all_cost_performance_plots(
            all_results,
            OUTPUT_BASE,
            config
        )

        total_time = time.time() - start_time

        print()
        print("=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        print(f"Total experiments: {len(all_results)}")
        print(f"Total runtime: {total_time/60:.1f} minutes")
        print()
        print("Generated files:")
        print(f"  - Combined heatmap: {viz_paths['heatmap']}")
        print(f"  - Cycle plot: {viz_paths['cycle_plot']}")
        print(f"  - Report: {viz_paths['report']}")
        print()
        print("Cost/Performance Analysis:")
        print(f"  - Heatmaps ({len(cost_perf_paths['heatmaps'])} files):")
        for heatmap_path in cost_perf_paths['heatmaps']:
            print(f"      {heatmap_path}")
        print()
        print(f"All results saved to: {OUTPUT_BASE}")
        print("=" * 80)

        return 0

    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        return 130

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

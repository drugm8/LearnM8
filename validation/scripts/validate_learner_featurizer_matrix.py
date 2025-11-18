#!/usr/bin/env python3
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from learnm8 import run_active_learning
from learnm8.api import LEARNER_REGISTRY, _create_learner
from learnm8.oracles import CSVOracle
from learnm8.learners.torch import ChempropLearner
from learnm8.learners.ensemble import ChempropEnsemble
from validation.lib import (
    load_validation_dataset,
    get_dataset_path,
    get_dataset_info
)
from validation.lib.featurizer_visualizations import (
    generate_comprehensive_visualizations,
    create_all_cost_performance_plots
)
from validation.lib.seed_aggregation import (
    aggregate_seed_results,
    check_existing_seed_results,
    save_aggregated_results,
    load_existing_seed_results
)

from learnm8 import setup_logging
setup_logging(level='DEBUG')


DATASET_NAME = 'ampc_30k'
N_CYCLES = 10
BATCH_FRACTION = 0.01
ACQUISITION = 'greedy'
RANDOM_SEEDS = [42, 123, 456]
OUTPUT_BASE = Path('validation/reports/learner_featurizer_matrix')


def print_header():
    print("=" * 80)
    print("LEARNER-FEATURIZER MATRIX VALIDATION (Multi-Seed)")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Cycles: {N_CYCLES}")
    print(f"Batch Fraction: {BATCH_FRACTION}")
    print(f"Acquisition: {ACQUISITION}")
    print(f"Random Seeds: {RANDOM_SEEDS}")
    print(f"Random Seeds: {RANDOM_SEEDS}")
    print("=" * 80)
    print()


def get_compatible_combinations() -> List[Tuple[Union[str, object], Optional[str], str]]:
    """Returns (learner_spec, featurizer, display_name) tuples."""
    combinations = []

    available_featurizers = ['morgan', 'maccs', 'ecfp6', 'descriptors', 'morgan_feat']

    # Create explicit learner instances with custom parameters
    learner_instances = {}
    for learner_name in sorted(LEARNER_REGISTRY.keys()):
        if learner_name in ['ensemble', 'gp']:
            continue

        try:
            learner = _create_learner(learner_name, RANDOM_SEEDS[0])

            if learner.requires_smiles():
                combinations.append((learner_name, None))
                for featurizer_name in available_featurizers:
                    combinations.append((learner_instance, featurizer_name, learner_name))
            else:
                for featurizer_name in available_featurizers:
                    combinations.append((learner_instance, featurizer_name, learner_name))

    # Fine-tuning versions with special markers for dynamic checkpoint directory creation
    combinations.append(('chemprop_with_ft', None, 'chemprop_ft'))
    for featurizer_name in available_featurizers:
        combinations.append(('chemprop_with_ft', featurizer_name, 'chemprop_ft'))

    combinations.append(('chemprop_ensemble_with_ft', None, 'chemprop_ensemble_ft'))
    for featurizer_name in available_featurizers:
        combinations.append(('chemprop_ensemble_with_ft', featurizer_name, 'chemprop_ensemble_ft'))

    return combinations


def check_existing_results(learner: str, featurizer: Optional[str], seed: int) -> bool:
    featurizer_str = featurizer if featurizer is not None else 'none'
    output_dir = OUTPUT_BASE / 'data' / f'{learner}_{featurizer_str}' / f'seed_{seed}'

    required_files = [
        'compounds_final.csv',
        'cycle_metrics.csv',
        'selection_history.csv'
    ]

    return all((output_dir / f).exists() for f in required_files)


def load_existing_results(learner: str, featurizer: Optional[str], seed: int) -> Dict:
    import polars as pl

    featurizer_str = featurizer if featurizer is not None else 'none'
    output_dir = OUTPUT_BASE / 'data' / f'{learner}_{featurizer_str}' / f'seed_{seed}'

    compounds_df = pl.read_csv(output_dir / 'compounds_final.csv', comment_prefix='#')
    cycle_metrics_df = pl.read_csv(output_dir / 'cycle_metrics.csv', comment_prefix='#')

    cycle_metrics = cycle_metrics_df.to_dicts()

    return {
        'compounds_df': compounds_df,
        'cycle_metrics': cycle_metrics,
        'output_dir': output_dir,
        'elapsed_time': None
    }


def run_single_experiment(
    learner_spec: Union[str, object],
    featurizer_name: Optional[str],
    display_name: str,
    compound_pool,
    oracle,
    target_col: str,
    score_direction: str,
    cache_dir: Path,
    seed: int
) -> Dict:
    featurizer_display = featurizer_name if featurizer_name is not None else 'none'
    print(f"  Running seed {seed}: {learner_name} + {featurizer_display}...", end=' ', flush=True)

    featurizer_str = featurizer_name if featurizer_name is not None else 'none'
    output_dir = OUTPUT_BASE / 'data' / f'{learner_name}_{featurizer_str}' / f'seed_{seed}'

        start_time = time.time()

    try:
        results = run_active_learning(
            compound_pool=compound_pool.clone(),
            oracle=oracle,
            learner=learner_name,
            target_col=target_col,
            featurizer_type=featurizer_name,
            n_cycles=N_CYCLES,
            batch_fraction=BATCH_FRACTION,
            strategy=ACQUISITION,
            score_direction=score_direction,
            random_state=seed,
            cache_dir=cache_dir,
            output_dir=output_dir,
            mode='benchmark'
        )

            elapsed_time = time.time() - start_time

            seed_results[seed] = {
                'compounds_df': results['compounds_df'],
                'cycle_metrics': results['cycle_metrics'],
                'elapsed_time': elapsed_time
            }

            print(f"✓ ({elapsed_time:.1f}s)")

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"✗ FAILED ({elapsed_time:.1f}s)")
            print(f"      Error: {str(e)}")
            raise

    if len(seed_results) < len(RANDOM_SEEDS):
        raise RuntimeError(
            f"Incomplete seed results: expected {len(RANDOM_SEEDS)}, got {len(seed_results)}"
        )

    aggregated = aggregate_seed_results(seed_results)
    save_aggregated_results(seed_results, aggregated, output_dir)

    return {
        'seed_results': seed_results,
        'aggregated': aggregated
    }


def run_all_experiments() -> Dict[Tuple[str, Optional[str]], Dict]:
    print_header()

    print("Loading dataset...")
    compound_pool, metadata = load_validation_dataset(
        dataset_name=DATASET_NAME,
        clean_invalid_scores=True,
        random_state=RANDOM_SEEDS[0]
    )
    target_col = metadata['target_column']
    score_direction = metadata['score_direction']
    n_compounds = len(compound_pool)
    print(f"✓ Loaded {n_compounds:,} compounds")
    print(f"  Score direction: {score_direction}")
    print()

    print("Creating oracle...")
    dataset_path = get_dataset_path(DATASET_NAME)
    oracle = CSVOracle(str(dataset_path), id_column='ID')
    print(f"✓ Oracle created from {dataset_path}")
    print()

    print("Setting up cache directory...")
    cache_dir = OUTPUT_BASE / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Cache directory: {cache_dir}")
    print()

    print("Identifying compatible learner-featurizer combinations...")
    combinations = get_compatible_combinations()
    print(f"✓ Found {len(combinations)} compatible combinations")
    print()

    print("Compatible combinations:")
    for learner_spec, featurizer, display_name in combinations:
        print(f"  - {display_name:20s} + {featurizer}")
    print()

    all_results = {}
    completed = 0
    skipped = 0
    failed = 0

    total_experiments = len(combinations) * len(RANDOM_SEEDS)
    print(f"Starting experiments ({len(combinations)} combinations × {len(RANDOM_SEEDS)} seeds = {total_experiments} total)...")
    print("=" * 80)
    print()

    for idx, (learner_name, featurizer_name) in enumerate(combinations, 1):
        featurizer_display = featurizer_name if featurizer_name is not None else 'none'
        print(f"[{idx}/{len(combinations)}] {learner_name} + {featurizer_display}")

        seed_results = {}

        for seed in RANDOM_SEEDS:
            if check_existing_results(learner_name, featurizer_name, seed):
                print(f"  Seed {seed}: Skipping (results already exist)")
                try:
                    result_data = load_existing_results(learner_name, featurizer_name, seed)
                    seed_results[seed] = result_data
                    skipped += 1
                except Exception as e:
                    print(f"  Warning: Could not load existing results for seed {seed}: {e}")
                    print(f"  Will re-run experiment...")

            if seed not in seed_results:
                try:
                    result_data = run_single_experiment(
                        learner_name,
                        featurizer_name,
                        compound_pool,
                        oracle,
                        target_col,
                        score_direction,
                        cache_dir,
                        seed
                    )
                    seed_results[seed] = result_data
                    completed += 1

                except Exception as e:
                    print(f"  Seed {seed}: Failed - {e}")
                    failed += 1

        if len(seed_results) > 0:
            print(f"  Aggregating results across {len(seed_results)} seeds...")
            aggregated = aggregate_seed_results(seed_results)

            featurizer_str = featurizer_name if featurizer_name is not None else 'none'
            combination_dir = OUTPUT_BASE / 'data' / f'{learner_name}_{featurizer_str}'
            save_aggregated_results(seed_results, aggregated, combination_dir)

            all_results[(learner_name, featurizer_name)] = {
                'seed_results': seed_results,
                'aggregated': aggregated,
                'cycle_metrics': aggregated['cycle_metrics_mean'],
                'compounds_df': seed_results[RANDOM_SEEDS[0]]['compounds_df'],
                'output_dir': combination_dir
            }
            print(f"  ✓ Aggregation complete")

        print()

    print("=" * 80)
    print(f"Individual experiments completed: {completed}")
    print(f"Individual experiments skipped (existing): {skipped}")
    print(f"Individual experiments failed: {failed}")
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
            'random_seeds': RANDOM_SEEDS
        }

        viz_paths = generate_comprehensive_visualizations(
            all_results,
            OUTPUT_BASE,
            config
        )

        # Generate cost/performance analysis
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
        print(f"  - Pareto plots ({len(cost_perf_paths['pareto_plots'])} files):")
        for pareto_path in cost_perf_paths['pareto_plots']:
            print(f"      {pareto_path}")
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

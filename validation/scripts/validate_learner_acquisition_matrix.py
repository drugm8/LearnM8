#!/usr/bin/env python3
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple, Union
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from learnm8 import run_active_learning
from learnm8.api import LEARNER_REGISTRY, _create_learner
from learnm8.acquisition import ACQUISITION_REGISTRY, get_acquisition_function
from learnm8.oracles import CSVOracle
from validation.lib import (
    load_validation_dataset,
    get_dataset_path,
    get_dataset_info
)
from validation.lib.matrix_visualizations import generate_comprehensive_visualizations
from validation.lib.seed_aggregation import (
    aggregate_seed_results,
    check_existing_seed_results,
    save_aggregated_results,
    load_existing_seed_results
)
from validation.lib.seed_aggregation import (
    aggregate_seed_results,
    check_existing_seed_results,
    save_aggregated_results,
    load_existing_seed_results
)


from learnm8 import setup_logging
setup_logging(level='INFO')

DATASET_NAME = 'ampc_30k'
N_CYCLES = 10
BATCH_FRACTION = 0.01
FEATURIZER_TYPE = 'morgan'
RANDOM_SEEDS = [42, 123, 456]
OUTPUT_BASE = Path('validation/reports/learner_acquisition_matrix')


def print_header():
    print("=" * 80)
    print("LEARNER-ACQUISITION MATRIX VALIDATION (Multi-Seed)")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Cycles: {N_CYCLES}")
    print(f"Batch Fraction: {BATCH_FRACTION}")
    print(f"Featurizer: {FEATURIZER_TYPE} (Chemprop learners use pure graph-based learning)")
    print(f"Random Seeds: {RANDOM_SEEDS}")
    print("=" * 80)
    print()


def get_compatible_combinations() -> List[Tuple[Union[str, object], str, str]]:
    """Returns (learner_spec, acquisition, display_name) tuples."""
    combinations = []

    basic_acquisitions = ['greedy', 'random', 'simulated_annealing']
    uncertainty_acquisitions = ['ucb', 'ei', 'pi', 'thompson', 'entropy']

    # Create explicit learner instances with custom parameters
    learner_instances = {}
    for learner_name in sorted(LEARNER_REGISTRY.keys()):
        if learner_name == 'ensemble' or learner_name == 'gp':
            continue

        try:
            learner = _create_learner(learner_name, RANDOM_SEEDS[0])
            supports_uncertainty = learner.supports_uncertainty()

            for acquisition_name in basic_acquisitions:
                combinations.append((learner_name, acquisition_name, learner_name))

            if supports_uncertainty:
                for acquisition_name in uncertainty_acquisitions:
                    combinations.append((learner_name, acquisition_name, learner_name))
        except Exception as e:
            print(f"Warning: Could not create learner {learner_name}: {e}")
            continue

    return combinations


def check_existing_results(learner: str, acquisition: str, seed: int) -> bool:
    output_dir = OUTPUT_BASE / 'data' / f'{learner}_{acquisition}' / f'seed_{seed}'

    required_files = [
        'compounds_final.csv',
        'cycle_metrics.csv',
        'selection_history.csv'
    ]

    return all((output_dir / f).exists() for f in required_files)


def load_existing_results(learner: str, acquisition: str, seed: int) -> Dict:
    import polars as pl

    output_dir = OUTPUT_BASE / 'data' / f'{learner}_{acquisition}' / f'seed_{seed}'

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
    learner_name: str,
    acquisition_name: str,
    compound_pool,
    oracle,
    target_col: str,
    score_direction: str,
    cache_dir: Path,
    seed: int
) -> Dict:
    output_dir = OUTPUT_BASE / 'data' / f'{learner_name}_{acquisition_name}' / f'seed_{seed}'

    start_time = time.time()

    featurizer = None if learner_name in ['chemprop', 'chemprop_ensemble'] else FEATURIZER_TYPE

    try:
        results = run_active_learning(
            compound_pool=compound_pool.clone(),
            oracle=oracle,
            learner=learner_name,
            target_col=target_col,
            featurizer=featurizer,
            n_cycles=N_CYCLES,
            batch_fraction=BATCH_FRACTION,
            strategy=acquisition_name,
            score_direction=score_direction,
            random_state=seed,
            cache_dir=cache_dir,
            output_dir=output_dir,
            mode='benchmark'
        )

        elapsed_time = time.time() - start_time

        return {
            'compounds_df': results['compounds_df'],
            'cycle_metrics': results['cycle_metrics'],
            'elapsed_time': elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        raise


def run_all_experiments() -> Dict[Tuple[str, str], Dict]:
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

    print("Identifying compatible learner-acquisition combinations...")
    combinations = get_compatible_combinations()
    print(f"✓ Found {len(combinations)} compatible combinations")
    print()

    print("Compatible combinations:")
    for learner_spec, acquisition, display_name in combinations:
        print(f"  - {display_name:20s} + {acquisition}")
    print()

    all_results = {}
    completed = 0
    skipped = 0
    failed = 0

    total_experiments = len(combinations) * len(RANDOM_SEEDS)
    print(f"Starting experiments ({len(combinations)} combinations × {len(RANDOM_SEEDS)} seeds = {total_experiments} total)...")
    print("=" * 80)
    print()

    pbar_combinations = tqdm(combinations, desc="Combinations", unit="comb", smoothing=0, position=0)

    for learner_spec, acquisition_name, display_name in pbar_combinations:
        pbar_combinations.set_description(f"{display_name} + {acquisition_name}")

        seed_results = {}

        pbar_seeds = tqdm(RANDOM_SEEDS, desc="  Seeds", unit="seed", smoothing=0, position=1, leave=False)
        for seed in pbar_seeds:
            pbar_seeds.set_description(f"  Seed {seed}")

            if check_existing_results(display_name, acquisition_name, seed):
                try:
                    result_data = load_existing_results(display_name, acquisition_name, seed)
                    seed_results[seed] = result_data
                    skipped += 1
                    pbar_seeds.write(f"  Seed {seed}: Skipping (results already exist)")
                except Exception as e:
                    pbar_seeds.write(f"  Warning: Could not load existing results for seed {seed}: {e}")
                    pbar_seeds.write(f"  Will re-run experiment...")

            if seed not in seed_results:
                try:
                    result_data = run_single_experiment(
                        display_name,
                        acquisition_name,
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
                    pbar_seeds.write(f"  Seed {seed}: Failed - {e}")
                    failed += 1

        pbar_seeds.close()

        if len(seed_results) > 0:
            pbar_combinations.write(f"  Aggregating results across {len(seed_results)} seeds...")
            aggregated = aggregate_seed_results(seed_results)

            combination_dir = OUTPUT_BASE / 'data' / f'{display_name}_{acquisition_name}'
            save_aggregated_results(seed_results, aggregated, combination_dir)

            all_results[(display_name, acquisition_name)] = {
                'seed_results': seed_results,
                'aggregated': aggregated,
                'cycle_metrics': aggregated['cycle_metrics_mean'],
                'compounds_df': seed_results[RANDOM_SEEDS[0]]['compounds_df'],
                'output_dir': combination_dir
            }
            pbar_combinations.write(f"  ✓ Aggregation complete")

    pbar_combinations.close()
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
            'featurizer': f"{FEATURIZER_TYPE} (chemprop: None)",
            'random_seeds': RANDOM_SEEDS
        }

        viz_paths = generate_comprehensive_visualizations(
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

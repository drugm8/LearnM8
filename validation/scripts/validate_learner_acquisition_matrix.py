#!/usr/bin/env python3
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple, Union
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from learnm8 import run_active_learning
from learnm8.api import LEARNER_REGISTRY, _create_learner
from learnm8.acquisition import ACQUISITION_REGISTRY, get_acquisition_function
from learnm8.oracles import CSVOracle
from learnm8.learners.torch import ChempropLearner
from learnm8.learners.ensemble import ChempropEnsemble
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


from learnm8 import setup_logging
setup_logging(level='INFO')

DATASET_NAME = 'ampc_30k'
N_CYCLES = 10
BATCH_FRACTION = 0.01
FEATURIZER_TYPE = 'morgan'
RANDOM_STATE = 42
RANDOM_SEEDS = [42, 123, 456]
OUTPUT_BASE = Path('validation/reports/learner_acquisition_matrix')


def print_header():
    print("=" * 80)
    print("LEARNER-ACQUISITION MATRIX VALIDATION (Multi-Seed)")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Cycles: {N_CYCLES}")
    print(f"Batch Fraction: {BATCH_FRACTION}")
    print(f"Featurizer: {FEATURIZER_TYPE}")
    print(f"Random Seeds: {RANDOM_SEEDS}")
    print("=" * 80)
    print()


def get_compatible_combinations() -> List[Tuple[Union[str, object], str, str]]:
    """Returns (learner_spec, acquisition, display_name) tuples."""
    combinations = []

    basic_acquisitions = ['greedy', 'random']
    uncertainty_acquisitions = ['ucb', 'ei', 'pi', 'thompson', 'entropy']

    # Create explicit learner instances with custom parameters
    learner_instances = {}
    for learner_name in sorted(LEARNER_REGISTRY.keys()):
        if learner_name == 'ensemble' or learner_name == 'gp':
            continue

        try:
            if learner_name == 'chemprop':
                learner_instances[learner_name] = ChempropLearner(
                    random_state=RANDOM_STATE,
                    early_stopping_patience=3
                )
            elif learner_name == 'chemprop_ensemble':
                learner_instances[learner_name] = ChempropEnsemble(
                    random_state=RANDOM_STATE,
                    early_stopping_patience=3
                )
            else:
                learner_instances[learner_name] = _create_learner(learner_name, RANDOM_STATE)
        except Exception as e:
            print(f"Warning: Could not create learner '{learner_name}': {e}")
            continue

    # Build combinations using learner instances
    for learner_name, learner_instance in learner_instances.items():
        supports_uncertainty = learner_instance.supports_uncertainty()

        if learner_name == 'chemprop':
            for acquisition_name in basic_acquisitions:
                combinations.append((learner_instance, acquisition_name, 'chemprop_no_ft'))

        elif learner_name == 'chemprop_ensemble':
            for acquisition_name in basic_acquisitions:
                combinations.append((learner_instance, acquisition_name, 'chemprop_ensemble_no_ft'))
            if supports_uncertainty:
                for acquisition_name in uncertainty_acquisitions:
                    combinations.append((learner_instance, acquisition_name, 'chemprop_ensemble_no_ft'))

        else:
            for acquisition_name in basic_acquisitions:
                combinations.append((learner_instance, acquisition_name, learner_name))

            if supports_uncertainty:
                for acquisition_name in uncertainty_acquisitions:
                    combinations.append((learner_instance, acquisition_name, learner_name))

    # Fine-tuning versions with special markers for dynamic checkpoint directory creation
    for acquisition_name in basic_acquisitions:
        combinations.append(('chemprop_with_ft', acquisition_name, 'chemprop_ft'))

    for acquisition_name in basic_acquisitions + uncertainty_acquisitions:
        combinations.append(('chemprop_ensemble_with_ft', acquisition_name, 'chemprop_ensemble_ft'))

    return combinations


def check_missing_seeds(display_name: str, acquisition: str) -> List[int]:
    """Check which random seeds need to be run for this experiment.

    Returns:
        List of seed values that are missing or incomplete.
    """
    return check_existing_seed_results(
        display_name=display_name,
        condition=acquisition,
        output_base=OUTPUT_BASE,
        seeds=RANDOM_SEEDS
    )


def run_single_experiment(
    learner_spec: Union[str, object],
    acquisition_name: str,
    display_name: str,
    compound_pool,
    oracle,
    target_col: str,
    cache_dir: Path,
    missing_seeds: List[int]
) -> Dict:
    """Run experiment for all missing random seeds and aggregate results.

    Args:
        missing_seeds: List of seed values that need to be run

    Returns:
        Dictionary with aggregated results across all seeds:
        {
            'seed_results': {seed: {...}, ...},
            'aggregated': {...}
        }
    """
    output_dir = OUTPUT_BASE / 'data' / f'{display_name}_{acquisition_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_results = load_existing_seed_results(
        display_name=display_name,
        condition=acquisition_name,
        output_base=OUTPUT_BASE,
        seeds=RANDOM_SEEDS
    )

    print(f"  Running {len(missing_seeds)} seed(s): {missing_seeds}")

    for seed in missing_seeds:
        seed_output_dir = output_dir / f'seed_{seed}'
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"    Seed {seed}...", end=' ', flush=True)

        if learner_spec == 'chemprop_with_ft':
            checkpoint_dir = seed_output_dir / '.checkpoints'
            learner_instance = ChempropLearner(
                random_state=seed,
                early_stopping_patience=3,
                enable_fine_tuning=True,
                checkpoint_dir=checkpoint_dir
            )
        elif learner_spec == 'chemprop_ensemble_with_ft':
            checkpoint_dir = seed_output_dir / '.checkpoints'
            learner_instance = ChempropEnsemble(
                random_state=seed,
                early_stopping_patience=3,
                enable_fine_tuning=True,
                checkpoint_dir=checkpoint_dir
            )
        else:
            learner_instance = learner_spec

        start_time = time.time()

        try:
            results = run_active_learning(
                compound_pool=compound_pool.copy(),
                oracle=oracle,
                learner=learner_instance,
                target_col=target_col,
                featurizer_type=FEATURIZER_TYPE,
                n_cycles=N_CYCLES,
                batch_fraction=BATCH_FRACTION,
                strategy=acquisition_name,
                random_state=seed,
                cache_dir=cache_dir,
                output_dir=seed_output_dir,
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


def run_all_experiments() -> Dict[Tuple[str, str], Dict]:
    print_header()

    print("Loading dataset...")
    compound_pool, metadata = load_validation_dataset(
        dataset_name=DATASET_NAME,
        clean_invalid_scores=True,
        random_state=RANDOM_STATE
    )
    target_col = metadata['target_column']
    n_compounds = len(compound_pool)
    print(f"✓ Loaded {n_compounds:,} compounds")
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

    print(f"Starting experiments ({len(combinations)} total)...")
    print("=" * 80)
    print()

    for idx, (learner_spec, acquisition_name, display_name) in enumerate(combinations, 1):
        print(f"[{idx}/{len(combinations)}] {display_name} + {acquisition_name}")

        missing_seeds = check_missing_seeds(display_name, acquisition_name)

        if not missing_seeds:
            print("  All seeds complete, loading existing results...")
            try:
                seed_results = load_existing_seed_results(
                    display_name=display_name,
                    condition=acquisition_name,
                    output_base=OUTPUT_BASE,
                    seeds=RANDOM_SEEDS
                )
                aggregated = aggregate_seed_results(seed_results)
                result_data = {
                    'seed_results': seed_results,
                    'aggregated': aggregated
                }
                all_results[(display_name, acquisition_name)] = result_data
                skipped += 1
            except Exception as e:
                print(f"  Warning: Could not load existing results: {e}")
                print("  Will re-run all seeds...")
                missing_seeds = RANDOM_SEEDS
            else:
                print()
                continue

        try:
            result_data = run_single_experiment(
                learner_spec,
                acquisition_name,
                display_name,
                compound_pool,
                oracle,
                target_col,
                cache_dir,
                missing_seeds
            )
            all_results[(display_name, acquisition_name)] = result_data
            completed += 1

        except Exception as e:
            print(f"  Failed to complete experiment: {e}")
            failed += 1

        print()

    print("=" * 80)
    print(f"Experiments completed: {completed}")
    print(f"Experiments skipped (existing): {skipped}")
    print(f"Experiments failed: {failed}")
    print(f"Total results: {len(all_results)}")
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
            'featurizer_type': FEATURIZER_TYPE,
            'random_state': RANDOM_STATE
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

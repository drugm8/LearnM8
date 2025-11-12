#!/usr/bin/env python3
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from learnm8 import run_active_learning
from learnm8.api import LEARNER_REGISTRY, _create_learner
from learnm8.oracles import CSVOracle
from validation.lib import (
    load_validation_dataset,
    get_dataset_path,
    get_dataset_info
)
from validation.lib.featurizer_visualizations import generate_comprehensive_visualizations

from learnm8 import setup_logging
setup_logging(level='INFO')


DATASET_NAME = 'ampc_30k'
N_CYCLES = 10
BATCH_FRACTION = 0.01
ACQUISITION = 'greedy'
RANDOM_STATE = 42
OUTPUT_BASE = Path('validation/reports/learner_featurizer_matrix')


def print_header():
    print("=" * 80)
    print("LEARNER-FEATURIZER MATRIX VALIDATION")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Cycles: {N_CYCLES}")
    print(f"Batch Fraction: {BATCH_FRACTION}")
    print(f"Acquisition: {ACQUISITION}")
    print(f"Random State: {RANDOM_STATE}")
    print("=" * 80)
    print()


def get_compatible_combinations() -> List[Tuple[str, Optional[str]]]:
    combinations = []

    available_featurizers = ['morgan', 'maccs', 'ecfp6', 'descriptors', 'morgan_feat']

    for learner_name in sorted(LEARNER_REGISTRY.keys()):
        if learner_name in ['ensemble', 'gp']:
            continue

        try:
            learner = _create_learner(learner_name, RANDOM_STATE)

            if learner.requires_smiles():
                combinations.append((learner_name, None))
                for featurizer_name in available_featurizers:
                    combinations.append((learner_name, featurizer_name))
            else:
                for featurizer_name in available_featurizers:
                    combinations.append((learner_name, featurizer_name))

        except Exception as e:
            print(f"Warning: Could not check learner '{learner_name}': {e}")
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


def run_single_experiment(
    learner_name: str,
    featurizer_name: Optional[str],
    compound_pool,
    oracle,
    target_col: str,
    cache_dir: Path
) -> Dict:
    featurizer_display = featurizer_name if featurizer_name is not None else 'none'
    print(f"  Running: {learner_name} + {featurizer_display}...", end=' ', flush=True)

    featurizer_str = featurizer_name if featurizer_name is not None else 'none'
    output_dir = OUTPUT_BASE / 'data' / f'{learner_name}_{featurizer_str}'

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
            random_state=RANDOM_STATE,
            cache_dir=cache_dir,
            output_dir=output_dir,
            mode='benchmark'
        )

        elapsed_time = time.time() - start_time

        print(f"✓ ({elapsed_time:.1f}s)")

        return {
            'compounds_df': results['compounds_df'],
            'cycle_metrics': results['cycle_metrics'],
            'output_dir': output_dir,
            'elapsed_time': elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ FAILED ({elapsed_time:.1f}s)")
        print(f"    Error: {str(e)}")
        raise


def run_all_experiments() -> Dict[Tuple[str, Optional[str]], Dict]:
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

    print("Identifying compatible learner-featurizer combinations...")
    combinations = get_compatible_combinations()
    print(f"✓ Found {len(combinations)} compatible combinations")
    print()

    print("Compatible combinations:")
    for learner, featurizer in combinations:
        print(f"  - {learner:20s} + {featurizer}")
    print()

    all_results = {}
    completed = 0
    skipped = 0
    failed = 0

    print(f"Starting experiments ({len(combinations)} total)...")
    print("=" * 80)
    print()

    for idx, (learner_name, featurizer_name) in enumerate(combinations, 1):
        print(f"[{idx}/{len(combinations)}] {learner_name} + {featurizer_name}")

        if check_existing_results(learner_name, featurizer_name):
            print(f"  Skipping: Results already exist")
            try:
                result_data = load_existing_results(learner_name, featurizer_name)
                all_results[(learner_name, featurizer_name)] = result_data
                skipped += 1
            except Exception as e:
                print(f"  Warning: Could not load existing results: {e}")
                print(f"  Will re-run experiment...")
            else:
                print()
                continue

        try:
            result_data = run_single_experiment(
                learner_name,
                featurizer_name,
                compound_pool,
                oracle,
                target_col,
                cache_dir
            )
            all_results[(learner_name, featurizer_name)] = result_data
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
            'acquisition': ACQUISITION,
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

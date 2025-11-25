import time
import yaml
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional

from learnm8 import run_active_learning, extract_features
from learnm8.oracles import CSVOracle
from learnm8.learners.ensemble import RFEnsemble
from validation.lib.data_loading import load_validation_dataset
from validation.lib.dataset_config import get_dataset_path


class ValidationRunner:

    def __init__(self, strategy_name: str, config_path: Optional[str] = None):
        self.strategy_name = strategy_name

        if config_path is None:
            validation_root = Path(__file__).parent.parent
            config_path = validation_root / 'config' / 'validation_strategies.yaml'
        else:
            config_path = Path(config_path)

        self.validation_root = Path(__file__).parent.parent
        self.config = self._load_config(config_path)
        self.global_config = self.config['global']
        self.strategy_config = self.config['strategies'][strategy_name]

        # Resolve output_base_dir relative to validation root
        output_base_dir = self.global_config['output_base_dir']
        if not Path(output_base_dir).is_absolute():
            self.global_config['output_base_dir'] = str(self.validation_root / output_base_dir)

        np.random.seed(self.global_config['numpy_seed'])

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def run_parameter_sweep(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Running {self.strategy_config['full_name']} validation")
        print(f"{'='*60}")
        print(f"Strategy: {self.strategy_name}")
        print(f"Parameter: {self.strategy_config['param_name']}")
        print(f"Values: {self.strategy_config['param_values']}")
        print(f"Dataset: {self.global_config['dataset_name']}")
        print(f"Cycles: {self.global_config['n_cycles']}")

        compound_pool, metadata = load_validation_dataset(
            dataset_name=self.global_config['dataset_name'],
            clean_invalid_scores=self.global_config['clean_invalid_scores'],
            random_state=self.global_config['numpy_seed']
        )

        target_column = metadata['target_column']
        score_direction = metadata['score_direction']

        print(f"\nDataset loaded:")
        print(f"  Compounds: {len(compound_pool):,}")
        print(f"  Target: {target_column}")
        print(f"  Direction: {score_direction}")

        dataset_path = get_dataset_path(self.global_config['dataset_name'])
        oracle = CSVOracle(str(dataset_path), id_column='ID')

        results = {}
        param_name = self.strategy_config['param_name']

        for param_value in self.strategy_config['param_values']:
            # Convert list to tuple for use as dictionary key (lists are unhashable)
            dict_key = tuple(param_value) if isinstance(param_value, list) else param_value

            print(f"\n{'-'*60}")
            print(f"Parameter: {param_name}={param_value}")
            print(f"{'-'*60}")

            output_dir = self._get_output_dir(param_value)

            if self._check_existing_outputs(output_dir):
                print(f"  ✓ Using existing data from {output_dir}")
                results[dict_key] = self._load_existing_results(output_dir, target_column)
            else:
                print(f"  Generating new data...")
                start_time = time.time()

                # Handle special case for multi-parameter configs
                if param_name == 'config' and isinstance(param_value, (list, tuple)):
                    # For simulated_annealing: [initial_temp, cooling_schedule]
                    acquisition_params = {
                        'initial_temp': param_value[0],
                        'cooling_schedule': param_value[1],
                        'max_iterations': 500  # Fixed for validation
                    }
                else:
                    acquisition_params = {param_name: param_value}

                al_results = run_active_learning(
                    compound_pool=compound_pool,
                    oracle=oracle,
                    learner=self.global_config['learner'],
                    target_col=target_column,
                    featurizer=self.global_config['featurizer'],
                    initial_strategy='random',
                    strategy=self.strategy_name,
                    n_cycles=self.global_config['n_cycles'],
                    batch_fraction=self.global_config['batch_fraction'],
                    score_direction=score_direction,
                    acquisition_params=acquisition_params,
                    output_dir=output_dir,
                    mode='benchmark',
                    cache_dir=Path(self.global_config['cache_dir'])
                )

                elapsed = time.time() - start_time
                print(f"  ✓ Completed in {elapsed:.1f}s")

                results[dict_key] = {
                    'output_dir': output_dir,
                    'cycle_metrics': al_results['cycle_metrics'],
                    'compounds_df': al_results['compounds_df'],
                    'target_column': target_column,
                    'score_direction': score_direction,
                    'elapsed_time': elapsed
                }

        print(f"\n{'='*60}")
        print(f"✓ Parameter sweep complete for {len(results)} values")
        print(f"{'='*60}")

        return results

    def _get_output_dir(self, param_value) -> Path:
        param_name = self.strategy_config['param_name']
        strategy_name = self.strategy_config['name']

        # Format param_value for directory name
        if isinstance(param_value, (list, tuple)):
            param_str = '_'.join(str(v) for v in param_value)
        else:
            param_str = str(param_value)

        dir_name = f"{strategy_name}_validation_{param_name}_{param_str}"
        return Path(self.global_config['output_base_dir']) / self.strategy_name / 'data' / dir_name

    def _check_existing_outputs(self, output_dir: Path) -> bool:
        required_files = ['compounds_final.csv', 'cycle_metrics.csv', 'selection_history.csv']
        return all((output_dir / f).exists() for f in required_files)

    def _load_existing_results(self, output_dir: Path, target_column: str) -> Dict[str, Any]:
        cycle_metrics = pl.read_csv(output_dir / 'cycle_metrics.csv', comment_prefix='#')
        compounds_df = pl.read_csv(output_dir / 'compounds_final.csv', comment_prefix='#')

        return {
            'output_dir': str(output_dir),
            'cycle_metrics': cycle_metrics.to_dicts(),
            'compounds_df': compounds_df,
            'target_column': target_column,
            'score_direction': 'lower',
            'elapsed_time': None
        }


class PruningValidationRunner(ValidationRunner):

    def run_parameter_sweep(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Running {self.strategy_config['full_name']}")
        print(f"{'='*60}")
        print(f"Parameter: {self.strategy_config['param_name']}")
        print(f"Values: {self.strategy_config['param_values']}")
        print(f"Dataset: {self.global_config['dataset_name']}")
        print(f"Cycles: {self.global_config['n_cycles']}")

        compound_pool, metadata = load_validation_dataset(
            dataset_name=self.global_config['dataset_name'],
            clean_invalid_scores=self.global_config['clean_invalid_scores'],
            random_state=self.global_config['numpy_seed']
        )

        target_column = metadata['target_column']
        score_direction = metadata['score_direction']

        print(f"\nDataset loaded:")
        print(f"  Compounds: {len(compound_pool):,}")
        print(f"  Target: {target_column}")
        print(f"  Direction: {score_direction}")

        dataset_path = get_dataset_path(self.global_config['dataset_name'])
        oracle = CSVOracle(str(dataset_path), id_column='ID')

        results = {}
        param_name = self.strategy_config['param_name']

        for param_value in self.strategy_config['param_values']:
            print(f"\n{'-'*60}")
            print(f"Pruning Fraction: {param_value:.1%}")
            print(f"{'-'*60}")

            output_dir = self._get_pruning_output_dir(param_value)

            if self._check_existing_outputs(output_dir):
                print(f"  ✓ Using existing data from {output_dir}")
                results[param_value] = self._load_existing_results(output_dir, target_column)
            else:
                print(f"  Generating new data...")
                start_time = time.time()

                pruning_strategy = 'score' if param_value > 0 else None
                pruning_params = {'pruning_fraction': param_value} if param_value > 0 else None

                al_results = run_active_learning(
                    compound_pool=compound_pool,
                    oracle=oracle,
                    learner=self.global_config['learner'],
                    target_col=target_column,
                    featurizer=self.global_config['featurizer'],
                    initial_strategy='random',
                    strategy='greedy',
                    n_cycles=self.global_config['n_cycles'],
                    batch_fraction=self.global_config['batch_fraction'],
                    score_direction=score_direction,
                    pruning_strategy=pruning_strategy,
                    pruning_params=pruning_params,
                    output_dir=output_dir,
                    mode='benchmark',
                    cache_dir=Path(self.global_config['cache_dir'])
                )

                elapsed = time.time() - start_time
                print(f"  ✓ Completed in {elapsed:.1f}s")

                results[param_value] = {
                    'output_dir': output_dir,
                    'cycle_metrics': al_results['cycle_metrics'],
                    'compounds_df': al_results['compounds_df'],
                    'target_column': target_column,
                    'score_direction': score_direction,
                    'elapsed_time': elapsed
                }

        print(f"\n{'='*60}")
        print(f"✓ Pruning validation complete for {len(results)} strategies")
        print(f"{'='*60}")

        return results

    def _get_pruning_output_dir(self, param_value: float) -> Path:
        pct_str = f"{int(param_value * 100)}pct" if param_value > 0 else "no_pruning"
        return Path(self.global_config['output_base_dir']) / 'score_based_pruning' / 'data' / pct_str

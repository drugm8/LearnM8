"""Main public API for LearnM8 active learning.

Provides a single function `run_active_learning()` that orchestrates the complete
active learning workflow:
1. Validate compounds (catch errors early)
2. Select initial training set
3. Measure initial compounds
4. Initialize master DataFrame
5. Execute active learning cycles
6. Save results to CSV files

Supports two APIs:
- Simple API: Specify n_cycles, batch_fraction, strategy
- Advanced API: Provide list of CycleConfig objects for full control

Examples:
    # Simple API
    results = run_active_learning(
        compound_pool='compounds.csv',
        oracle='oracle.csv',
        learner='rf',
        target_col='Activity',
        featurizer_type='morgan',
        n_cycles=10,
        batch_fraction=0.01
    )

    # Advanced API
    from learnm8.core.config import CycleConfig
    results = run_active_learning(
        compound_pool=df,
        oracle=my_oracle,
        learner=my_learner,
        target_col='Activity',
        featurizer_type='morgan',
        cycles=[
            CycleConfig('random', n_cycles=3, batch_fraction=0.02),
            CycleConfig('greedy', n_cycles=5, batch_fraction=0.01,
                       pruning_strategy='score',
                       pruning_params={'pruning_fraction': 0.3})
        ]
    )
"""

import logging
import json
import inspect
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Literal
import pandas as pd

from learnm8.core.validation import validate_compound_pool, ValidationResult
from learnm8.core.initialization import initialize_master_dataframe, select_initial_sample
from learnm8.core.config import parse_cycle_schedule, CycleConfig
from learnm8.core.cycle import execute_cycle
from learnm8.core.persistence import save_results
from learnm8.core.interfaces import Oracle, Learner
from learnm8.oracles import CSVOracle
from learnm8.learners import (
    RandomForestLearner, GaussianProcessLearner, XGBoostLearner,
    MLPLearner, MCDropoutLearner, EnsembleLearner,
    RFEnsemble, LREnsemble, XGBEnsemble, DTEnsemble, MixedEnsemble
)

logger = logging.getLogger(__name__)


LEARNER_REGISTRY = {}

# Register sklearn-based learners (always available)
try:
    LEARNER_REGISTRY.update({
        'rf': RandomForestLearner,
        'gp': GaussianProcessLearner,
        'xgb': XGBoostLearner,
    })
except NameError:
    pass

# Register pytorch-based learners (optional)
try:
    LEARNER_REGISTRY.update({
        'mlp': MLPLearner,
        'mc_dropout': MCDropoutLearner,
    })
except NameError:
    pass

# Register ensemble learners (optional)
try:
    LEARNER_REGISTRY.update({
        'ensemble': EnsembleLearner,
        'rf_ensemble': RFEnsemble,
        'lr_ensemble': LREnsemble,
        'xgb_ensemble': XGBEnsemble,
        'dt_ensemble': DTEnsemble,
        'mixed_ensemble': MixedEnsemble
    })
except NameError:
    pass


def list_available_learners():
    """Return list of available learner shortcuts.

    Returns only learners whose imports succeeded.

    Returns:
        List of learner shortcut strings
    """
    return sorted(LEARNER_REGISTRY.keys())


def _create_learner(learner_str: str, random_state: int) -> Learner:
    """Instantiate learner from string shortcut.

    Automatically detects whether learner accepts random_state (singular)
    or random_states (plural) parameter through signature inspection.

    For ensemble learners requiring random_states, derives a deterministic
    list from the provided random_state: [seed, seed+81, seed+314]

    Supported learner shortcuts:
    - 'rf': Random Forest
    - 'gp': Gaussian Process
    - 'xgb': XGBoost
    - 'mlp': Multi-Layer Perceptron
    - 'mc_dropout': Monte Carlo Dropout
    - 'ensemble': Ensemble (default configuration)
    - 'rf_ensemble': Random Forest Ensemble
    - 'lr_ensemble': Linear Regression Ensemble
    - 'xgb_ensemble': XGBoost Ensemble
    - 'dt_ensemble': Decision Tree Ensemble
    - 'mixed_ensemble': Mixed Model Ensemble

    Args:
        learner_str: Learner shortcut string
        random_state: Random seed for reproducibility

    Returns:
        Instantiated Learner object

    Raises:
        ValueError: If learner_str not recognized
        RuntimeError: If learner instantiation fails

    Notes:
        Ensemble learners automatically receive random_states derived from
        random_state using deterministic offsets (81, 314) to ensure diversity
        while maintaining reproducibility.
    """
    if learner_str not in LEARNER_REGISTRY:
        available = ', '.join(sorted(LEARNER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown learner '{learner_str}'. "
            f"Available learners: {available}"
        )

    learner_class = LEARNER_REGISTRY[learner_str]

    try:
        sig = inspect.signature(learner_class.__init__)
        params = sig.parameters

        if 'random_states' in params:
            random_states = [random_state, random_state + 81, random_state + 314]
            logger.debug(
                f"Creating {learner_class.__name__} with random_states={random_states} "
                f"(derived from random_state={random_state})"
            )
            return learner_class(random_states=random_states)

        elif 'random_state' in params:
            logger.debug(f"Creating {learner_class.__name__} with random_state={random_state}")
            return learner_class(random_state=random_state)

        else:
            logger.debug(f"Creating {learner_class.__name__} without random state parameter")
            return learner_class()

    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {learner_class.__name__}: {e}"
        ) from e


def _calculate_aggregate_metrics(
    all_metrics: List[Dict[str, Any]],
    compounds_df: pd.DataFrame,
    mode: str
) -> Dict[str, Any]:
    """Calculate aggregate metrics across all cycles.

    Args:
        all_metrics: List of cycle metrics dictionaries
        compounds_df: Final master DataFrame
        mode: Execution mode ('run' or 'benchmark')

    Returns:
        Dictionary with aggregate metrics
    """
    import numpy as np

    if not all_metrics:
        return {}

    agg = {}

    rmse_values = [m.get('rmse') for m in all_metrics if m.get('rmse') is not None]
    if rmse_values:
        agg['initial_rmse'] = rmse_values[0]
        agg['final_rmse'] = rmse_values[-1]
        agg['rmse_improvement'] = rmse_values[0] - rmse_values[-1]
        agg['rmse_improvement_pct'] = ((rmse_values[0] - rmse_values[-1]) / rmse_values[0] * 100) if rmse_values[0] != 0 else 0

    r2_values = [m.get('r2_score') for m in all_metrics if m.get('r2_score') is not None]
    if r2_values:
        agg['initial_r2'] = r2_values[0]
        agg['final_r2'] = r2_values[-1]
        agg['r2_improvement'] = r2_values[-1] - r2_values[0]

    mae_values = [m.get('mae') for m in all_metrics if m.get('mae') is not None]
    if mae_values:
        agg['initial_mae'] = mae_values[0]
        agg['final_mae'] = mae_values[-1]
        agg['mae_improvement'] = mae_values[0] - mae_values[-1]

    spearman_values = [m.get('spearman_correlation') for m in all_metrics if m.get('spearman_correlation') is not None]
    if spearman_values:
        agg['initial_spearman'] = spearman_values[0]
        agg['final_spearman'] = spearman_values[-1]
        agg['spearman_improvement'] = spearman_values[-1] - spearman_values[0]

    avg_score_values = [m.get('avg_score_selected') for m in all_metrics if m.get('avg_score_selected') is not None]
    if avg_score_values:
        agg['avg_selection_quality'] = float(np.mean(avg_score_values))
        agg['std_selection_quality'] = float(np.std(avg_score_values))

    best_values = [m.get('best_so_far') for m in all_metrics if m.get('best_so_far') is not None]
    if best_values:
        agg['best_compound_value'] = best_values[-1]
        best_cycle = next((i for i, m in enumerate(all_metrics) if m.get('best_so_far') == best_values[-1]), None)
        agg['best_compound_found_cycle'] = best_cycle

    if mode == 'benchmark':
        top100_values = [m.get('top_100_overlap') for m in all_metrics if m.get('top_100_overlap') is not None]
        if top100_values:
            agg['final_top_100_overlap'] = top100_values[-1]
            agg['avg_top_100_overlap'] = float(np.mean(top100_values))

        top1k_values = [m.get('top_1000_overlap') for m in all_metrics if m.get('top_1000_overlap') is not None]
        if top1k_values:
            agg['final_top_1000_overlap'] = top1k_values[-1]
            agg['avg_top_1000_overlap'] = float(np.mean(top1k_values))

    agg['total_cycles'] = len(all_metrics)
    agg['total_labeled'] = int((compounds_df['status'] == 'labeled').sum())
    agg['total_pruned'] = int((compounds_df['status'] == 'pruned').sum())

    return agg


def run_active_learning(
    compound_pool: Union[str, Path, pd.DataFrame],
    oracle: Union[str, Path, Oracle],
    learner: Union[str, Learner],
    target_col: str,
    featurizer_type: str,
    # Advanced API
    cycles: Optional[List[CycleConfig]] = None,
    # Simple API
    n_cycles: int = 10,
    batch_fraction: float = 0.01,
    strategy: str = 'greedy',
    initial_strategy: str = 'random',
    initial_batch_fraction: float = 0.01,
    # Initial sampling
    n_initial: int = 10,
    initial_sampling_strategy: Literal['random', 'diverse'] = 'random',
    # Common parameters
    score_direction: str = 'higher',
    mode: Optional[Literal['run', 'benchmark']] = None,
    output_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    # Pruning
    pruning_fraction: Optional[float] = None,
    pruning_strategy: Optional[str] = None,
    pruning_params: Optional[Dict] = None,
    # Acquisition
    acquisition_params: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Execute active learning experiment.

    This is the main entry point for the LearnM8 API. It orchestrates the complete
    active learning workflow from validation to final results.

    Two APIs are supported:

    **Simple API** - Specify basic parameters:
        run_active_learning(
            compound_pool='compounds.csv',
            oracle='oracle.csv',
            learner='rf',
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=10,
            batch_fraction=0.01
        )

    **Advanced API** - Provide CycleConfig list for full control:
        run_active_learning(
            compound_pool=df,
            oracle=my_oracle,
            learner=my_learner,
            target_col='Activity',
            featurizer_type='morgan',
            cycles=[
                CycleConfig('random', n_cycles=3, batch_fraction=0.02),
                CycleConfig('greedy', n_cycles=5, batch_fraction=0.01)
            ]
        )

    Args:
        compound_pool: Compound pool specification:
            - str/Path: Path to CSV file (must have 'ID' and 'SMILES' columns)
            - pd.DataFrame: DataFrame with 'ID' and 'SMILES' columns

        oracle: Oracle specification:
            - None: Auto-detect from compound_pool (benchmark mode)
            - str/Path ending in .csv: CSV file oracle (benchmark mode)
            - str 'module.py:function': Python function oracle (run mode)
            - Oracle instance: Custom oracle

        learner: Learner specification:
            - str: Learner shortcut (see _create_learner for options)
            - Learner instance: Custom learner

        target_col: Target property column name

        featurizer_type: Molecular featurizer type
            ('morgan', 'maccs', 'ecfp6', 'descriptors', 'morgan_feat')

        cycles: Advanced API - List of CycleConfig objects
            If provided, overrides simple API parameters

        n_cycles: Simple API - Number of cycles (default: 10)

        batch_fraction: Simple API - Fraction per cycle (default: 0.01)

        strategy: Simple API - Acquisition strategy (default: 'greedy')

        initial_strategy: Simple API - First cycle strategy (default: 'random')

        initial_batch_fraction: Simple API - First cycle fraction (default: 0.01)

        n_initial: Initial training set size (default: 10)

        initial_sampling_strategy: Initial sampling method ('random' or 'diverse')

        score_direction: Optimization direction ('higher' or 'lower')

        mode: Execution mode:
            - None: Auto-detect from oracle type
            - 'run': Production screening (predict unlabeled only)
            - 'benchmark': Evaluation (predict full dataset)

        output_dir: Output directory path
            If None, auto-generates timestamped directory

        cache_dir: Cache directory for feature storage
            If None, uses output_dir/.cache

        random_state: Random seed (default: 42)

        pruning_fraction: Fraction to prune per cycle (0.0-0.9)
            If provided, enables pruning with score_based strategy

        pruning_strategy: Pruning strategy (default: 'score_based')

        pruning_params: Additional pruning parameters

        acquisition_params: Additional acquisition parameters

        **kwargs: Additional parameters passed to cycle execution

    Returns:
        Dictionary with:
            - compounds_df: Final master DataFrame
            - cycle_metrics: List of metrics per cycle
            - validation_result: ValidationResult object
            - output_dir: Path to output directory
            - saved_files: Dictionary of saved file paths
            - labeled_data: Convenience accessor for labeled compounds
            - unlabeled_data: Convenience accessor for unlabeled compounds

    Raises:
        FileNotFoundError: If compound_pool file not found
        ValueError: If validation fails or invalid parameters
        TypeError: If invalid input types
        RuntimeError: If cycle execution fails

    Notes:
        **Oracle Auto-Detection:**
        - When oracle=None and compound_pool is a CSV path → uses that CSV as benchmark oracle
        - When oracle=None and compound_pool is a DataFrame → raises error (oracle required)
        - CSV oracle files automatically trigger benchmark mode
        - Python function oracles (module.py:function) automatically trigger run mode

        **Other Notes:**
        - Pruning is enabled automatically if pruning_fraction is provided
        - All learners receive featurizer_type and random_state parameters
        - Results are automatically saved to CSV files in output_dir
    """

    try:
        DEPRECATED_PARAMS = {
            'export_csv': 'CSV export is now automatic via save_results(). This parameter has no effect.',
            'console_output': 'Console output is controlled by Python logging configuration.',
            'target_column': "Use 'target_col' instead (renamed in v1.0.0).",
            'featurizer': "Use 'featurizer_type' instead (renamed in v1.0.0)."
        }

        for param, message in DEPRECATED_PARAMS.items():
            if param in kwargs:
                import warnings
                warnings.warn(
                    f"Parameter '{param}' is deprecated: {message}",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning(f"Deprecated parameter '{param}' passed: {message}")
                kwargs.pop(param)

        original_compound_pool_path = None

        if output_dir is None:
            output_dir = Path(f'learnm8_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if cache_dir is None:
            cache_dir = output_dir / '.cache'
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / 'learnm8.log'
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == str(log_file) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

        logger.info(f"Starting active learning experiment at {datetime.now()}")
        logger.info(f"Output directory: {output_dir}")

        if isinstance(compound_pool, (str, Path)):
            compound_pool_path = Path(compound_pool)
            original_compound_pool_path = compound_pool_path
            if not compound_pool_path.exists():
                raise FileNotFoundError(f"Compound pool file not found: {compound_pool_path}")
            compound_pool = pd.read_csv(compound_pool_path)
            logger.info(f"Loaded compound pool from file: {compound_pool_path}")
        elif isinstance(compound_pool, pd.DataFrame):
            logger.info("Using provided DataFrame as compound pool")
        else:
            raise TypeError(
                f"compound_pool must be str, Path, or pd.DataFrame, got {type(compound_pool)}"
            )

        if 'ID' not in compound_pool.columns or 'SMILES' not in compound_pool.columns:
            raise ValueError(
                f"Compound pool must have 'ID' and 'SMILES' columns. "
                f"Found columns: {list(compound_pool.columns)}"
            )

        mode_detected = False

        if oracle is None:
            logger.info("Oracle not provided, attempting auto-detection")
            if original_compound_pool_path is not None:
                oracle = CSVOracle(str(original_compound_pool_path))
                mode = 'benchmark'
                mode_detected = True
                logger.info(f"Auto-detected CSV oracle from compound pool, mode=benchmark")
            else:
                raise ValueError(
                    "Oracle required when compound_pool is DataFrame. "
                    "Provide oracle parameter or use CSV file for compound_pool."
                )
        elif isinstance(oracle, (str, Path)):
            oracle_path = Path(oracle)
            if oracle_path.suffix == '.csv':
                oracle = CSVOracle(str(oracle_path))
                if mode is None:
                    mode = 'benchmark'
                    mode_detected = True
                logger.info(f"Using CSV oracle: {oracle_path}, mode={mode}")
            else:
                if ':' not in str(oracle):
                    raise ValueError(
                        f"Non-CSV oracle must be 'module.py:function', got: {oracle}"
                    )
                module_path, function_name = str(oracle).split(':', 1)
                from learnm8.oracles.python_oracle import PythonOracle
                oracle = PythonOracle(module_path, function_name)
                if mode is None:
                    mode = 'run'
                    mode_detected = True
                logger.info(f"Using Python oracle: {module_path}:{function_name}, mode={mode}")
        elif isinstance(oracle, Oracle):
            logger.info(f"Using provided oracle instance: {oracle.__class__.__name__}")
            if mode is None:
                mode = 'run'
        else:
            raise TypeError(
                f"oracle must be None, str, Path, or Oracle instance, got {type(oracle)}"
            )

        if mode is None:
            mode = 'run'
            logger.info(f"Mode not specified, defaulting to: {mode}")
        elif mode_detected:
            logger.info(f"Mode auto-detected: {mode}")
        else:
            logger.info(f"Mode explicitly set: {mode}")

        if isinstance(learner, str):
            learner = _create_learner(learner, random_state)
            logger.info(f"Instantiated learner from string: {learner.__class__.__name__}")
        elif isinstance(learner, Learner):
            logger.info(f"Using provided learner instance: {learner.__class__.__name__}")
        else:
            raise TypeError(
                f"learner must be str or Learner instance, got {type(learner)}"
            )

        logger.info("Phase 1: Validating compound pool")
        validation_result = validate_compound_pool(compound_pool, featurizer_type, cache_dir)

        if len(validation_result.valid_compounds) == 0:
            logger.error(
                f"No valid compounds after validation. "
                f"Total: {len(compound_pool)}, "
                f"Invalid: {len(validation_result.invalid_compounds)}"
            )
            raise ValueError(
                "No valid compounds after validation. Check validation_report for details."
            )

        logger.info(
            f"Validation complete: {len(validation_result.valid_compounds)} valid "
            f"({validation_result.success_rate:.1%}), "
            f"{len(validation_result.invalid_compounds)} invalid"
        )

        if len(validation_result.invalid_compounds) > 0:
            logger.warning(f"Found {len(validation_result.invalid_compounds)} invalid compounds")

        logger.info("Phase 2: Selecting initial training set")

        if n_initial > len(validation_result.valid_compounds):
            logger.warning(
                f"n_initial ({n_initial}) > valid compounds ({len(validation_result.valid_compounds)}), "
                f"using all valid compounds"
            )
            n_initial = len(validation_result.valid_compounds)

        initial_ids = select_initial_sample(
            validation_result.valid_compounds,
            n_initial,
            initial_sampling_strategy,
            featurizer_type,
            cache_dir,
            random_state
        )

        if len(initial_ids) == 0:
            raise ValueError("Initial sampling returned 0 compounds")

        logger.info(
            f"Selected {len(initial_ids)} initial compounds using {initial_sampling_strategy} strategy"
        )

        logger.info("Phase 3: Measuring initial compounds")
        initial_compounds = validation_result.valid_compounds[
            validation_result.valid_compounds['ID'].isin(initial_ids)
        ]
        initial_measurements = oracle.measure(initial_compounds, [target_col])

        measured_ids = set(initial_measurements['ID'])
        missing_ids = set(initial_ids) - measured_ids
        if missing_ids:
            raise ValueError(
                f"Oracle failed to measure {len(missing_ids)} initial compounds: {missing_ids}"
            )

        initial_values = initial_measurements.set_index('ID')[target_col]
        logger.info(
            f"Measured initial compounds: mean={initial_values.mean():.3f}, "
            f"std={initial_values.std():.3f}, "
            f"min={initial_values.min():.3f}, "
            f"max={initial_values.max():.3f}"
        )

        logger.info("Phase 4: Initializing master DataFrame")
        compounds_df = initialize_master_dataframe(
            validation_result.valid_compounds,
            initial_ids,
            initial_values,
            target_col
        )

        expected_cols = {'ID', 'SMILES', 'status', 'selected_cycle', 'labeled_cycle', 'pruned_cycle', target_col}
        if not expected_cols.issubset(compounds_df.columns):
            missing = expected_cols - set(compounds_df.columns)
            raise ValueError(f"Master DataFrame missing columns: {missing}")

        logger.info(
            f"Initialized master DataFrame: shape={compounds_df.shape}, "
            f"labeled={len(compounds_df[compounds_df['status'] == 'labeled'])}"
        )

        logger.info("Phase 5: Parsing cycle schedule")

        if pruning_fraction is not None or pruning_strategy is not None:
            # Validate pruning_fraction if provided
            if pruning_fraction is not None:
                if not (0.0 <= pruning_fraction <= 0.9):
                    raise ValueError(f"pruning_fraction must be in [0.0, 0.9], got {pruning_fraction}")

            # Set defaults
            if pruning_strategy is None:
                pruning_strategy = 'score'
            if pruning_params is None:
                pruning_params = {}

            # Add pruning_fraction to params if provided via Simple API
            if pruning_fraction is not None:
                pruning_params['pruning_fraction'] = pruning_fraction

            # Add to kwargs for parse_cycle_schedule
            kwargs['pruning_strategy'] = pruning_strategy
            kwargs['pruning_params'] = pruning_params

            # Log configuration
            fraction_str = f"{pruning_params.get('pruning_fraction', 'N/A'):.1%}" if isinstance(pruning_params.get('pruning_fraction'), (int, float)) else str(pruning_params.get('pruning_fraction'))
            logger.info(f"Pruning enabled: {fraction_str} per cycle using {pruning_strategy}")

        if acquisition_params is not None:
            kwargs['acquisition_params'] = acquisition_params

        cycle_schedule = parse_cycle_schedule(
            cycles, strategy, n_cycles, batch_fraction,
            initial_strategy, initial_batch_fraction,
            **kwargs
        )

        if len(cycle_schedule) == 0:
            raise ValueError("Cycle schedule is empty")

        from collections import Counter
        strategy_counts = Counter(config.strategy for config in cycle_schedule)
        logger.info(
            f"Cycle schedule parsed: {len(cycle_schedule)} total cycles, "
            f"strategies: {dict(strategy_counts)}"
        )

        logger.info("Phase 6: Executing active learning cycles")
        original_pool = validation_result.valid_compounds.copy() if mode == 'benchmark' else None
        original_pool_size = len(validation_result.valid_compounds)
        all_metrics = []

        for cycle_num, config in enumerate(cycle_schedule):
            logger.info(f"Executing cycle {cycle_num + 1}/{len(cycle_schedule)}")

            try:
                compounds_df, metrics = execute_cycle(
                    compounds_df=compounds_df,
                    cycle=cycle_num,
                    config=config,
                    learner=learner,
                    oracle=oracle,
                    target_col=target_col,
                    featurizer_type=featurizer_type,
                    cache_dir=cache_dir,
                    original_pool_size=original_pool_size,
                    score_direction=score_direction,
                    mode=mode,
                    original_pool=original_pool
                )
                all_metrics.append(metrics)

                logger.info(
                    f"Cycle {cycle_num} complete: "
                    f"{metrics['selected_count']} selected, "
                    f"{metrics['remaining_unlabeled']} remaining"
                )

                if metrics['remaining_unlabeled'] == 0:
                    logger.info("Pool exhausted, stopping early")
                    break

            except Exception as e:
                logger.error(f"Cycle {cycle_num} failed: {e}")
                raise RuntimeError(f"Cycle {cycle_num} failed: {e}") from e

        logger.info("Phase 7: Calculating aggregate metrics")
        aggregate_metrics = _calculate_aggregate_metrics(all_metrics, compounds_df, mode)

        if aggregate_metrics:
            logger.info("Aggregate metrics:")
            if 'rmse_improvement' in aggregate_metrics:
                logger.info(f"  RMSE: {aggregate_metrics.get('initial_rmse', 'N/A'):.4f} → {aggregate_metrics.get('final_rmse', 'N/A'):.4f} (improvement: {aggregate_metrics.get('rmse_improvement', 0):.4f})")
            if 'r2_improvement' in aggregate_metrics:
                logger.info(f"  R²: {aggregate_metrics.get('initial_r2', 'N/A'):.4f} → {aggregate_metrics.get('final_r2', 'N/A'):.4f} (improvement: {aggregate_metrics.get('r2_improvement', 0):.4f})")

        logger.info("Phase 8: Saving results")
        config_dict = {
            'target_col': target_col,
            'featurizer_type': featurizer_type,
            'score_direction': score_direction,
            'mode': mode,
            'n_cycles': len(all_metrics),
            'random_state': random_state,
            'learner': learner.__class__.__name__,
            'oracle': oracle.__class__.__name__,
            'initial_sampling_strategy': initial_sampling_strategy,
            'n_initial': n_initial
        }

        saved_files = save_results(
            compounds_df, all_metrics, validation_result, config_dict, output_dir
        )

        logger.info(f"Results saved to: {output_dir}")
        for file_type, file_path in saved_files.items():
            logger.info(f"  {file_type}: {file_path}")

        logger.info("Active learning complete")

        return {
            'compounds_df': compounds_df,
            'cycle_metrics': all_metrics,
            'aggregate_metrics': aggregate_metrics,
            'validation_result': validation_result,
            'output_dir': output_dir,
            'saved_files': saved_files,
            'labeled_data': compounds_df[compounds_df['status'] == 'labeled'],
            'unlabeled_data': compounds_df[compounds_df['status'] == 'unlabeled']
        }

    except Exception as e:
        logger.error(f"Active learning failed: {e}", exc_info=True)
        raise

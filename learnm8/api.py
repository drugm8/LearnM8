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
import inspect
import time
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Literal
import polars as pl

from learnm8.core.validation import validate_compound_pool, ValidationResult
from learnm8.core.initialization import initialize_master_dataframe_empty, select_initial_batch
from learnm8.core.config import parse_cycle_schedule, CycleConfig
from learnm8.core.cycle import execute_cycle
from learnm8.core.persistence import save_results
from learnm8.core.interfaces import Oracle, Learner
from learnm8.oracles import CSVOracle
from learnm8.learners import (
    RandomForestLearner, GaussianProcessLearner, XGBoostLearner,
    MLPLearner, MCDropoutLearner, FastpropLearner, EnsembleLearner,
    RFEnsemble, LREnsemble, XGBEnsemble, DTEnsemble, MixedEnsemble,
    FastpropEnsemble
)

try:
    from learnm8.learners.torch.chemprop_learner import ChempropLearner
    from learnm8.learners.ensemble.chemprop_ensemble import ChempropEnsemble
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False
from learnm8.utils.logging_formatters import (
    format_cycle_schedule,
    format_duration,
    format_experiment_summary
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
        'fastprop': FastpropLearner,
    })
except NameError:
    pass

# Register chemprop learners (optional)
if CHEMPROP_AVAILABLE:
    LEARNER_REGISTRY['chemprop'] = ChempropLearner
    LEARNER_REGISTRY['chemprop_ensemble'] = ChempropEnsemble

# Register ensemble learners (optional)
try:
    LEARNER_REGISTRY.update({
        'ensemble': EnsembleLearner,
        'rf_ensemble': RFEnsemble,
        'lr_ensemble': LREnsemble,
        'xgb_ensemble': XGBEnsemble,
        'dt_ensemble': DTEnsemble,
        'mixed_ensemble': MixedEnsemble,
        'fastprop_ensemble': FastpropEnsemble
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


def _create_learner(
    learner_str: str,
    random_state: int,
    enable_chemprop_fine_tuning: bool = False,
    checkpoint_dir: Optional[Path] = None
) -> Learner:
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
    - 'chemprop': Chemprop Ensemble (supports fine-tuning)
    - 'ensemble': Ensemble (default configuration)
    - 'rf_ensemble': Random Forest Ensemble
    - 'lr_ensemble': Linear Regression Ensemble
    - 'xgb_ensemble': XGBoost Ensemble
    - 'dt_ensemble': Decision Tree Ensemble
    - 'mixed_ensemble': Mixed Model Ensemble

    Args:
        learner_str: Learner shortcut string
        random_state: Random seed for reproducibility
        enable_chemprop_fine_tuning: Enable fine-tuning for Chemprop (default: False)
        checkpoint_dir: Checkpoint directory for Chemprop fine-tuning

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

    # Special handling for ChempropEnsemble with fine-tuning
    if CHEMPROP_AVAILABLE and learner_class == ChempropEnsemble:
        if enable_chemprop_fine_tuning:
            if checkpoint_dir is None:
                raise ValueError(
                    "checkpoint_dir required when enable_chemprop_fine_tuning=True. "
                    "This should be set automatically by run_active_learning()."
                )
            logger.info(f"Creating ChempropEnsemble with fine-tuning enabled (checkpoint_dir={checkpoint_dir})")
            return ChempropEnsemble(
                enable_fine_tuning=True,
                checkpoint_dir=checkpoint_dir
            )
        else:
            logger.debug("Creating ChempropEnsemble without fine-tuning")
            return ChempropEnsemble()

    # Standard learner instantiation
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
    compounds_df: pl.DataFrame,
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
        top10_disc_values = [m.get('top_10_discovery') for m in all_metrics if m.get('top_10_discovery') is not None]
        if top10_disc_values:
            agg['final_top_10_discovery'] = top10_disc_values[-1]
            agg['avg_top_10_discovery'] = float(np.mean(top10_disc_values))

        top100_disc_values = [m.get('top_100_discovery') for m in all_metrics if m.get('top_100_discovery') is not None]
        if top100_disc_values:
            agg['final_top_100_discovery'] = top100_disc_values[-1]
            agg['avg_top_100_discovery'] = float(np.mean(top100_disc_values))

        cumulative_ef_values = [m.get('cumulative_ef') for m in all_metrics if m.get('cumulative_ef') is not None]
        if cumulative_ef_values:
            agg['final_cumulative_ef'] = cumulative_ef_values[-1]
            agg['avg_cumulative_ef'] = float(np.mean(cumulative_ef_values))

        batch_avg_ratio_values = [m.get('batch_avg_score_ratio') for m in all_metrics if m.get('batch_avg_score_ratio') is not None]
        if batch_avg_ratio_values:
            agg['avg_batch_score_ratio'] = float(np.mean(batch_avg_ratio_values))

        unlabeled_spearman_values = [m.get('unlabeled_spearman_correlation') for m in all_metrics if m.get('unlabeled_spearman_correlation') is not None]
        if unlabeled_spearman_values:
            agg['final_unlabeled_spearman'] = unlabeled_spearman_values[-1]
            agg['avg_unlabeled_spearman'] = float(np.mean(unlabeled_spearman_values))

        unlabeled_top100_values = [m.get('unlabeled_top_100_overlap') for m in all_metrics if m.get('unlabeled_top_100_overlap') is not None]
        if unlabeled_top100_values:
            agg['final_unlabeled_top_100_overlap'] = unlabeled_top100_values[-1]
            agg['avg_unlabeled_top_100_overlap'] = float(np.mean(unlabeled_top100_values))

        unlabeled_ef_values = [m.get('unlabeled_ef_1_0') for m in all_metrics if m.get('unlabeled_ef_1_0') is not None]
        if unlabeled_ef_values:
            agg['final_unlabeled_ef_1_0'] = unlabeled_ef_values[-1]
            agg['avg_unlabeled_ef_1_0'] = float(np.mean(unlabeled_ef_values))

    agg['total_cycles'] = len(all_metrics)
    agg['total_labeled'] = int(compounds_df.filter(pl.col('status') == 'labeled').height)
    agg['total_pruned'] = int(compounds_df.filter(pl.col('status') == 'pruned').height)

    return agg


def run_active_learning(
    compound_pool: Union[str, Path, pl.DataFrame],
    oracle: Union[str, Path, Oracle],
    learner: Union[str, Learner],
    target_col: str,
    featurizer_type: Optional[str] = None,
    # Advanced API
    cycles: Optional[List[CycleConfig]] = None,
    # Simple API
    n_cycles: int = 10,
    batch_fraction: float = 0.01,
    strategy: str = 'greedy',
    initial_strategy: str = 'random',
    # Common parameters
    score_direction: str = 'higher',
    mode: Optional[Literal['run', 'benchmark']] = None,
    output_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    # Chemprop fine-tuning
    enable_chemprop_fine_tuning: bool = False,
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

    **Simple API** - Consistent batch size, flexible strategies:
        run_active_learning(
            compound_pool='compounds.csv',
            oracle='oracle.csv',
            learner='rf',
            target_col='Activity',
            featurizer_type='morgan',
            n_cycles=10,
            batch_fraction=0.01,       # 1% per cycle (ALL cycles)
            initial_strategy='random',  # Strategy for cycle 0
            strategy='greedy'          # Strategy for cycles 1-9
        )

    **Advanced API** - Provide CycleConfig list for full control:
        run_active_learning(
            compound_pool=df,
            oracle=my_oracle,
            learner=my_learner,
            target_col='Activity',
            featurizer_type='morgan',
            cycles=[
                CycleConfig('random', n_cycles=1, batch_fraction=0.02),
                CycleConfig('greedy', n_cycles=5, batch_fraction=0.01),
                CycleConfig('ucb', n_cycles=10, batch_fraction=0.005)
            ]
        )

    Args:
        compound_pool: Compound pool specification:
            - str/Path: Path to CSV file (must have 'ID' and 'SMILES' columns)
            - pl.DataFrame: DataFrame with 'ID' and 'SMILES' columns

        oracle: Oracle specification:
            - None: Auto-detect from compound_pool (benchmark mode)
            - str/Path ending in .csv: CSV file oracle (benchmark mode)
            - str 'module.py:function': Python function oracle (run mode)
            - Oracle instance: Custom oracle

        learner: Learner specification:
            - str: Learner shortcut (see _create_learner for options)
            - Learner instance: Custom learner

        target_col: Target property column name

        featurizer_type: Molecular featurizer type. Optional for SMILES-aware
            learners (e.g., 'chemprop'). Required for feature-based learners.
            Valid options: 'morgan', 'maccs', 'ecfp6', 'descriptors', 'morgan_feat'

        cycles: Advanced API - List of CycleConfig objects
            If provided, overrides simple API parameters

        n_cycles: Simple API - Total number of cycles (default: 10)

        batch_fraction: Simple API - Fraction per cycle for ALL cycles (default: 0.01)

        strategy: Simple API - Acquisition strategy for cycles 1+ (default: 'greedy')

        initial_strategy: Simple API - Acquisition strategy for cycle 0 (default: 'random')

        score_direction: Optimization direction ('higher' or 'lower')

        mode: Execution mode:
            - None: Auto-detect from oracle type
            - 'run': Production screening (basic metrics only, no ground truth needed)
            - 'benchmark': Evaluation (includes discovery/ranking metrics using ground truth)

            Note: Both modes predict on unlabeled compounds only. The difference is in
            metric computation: benchmark mode calculates additional discovery and ranking
            metrics using the ground truth reference (original_pool).

        output_dir: Output directory path
            If None, auto-generates timestamped directory

        cache_dir: Cache directory for feature storage
            If None, uses output_dir/.cache

        random_state: Random seed (default: 42)

        enable_chemprop_fine_tuning: Enable incremental fine-tuning for Chemprop models (default: False)
            When enabled with learner='chemprop', models are saved after each cycle and loaded
            for fine-tuning in subsequent cycles, potentially reducing training time.
            Checkpoints saved to: {output_dir}/.checkpoints/chemprop/
            Ignored if learner is pre-instantiated (configure on learner instance instead).

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

    start_time = time.time()

    try:
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

        # Setup checkpoint directory for Chemprop fine-tuning
        chemprop_checkpoint_dir = output_dir / '.checkpoints' / 'chemprop'

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

        logger.debug(f"Normalizing compound_pool input (type: {type(compound_pool).__name__})")

        if isinstance(compound_pool, (str, Path)):
            compound_pool_path = Path(compound_pool)
            original_compound_pool_path = compound_pool_path
            if not compound_pool_path.exists():
                raise FileNotFoundError(f"Compound pool file not found: {compound_pool_path}")
            compound_pool = pl.read_csv(compound_pool_path)
            logger.debug(f"Loading DataFrame from CSV: {len(compound_pool)} rows, {len(compound_pool.columns)} columns")
        elif isinstance(compound_pool, pl.DataFrame):
            logger.debug("Using provided DataFrame as compound pool")
        else:
            raise TypeError(
                f"compound_pool must be str, Path, or pl.DataFrame, got {type(compound_pool)}"
            )

        if 'ID' not in compound_pool.columns or 'SMILES' not in compound_pool.columns:
            raise ValueError(
                f"Compound pool must have 'ID' and 'SMILES' columns. "
                f"Found columns: {list(compound_pool.columns)}"
            )

        mode_detected = False

        if oracle is None:
            logger.debug("Oracle not provided, attempting auto-detection from compound_pool")
            if original_compound_pool_path is not None:
                oracle = CSVOracle(str(original_compound_pool_path))
                mode = 'benchmark'
                mode_detected = True
                logger.debug(f"Auto-detected CSV oracle from compound pool, mode=benchmark")
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
                logger.debug(f"Using CSV oracle: {oracle_path}, mode={mode}")
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
                logger.debug(f"Using Python oracle: {module_path}:{function_name}, mode={mode}")
        elif isinstance(oracle, Oracle):
            logger.debug(f"Using provided oracle instance: {oracle.__class__.__name__}")
            if mode is None:
                mode = 'run'
        else:
            raise TypeError(
                f"oracle must be None, str, Path, or Oracle instance, got {type(oracle)}"
            )

        logger.debug(f"Detected {mode} mode (oracle type: {type(oracle).__name__})")

        if mode is None:
            mode = 'run'
        elif mode_detected:
            pass
        else:
            pass

        if isinstance(learner, str):
            learner_str = learner
            learner = _create_learner(
                learner,
                random_state,
                enable_chemprop_fine_tuning=enable_chemprop_fine_tuning,
                checkpoint_dir=chemprop_checkpoint_dir if enable_chemprop_fine_tuning else None
            )
            logger.info(f"Using learner: {learner.__class__.__name__}")
            logger.debug(f"Instantiated learner from string: {learner_str}")
        elif isinstance(learner, Learner):
            logger.info(f"Using learner: {learner.__class__.__name__}")
            logger.debug(f"Using provided learner instance")
            if enable_chemprop_fine_tuning:
                logger.warning(
                    "enable_chemprop_fine_tuning=True ignored because learner is "
                    "pre-instantiated. Configure fine-tuning on the learner instance directly."
                )
        else:
            raise TypeError(
                f"learner must be str or Learner instance, got {type(learner)}"
            )

        if learner.requires_smiles():
            if featurizer_type is not None:
                logger.info(
                    f"{learner.get_name()} will use {featurizer_type} features as extra descriptors (x_d). "
                    f"For pure graph-based learning, set featurizer_type=None."
                )
        else:
            if featurizer_type is None:
                raise ValueError(
                    f"{learner.get_name()} requires a featurizer. "
                    f"Specify featurizer_type parameter. "
                    f"Valid options: morgan, maccs, ecfp6, descriptors, morgan_feat"
                )

        oracle_desc = f"{oracle.__class__.__name__}"
        if hasattr(oracle, 'source_file'):
            oracle_desc += f" ({oracle.source_file})"
        logger.info(f"Using oracle: {oracle_desc}")

        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("Phase 1: Validating compound pool")
        logger.info("═══════════════════════════════════════════════════════════════")
        validation_result = validate_compound_pool(compound_pool, n_jobs=-1, progress=True)

        if len(validation_result.valid_compounds) == 0:
            logger.error(
                f"No valid compounds after validation. "
                f"Total: {len(compound_pool)}, "
                f"Invalid: {len(validation_result.invalid_compounds)}"
            )
            raise ValueError(
                "No valid compounds after validation. Check validation_report for details."
            )

        if len(validation_result.invalid_compounds) > 0:
            logger.warning(f"Found {len(validation_result.invalid_compounds)} invalid compounds")

        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("Phase 2: Initializing master DataFrame (all compounds unlabeled)")
        logger.info("═══════════════════════════════════════════════════════════════")

        compounds_df = initialize_master_dataframe_empty(
            validation_result.valid_compounds,
            target_col
        )

        expected_cols = {'ID', 'SMILES', 'status', 'selected_cycle', 'labeled_cycle', 'pruned_cycle', target_col}
        if not expected_cols.issubset(compounds_df.columns):
            missing = expected_cols - set(compounds_df.columns)
            raise ValueError(f"Master DataFrame missing columns: {missing}")

        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("Phase 2b: Initialization (Cycle 0) - selecting and measuring initial batch")
        logger.info("═══════════════════════════════════════════════════════════════")

        if pruning_fraction is not None or pruning_strategy is not None:
            if pruning_fraction is not None:
                if not (0.0 <= pruning_fraction <= 0.9):
                    raise ValueError(f"pruning_fraction must be in [0.0, 0.9], got {pruning_fraction}")

            if pruning_strategy is None:
                pruning_strategy = 'score'
            if pruning_params is None:
                pruning_params = {}

            if pruning_fraction is not None:
                pruning_params['pruning_fraction'] = pruning_fraction

            kwargs['pruning_strategy'] = pruning_strategy
            kwargs['pruning_params'] = pruning_params

            fraction_str = f"{pruning_params.get('pruning_fraction', 'N/A'):.1%}" if isinstance(pruning_params.get('pruning_fraction'), (int, float)) else str(pruning_params.get('pruning_fraction'))
            logger.info(f"Pruning enabled: {fraction_str} per cycle using {pruning_strategy}")

        if acquisition_params is not None:
            kwargs['acquisition_params'] = acquisition_params

        cycle_schedule = parse_cycle_schedule(
            cycles, strategy, n_cycles, batch_fraction,
            initial_strategy,
            **kwargs
        )

        if len(cycle_schedule) == 0:
            raise ValueError("Cycle schedule is empty - cannot determine initialization strategy")

        init_config = cycle_schedule[0]

        logger.info(
            f"Cycle 0 (initialization) config: "
            f"strategy={init_config.strategy}, batch_fraction={init_config.batch_fraction}"
        )

        original_pool = validation_result.valid_compounds.clone() if mode == 'benchmark' else None
        original_pool_size = len(validation_result.valid_compounds)

        compounds_df, cycle_0_metrics = select_initial_batch(
            compounds_df=compounds_df,
            oracle=oracle,
            target_col=target_col,
            strategy=init_config.strategy,
            batch_fraction=init_config.batch_fraction,
            featurizer_type=featurizer_type,
            cache_dir=cache_dir,
            original_pool_size=original_pool_size,
            random_state=random_state,
            acquisition_params=init_config.acquisition_params,
            score_direction=score_direction,
            mode=mode,
            original_pool=original_pool
        )

        # Initialize metrics list with cycle 0
        all_metrics = [cycle_0_metrics]
        logger.info(
            f"Cycle 0 metrics captured: "
            f"avg_score={cycle_0_metrics.get('avg_score_selected', 'N/A')}, "
            f"best={cycle_0_metrics.get('best_so_far', 'N/A')}"
        )

        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("Phase 3: Cycle Schedule")
        logger.info("═══════════════════════════════════════════════════════════════")

        for i, config in enumerate(cycle_schedule, 1):
            schedule_msg = format_cycle_schedule(i, config, original_pool_size)
            logger.info(schedule_msg)

        total_cycles = len(cycle_schedule)
        total_compounds_to_label = sum(
            int(original_pool_size * c.batch_fraction) * c.n_cycles
            for c in cycle_schedule
        )
        logger.info(f"Total: {total_cycles} cycles, {total_compounds_to_label} compounds to be labeled")

        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("Phase 4: Active Learning Cycles")
        logger.info("═══════════════════════════════════════════════════════════════")

        cumulative_selected_ids = set(compounds_df.filter(pl.col('status') == 'labeled')['ID'].to_list())

        for cycle_num, config in enumerate(cycle_schedule[1:], start=1):
            logger.info("─────────────────────────────────────────────────────────────")
            logger.info(f"Cycle {cycle_num} of {len(cycle_schedule) - 1}")
            logger.info("─────────────────────────────────────────────────────────────")

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
                    original_pool=original_pool,
                    cumulative_selected_ids=cumulative_selected_ids
                )
                all_metrics.append(metrics)

                # Update cumulative_selected_ids after cycle
                cumulative_selected_ids = set(compounds_df.filter(pl.col('status') == 'labeled')['ID'].to_list())

                if metrics['remaining_unlabeled'] == 0:
                    logger.info("Pool exhausted, stopping early")
                    break

            except Exception as e:
                logger.error(f"Cycle {cycle_num} failed: {e}")
                raise RuntimeError(f"Cycle {cycle_num} failed: {e}") from e

        logger.debug("Calculating aggregate metrics")
        aggregate_metrics = _calculate_aggregate_metrics(all_metrics, compounds_df, mode)

        if aggregate_metrics:
            logger.debug("Aggregate metrics:")
            if 'rmse_improvement' in aggregate_metrics:
                logger.debug(f"  RMSE: {aggregate_metrics.get('initial_rmse', 'N/A'):.4f} → {aggregate_metrics.get('final_rmse', 'N/A'):.4f} (improvement: {aggregate_metrics.get('rmse_improvement', 0):.4f})")
            if 'r2_improvement' in aggregate_metrics:
                logger.debug(f"  R²: {aggregate_metrics.get('initial_r2', 'N/A'):.4f} → {aggregate_metrics.get('final_r2', 'N/A'):.4f} (improvement: {aggregate_metrics.get('r2_improvement', 0):.4f})")

        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("Experiment Complete")
        logger.info("═══════════════════════════════════════════════════════════════")

        summary_lines = format_experiment_summary(compounds_df, time.time() - start_time, len(all_metrics))
        for line in summary_lines:
            logger.info(line)

        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("Phase 5: Saving Results")
        logger.info("═══════════════════════════════════════════════════════════════")
        config_dict = {
            'target_col': target_col,
            'featurizer_type': featurizer_type,
            'score_direction': score_direction,
            'mode': mode,
            'n_cycles': len(all_metrics),
            'random_state': random_state,
            'learner': learner.__class__.__name__,
            'oracle': oracle.__class__.__name__
        }

        saved_files = save_results(
            compounds_df, all_metrics, validation_result, config_dict, output_dir
        )

        logger.info(f"All results saved to: {output_dir}")
        logger.debug("Saved files:")
        for file_type, file_path in saved_files.items():
            logger.debug(f"  {file_type}: {file_path}")

        return {
            'compounds_df': compounds_df,
            'cycle_metrics': all_metrics,
            'aggregate_metrics': aggregate_metrics,
            'validation_result': validation_result,
            'output_dir': output_dir,
            'saved_files': saved_files,
            'labeled_data': compounds_df.filter(pl.col('status') == 'labeled'),
            'unlabeled_data': compounds_df.filter(pl.col('status') == 'unlabeled')
        }

    except Exception as e:
        logger.error(f"Active learning failed: {e}", exc_info=True)
        raise

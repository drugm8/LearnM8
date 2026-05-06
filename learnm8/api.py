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
        featurizer='morgan',
        n_cycles=10,
        batch_fraction=0.01
    )

    # Advanced API with custom featurizer
    from learnm8.core.config import CycleConfig
    from learnm8.features import MorganFeaturizer

    custom_featurizer = MorganFeaturizer(radius=3, fp_size=4096)
    results = run_active_learning(
        compound_pool=df,
        oracle=my_oracle,
        learner=my_learner,
        target_col='Activity',
        featurizer=custom_featurizer,
        cycles=[
            CycleConfig('random', n_cycles=3, batch_fraction=0.02),
            CycleConfig('greedy', n_cycles=5, batch_fraction=0.01,
                       pruning_strategy='score',
                       pruning_params={'pruning_fraction': 0.3})
        ]
    )
"""
from __future__ import annotations

import inspect
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

import polars as pl

from learnm8.core.config import CycleConfig, parse_cycle_schedule
from learnm8.core.cycle import execute_cycle
from learnm8.core.initialization import (
    initialize_master_dataframe_empty,
    select_initial_batch,
)
from learnm8.core.interfaces import Featurizer, Learner, Oracle
from learnm8.evaluation.metrics.similarity import RunCache
from learnm8.core.persistence import save_results
from learnm8.core.resources import validate_device, validate_n_jobs
from learnm8.core.validation import validate_compound_pool
from learnm8.exceptions import ConfigurationError, LearnM8Error
from learnm8.learners import (
    DecisionTreeLearner,
    DTEnsemble,
    EnsembleLearner,
    FastpropEnsemble,
    FastpropLearner,
    GaussianProcessLearner,
    LinearRegressionLearner,
    LREnsemble,
    MCDropoutLearner,
    MixedEnsemble,
    MLPLearner,
    RandomForestLearner,
    RFEnsemble,
    RfFilLearner,
    RidgeCumlLearner,
    XGBEnsemble,
    XGBoostLearner,
)
from learnm8.learners.ensemble.chemprop_ensemble import ChempropEnsemble
from learnm8.learners.torch.chemprop_learner import ChempropLearner
from learnm8.oracles import CSVOracle
from learnm8.utils.file_loaders import load_compound_file
from learnm8.utils.logging import configure_learnm8_logging
from learnm8.utils.logging_formatters import (
    format_cycle_schedule,
    format_experiment_summary,
)

logger = logging.getLogger(__name__)


LEARNER_REGISTRY = {
    # Sklearn-based learners
    'rf': RandomForestLearner,
    'gp': GaussianProcessLearner,
    'xgb': XGBoostLearner,
    'lr': LinearRegressionLearner,
    'dt': DecisionTreeLearner,

    # GPU-accelerated learners
    'rf_fil': RfFilLearner,
    'ridge_cuml': RidgeCumlLearner,

    # PyTorch-based learners
    'mlp': MLPLearner,
    'mc_dropout': MCDropoutLearner,
    'fastprop': FastpropLearner,

    # Chemprop learners
    'chemprop': ChempropLearner,
    'chemprop_ensemble': ChempropEnsemble,

    # Ensemble learners
    'ensemble': EnsembleLearner,
    'rf_ensemble': RFEnsemble,
    'lr_ensemble': LREnsemble,
    'xgb_ensemble': XGBEnsemble,
    'dt_ensemble': DTEnsemble,
    'mixed_ensemble': MixedEnsemble,
    'fastprop_ensemble': FastpropEnsemble,
}

try:
    from learnm8.learners.gpytorch.gpu_gp import GPyTorchGPLearner
    LEARNER_REGISTRY['gpu_gp'] = GPyTorchGPLearner
except ImportError:
    pass

try:
    from learnm8.learners.gpytorch.svgp import SVGPLearner
    LEARNER_REGISTRY['svgp'] = SVGPLearner
except ImportError:
    pass


LEARNER_DISPLAY_NAMES = {
    'rf': 'Random Forest',
    'gp': 'Gaussian Process',
    'xgb': 'XGBoost',
    'rf_fil': 'RF FIL (GPU)',
    'ridge_cuml': 'Ridge cuML (GPU)',
    'mlp': 'MLP',
    'mc_dropout': 'MC Dropout',
    'fastprop': 'FastProp',
    'chemprop': 'Chemprop',
    'chemprop_ensemble': 'Chemprop Ensemble',
    'ensemble': 'Mixed Ensemble',
    'rf_ensemble': 'RF Ensemble',
    'lr_ensemble': 'LR Ensemble',
    'xgb_ensemble': 'XGBoost Ensemble',
    'dt_ensemble': 'DT Ensemble',
    'mixed_ensemble': 'Mixed Ensemble',
    'fastprop_ensemble': 'FastProp Ensemble',
    'gpu_gp': 'GPyTorch GPU GP',
    'svgp': 'SVGP (Sparse Variational GP)',
}


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
    checkpoint_dir: Path | None = None,
    n_jobs: int = -1,
    device: str = 'auto',
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
            f"Available learners: {available}. "
            f"Uncertainty-capable: gp, mc_dropout, *_ensemble. "
            f"SMILES-native (no featurizer needed): chemprop, fastprop."
        )

    learner_class = LEARNER_REGISTRY[learner_str]

    # Special handling for ChempropEnsemble with fine-tuning
    if learner_class == ChempropEnsemble:
        if enable_chemprop_fine_tuning:
            if checkpoint_dir is None:
                raise ValueError(
                    "checkpoint_dir required when enable_chemprop_fine_tuning=True. "
                    "This should be set automatically by run_active_learning()."
                )
            logger.info(f"Creating ChempropEnsemble with fine-tuning enabled (checkpoint_dir={checkpoint_dir})")
            return ChempropEnsemble(
                enable_fine_tuning=True,
                checkpoint_dir=checkpoint_dir,
                device=device
            )
        else:
            logger.debug("Creating ChempropEnsemble without fine-tuning")
            return ChempropEnsemble(device=device)

    # Standard learner instantiation
    try:
        sig = inspect.signature(learner_class.__init__)
        params = sig.parameters

        kwargs = {}

        if 'n_jobs' in params:
            kwargs['n_jobs'] = n_jobs
        if 'device' in params:
            kwargs['device'] = device

        if 'random_states' in params:
            random_states = [random_state, random_state + 81, random_state + 314]
            logger.debug(
                f"Creating {learner_class.__name__} with random_states={random_states} "
                f"(derived from random_state={random_state})"
            )
            return learner_class(random_states=random_states, **kwargs)

        elif 'random_state' in params:
            logger.debug(f"Creating {learner_class.__name__} with random_state={random_state}")
            return learner_class(random_state=random_state, **kwargs)

        else:
            logger.debug(f"Creating {learner_class.__name__} without random state parameter")
            return learner_class(**kwargs)

    except (ValueError, TypeError, ImportError, RuntimeError) as e:
        raise ConfigurationError(
            f"Failed to instantiate learner '{learner_str}' ({learner_class.__name__}): {e}. "
            f"Check that all required dependencies are installed and parameters are valid."
        ) from e


def _calculate_aggregate_metrics(
    all_metrics: list[dict[str, Any]],
    compounds_df: pl.DataFrame,
    mode: str
) -> dict[str, Any]:
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
    compound_pool: str | Path | pl.DataFrame,
    oracle: str | Path | Oracle,
    learner: str | Learner,
    target_col: str,
    featurizer: str | Featurizer | None = None,
    smiles_column: str | None = None,
    id_column: str | None = None,
    # Advanced API
    cycles: list[CycleConfig] | None = None,
    # Simple API
    n_cycles: int = 10,
    batch_fraction: float = 0.01,
    strategy: str = 'greedy',
    initial_strategy: str = 'random',
    # Common parameters
    score_direction: str = 'higher',
    mode: Literal['run', 'benchmark'] | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    random_state: int = 42,
    # Chemprop fine-tuning
    enable_chemprop_fine_tuning: bool = False,
    # Pruning
    pruning_fraction: float | None = None,
    pruning_strategy: str | None = None,
    pruning_params: dict | None = None,
    # Acquisition
    acquisition_params: dict | None = None,
    # Memory management
    memory_safety_factor: float = 0.7,
    # Resource control
    n_jobs: int = -1,
    device: str = 'auto',
    # Diversity metrics (feature 013)
    disable_molecular_similarity: bool | Iterable[str] = False,
    **kwargs
) -> dict[str, Any]:
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
            featurizer='morgan',
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
            featurizer='morgan',
            cycles=[
                CycleConfig('random', n_cycles=1, batch_fraction=0.02),
                CycleConfig('greedy', n_cycles=5, batch_fraction=0.01),
                CycleConfig('ucb', n_cycles=10, batch_fraction=0.005)
            ]
        )

    Args:
        compound_pool: Compound pool specification:
            - str/Path: Path to compound file (CSV/SDF/SMI) - must have 'ID' and 'SMILES' columns
            - pl.DataFrame: DataFrame with 'ID' and 'SMILES' columns

        oracle: Oracle specification:
            - None: Auto-detect from compound_pool (benchmark mode)
            - str/Path ending in .csv: CSV file oracle (benchmark mode)
            - str 'module.py:function': Python function oracle (run mode)
            - Oracle instance: Custom oracle

        learner: Learner specification:
            - str: Learner shortcut (see _create_learner for options)
            - Learner instance: Custom learner

            GPU-accelerated learners (require RAPIDS/CUDA):
            - 'rf_fil': sklearn RF trained on float32 + RAPIDS FIL GPU inference.
              Provides ordinal uncertainty via per-tree std (ddof=0). Note: float32
              training produces minor numerical divergence from the CPU 'rf' learner.
            - 'ridge_cuml': cuML Ridge regression with CuPy leverage-based uncertainty.
              Ordinal uncertainty (not calibrated). Requires RAPIDS cuML >= 25.04.

            Ensemble learners with device parameter support (lr_ensemble, xgb_ensemble,
            mixed_ensemble accept device='cpu', 'cuda', or 'auto'):
            - device='auto' resolves to CPU member types for all ensembles.
            - device='cuda' activates GPU members (RfFilLearner, RidgeCumlLearner, or
              XGBoostLearner with device='cuda').

        target_col: Target property column name

        featurizer: Molecular featurizer specification. Optional for SMILES-aware
            learners (e.g., 'chemprop'). Required for feature-based learners.
            Can be:
            - str: Featurizer name ('morgan', 'maccs', 'ecfp6', 'descriptors', etc.)
            - Featurizer instance: Custom featurizer with specific configuration
            - None: No featurization (only valid for SMILES-aware learners)

            Examples:
                featurizer='morgan'  # Default Morgan fingerprints
                featurizer=MorganFeaturizer(radius=3, fp_size=4096)  # Custom config

        smiles_column: Column name for SMILES in the input file (CSV only).
            If None, auto-detects from common names ('SMILES', 'smiles', 'Smiles').
            Ignored when compound_pool is a DataFrame or a non-CSV file.

        id_column: Column name for compound ID in the input file (CSV/SDF).
            If None, auto-detects from common names ('ID') or generates synthetic IDs.
            Ignored when compound_pool is a DataFrame.

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

        memory_safety_factor: Fraction of available memory to use for prediction
            batching (0.0, 1.0]. Default 0.7. Higher values use more memory for
            larger batches. Use 0.85 on dedicated hardware, 0.5 on shared clusters.

        **kwargs: Additional parameters passed to cycle execution

    Returns:
        Dictionary with:
            - compounds_df: Final master DataFrame (narrow: 7 base columns + target)
            - cycle_metrics: List of metrics per cycle
            - aggregate_metrics: Aggregated metrics across cycles
            - validation_result: ValidationResult object
            - output_dir: Path to output directory
            - saved_files: Dictionary of saved file paths
            - prediction_files: list[Path] of per-cycle parquet files
            - labeled_count: int, count of labeled compounds at end
            - unlabeled_count: int, count of remaining unlabeled compounds

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
        - Featurizer instances are used for feature extraction and caching
        - Results are automatically saved to CSV files in output_dir
    """

    start_time = time.time()

    # FR-026: per-run diversity-metric cache, lifetime owned here. The
    # try/finally below guarantees prompt eviction independent of GC timing
    # so cumulative buffers never bleed across runs in long-lived processes.
    run_cache = RunCache()

    try:
        n_jobs = validate_n_jobs(n_jobs)
        device = validate_device(device)

        original_compound_pool_path = None

        if output_dir is None:
            output_dir = Path(f'learnm8_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = output_dir / '.cache' if cache_dir is None else Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup checkpoint directory for Chemprop fine-tuning
        chemprop_checkpoint_dir = output_dir / '.checkpoints' / 'chemprop'

        configure_learnm8_logging(
            output_dir=output_dir,
            level='INFO',
            console_type='auto'
        )

        logger.info(f"Starting active learning experiment at {datetime.now()}")
        logger.info(f"Output directory: {output_dir}")

        logger.debug(f"Normalizing compound_pool input (type: {type(compound_pool).__name__})")

        if isinstance(compound_pool, (str, Path)):
            compound_pool_path = Path(compound_pool)
            original_compound_pool_path = compound_pool_path
            if not compound_pool_path.exists():
                raise FileNotFoundError(
                    f"Compound pool file not found: {compound_pool_path}. "
                    f"Verify the file path and ensure the file exists. "
                    f"Supported formats: CSV, SDF, SMI."
                )
            compound_pool = load_compound_file(
                compound_pool_path,
                smiles_column=smiles_column,
                id_column=id_column,
                progress=True,
            )
            logger.debug(f"Loaded DataFrame: {len(compound_pool)} rows, {len(compound_pool.columns)} columns")
        elif isinstance(compound_pool, pl.DataFrame):
            logger.debug("Using provided DataFrame as compound pool")
        else:
            raise TypeError(
                f"compound_pool must be str, Path, or pl.DataFrame, got {type(compound_pool).__name__}. "
                f"Pass a file path (CSV/SDF/SMI) or a Polars DataFrame with 'ID' and 'SMILES' columns."
            )

        if 'ID' not in compound_pool.columns or 'SMILES' not in compound_pool.columns:
            missing = {'ID', 'SMILES'} - set(compound_pool.columns)
            raise ValueError(
                f"Compound pool missing required column(s): {missing}. "
                f"Found columns: {list(compound_pool.columns)}. "
                f"Ensure your data has 'ID' (unique identifier) and 'SMILES' (molecular structure) columns."
            )

        mode_detected = False

        if oracle is None:
            logger.debug("Oracle not provided, attempting auto-detection from compound_pool")
            if original_compound_pool_path is not None:
                oracle = CSVOracle(str(original_compound_pool_path), id_column=id_column or 'ID')
                mode = 'benchmark'
                mode_detected = True
                logger.debug("Auto-detected CSV oracle from compound pool, mode=benchmark")
            else:
                raise ValueError(
                    "Oracle is required when compound_pool is a DataFrame. "
                    "Provide an oracle parameter: a CSV path (benchmark mode), "
                    "'module.py:function' (run mode), or an Oracle instance. "
                    "Alternatively, pass compound_pool as a CSV file path for auto-detection."
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
                        f"Non-CSV oracle path must use 'module.py:function' format, got: '{oracle}'. "
                        f"Example: oracle='my_oracle.py:measure_activity'. "
                        f"For CSV-based oracles, use a .csv file extension."
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
        elif isinstance(oracle, pl.DataFrame):
            # DataFrame oracle for benchmark mode (in-memory)
            logger.debug("Creating in-memory oracle from DataFrame")
            # Note: CSVOracle already imported at module level
            # Save to temp file for CSVOracle
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_oracle_path = Path(f.name)
                oracle.write_csv(temp_oracle_path)
            oracle = CSVOracle(str(temp_oracle_path))
            if mode is None:
                mode = 'benchmark'
                mode_detected = True
            logger.debug(f"Created CSV oracle from DataFrame, mode={mode}")
        else:
            raise TypeError(
                f"oracle must be None, str, Path, DataFrame, or Oracle instance, "
                f"got {type(oracle).__name__}. "
                f"Use a CSV path for benchmark mode, 'module.py:function' for run mode, "
                f"or pass an Oracle instance."
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
                checkpoint_dir=chemprop_checkpoint_dir if enable_chemprop_fine_tuning else None,
                n_jobs=n_jobs,
                device=device
            )
            logger.info(f"Using learner: {learner.__class__.__name__}")
            logger.debug(f"Instantiated learner from string: {learner_str}")
        elif isinstance(learner, Learner):
            logger.info(f"Using learner: {learner.__class__.__name__}")
            logger.debug("Using provided learner instance")
            if enable_chemprop_fine_tuning:
                logger.warning(
                    "enable_chemprop_fine_tuning=True ignored because learner is "
                    "pre-instantiated. Configure fine-tuning on the learner instance directly."
                )
        else:
            raise TypeError(
                f"learner must be str or Learner instance, got {type(learner).__name__}. "
                f"Use a string shortcut (e.g., 'rf', 'gp', 'chemprop') or pass "
                f"a Learner instance."
            )

        if learner.requires_smiles():
            if featurizer is not None:
                featurizer_name = featurizer if isinstance(featurizer, str) else featurizer.get_name()
                logger.info(
                    f"{learner.get_name()} will use {featurizer_name} features as extra descriptors (x_d). "
                    f"For pure graph-based learning, set featurizer=None."
                )
        else:
            if featurizer is None:
                from learnm8.features import list_available_featurizers
                available = list_available_featurizers()
                featurizer_options = ', '.join(sorted(available['all']))
                raise ValueError(
                    f"{learner.get_name()} requires a featurizer because it is a feature-based "
                    f"learner. Specify featurizer parameter (e.g., featurizer='morgan'). "
                    f"Valid options: {featurizer_options}. "
                    f"Learners that work without a featurizer: chemprop, fastprop."
                )

        # Convert string featurizer to instance if needed
        featurizer_obj = None
        if featurizer is not None:
            if isinstance(featurizer, str):
                from learnm8.features import FEATURIZER_REGISTRY
                if featurizer not in FEATURIZER_REGISTRY:
                    from learnm8.features import list_available_featurizers
                    available = list_available_featurizers()
                    featurizer_options = ', '.join(sorted(available['all']))
                    raise ValueError(
                        f"Unknown featurizer '{featurizer}'. "
                        f"Valid options: {featurizer_options}. "
                        f"Use 'learnm8 list featurizers' to see all available featurizers."
                    )
                featurizer_class = FEATURIZER_REGISTRY[featurizer]
                featurizer_obj = featurizer_class(n_jobs=n_jobs)
                logger.debug(f"Instantiated featurizer: {featurizer_obj.get_name()}")
            else:
                from learnm8.core.interfaces import Featurizer
                if not isinstance(featurizer, Featurizer):
                    raise TypeError(
                        f"featurizer must be a string name or Featurizer instance, "
                        f"got {type(featurizer).__name__}. "
                        f"Use a string (e.g., 'morgan') or a Featurizer subclass instance."
                    )
                featurizer_obj = featurizer
                logger.debug(f"Using provided featurizer instance: {featurizer_obj.get_name()}")

        if not (0.0 < memory_safety_factor <= 1.0):
            raise ConfigurationError(
                f"memory_safety_factor must be in (0.0, 1.0], got {memory_safety_factor}. "
                f"Use 0.7 (default) for most cases, 0.85 for dedicated hardware, "
                f"0.5 for shared clusters."
            )

        oracle_desc = f"{oracle.__class__.__name__}"
        if hasattr(oracle, 'source_file'):
            oracle_desc += f" ({oracle.source_file})"
        logger.info(f"Using oracle: {oracle_desc}")

        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("Phase 1: Validating compound pool")
        logger.info("═══════════════════════════════════════════════════════════════")
        validation_result = validate_compound_pool(compound_pool, n_jobs=n_jobs, progress=True)

        if len(validation_result.valid_compounds) == 0:
            logger.error(
                f"No valid compounds after validation. "
                f"Total: {len(compound_pool)}, "
                f"Invalid: {len(validation_result.invalid_compounds)}"
            )
            raise ValueError(
                f"No valid compounds after validation. All {len(compound_pool)} compounds failed "
                f"SMILES validation. Run 'learnm8 validate <file>' to identify invalid compounds, "
                f"or check that SMILES strings are valid molecular structures."
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
            raise ValueError(
                f"Master DataFrame missing required columns: {missing}. "
                f"This is an internal error — initialization may have failed. "
                f"Check that target_col='{target_col}' matches a column in your data."
            )

        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("Phase 2b: Initialization (Cycle 0) - selecting and measuring initial batch")
        logger.info("═══════════════════════════════════════════════════════════════")

        if pruning_fraction is not None or pruning_strategy is not None:
            if pruning_fraction is not None and not (0.0 <= pruning_fraction <= 0.9):
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
            featurizer=featurizer_obj.get_name() if featurizer_obj else None,
            cache_dir=cache_dir,
            original_pool_size=original_pool_size,
            random_state=random_state,
            acquisition_params=init_config.acquisition_params,
            score_direction=score_direction,
            mode=mode,
            original_pool=original_pool,
            run_cache=run_cache,
            featurizer_obj=featurizer_obj,
            disable_molecular_similarity=disable_molecular_similarity,
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
        prediction_files: list[Path] = []

        for cycle_num, config in enumerate(cycle_schedule[1:], start=1):
            logger.info("─────────────────────────────────────────────────────────────")
            logger.info(f"Cycle {cycle_num} of {len(cycle_schedule) - 1}")
            logger.info("─────────────────────────────────────────────────────────────")

            try:
                # Get previous metrics for change indicators in logging
                previous_metrics = all_metrics[-1] if all_metrics else None

                compounds_df, metrics = execute_cycle(
                    compounds_df=compounds_df,
                    cycle=cycle_num,
                    config=config,
                    learner=learner,
                    oracle=oracle,
                    target_col=target_col,
                    featurizer=featurizer_obj.get_name() if featurizer_obj else None,
                    cache_dir=cache_dir,
                    original_pool_size=original_pool_size,
                    score_direction=score_direction,
                    mode=mode,
                    original_pool=original_pool,
                    cumulative_selected_ids=cumulative_selected_ids,
                    memory_safety_factor=memory_safety_factor,
                    run_cache=run_cache,
                    featurizer_obj=featurizer_obj,
                    disable_molecular_similarity=disable_molecular_similarity,
                    random_state=random_state,
                    previous_metrics=previous_metrics,
                    n_jobs=n_jobs,
                    output_dir=output_dir,
                )
                all_metrics.append(metrics)

                cycle_parquet_path = metrics.get('parquet_path')
                if cycle_parquet_path is not None:
                    prediction_files.append(cycle_parquet_path)

                # Update cumulative_selected_ids after cycle
                cumulative_selected_ids = set(compounds_df.filter(pl.col('status') == 'labeled')['ID'].to_list())

                if metrics['remaining_unlabeled'] == 0:
                    logger.info("Pool exhausted, stopping early")
                    break

            except LearnM8Error as e:
                e.add_note(
                    f"Failed during cycle {cycle_num} of {total_cycles} "
                    f"(strategy: {config.strategy}, batch_fraction: {config.batch_fraction})"
                )
                raise
            except (ValueError, RuntimeError, TypeError, OSError) as e:
                logger.error(f"Cycle {cycle_num} failed: {e}")
                err = LearnM8Error(
                    f"Cycle {cycle_num} of {total_cycles} failed: {e}. "
                    f"Check the error details above for the specific cause."
                )
                err.add_note(
                    f"Failed during cycle {cycle_num} of {total_cycles} "
                    f"(strategy: {config.strategy}, batch_fraction: {config.batch_fraction})"
                )
                raise err from e

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
            'featurizer': featurizer_obj.get_name() if featurizer_obj else None,
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
            'prediction_files': prediction_files,
            'labeled_count': int(compounds_df.filter(pl.col('status') == 'labeled').height),
            'unlabeled_count': int(compounds_df.filter(pl.col('status') == 'unlabeled').height),
        }

    except Exception as e:
        logger.error(f"Active learning failed: {e}", exc_info=True)
        raise
    finally:
        # Release diversity-metric caches even on success/failure paths.
        run_cache.clear()

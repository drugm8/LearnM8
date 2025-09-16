"""
Active learning implementation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from pathlib import Path
from .core.interfaces import Oracle, Learner
from .core.data_manager import DataManager
from .evaluation import evaluate_cycle, format_progress_output

logger = logging.getLogger(__name__)


def create_learner_from_string(learner_name: str, **kwargs) -> Learner:
    """Create a learner instance from string name.
    
    Args:
        learner_name: String name of the learner
        **kwargs: Additional arguments passed to learner constructor
        
    Returns:
        Learner instance
        
    Raises:
        ValueError: If learner_name is not recognized
    """
    from .learners.sklearn import RandomForestLearner, GaussianProcessLearner, XGBoostLearner
    from .learners.torch import MLPLearner, MCDropoutLearner
    from .learners.ensemble import (
        EnsembleLearner, RFEnsemble, LREnsemble, XGBEnsemble, DTEnsemble, MixedEnsemble
    )
    
    learner_map = {
        'rf': RandomForestLearner,
        'gp': GaussianProcessLearner,
        'xgb': XGBoostLearner,
        'mlp': MLPLearner,
        'mc_dropout': MCDropoutLearner,
        'ensemble': MixedEnsemble,
        'rf_ensemble': RFEnsemble,
        'lr_ensemble': LREnsemble,
        'xgb_ensemble': XGBEnsemble,
        'dt_ensemble': DTEnsemble,
        'mixed_ensemble': MixedEnsemble
    }
    
    if learner_name not in learner_map:
        available = ', '.join(learner_map.keys())
        raise ValueError(f"Unknown learner '{learner_name}'. Available: {available}")
    
    learner_class = learner_map[learner_name]
    return learner_class(**kwargs)


def run_active_learning(
    compound_pool: pd.DataFrame,
    oracle: Oracle,
    learner: Union[str, Learner],
    target_column: str,
    # Simple API parameters (recommended for most users)
    initial_strategy: str = 'random',
    strategy: str = 'greedy',
    n_cycles: int = 10,
    initial_batch_fraction: float = 0.01,
    batch_fraction: float = 0.01,
    # Advanced scheduling (overrides simple parameters if provided)
    cycles: Optional[List[Tuple[str, float]]] = None,
    # Common parameters
    initial_size: Optional[int] = None,
    random_state: int = 42,
    output_dir: Optional[str] = None,
    score_direction: str = 'higher',
    # Pruning parameters
    pruning_strategy: Optional[str] = None,
    pruning_params: Optional[Dict[str, Any]] = None,
    # Acquisition parameters
    default_acquisition: Optional[str] = None,
    acquisition_params: Optional[Dict[str, Any]] = None,
    # Evaluation parameters
    enable_evaluation: bool = True,
    console_output: bool = True,
    export_csv: bool = False
) -> Dict[str, Any]:
    """Active learning with simplified and advanced interfaces.
    
    This function provides both a simple interface for common use cases and an 
    advanced interface for complex scheduling needs.
    
    Args:
        compound_pool: DataFrame with all compounds (must have ID, SMILES columns)
        oracle: Oracle instance for measuring compounds  
        learner: Learner instance or string name ('rf', 'gp', 'xgb', 'mlp', 'mc_dropout', 
                'ensemble', 'rf_ensemble', 'lr_ensemble', 'xgb_ensemble', 'dt_ensemble', 'mixed_ensemble')
        target_column: Name of target property column
        
        # Simple interface (recommended for most users):
        initial_strategy: Strategy for first cycle (supports all acquisition methods: 'random', 'greedy', 'ucb', 'ei', 'pi', 'thompson', 'entropy', 'pca_dbscan', 'umap_dbscan', 'tsne_dbscan', 'bitbirch', etc.)
        strategy: Strategy for remaining cycles (supports all acquisition methods)
        n_cycles: Total number of active learning cycles
        initial_batch_fraction: Fraction of original pool to select in first cycle
        batch_fraction: Fraction of original pool to select in remaining cycles (maintains consistent batch sizes)
        
        # Advanced interface (overrides simple parameters):
        cycles: List of (strategy, batch_fraction) tuples defining each cycle (fractions calculated from original pool size)
        
        # Common parameters:
        initial_size: Initial training set size (default: 1% of pool, min 10)
        random_state: Random seed for reproducibility
        output_dir: Output directory for DataManager cache (optional, creates temp if None)
        score_direction: Direction of score optimization ('higher' or 'lower' is better)
        
        # Pruning parameters:
        pruning_strategy: Pruning strategy name (None or 'score_based')
        pruning_params: Dictionary of strategy-specific parameters (pruning_fraction, score_direction)
        
        # Acquisition parameters:
        default_acquisition: Default acquisition method override (optional, supports all methods in acquisition registry)
        acquisition_params: Dictionary of acquisition method parameters (e.g., {'diversity_weight': 0.3}) (default: None)
        
        enable_evaluation: Enable comprehensive evaluation metrics (default: True)
        console_output: Display progress output to console (default: True) 
        export_csv: Export detailed metrics to CSV file (default: False)
        
    Returns:
        Dictionary containing:
        - labeled_data: Final labeled compounds DataFrame
        - unlabeled_data: Remaining unlabeled compounds DataFrame  
        - cycle_metrics: List of metrics for each completed cycle
        - total_cycles: Number of cycles completed
    """
    # Validate score direction
    valid_directions = ['higher', 'lower']
    if score_direction not in valid_directions:
        raise ValueError(f"score_direction must be one of {valid_directions}, got '{score_direction}'")
    
    # Convert simple parameters to cycles format if cycles not provided
    if cycles is None:
        # Validate simple parameters using acquisition registry
        from .acquisition import list_acquisition_functions
        valid_strategies = list_acquisition_functions()
        if initial_strategy not in valid_strategies:
            raise ValueError(f"initial_strategy must be one of {valid_strategies}, got '{initial_strategy}'")
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got '{strategy}'")
        if n_cycles < 1:
            raise ValueError(f"n_cycles must be at least 1, got {n_cycles}")
        if not 0 < initial_batch_fraction <= 1:
            raise ValueError(f"initial_batch_fraction must be in (0, 1], got {initial_batch_fraction}")
        if not 0 < batch_fraction <= 1:
            raise ValueError(f"batch_fraction must be in (0, 1], got {batch_fraction}")
            
        # Create cycles list from simple parameters
        cycles = [(initial_strategy, initial_batch_fraction)]  # First cycle
        if n_cycles > 1:
            # Add remaining cycles with specified strategy
            for _ in range(n_cycles - 1):
                cycles.append((strategy, batch_fraction))
    
    # Convert string learner to Learner instance if needed
    if isinstance(learner, str):
        learner = create_learner_from_string(learner)
    
    logger.info(f"Starting active learning with {len(cycles)} cycles")
    
    # Initialize DataManager for feature extraction and caching
    from .core.data_manager import DataManager
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix='learnm8_')
        logger.info(f"Using temporary output directory: {output_dir}")
    
    data_manager = DataManager(output_dir)
    logger.info(f"DataManager initialized with cache directory: {output_dir}")
    
    # Initialize with random selection for training set
    if initial_size is None:
        initial_size = max(10, int(len(compound_pool) * 0.01))
    
    logger.info(f"Selecting initial training set: {initial_size} compounds")
    np.random.seed(random_state)
    initial_compounds = compound_pool.sample(n=initial_size, random_state=random_state)
    
    # Measure initial compounds
    initial_labeled = oracle.measure(initial_compounds[['ID', 'SMILES']], [target_column])
    
    # Simple state variables (no complex state objects!)
    labeled_data = initial_labeled.copy()
    unlabeled_pool = compound_pool[~compound_pool['ID'].isin(initial_labeled['ID'])].copy()
    all_metrics = []
    
    # CSV tracking data structures (added for comprehensive CSV export)
    prediction_history = {}  # cycle -> DataFrame with ID, SMILES, prediction columns
    uncertainty_history = {}  # cycle -> DataFrame with ID, SMILES, uncertainty columns
    selection_history = []    # List of selection records with context
    
    # Store original pool size for consistent batch calculations
    original_pool_size = len(compound_pool)
    
    # Store original compound pool for benchmark mode (Alternative 1)
    original_compound_pool = compound_pool.copy()
    
    logger.info(f"Initial state: {len(labeled_data)} labeled, {len(unlabeled_pool)} unlabeled")
    
    # Detect oracle type for evaluation
    oracle_type = 'benchmark' if hasattr(oracle, 'ground_truth') else 'run'
    ground_truth_data = getattr(oracle, 'ground_truth', None)
    
    # Execute cycles
    for cycle_num, (strategy, batch_fraction) in enumerate(cycles):
        logger.info(f"Starting cycle {cycle_num + 1}/{len(cycles)}: {strategy} strategy, {batch_fraction:.1%} batch")
        
        cycle_result = execute_single_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            original_compound_pool=original_compound_pool,
            strategy=strategy,
            batch_fraction=batch_fraction,
            cycle=cycle_num,
            oracle=oracle,
            learner=learner,
            target_column=target_column,
            data_manager=data_manager,
            original_pool_size=original_pool_size,
            score_direction=score_direction,
            pruning_strategy=pruning_strategy,
            pruning_params=pruning_params,
            acquisition_params=acquisition_params,
            enable_evaluation=enable_evaluation,
            console_output=console_output,
            ground_truth_data=ground_truth_data,
            oracle_type=oracle_type,
            export_csv=export_csv  # Pass CSV export flag for data collection
        )
        
        # Unpack results including CSV tracking data
        labeled_data, unlabeled_pool, cycle_metrics = cycle_result[:3]
        if export_csv and len(cycle_result) > 3:
            cycle_predictions, cycle_uncertainties, cycle_selections = cycle_result[3:6]
            # Store prediction and uncertainty history
            if cycle_predictions is not None:
                prediction_history[cycle_num] = cycle_predictions
            if cycle_uncertainties is not None:
                uncertainty_history[cycle_num] = cycle_uncertainties
            if cycle_selections:
                selection_history.extend(cycle_selections)
        
        all_metrics.append(cycle_metrics)
        logger.info(f"Cycle {cycle_num + 1} completed: {cycle_metrics['selected_count']} compounds selected")
        
        # Early termination if pool exhausted
        if unlabeled_pool.empty:
            logger.info("Unlabeled pool exhausted, stopping early")
            break
    
    logger.info(f"Active learning completed: {len(all_metrics)} cycles, {len(labeled_data)} total labeled compounds")
    
    # Export comprehensive CSV files if requested
    csv_files = {}
    if export_csv:
        csv_files = _export_comprehensive_csv(
            output_dir=output_dir,
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            all_metrics=all_metrics,
            prediction_history=prediction_history,
            uncertainty_history=uncertainty_history,
            selection_history=selection_history,
            compound_pool=compound_pool,
            target_column=target_column,
            oracle_type=oracle_type,
            score_direction=score_direction
        )
        logger.info(f"CSV files exported to: {output_dir}")
    
    return {
        'labeled_data': labeled_data,
        'unlabeled_data': unlabeled_pool,
        'cycle_metrics': all_metrics,
        'total_cycles': len(all_metrics),
        'output_dir': output_dir,
        'csv_files': csv_files
    }


def execute_single_cycle(
    labeled_data: pd.DataFrame,
    unlabeled_pool: pd.DataFrame,
    original_compound_pool: pd.DataFrame,
    strategy: str,
    batch_fraction: float,
    cycle: int,
    oracle: Oracle,
    learner: Learner,
    target_column: str,
    data_manager: 'DataManager',
    original_pool_size: int,
    score_direction: str,
    pruning_strategy: Optional[str] = None,
    pruning_params: Optional[Dict[str, Any]] = None,
    acquisition_params: Optional[Dict[str, Any]] = None,
    enable_evaluation: bool = True,
    console_output: bool = True,
    ground_truth_data: Optional[pd.DataFrame] = None,
    oracle_type: str = 'auto',
    export_csv: bool = False
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame], List[Dict[str, Any]]]]:
    """Execute single active learning cycle - pure function with no side effects.
    
    Args:
        labeled_data: Current labeled compounds DataFrame
        unlabeled_pool: Current unlabeled compounds DataFrame
        original_compound_pool: Original full compound pool (for benchmark mode)
        strategy: Selection strategy name ('random', 'greedy', 'ucb', 'ei', 'pi', 'thompson', etc.)
        batch_fraction: Fraction of original pool to select (maintains consistent batch sizes)
        cycle: Current cycle number (0-indexed)
        oracle: Oracle for measuring compounds
        learner: Learner for training/prediction
        target_column: Target property column name
        data_manager: DataManager for feature extraction
        original_pool_size: Size of the original compound pool for consistent batch calculation
        score_direction: Direction of score optimization ('higher' or 'lower' is better)
        pruning_strategy: Pruning strategy name (None for no pruning)
        pruning_params: Dictionary of strategy-specific parameters
        enable_evaluation: Enable comprehensive evaluation metrics
        console_output: Display progress output to console
        ground_truth_data: Full ground truth data (for benchmark mode)
        oracle_type: Oracle type ('benchmark', 'run', or 'auto')
        export_csv: Enable CSV export tracking
        
    Returns:
        Tuple of (updated_labeled_data, updated_unlabeled_pool, cycle_metrics)
    """
    # Dispatch to mode-specific cycle execution
    if oracle_type == 'benchmark':
        return execute_benchmark_mode_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            original_compound_pool=original_compound_pool,
            strategy=strategy,
            batch_fraction=batch_fraction,
            cycle=cycle,
            oracle=oracle,
            learner=learner,
            target_column=target_column,
            data_manager=data_manager,
            original_pool_size=original_pool_size,
            score_direction=score_direction,
            pruning_strategy=pruning_strategy,
            pruning_params=pruning_params,
            acquisition_params=acquisition_params,
            enable_evaluation=enable_evaluation,
            console_output=console_output,
            ground_truth_data=ground_truth_data,
            oracle_type=oracle_type,
            export_csv=export_csv
        )
    else:
        return execute_run_mode_cycle(
            labeled_data=labeled_data,
            unlabeled_pool=unlabeled_pool,
            strategy=strategy,
            batch_fraction=batch_fraction,
            cycle=cycle,
            oracle=oracle,
            learner=learner,
            target_column=target_column,
            data_manager=data_manager,
            original_pool_size=original_pool_size,
            score_direction=score_direction,
            pruning_strategy=pruning_strategy,
            pruning_params=pruning_params,
            acquisition_params=acquisition_params,
            enable_evaluation=enable_evaluation,
            console_output=console_output,
            ground_truth_data=ground_truth_data,
            oracle_type=oracle_type,
            export_csv=export_csv
        )


def execute_run_mode_cycle(
    labeled_data: pd.DataFrame,
    unlabeled_pool: pd.DataFrame,
    strategy: str,
    batch_fraction: float,
    cycle: int,
    oracle: Oracle,
    learner: Learner,
    target_column: str,
    data_manager: 'DataManager',
    original_pool_size: int,
    score_direction: str,
    pruning_strategy: Optional[str] = None,
    pruning_params: Optional[Dict[str, Any]] = None,
    acquisition_params: Optional[Dict[str, Any]] = None,
    enable_evaluation: bool = True,
    console_output: bool = True,
    ground_truth_data: Optional[pd.DataFrame] = None,
    oracle_type: str = 'run',
    export_csv: bool = False
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame], List[Dict[str, Any]]]]:
    """Execute single active learning cycle optimized for real-world screening (MolPAL-style).
    
    This function implements efficient prediction on unlabeled pool only, optimized for
    production screening scenarios where computational efficiency is critical.
    
    Args:
        labeled_data: Current labeled compounds DataFrame
        unlabeled_pool: Current unlabeled compounds DataFrame
        strategy: Selection strategy name ('random', 'greedy', 'ucb', 'ei', 'pi', 'thompson', etc.)
        batch_fraction: Fraction of original pool to select (maintains consistent batch sizes)
        cycle: Current cycle number (0-indexed)
        oracle: Oracle for measuring compounds
        learner: Learner for training/prediction
        target_column: Target property column name
        data_manager: DataManager for feature extraction
        original_pool_size: Size of the original compound pool for consistent batch calculation
        score_direction: Direction of score optimization ('higher' or 'lower' is better)
        pruning_strategy: Pruning strategy name (None for no pruning)
        pruning_params: Dictionary of strategy-specific parameters
        enable_evaluation: Enable comprehensive evaluation metrics
        console_output: Display progress output to console
        ground_truth_data: Full ground truth data (for benchmark mode)
        oracle_type: Oracle type ('run')
        export_csv: Enable CSV export tracking
        
    Returns:
        Tuple of (updated_labeled_data, updated_unlabeled_pool, cycle_metrics)
    """
    
    # Train model if we have labeled data
    if not labeled_data.empty:
        learner.train(labeled_data, target_column, data_manager)
    else:
        logger.warning(f"Cycle {cycle}: No labeled data available for training")
    
    if unlabeled_pool.empty:
        logger.warning(f"Cycle {cycle}: No unlabeled compounds remaining")
        empty_metrics = {
            'cycle': cycle,
            'strategy': strategy,
            'batch_fraction': batch_fraction,
            'selected_count': 0,
            'remaining_pool': 0,
            'cumulative_labeled': len(labeled_data)
        }
        if export_csv:
            return labeled_data, unlabeled_pool, empty_metrics, None, None, []
        else:
            return labeled_data, unlabeled_pool, empty_metrics
    
    # Get predictions on unlabeled pool ONLY (MolPAL-style efficiency)
    predictions, uncertainties = learner.predict(unlabeled_pool, data_manager)
    
    # Prepare CSV tracking data if export is enabled
    cycle_predictions = None
    cycle_uncertainties = None
    cycle_selections = []
    
    if export_csv:
        # Store predictions with compound info for CSV export
        cycle_predictions = unlabeled_pool[['ID', 'SMILES']].copy()
        if predictions is not None:
            if isinstance(predictions, np.ndarray):
                cycle_predictions[f'prediction_cycle_{cycle}'] = predictions
            elif hasattr(predictions, 'iloc'):
                cycle_predictions[f'prediction_cycle_{cycle}'] = predictions.iloc[:, 0].values
        
        # Store uncertainties if available
        if uncertainties is not None:
            cycle_uncertainties = unlabeled_pool[['ID', 'SMILES']].copy()
            if isinstance(uncertainties, np.ndarray):
                cycle_uncertainties[f'uncertainty_cycle_{cycle}'] = uncertainties
            elif hasattr(uncertainties, 'iloc'):
                cycle_uncertainties[f'uncertainty_cycle_{cycle}'] = uncertainties.iloc[:, 0].values
    
    # Apply pruning if specified
    pruned_pool = unlabeled_pool
    pruning_stats = {'pruned_count': 0, 'original_pool_size': len(unlabeled_pool)}
    
    if pruning_strategy is not None:
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=unlabeled_pool,
                predictions=predictions,
                uncertainties=uncertainties,
                strategy=pruning_strategy,
                params=pruning_params or {},
                score_direction=score_direction
            )
            pruning_stats.update(pruning_info)
            logger.info(f"Cycle {cycle}: Pruning removed {pruning_stats['pruned_count']} compounds "
                       f"({pruning_stats['pruned_count']/len(unlabeled_pool)*100:.1f}% of pool)")
        except Exception as e:
            logger.warning(f"Cycle {cycle}: Pruning failed ({e}), continuing without pruning")
            pruned_pool = unlabeled_pool
    
    # Calculate batch size based on original pool size for consistent batch sizes across cycles
    batch_size = max(1, int(original_pool_size * batch_fraction))
    batch_size = min(batch_size, len(pruned_pool))  # Don't exceed available compounds
    
    logger.debug(f"Cycle {cycle}: Selecting {batch_size} compounds from {len(pruned_pool)} candidates using {strategy} strategy")
    
    # Select compounds using strategy (from pruned pool)
    selected_compounds = select_compounds_by_strategy(
        pool=pruned_pool,
        predictions=predictions if pruned_pool is unlabeled_pool else None,  # Predictions might not align with pruned pool
        uncertainties=uncertainties if pruned_pool is unlabeled_pool else None,
        strategy=strategy,
        batch_size=batch_size,
        score_direction=score_direction,
        data_manager=data_manager,
        acquisition_params=acquisition_params
    )
    
    # Measure selected compounds
    measured_compounds = oracle.measure(selected_compounds[['ID', 'SMILES']], [target_column])
    
    # Track selection history for CSV export
    if export_csv:
        # Get predictions and uncertainties for selected compounds
        selected_ids = selected_compounds['ID'].values
        for i, compound_id in enumerate(selected_ids):
            # Find the prediction and uncertainty for this compound
            compound_idx = unlabeled_pool[unlabeled_pool['ID'] == compound_id].index
            if len(compound_idx) > 0:
                idx = compound_idx[0]
                pool_idx = unlabeled_pool.index.get_loc(idx)
                
                selection_record = {
                    'ID': compound_id,
                    'SMILES': selected_compounds.iloc[i]['SMILES'],
                    'selected_cycle': cycle,
                    'strategy': strategy,
                    'prediction_at_selection': predictions[pool_idx] if predictions is not None else None,
                    'uncertainty_at_selection': uncertainties[pool_idx] if uncertainties is not None else None,
                    'oracle_measured_value': measured_compounds[measured_compounds['ID'] == compound_id][target_column].iloc[0] if target_column in measured_compounds.columns else None
                }
                cycle_selections.append(selection_record)
    
    # Update data (functional updates - create new DataFrames)
    new_labeled_data = pd.concat([labeled_data, measured_compounds], ignore_index=True)
    
    # Remove selected compounds from the original unlabeled pool by ID (not index)
    selected_ids = set(selected_compounds['ID'])
    new_unlabeled_pool = unlabeled_pool[~unlabeled_pool['ID'].isin(selected_ids)].reset_index(drop=True)
    
    # Calculate cycle metrics
    cycle_metrics = {
        'cycle': cycle,
        'strategy': strategy,
        'batch_fraction': batch_fraction,
        'selected_count': len(selected_compounds),
        'remaining_pool': len(new_unlabeled_pool),
        'cumulative_labeled': len(new_labeled_data),
        'pruning_strategy': pruning_strategy,
        'original_pool_size': pruning_stats['original_pool_size'],
        'pruned_count': pruning_stats['pruned_count'],
        'pruned_pool_size': len(pruned_pool)
    }
    
    # Add prediction statistics if available
    if predictions is not None:
        if hasattr(predictions, 'empty') and predictions.empty:
            pass  # Skip empty DataFrame/Series
        elif isinstance(predictions, np.ndarray) and len(predictions) > 0:
            cycle_metrics.update({
                'prediction_mean': float(np.mean(predictions)),
                'prediction_std': float(np.std(predictions))
            })
        elif hasattr(predictions, 'iloc') and len(predictions) > 0:
            cycle_metrics.update({
                'prediction_mean': float(predictions.iloc[:, 0].mean()),
                'prediction_std': float(predictions.iloc[:, 0].std())
            })
    
    if uncertainties is not None:
        if hasattr(uncertainties, 'empty') and uncertainties.empty:
            pass  # Skip empty DataFrame/Series
        elif isinstance(uncertainties, np.ndarray) and len(uncertainties) > 0:
            cycle_metrics.update({
                'uncertainty_mean': float(np.mean(uncertainties)),
                'uncertainty_std': float(np.std(uncertainties))
            })
        elif hasattr(uncertainties, 'iloc') and len(uncertainties) > 0:
            cycle_metrics.update({
                'uncertainty_mean': float(uncertainties.iloc[:, 0].mean()),
                'uncertainty_std': float(uncertainties.iloc[:, 0].std())
            })
    
    # Add measured value statistics
    if target_column in measured_compounds.columns:
        measured_values = measured_compounds[target_column]
        cycle_metrics.update({
            'measured_mean': float(measured_values.mean()),
            'measured_std': float(measured_values.std()),
            'measured_min': float(measured_values.min()),
            'measured_max': float(measured_values.max())
        })
    
    # Comprehensive evaluation integration
    if enable_evaluation and not labeled_data.empty:
        try:
            # Get model predictions on labeled data for model performance assessment
            model_pred_on_labeled, _ = learner.predict(new_labeled_data, data_manager)
            
            # Call comprehensive evaluation with unlabeled pool predictions (for run mode)
            from .evaluation import evaluate_cycle, format_progress_output
            eval_metrics = evaluate_cycle(
                cycle=cycle,
                predictions=model_pred_on_labeled,
                ground_truth=new_labeled_data[target_column].values,
                labeled_data=new_labeled_data,
                selected_compounds=measured_compounds,
                target_column=target_column,
                oracle_type=oracle_type,
                ground_truth_data=ground_truth_data,
                pool_predictions=predictions,  # Use unlabeled pool predictions for run mode
                pool_ids=unlabeled_pool['ID'].values if not unlabeled_pool.empty else [],
                uncertainties=uncertainties,
                score_direction=score_direction
            )
            
            # Add evaluation metrics to cycle metrics
            cycle_metrics.update(eval_metrics)
            
            # Display progress output if enabled
            if console_output:
                progress_output = format_progress_output(eval_metrics, oracle_type)
                print(progress_output)
                
        except Exception as e:
            logger.warning(f"Evaluation failed for cycle {cycle}: {e}")
            # Continue without evaluation metrics
    
    if export_csv:
        return new_labeled_data, new_unlabeled_pool, cycle_metrics, cycle_predictions, cycle_uncertainties, cycle_selections
    else:
        return new_labeled_data, new_unlabeled_pool, cycle_metrics


def execute_benchmark_mode_cycle(
    labeled_data: pd.DataFrame,
    unlabeled_pool: pd.DataFrame,
    original_compound_pool: pd.DataFrame,
    strategy: str,
    batch_fraction: float,
    cycle: int,
    oracle: Oracle,
    learner: Learner,
    target_column: str,
    data_manager: 'DataManager',
    original_pool_size: int,
    score_direction: str,
    pruning_strategy: Optional[str] = None,
    pruning_params: Optional[Dict[str, Any]] = None,
    acquisition_params: Optional[Dict[str, Any]] = None,
    enable_evaluation: bool = True,
    console_output: bool = True,
    ground_truth_data: Optional[pd.DataFrame] = None,
    oracle_type: str = 'benchmark',
    export_csv: bool = False
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame], List[Dict[str, Any]]]]:
    """Execute single active learning cycle optimized for benchmarking (Alternative 1).
    
    This function implements Alternative 1: always predict the full original dataset 
    for scientifically correct EF calculation in benchmark scenarios.
    
    Args:
        labeled_data: Current labeled compounds DataFrame
        unlabeled_pool: Current unlabeled compounds DataFrame
        original_compound_pool: Original full compound pool (for full dataset prediction)
        strategy: Selection strategy name ('random', 'greedy', 'ucb', 'ei', 'pi', 'thompson', etc.)
        batch_fraction: Fraction of original pool to select (maintains consistent batch sizes)
        cycle: Current cycle number (0-indexed)
        oracle: Oracle for measuring compounds
        learner: Learner for training/prediction
        target_column: Target property column name
        data_manager: DataManager for feature extraction
        original_pool_size: Size of the original compound pool for consistent batch calculation
        score_direction: Direction of score optimization ('higher' or 'lower' is better)
        pruning_strategy: Pruning strategy name (None for no pruning)
        pruning_params: Dictionary of strategy-specific parameters
        enable_evaluation: Enable comprehensive evaluation metrics
        console_output: Display progress output to console
        ground_truth_data: Full ground truth data (for benchmark mode)
        oracle_type: Oracle type ('benchmark')
        export_csv: Enable CSV export tracking
        
    Returns:
        Tuple of (updated_labeled_data, updated_unlabeled_pool, cycle_metrics)
    """
    
    # Train model if we have labeled data
    if not labeled_data.empty:
        learner.train(labeled_data, target_column, data_manager)
    else:
        logger.warning(f"Cycle {cycle}: No labeled data available for training")
    
    if unlabeled_pool.empty:
        logger.warning(f"Cycle {cycle}: No unlabeled compounds remaining")
        empty_metrics = {
            'cycle': cycle,
            'strategy': strategy,
            'batch_fraction': batch_fraction,
            'selected_count': 0,
            'remaining_pool': 0,
            'cumulative_labeled': len(labeled_data)
        }
        if export_csv:
            return labeled_data, unlabeled_pool, empty_metrics, None, None, []
        else:
            return labeled_data, unlabeled_pool, empty_metrics
    
    # Alternative 1: Predict FULL original dataset for correct EF calculation
    full_predictions, full_uncertainties = learner.predict(original_compound_pool, data_manager)
    
    # Extract unlabeled subset for selection logic
    unlabeled_mask = original_compound_pool['ID'].isin(unlabeled_pool['ID'])
    unlabeled_indices = np.where(unlabeled_mask)[0]
    
    # Extract predictions for unlabeled compounds only
    if full_predictions is not None:
        if isinstance(full_predictions, np.ndarray):
            unlabeled_predictions = full_predictions[unlabeled_indices]
        else:
            unlabeled_predictions = full_predictions.iloc[unlabeled_indices]
    else:
        unlabeled_predictions = None
    
    if full_uncertainties is not None:
        if isinstance(full_uncertainties, np.ndarray):
            unlabeled_uncertainties = full_uncertainties[unlabeled_indices]
        else:
            unlabeled_uncertainties = full_uncertainties.iloc[unlabeled_indices]
    else:
        unlabeled_uncertainties = None
    
    # Prepare CSV tracking data if export is enabled
    cycle_predictions = None
    cycle_uncertainties = None
    cycle_selections = []
    
    if export_csv:
        # Store predictions with compound info for CSV export (unlabeled pool only for consistency)
        cycle_predictions = unlabeled_pool[['ID', 'SMILES']].copy()
        if unlabeled_predictions is not None:
            if isinstance(unlabeled_predictions, np.ndarray):
                cycle_predictions[f'prediction_cycle_{cycle}'] = unlabeled_predictions
            elif hasattr(unlabeled_predictions, 'iloc'):
                cycle_predictions[f'prediction_cycle_{cycle}'] = unlabeled_predictions.iloc[:, 0].values
        
        # Store uncertainties if available
        if unlabeled_uncertainties is not None:
            cycle_uncertainties = unlabeled_pool[['ID', 'SMILES']].copy()
            if isinstance(unlabeled_uncertainties, np.ndarray):
                cycle_uncertainties[f'uncertainty_cycle_{cycle}'] = unlabeled_uncertainties
            elif hasattr(unlabeled_uncertainties, 'iloc'):
                cycle_uncertainties[f'uncertainty_cycle_{cycle}'] = unlabeled_uncertainties.iloc[:, 0].values
    
    # Apply pruning if specified (to unlabeled subset)
    pruned_pool = unlabeled_pool
    pruning_stats = {'pruned_count': 0, 'original_pool_size': len(unlabeled_pool)}
    
    if pruning_strategy is not None:
        try:
            pruned_pool, pruning_info = apply_pruning_strategy(
                pool=unlabeled_pool,
                predictions=unlabeled_predictions,
                uncertainties=unlabeled_uncertainties,
                strategy=pruning_strategy,
                params=pruning_params or {},
                score_direction=score_direction
            )
            pruning_stats.update(pruning_info)
            logger.info(f"Cycle {cycle}: Pruning removed {pruning_stats['pruned_count']} compounds "
                       f"({pruning_stats['pruned_count']/len(unlabeled_pool)*100:.1f}% of pool)")
        except Exception as e:
            logger.warning(f"Cycle {cycle}: Pruning failed ({e}), continuing without pruning")
            pruned_pool = unlabeled_pool
    
    # Calculate batch size based on original pool size for consistent batch sizes across cycles
    batch_size = max(1, int(original_pool_size * batch_fraction))
    batch_size = min(batch_size, len(pruned_pool))  # Don't exceed available compounds
    
    logger.debug(f"Cycle {cycle}: Selecting {batch_size} compounds from {len(pruned_pool)} candidates using {strategy} strategy")
    
    # Select compounds using strategy (from pruned pool)
    # Need to align predictions with pruned pool if pruning was applied
    if pruned_pool is unlabeled_pool:
        # No pruning - use unlabeled predictions directly
        selection_predictions = unlabeled_predictions
        selection_uncertainties = unlabeled_uncertainties
    else:
        # Pruning was applied - need to extract relevant subset
        pruned_ids = set(pruned_pool['ID'])
        unlabeled_ids = list(unlabeled_pool['ID'])
        
        # Find indices of pruned compounds in unlabeled pool
        pruned_indices = [i for i, id_val in enumerate(unlabeled_ids) if id_val in pruned_ids]
        
        if unlabeled_predictions is not None:
            if isinstance(unlabeled_predictions, np.ndarray):
                selection_predictions = unlabeled_predictions[pruned_indices]
            else:
                selection_predictions = unlabeled_predictions.iloc[pruned_indices]
        else:
            selection_predictions = None
            
        if unlabeled_uncertainties is not None:
            if isinstance(unlabeled_uncertainties, np.ndarray):
                selection_uncertainties = unlabeled_uncertainties[pruned_indices]
            else:
                selection_uncertainties = unlabeled_uncertainties.iloc[pruned_indices]
        else:
            selection_uncertainties = None
    
    selected_compounds = select_compounds_by_strategy(
        pool=pruned_pool,
        predictions=selection_predictions,
        uncertainties=selection_uncertainties,
        strategy=strategy,
        batch_size=batch_size,
        score_direction=score_direction,
        data_manager=data_manager,
        acquisition_params=acquisition_params
    )
    
    # Measure selected compounds
    measured_compounds = oracle.measure(selected_compounds[['ID', 'SMILES']], [target_column])
    
    # Track selection history for CSV export
    if export_csv:
        # Get predictions and uncertainties for selected compounds
        selected_ids = selected_compounds['ID'].values
        for i, compound_id in enumerate(selected_ids):
            # Find the prediction and uncertainty for this compound in unlabeled pool
            compound_idx = unlabeled_pool[unlabeled_pool['ID'] == compound_id].index
            if len(compound_idx) > 0:
                idx = compound_idx[0]
                pool_idx = unlabeled_pool.index.get_loc(idx)
                
                selection_record = {
                    'ID': compound_id,
                    'SMILES': selected_compounds.iloc[i]['SMILES'],
                    'selected_cycle': cycle,
                    'strategy': strategy,
                    'prediction_at_selection': unlabeled_predictions[pool_idx] if unlabeled_predictions is not None else None,
                    'uncertainty_at_selection': unlabeled_uncertainties[pool_idx] if unlabeled_uncertainties is not None else None,
                    'oracle_measured_value': measured_compounds[measured_compounds['ID'] == compound_id][target_column].iloc[0] if target_column in measured_compounds.columns else None
                }
                cycle_selections.append(selection_record)
    
    # Update data (functional updates - create new DataFrames)
    new_labeled_data = pd.concat([labeled_data, measured_compounds], ignore_index=True)
    
    # Remove selected compounds from the original unlabeled pool by ID (not index)
    selected_ids = set(selected_compounds['ID'])
    new_unlabeled_pool = unlabeled_pool[~unlabeled_pool['ID'].isin(selected_ids)].reset_index(drop=True)
    
    # Calculate cycle metrics
    cycle_metrics = {
        'cycle': cycle,
        'strategy': strategy,
        'batch_fraction': batch_fraction,
        'selected_count': len(selected_compounds),
        'remaining_pool': len(new_unlabeled_pool),
        'cumulative_labeled': len(new_labeled_data),
        'pruning_strategy': pruning_strategy,
        'original_pool_size': pruning_stats['original_pool_size'],
        'pruned_count': pruning_stats['pruned_count'],
        'pruned_pool_size': len(pruned_pool)
    }
    
    # Add prediction statistics if available (using unlabeled predictions for consistency)
    if unlabeled_predictions is not None:
        if hasattr(unlabeled_predictions, 'empty') and unlabeled_predictions.empty:
            pass  # Skip empty DataFrame/Series
        elif isinstance(unlabeled_predictions, np.ndarray) and len(unlabeled_predictions) > 0:
            cycle_metrics.update({
                'prediction_mean': float(np.mean(unlabeled_predictions)),
                'prediction_std': float(np.std(unlabeled_predictions))
            })
        elif hasattr(unlabeled_predictions, 'iloc') and len(unlabeled_predictions) > 0:
            cycle_metrics.update({
                'prediction_mean': float(unlabeled_predictions.iloc[:, 0].mean()),
                'prediction_std': float(unlabeled_predictions.iloc[:, 0].std())
            })
    
    if unlabeled_uncertainties is not None:
        if hasattr(unlabeled_uncertainties, 'empty') and unlabeled_uncertainties.empty:
            pass  # Skip empty DataFrame/Series
        elif isinstance(unlabeled_uncertainties, np.ndarray) and len(unlabeled_uncertainties) > 0:
            cycle_metrics.update({
                'uncertainty_mean': float(np.mean(unlabeled_uncertainties)),
                'uncertainty_std': float(np.std(unlabeled_uncertainties))
            })
        elif hasattr(unlabeled_uncertainties, 'iloc') and len(unlabeled_uncertainties) > 0:
            cycle_metrics.update({
                'uncertainty_mean': float(unlabeled_uncertainties.iloc[:, 0].mean()),
                'uncertainty_std': float(unlabeled_uncertainties.iloc[:, 0].std())
            })
    
    # Add measured value statistics
    if target_column in measured_compounds.columns:
        measured_values = measured_compounds[target_column]
        cycle_metrics.update({
            'measured_mean': float(measured_values.mean()),
            'measured_std': float(measured_values.std()),
            'measured_min': float(measured_values.min()),
            'measured_max': float(measured_values.max())
        })
    
    # Comprehensive evaluation integration
    if enable_evaluation and not labeled_data.empty:
        try:
            # Get model predictions on labeled data for model performance assessment
            model_pred_on_labeled, _ = learner.predict(new_labeled_data, data_manager)
            
            # Call comprehensive evaluation with FULL dataset predictions (for correct EF calculation)
            from .evaluation import evaluate_cycle, format_progress_output
            eval_metrics = evaluate_cycle(
                cycle=cycle,
                predictions=model_pred_on_labeled,
                ground_truth=new_labeled_data[target_column].values,
                labeled_data=new_labeled_data,
                selected_compounds=measured_compounds,
                target_column=target_column,
                oracle_type=oracle_type,
                ground_truth_data=ground_truth_data,
                pool_predictions=full_predictions,  # Use FULL dataset predictions for correct EF
                pool_ids=original_compound_pool['ID'].values,  # Use FULL dataset IDs
                uncertainties=full_uncertainties,
                score_direction=score_direction
            )
            
            # Add evaluation metrics to cycle metrics
            cycle_metrics.update(eval_metrics)
            
            # Display progress output if enabled
            if console_output:
                progress_output = format_progress_output(eval_metrics, oracle_type)
                print(progress_output)
                
        except Exception as e:
            logger.warning(f"Evaluation failed for cycle {cycle}: {e}")
            # Continue without evaluation metrics
    
    if export_csv:
        return new_labeled_data, new_unlabeled_pool, cycle_metrics, cycle_predictions, cycle_uncertainties, cycle_selections
    else:
        return new_labeled_data, new_unlabeled_pool, cycle_metrics


def select_compounds_by_strategy(
    pool: pd.DataFrame,
    predictions: Optional[pd.DataFrame],
    uncertainties: Optional[pd.DataFrame], 
    strategy: str,
    batch_size: int,
    score_direction: str = 'higher',
    data_manager: Optional[DataManager] = None,
    acquisition_params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Select compounds using specified strategy with full acquisition module support.
    
    Args:
        pool: Unlabeled compounds pool
        predictions: Model predictions on pool
        uncertainties: Model uncertainties on pool (optional)
        strategy: Selection strategy name
        batch_size: Number of compounds to select
        score_direction: Direction of score optimization ('higher' or 'lower' is better)
        data_manager: DataManager instance for advanced acquisition methods
        acquisition_params: Parameters for acquisition function initialization
        
    Returns:
        Selected compounds DataFrame
        
    Raises:
        ValueError: If strategy is unknown
    """
    from .acquisition import get_acquisition_function, list_acquisition_functions
    
    # Ensure we don't select more than available
    actual_batch_size = min(batch_size, len(pool))
    
    # Prepare compound data with predictions and uncertainties
    compound_data = pool.copy()
    
    # Add predictions to compound data if available
    if predictions is not None:
        if isinstance(predictions, np.ndarray):
            compound_data['prediction'] = predictions
        elif isinstance(predictions, pd.DataFrame):
            # Use first column as prediction
            pred_col = predictions.columns[0]
            compound_data['prediction'] = predictions[pred_col].values
        else:
            # Try to convert to array
            compound_data['prediction'] = np.array(predictions)
    else:
        # Use random predictions if none available
        compound_data['prediction'] = np.random.uniform(0, 1, len(pool))
        logger.warning(f"No predictions available for {strategy} selection, using random predictions")
    
    # Add uncertainties to compound data if available
    if uncertainties is not None:
        if isinstance(uncertainties, np.ndarray):
            compound_data['uncertainty'] = uncertainties
        elif isinstance(uncertainties, pd.DataFrame):
            # Use first column as uncertainty
            unc_col = uncertainties.columns[0]
            compound_data['uncertainty'] = uncertainties[unc_col].values
        else:
            # Try to convert to array
            compound_data['uncertainty'] = np.array(uncertainties)
    else:
        # Use random uncertainties if none available
        compound_data['uncertainty'] = np.random.uniform(0.1, 0.5, len(pool))
        if strategy in ['ucb', 'ei', 'pi', 'thompson', 'entropy']:
            logger.warning(f"No uncertainties available for {strategy} selection, using random uncertainties")
    
    # Adjust predictions based on score direction for greedy-like methods
    if score_direction == 'lower' and strategy in ['greedy', 'topk']:
        compound_data['prediction'] = 1.0 - compound_data['prediction']
    
    try:
        # Get acquisition function from registry
        acquisition_class = get_acquisition_function(strategy)
        
        # Initialize acquisition function with appropriate parameters
        acquisition_params = acquisition_params or {}
        
        # Always pass data_manager to acquisition functions (standardized interface)
        if data_manager is None:
            logger.warning(f"DataManager not available for {strategy}, some methods may have limited functionality")
        
        acquisition_function = acquisition_class(data_manager=data_manager, **acquisition_params)
        
        # Perform selection
        selected = acquisition_function.select(compound_data, n_select=actual_batch_size)
        
        # Return original pool data (without added prediction/uncertainty columns)
        selected_ids = selected['ID'].tolist()
        return pool[pool['ID'].isin(selected_ids)]
        
    except KeyError:
        # Unknown strategy - provide helpful error message
        available_strategies = list_acquisition_functions()
        raise ValueError(f"Unknown strategy '{strategy}'. Available strategies: {available_strategies}")
    except Exception as e:
        # Fallback on any error
        logger.warning(f"Error with {strategy} acquisition: {e}. Falling back to greedy selection.")
        return _fallback_greedy_selection(compound_data, actual_batch_size, score_direction)


def _fallback_greedy_selection(
    compound_data: pd.DataFrame, 
    batch_size: int, 
    score_direction: str
) -> pd.DataFrame:
    """Fallback greedy selection when advanced methods fail."""
    if compound_data.empty:
        return compound_data.copy()
    
    if 'prediction' in compound_data.columns and len(compound_data) > 0:
        try:
            # Ensure prediction column is numeric
            compound_data = compound_data.copy()
            compound_data['prediction'] = pd.to_numeric(compound_data['prediction'], errors='coerce')
            
            # Remove rows with NaN predictions
            compound_data = compound_data.dropna(subset=['prediction'])
            
            if compound_data.empty:
                return compound_data
            
            batch_size = min(batch_size, len(compound_data))
            if score_direction == 'higher':
                return compound_data.nlargest(batch_size, 'prediction')
            else:
                return compound_data.nsmallest(batch_size, 'prediction')
        except (TypeError, ValueError):
            # Fall back to random sampling if numeric conversion fails
            batch_size = min(batch_size, len(compound_data))
            return compound_data.sample(n=batch_size) if batch_size > 0 else compound_data.iloc[:0]
    else:
        batch_size = min(batch_size, len(compound_data))
        return compound_data.sample(n=batch_size) if batch_size > 0 else compound_data.iloc[:0]


# Removed select_diverse_compounds - now using acquisition module directly


def apply_pruning_strategy(
    pool: pd.DataFrame,
    predictions: Optional[pd.DataFrame],
    uncertainties: Optional[pd.DataFrame],
    strategy: str,
    params: Dict[str, Any],
    score_direction: str = 'higher'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply pruning strategy to reduce compound pool size.
    
    Args:
        pool: Unlabeled compounds pool
        predictions: Model predictions on pool
        uncertainties: Model uncertainties on pool (optional)
        strategy: Pruning strategy name
        params: Strategy-specific parameters
        score_direction: Direction of score optimization ('higher' or 'lower')
        
    Returns:
        Tuple of (pruned_pool, pruning_info_dict)
        
    Raises:
        ValueError: If strategy is unknown
    """
    from .pruning import create_pruning_strategy
    
    # Map CLI-friendly names to class names
    strategy_mapping = {
        'score_based': 'ScoreBasedPruner'
    }
    
    strategy_class_name = strategy_mapping.get(strategy, strategy)
    
    # Ensure score_direction is in params for score-based pruning
    if strategy in ['score_based', 'ScoreBasedPruner']:
        params = params.copy()
        if 'score_direction' not in params:
            params['score_direction'] = score_direction
    
    # Create pruner instance
    try:
        pruner = create_pruning_strategy(strategy_class_name, params)
    except Exception as e:
        raise ValueError(f"Failed to create pruning strategy '{strategy}': {e}")
    
    # Apply pruning
    original_size = len(pool)
    
    # Convert predictions and uncertainties to numpy arrays if they're DataFrames
    if hasattr(predictions, 'values'):
        pred_array = predictions.values.flatten() if predictions.values.ndim > 1 else predictions.values
    else:
        pred_array = predictions
    
    if uncertainties is not None and hasattr(uncertainties, 'values'):
        unc_array = uncertainties.values.flatten() if uncertainties.values.ndim > 1 else uncertainties.values
    else:
        unc_array = uncertainties
    
    pruned_pool = pruner.prune(
        compounds=pool,
        predictions=pred_array,
        uncertainties=unc_array
    )
    
    pruned_count = original_size - len(pruned_pool)
    
    pruning_info = {
        'pruned_count': pruned_count,
        'original_pool_size': original_size,
        'pruned_pool_size': len(pruned_pool),
        'pruning_fraction': pruned_count / original_size if original_size > 0 else 0.0
    }
    
    return pruned_pool, pruning_info


def _export_comprehensive_csv(
    output_dir: str,
    labeled_data: pd.DataFrame,
    unlabeled_pool: pd.DataFrame,
    all_metrics: List[Dict[str, Any]],
    prediction_history: Dict[int, pd.DataFrame],
    uncertainty_history: Dict[int, pd.DataFrame],
    selection_history: List[Dict[str, Any]],
    compound_pool: pd.DataFrame,
    target_column: str,
    oracle_type: str,
    score_direction: str
) -> Dict[str, str]:
    """Export comprehensive CSV files for active learning results.
    
    Args:
        output_dir: Output directory path
        labeled_data: Final labeled compounds
        unlabeled_pool: Final unlabeled compounds
        all_metrics: Cycle metrics list
        prediction_history: Predictions by cycle
        uncertainty_history: Uncertainties by cycle
        selection_history: Selection tracking records
        compound_pool: Original compound pool
        target_column: Target property column name
        oracle_type: Oracle type ('run' or 'benchmark')
        score_direction: Score direction ('higher' or 'lower')
        
    Returns:
        Dictionary mapping file type to file path
    """
    from pathlib import Path
    import os
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = {}
    
    try:
        # 1. Export enhanced cycle metrics using evaluation module
        if all_metrics:
            metrics_path = output_path / 'cycle_metrics.csv'
            from .evaluation.core import export_metrics_csv
            export_metrics_csv(all_metrics, str(metrics_path), oracle_type, score_direction, target_column)
            csv_files['cycle_metrics'] = str(metrics_path)
        
        # 2. Export predictions by cycle
        if prediction_history:
            predictions_path = output_path / 'predictions_by_cycle.csv'
            _export_predictions_by_cycle_csv(prediction_history, uncertainty_history, compound_pool, labeled_data, target_column, predictions_path)
            csv_files['predictions_by_cycle'] = str(predictions_path)
            logger.info(f"Exported predictions for {len(prediction_history)} cycles to {predictions_path}")
        
        # 3. Export selection history
        if selection_history:
            selection_path = output_path / 'selection_history.csv'
            _export_selection_history_csv(selection_history, selection_path)
            csv_files['selection_history'] = str(selection_path)
            logger.info(f"Exported selection history for {len(selection_history)} compounds to {selection_path}")
        
        # 4. Export best compounds (run mode only)
        if oracle_type == 'run' and not labeled_data.empty:
            best_path = output_path / 'best_compounds.csv'
            _export_best_compounds_csv(labeled_data, target_column, score_direction, best_path)
            csv_files['best_compounds'] = str(best_path)
            logger.info(f"Exported {len(labeled_data)} best compounds to {best_path}")
        
        return csv_files
        
    except Exception as e:
        logger.error(f"Error during CSV export: {e}")
        return csv_files



def _export_predictions_by_cycle_csv(prediction_history: Dict[int, pd.DataFrame], uncertainty_history: Dict[int, pd.DataFrame], compound_pool: pd.DataFrame, labeled_data: pd.DataFrame, target_column: str, output_path: Path) -> None:
    """Export predictions and uncertainties by cycle with final oracle values."""
    try:
        # Start with original compound pool structure
        result_df = compound_pool[['ID', 'SMILES']].copy()
        
        # Add prediction columns for each cycle
        for cycle in sorted(prediction_history.keys()):
            cycle_preds = prediction_history[cycle]
            pred_col = f'prediction_cycle_{cycle}'
            
            # Merge predictions
            if pred_col in cycle_preds.columns:
                result_df = result_df.merge(
                    cycle_preds[['ID', pred_col]], 
                    on='ID', 
                    how='left'
                )
        
        # Add uncertainty columns for each cycle (if available)
        for cycle in sorted(uncertainty_history.keys()):
            cycle_uncs = uncertainty_history[cycle]
            unc_col = f'uncertainty_cycle_{cycle}'
            
            # Merge uncertainties
            if unc_col in cycle_uncs.columns:
                result_df = result_df.merge(
                    cycle_uncs[['ID', unc_col]], 
                    on='ID', 
                    how='left'
                )
        
        # Add final oracle values for labeled compounds
        if target_column in labeled_data.columns:
            oracle_values = labeled_data[['ID', target_column]].rename(columns={target_column: 'final_oracle_value'})
            result_df = result_df.merge(oracle_values, on='ID', how='left')
        
        result_df.to_csv(output_path, index=False)
        
    except Exception as e:
        logger.error(f"Error exporting predictions by cycle to CSV: {e}")


def _export_selection_history_csv(selection_history: List[Dict[str, Any]], output_path: Path) -> None:
    """Export compound selection history with predictions and oracle values."""
    try:
        if not selection_history:
            logger.warning("No selection history to export")
            return
        
        selection_df = pd.DataFrame(selection_history)
        selection_df.to_csv(output_path, index=False)
        
    except Exception as e:
        logger.error(f"Error exporting selection history to CSV: {e}")


def _export_best_compounds_csv(labeled_data: pd.DataFrame, target_column: str, score_direction: str, output_path: Path) -> None:
    """Export best compounds ranked by oracle score (run mode only)."""
    try:
        if target_column not in labeled_data.columns:
            logger.warning(f"Target column '{target_column}' not found in labeled data")
            return
        
        # Sort by oracle score according to score direction
        ascending = (score_direction == 'lower')
        best_compounds = labeled_data.sort_values(target_column, ascending=ascending).copy()
        
        # Add ranking column
        best_compounds['rank'] = range(1, len(best_compounds) + 1)
        
        # Reorder columns to put rank first
        cols = ['rank'] + [col for col in best_compounds.columns if col != 'rank']
        best_compounds = best_compounds[cols]
        
        best_compounds.to_csv(output_path, index=False)
        
    except Exception as e:
        logger.error(f"Error exporting best compounds to CSV: {e}")
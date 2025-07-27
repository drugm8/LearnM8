"""Main active learning loop implementation."""

import pandas as pd
from pathlib import Path
from typing import Callable, Optional
from rich.console import Console

from learnm8.core.interfaces import Oracle, Learner
from learnm8.oracles.csv_oracle import CSVOracle
from learnm8.utils.logging import (
    setup_logging, get_logger, log_experiment_start, log_cycle_start,
    log_selection, log_file_operation
)
from learnm8.evaluation.monitor import (
    create_cycle_report, save_monitoring_results, print_cycle_report,
    save_consolidated_predictions, save_monitoring_csv
)
from learnm8.evaluation.metrics import calculate_ground_truth_enrichment_factors


def run_active_learning(
    compound_pool: pd.DataFrame,
    oracle: Oracle,
    learner: Learner,
    target_column: str,
    n_cycles: int,
    batch_size: int,
    selection_strategy: Callable,
    initial_selection_strategy: Optional[Callable] = None,
    score_direction: str = 'higher',
    random_state: int = 42,
    output_dir: Optional[Path] = None,
    monitoring_config: Optional[dict] = None
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run active learning experiment.
    
    Args:
        compound_pool: DataFrame with 'ID' and 'SMILES' columns
        oracle: Oracle instance for measuring compounds
        learner: Learner instance for making predictions
        target_column: Target property to learn
        n_cycles: Number of active learning cycles
        batch_size: Number of compounds to select per cycle
        selection_strategy: Function to select compounds (receives compounds with predictions)
        initial_selection_strategy: Strategy for initial selection (defaults to selection_strategy)
        score_direction: 'higher' or 'lower' for score interpretation
        random_state: Random seed
        output_dir: Directory to save results
        monitoring_config: Dict with monitoring parameters (top_k)
        
    Returns:
        Tuple of (final predictions DataFrame, monitoring results list)
    """
    # Setup logging
    console = Console()
    logger = setup_logging(console=console)
    
    # Setup
    if initial_selection_strategy is None:
        initial_selection_strategy = selection_strategy
    
    if monitoring_config is None:
        monitoring_config = {'top_k': 100}
    
    # Initialize tracking
    available_pool = compound_pool.copy()
    labeled_compounds = pd.DataFrame()
    monitoring_results = []
    all_cycle_predictions = []  # Store predictions from all cycles
    cumulative_compounds_selected = 0  # Track cumulative compounds
    
    # Get full ground truth for monitoring (in real scenarios, this wouldn't be available)
    all_properties = [target_column]
    if hasattr(oracle, 'ground_truth') and 'Activity' in oracle.ground_truth.columns:
        all_properties.append('Activity')
    ground_truth_full = oracle.measure(compound_pool, all_properties)
    
    # Calculate ground truth enrichment factors once (constant across all cycles)
    ground_truth_enrichment_factors = calculate_ground_truth_enrichment_factors(
        ground_truth_full, target_column, score_direction
    )
    
    # Log experiment start
    config = {
        'target_column': target_column,
        'n_compounds': len(compound_pool),
        'n_cycles': n_cycles,
        'selection_strategy': selection_strategy.__name__,
        'score_direction': score_direction,
        'batch_size': batch_size
    }
    log_experiment_start(logger, config)
    
    # Run active learning cycles
    for cycle in range(n_cycles):
        log_cycle_start(logger, cycle, n_cycles)
        
        if cycle == 0:
            # Initial selection (usually random)
            selected = initial_selection_strategy(available_pool, batch_size, random_state)
            log_selection(logger, len(selected), f"initial ({initial_selection_strategy.__name__})")
        else:
            # Make predictions on available pool
            predictions, uncertainty = learner.predict(available_pool)
            available_pool['prediction'] = predictions
            # Store uncertainty if available
            if uncertainty is not None:
                available_pool['uncertainty'] = uncertainty
            
            # Select next batch - call strategy with correct parameters
            if selection_strategy.__name__ == 'select_random':
                selected = selection_strategy(available_pool, batch_size, random_state)
            elif selection_strategy.__name__ == 'select_greedy':
                selected = selection_strategy(available_pool, batch_size, score_direction)
            elif selection_strategy.__name__ == 'select_diverse':
                selected = selection_strategy(available_pool, batch_size, random_state=random_state)
            else:
                # Fallback: try with score_direction first, then random_state
                try:
                    selected = selection_strategy(available_pool, batch_size, score_direction)
                except (TypeError, ValueError):
                    selected = selection_strategy(available_pool, batch_size, random_state)
            
            log_selection(logger, len(selected), selection_strategy.__name__)
        
        # Measure selected compounds
        measured = oracle.measure(selected, [target_column])
        selected_with_labels = pd.merge(selected[['ID', 'SMILES']], measured, on='ID')
        
        # Update labeled data and cumulative count
        if labeled_compounds.empty:
            labeled_compounds = selected_with_labels.copy()
        else:
            labeled_compounds = pd.concat([labeled_compounds, selected_with_labels], ignore_index=True)
        
        cumulative_compounds_selected += len(selected_with_labels)
        
        # Remove selected from available pool
        available_pool = available_pool[~available_pool['ID'].isin(selected['ID'])]
        
        # Train learner
        learner.train(selected_with_labels, target_column)
        
        # Make predictions on entire pool for monitoring
        all_predictions, all_uncertainty = learner.predict(compound_pool)
        
        # Create comprehensive predictions DataFrame for both monitoring and export
        cycle_prediction_df = pd.DataFrame({
            'cycle': cycle + 1,
            'ID': compound_pool['ID'],
            'SMILES': compound_pool['SMILES'],
            'prediction': all_predictions
        })
        
        # Add uncertainty column if available
        if all_uncertainty is not None:
            cycle_prediction_df['uncertainty'] = all_uncertainty
        
        # Extract predictions_df for monitoring (subset of cycle_prediction_df)
        predictions_df = cycle_prediction_df[['ID', 'prediction']]
        
        # Store cycle predictions for consolidated export
        all_cycle_predictions.extend(cycle_prediction_df.to_dict('records'))
        
        # Create cycle report  
        # Get previously selected compounds (all compounds selected before this cycle)
        previously_selected_df = labeled_compounds.copy() if not labeled_compounds.empty else None
        
        # Detect oracle mode for appropriate metric calculation
        oracle_mode = 'benchmark' if isinstance(oracle, CSVOracle) else 'run'
        
        report = create_cycle_report(
            cycle=cycle + 1,
            predictions_df=predictions_df,
            ground_truth_df=ground_truth_full,
            newly_selected_df=selected,
            target_column=target_column,
            score_direction=score_direction,
            cumulative_compounds_selected=cumulative_compounds_selected,
            ground_truth_enrichment_factors=ground_truth_enrichment_factors,
            previously_selected_df=previously_selected_df,
            oracle_mode=oracle_mode
        )
        
        # Add total compounds info for parameter sweep analysis
        report['total_compounds'] = len(compound_pool)
        
        # Add previous average score for trend
        if monitoring_results:
            report['prev_avg_score'] = monitoring_results[-1]['avg_score_selected']
        
        monitoring_results.append(report)
        print_cycle_report(report, score_direction)
        
        # Skip intermediate file saves - only export final predictions.csv and monitoring.csv
    
    # Final predictions
    final_predictions_array, final_uncertainty = learner.predict(compound_pool)
    final_predictions = pd.DataFrame({
        'ID': compound_pool['ID'],
        'SMILES': compound_pool['SMILES'],
        'prediction': final_predictions_array
    })
    
    # Add uncertainty column if available
    if final_uncertainty is not None:
        final_predictions['uncertainty'] = final_uncertainty
    
    # Save only the requested output files
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export only predictions.csv and monitoring.csv
        predictions_path = output_dir / "predictions.csv"
        monitoring_path = output_dir / "monitoring.csv"
        
        save_consolidated_predictions(
            all_cycle_predictions, 
            ground_truth_full, 
            target_column, 
            predictions_path
        )
        save_monitoring_csv(monitoring_results, monitoring_path)
        
        log_file_operation(logger, "Saved predictions", str(predictions_path))
        log_file_operation(logger, "Saved monitoring", str(monitoring_path))
    
    return final_predictions, monitoring_results
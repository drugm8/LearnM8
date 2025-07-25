"""Main active learning loop implementation."""

import pandas as pd
from pathlib import Path
from typing import Callable, Optional
from core.interfaces import Oracle, Learner
from evaluation.monitor import create_cycle_report, save_monitoring_results, print_cycle_report


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
        monitoring_config: Dict with monitoring parameters (top_k, enrichment_percentile)
        
    Returns:
        Tuple of (final predictions DataFrame, monitoring results list)
    """
    # Setup
    if initial_selection_strategy is None:
        initial_selection_strategy = selection_strategy
    
    if monitoring_config is None:
        monitoring_config = {'top_k': 100, 'enrichment_percentile': 1.0}
    
    # Initialize tracking
    available_pool = compound_pool.copy()
    labeled_compounds = pd.DataFrame()
    monitoring_results = []
    
    # Get full ground truth for monitoring (in real scenarios, this wouldn't be available)
    all_properties = [target_column]
    if hasattr(oracle, 'ground_truth') and 'Activity' in oracle.ground_truth.columns:
        all_properties.append('Activity')
    ground_truth_full = oracle.measure(compound_pool, all_properties)
    
    print(f"Starting active learning experiment")
    print(f"Target: {target_column} ({'higher' if score_direction == 'higher' else 'lower'} is better)")
    print(f"Total compounds: {len(compound_pool)}")
    print(f"Cycles: {n_cycles}")
    print(f"Batch size: {batch_size}")
    print("-" * 60)
    
    for cycle in range(n_cycles):
        print(f"\nCycle {cycle + 1}/{n_cycles}")
        
        if cycle == 0:
            # Initial selection (usually random)
            selected = initial_selection_strategy(available_pool, batch_size, random_state)
            print(f"Initial selection: {len(selected)} compounds")
        else:
            # Make predictions on available pool
            predictions = learner.predict(available_pool)
            available_pool['prediction'] = predictions
            
            # Select next batch
            selected = selection_strategy(available_pool, batch_size, score_direction)
            print(f"Selected {len(selected)} compounds using {selection_strategy.__name__}")
        
        # Measure selected compounds
        measured = oracle.measure(selected, [target_column])
        selected_with_labels = pd.merge(selected[['ID', 'SMILES']], measured, on='ID')
        
        # Update labeled data
        if labeled_compounds.empty:
            labeled_compounds = selected_with_labels.copy()
        else:
            labeled_compounds = pd.concat([labeled_compounds, selected_with_labels], ignore_index=True)
        
        # Remove selected from available pool
        available_pool = available_pool[~available_pool['ID'].isin(selected['ID'])]
        
        # Train learner
        learner.train(selected_with_labels, target_column)
        
        # Make predictions on entire pool for monitoring
        all_predictions = learner.predict(compound_pool)
        predictions_df = pd.DataFrame({
            'ID': compound_pool['ID'],
            'prediction': all_predictions
        })
        print(predictions_df.head())  # Debugging line to check predictions
        
        # Create cycle report
        report = create_cycle_report(
            cycle=cycle + 1,
            predictions_df=predictions_df,
            ground_truth_df=ground_truth_full,
            newly_selected_df=selected,
            target_column=target_column,
            score_direction=score_direction,
            top_k=monitoring_config['top_k'],
            enrichment_percentile=monitoring_config['enrichment_percentile']
        )
        
        # Add previous average score for trend
        if monitoring_results:
            report['prev_avg_score'] = monitoring_results[-1]['avg_score_selected']
        
        monitoring_results.append(report)
        print_cycle_report(report, score_direction)
        
        # Save intermediate results if output directory provided
        if output_dir and cycle % 5 == 0:
            save_monitoring_results(monitoring_results, output_dir / "monitoring_intermediate.csv")
            predictions_df.to_csv(output_dir / f"predictions_cycle_{cycle + 1}.csv", index=False)
    
    # Final predictions
    final_predictions = pd.DataFrame({
        'ID': compound_pool['ID'],
        'SMILES': compound_pool['SMILES'],
        'prediction': learner.predict(compound_pool)
    })
    
    # Save final results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_monitoring_results(monitoring_results, output_dir / "monitoring_results.csv")
        final_predictions.to_csv(output_dir / "final_predictions.csv", index=False)
        labeled_compounds.to_csv(output_dir / "labeled_compounds.csv", index=False)
    
    return final_predictions, monitoring_results
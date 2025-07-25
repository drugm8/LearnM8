"""Experiment runner for active learning workflows."""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

from core.active_learning import run_active_learning
from core.interfaces import Oracle, Learner
from oracles.csv_oracle import CSVOracle
from learners.random_forest import RandomForestLearner
from strategies.greedy import select_greedy
from strategies.random import select_random
from strategies.diversity import select_diverse


def create_experiment_config(
    compound_pool_path: str,
    ground_truth_path: str,
    target_column: str,
    n_cycles: int = 10,
    batch_size_fraction: float = 0.1,
    selection_strategy: str = 'greedy',
    initial_strategy: str = 'random',
    score_direction: str = 'higher',
    learner_type: str = 'random_forest',
    random_state: int = 42,
    top_k: int = 100,
    enrichment_percentile: float = 1.0
) -> dict:
    """Create experiment configuration dictionary."""
    return {
        'compound_pool_path': compound_pool_path,
        'ground_truth_path': ground_truth_path,
        'target_column': target_column,
        'n_cycles': n_cycles,
        'batch_size_fraction': batch_size_fraction,
        'selection_strategy': selection_strategy,
        'initial_strategy': initial_strategy,
        'score_direction': score_direction,
        'learner_type': learner_type,
        'random_state': random_state,
        'monitoring': {
            'top_k': top_k,
            'enrichment_percentile': enrichment_percentile
        },
        'timestamp': datetime.now().isoformat()
    }


def get_selection_strategy(name: str):
    """Get selection strategy function by name."""
    strategies = {
        'greedy': select_greedy,
        'random': select_random,
        'diverse': select_diverse,
        'diversity': select_diverse
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name]


def get_learner(name: str, random_state: int) -> Learner:
    """Get learner instance by name."""
    learners = {
        'random_forest': RandomForestLearner,
        'rf': RandomForestLearner
    }
    
    if name not in learners:
        raise ValueError(f"Unknown learner: {name}. Available: {list(learners.keys())}")
    
    return learners[name](random_state=random_state)


def run_experiment(config: dict, output_dir: Optional[Path] = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run a complete active learning experiment.
    
    Args:
        config: Experiment configuration dictionary
        output_dir: Directory to save results
        
    Returns:
        Tuple of (final predictions, monitoring results)
    """
    # Load compound pool
    compound_pool = pd.read_csv(config['compound_pool_path'])
    
    # Validate required columns
    required_cols = ['ID', 'SMILES']
    missing_cols = [col for col in required_cols if col not in compound_pool.columns]
    if missing_cols:
        raise ValueError(f"Compound pool missing columns: {missing_cols}")
    
    # Calculate batch size
    batch_size = int(len(compound_pool) * config['batch_size_fraction'])
    if batch_size < 1:
        raise ValueError(f"Batch size too small: {batch_size}. Increase batch_size_fraction.")
    
    print(f"Calculated batch size: {batch_size} ({config['batch_size_fraction']*100:.1f}% of {len(compound_pool)} compounds)")
    
    # Create components
    oracle = CSVOracle(config['ground_truth_path'])
    learner = get_learner(config['learner_type'], config['random_state'])
    selection_strategy = get_selection_strategy(config['selection_strategy'])
    initial_strategy = get_selection_strategy(config['initial_strategy'])
    
    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(output_dir / 'experiment_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    # Run active learning
    final_predictions, monitoring_results = run_active_learning(
        compound_pool=compound_pool,
        oracle=oracle,
        learner=learner,
        target_column=config['target_column'],
        n_cycles=config['n_cycles'],
        batch_size=batch_size,
        selection_strategy=selection_strategy,
        initial_selection_strategy=initial_strategy,
        score_direction=config['score_direction'],
        random_state=config['random_state'],
        output_dir=output_dir,
        monitoring_config=config['monitoring']
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    if monitoring_results:
        first_result = monitoring_results[0]
        last_result = monitoring_results[-1]
        
        print(f"Top-{config['monitoring']['top_k']} Overlap:")
        print(f"  First cycle: {first_result['top_k_overlap']:.2f}%")
        print(f"  Last cycle:  {last_result['top_k_overlap']:.2f}%")
        print(f"  Improvement: {last_result['top_k_overlap'] - first_result['top_k_overlap']:+.2f}%")
        
        print(f"\nAverage Score of Selected:")
        print(f"  First cycle: {first_result['avg_score_selected']:.4f}")
        print(f"  Last cycle:  {last_result['avg_score_selected']:.4f}")
        
        direction_better = config['score_direction'] == 'higher'
        score_improved = (last_result['avg_score_selected'] > first_result['avg_score_selected']) == direction_better
        print(f"  Trend: {'Better' if score_improved else 'Worse'}")
        
        if last_result.get('enrichment_factor') is not None:
            print(f"\nEnrichment Factor (top {config['monitoring']['enrichment_percentile']}%):")
            print(f"  First cycle: {first_result.get('enrichment_factor', 'N/A')}")
            print(f"  Last cycle:  {last_result['enrichment_factor']:.2f}")
    
    return final_predictions, monitoring_results
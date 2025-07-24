"""
LearnM8 Active Learning Module

This module provides the core active learning functionality for molecular compound selection
and iterative model training using ground truth data from CSV files.
"""

import pandas as pd
import gc
import math

from helpers.oracle import get_ground_truth_data
from helpers.helpers import remove_right_df_from_left_df
from helpers.query_functions import random_query_function
from learners.learner_abc import learner

def learnM8(learner: learner, 
           compound_pool_csv_path,
           ground_truth_path, 
           target_column,
           batch_size_fraction=0.1,
           cycles=10,
           first_query_function=None,
           query_function=None,
           seed=None):
    """
    Run LearnM8 active learning on existing ground truth data.
    
    This simplified function performs iterative active learning by querying compounds
    from a pool, getting ground truth labels from existing data, and training a model
    to select the most informative compounds for future iterations.
    
    Args:
        learner: Active learning model instance implementing the learner interface.
                Must have methods: teach(), query(), set_seed(), set_query_function()
        compound_pool_csv_path (str): Path to CSV file containing the compound pool.
                                    Must have 'ID' and 'SMILES' columns
        ground_truth_path (str): Path to CSV file containing ground truth data.
                               Must have 'ID' column and the target_column
        target_column (str): Column name in ground truth data to learn/predict.
                           Examples: 'Activity', 'CHEMPLP', 'LinF9'
        batch_size_fraction (float): Fraction of total compounds to query per iteration.
                                   Default 0.1 means 10% of pool size per batch
        cycles (int): Number of active learning iterations to perform.
                     Use -1 for single large batch mode. Default: 10
        first_query_function: Function for initial compound selection strategy.
                            Default: random selection
        query_function: Function for subsequent compound selection strategies.
                       Default: greedy selection based on model predictions
        seed (int): Random seed for reproducibility. Default: None
        
    Returns:
        bool: True if active learning completed successfully
        
    Example:
        >>> from learners.rf_learner import rf_learner
        >>> from helpers.query_functions import random_query_function, greedy_query_function
        >>> 
        >>> model = rf_learner()
        >>> success = learnM8(
        ...     learner=model,
        ...     compound_pool_csv_path='data/compounds.csv',
        ...     ground_truth_path='data/ground_truth.csv', 
        ...     target_column='Activity',
        ...     batch_size_fraction=0.1,
        ...     cycles=5,
        ...     first_query_function=random_query_function,
        ...     query_function=greedy_query_function,
        ...     seed=42
        ... )
    
    Note:
        This simplified version focuses on single target column learning and removes
        complexity around hyperparameter tuning and multiple scoring function prediction.
        It works directly with ground truth data without consensus scoring.
    """
    # Validate input parameters
    if learner is None:
        raise ValueError("Learner instance is required")
    if not compound_pool_csv_path or not ground_truth_path:
        raise ValueError("Both compound pool and ground truth CSV paths are required")
    if not target_column:
        raise ValueError("Target column name is required")
    
    # Set up learner configuration
    learner.set_seed(seed)
    learner.set_query_function(query_function)
    
    # Load compound pool (full dataset of available compounds)
    try:
        compound_pool = pd.read_csv(compound_pool_csv_path)
    except Exception as e:
        raise IOError(f"Failed to read compound pool file {compound_pool_csv_path}: {e}")
    
    # Validate compound pool structure
    required_columns = ['ID', 'SMILES']
    missing_columns = [col for col in required_columns if col not in compound_pool.columns]
    if missing_columns:
        raise KeyError(f"Compound pool missing required columns: {missing_columns}")

    # Extract only ID and SMILES for the active learning pool
    compound_pool = compound_pool.loc[:, ['ID', 'SMILES']]
    
    # Configure learner for single target column learning
    learner.set_column_to_learn(target_column)
    learner.set_do_scoring_function_list_prediction(False)  # Simplified: single target only
    
    # Calculate actual batch size from fraction
    total_compounds = compound_pool.shape[0]
    actual_batch_size = math.floor(total_compounds * batch_size_fraction)
    
    if actual_batch_size < 1:
        raise ValueError(f"Batch size too small: {actual_batch_size}. "
                        f"Try increasing batch_size_fraction or using more compounds")
    
    # Get initial sample using first query function (typically random)
    if first_query_function is None:
        first_query_function = random_query_function
        
    initial_sample = first_query_function(compound_pool, actual_batch_size, seed)
    
    # Handle single batch mode (cycles = -1)
    if cycles == -1:
        # Single batch mode: use 10x normal batch size for comprehensive sampling
        actual_batch_size *= 10
        cycles = 1
    
    # Configure learner with batch size and target columns
    learner.set_int_batch_size(batch_size=actual_batch_size)
    learner.set_scoring_functions([target_column])  # Single target column
    
    # Remove initial sample from available pool
    compound_pool = remove_right_df_from_left_df(compound_pool, initial_sample)
    
    # Get ground truth labels for initial sample
    try:
        initial_labeled_data = get_ground_truth_data(
            ground_truth_path, 
            initial_sample, 
            ['SMILES', target_column]
        )
    except Exception as e:
        raise IOError(f"Failed to get ground truth data for initial sample: {e}")
    
    # Train learner with initial labeled data
    learner.teach(initial_labeled_data)
    
    # Main active learning loop
    for iteration in range(cycles + 1):
        # Query next batch of compounds using learner's strategy
        try:
            queried_compounds = learner.query(
                compound_pool, 
                compound_pool_csv_path,
                target_column
            )
        except Exception as e:
            print(f"Warning: Query failed at iteration {iteration}: {e}")
            break
        
        # Save predictions from final iteration (for evaluation)
        if iteration == cycles:
            break
        
        # Remove queried compounds from pool to avoid re-selection
        compound_pool = remove_right_df_from_left_df(compound_pool, queried_compounds)
        
        # Get ground truth labels for newly queried compounds
        try:
            queried_labeled_data = get_ground_truth_data(
                ground_truth_path, 
                queried_compounds, 
                ['SMILES', target_column]
            )
        except Exception as e:
            print(f"Warning: Failed to get ground truth for iteration {iteration}: {e}")
            break
        
        # Retrain learner with new labeled data
        learner.teach(queried_labeled_data)
        
        # Force garbage collection to manage memory usage
        # Important for large molecular datasets
        gc.collect()
    
    return True
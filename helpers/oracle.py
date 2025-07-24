"""
Oracle module for retrieving ground truth molecular data in active learning experiments.

This module provides functionality to look up ground truth scoring data for molecular
compounds, acting as an oracle that provides "experimental" results in the active
learning pipeline for drug discovery and molecular screening.
"""

import pandas as pd
from typing import List
import os


def get_ground_truth_data(ground_truth_csv_path: str, 
                         lookup_data: pd.DataFrame, 
                         requested_columns: List[str]) -> pd.DataFrame:
    """
    Retrieve ground truth data for specified compounds and columns from any CSV file.
    
    Acts as a generic oracle that provides ground truth data for active learning
    experiments. Takes a dataset of compound IDs and returns the corresponding
    data from specified columns in a ground truth CSV file.
    
    Args:
        ground_truth_csv_path (str): Path to CSV file containing ground truth data.
                                   Must contain an 'ID' column for matching compounds
        lookup_data (pd.DataFrame): DataFrame containing compound identifiers.
                                       Must have an 'ID' column with compound identifiers
        requested_columns (List[str]): List of column names to retrieve from ground truth data.
                                     Can be any columns present in the CSV file.
                                     Examples: ['CHEMPLP', 'LinF9', 'Activity', 'pIC50']
    
    Returns:
        pd.DataFrame: DataFrame containing the requested data with columns:
                     - 'ID': Compound identifier (always included)
                     - Additional columns for each requested column from the CSV
                     
    Raises:
        FileNotFoundError: If ground truth CSV file doesn't exist
        KeyError: If required columns are missing from either dataset
        ValueError: If no compounds match between datasets or input validation fails
        
    Example:
        >>> compounds = pd.DataFrame({'ID': ['COMP_001', 'COMP_002']})
        >>> columns = ['SMILES', 'Activity', 'pIC50']
        >>> result = get_ground_truth_data('data/ground_truth.csv', compounds, columns)
        >>> print(result.columns)
        Index(['ID', 'SMILES', 'Activity', 'pIC50'], dtype='object')
    """
    # Validate input parameters
    if not os.path.exists(ground_truth_csv_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_csv_path}")
    
    if lookup_data.empty:
        raise ValueError("Compound dataset is empty")
        
    if 'ID' not in lookup_data.columns:
        raise KeyError("Compound dataset must contain an 'ID' column")
    
    if not requested_columns:
        raise ValueError("At least one column must be requested")
    
    # Load ground truth data from CSV file
    try:
        ground_truth_dataframe = pd.read_csv(ground_truth_csv_path)
    except Exception as e:
        raise IOError(f"Failed to read ground truth file {ground_truth_csv_path}: {e}")
    
    # Validate ground truth data structure
    if ground_truth_dataframe.empty:
        raise ValueError(f"Ground truth file is empty: {ground_truth_csv_path}")
        
    # Check for required ID column and requested columns
    all_required_columns = ['ID'] + requested_columns
    missing_columns = [col for col in all_required_columns if col not in ground_truth_dataframe.columns]
    
    if missing_columns:
        available_columns = list(ground_truth_dataframe.columns)
        raise KeyError(f"Missing columns in ground truth data: {missing_columns}. "
                      f"Available columns: {available_columns}")
    
    # Extract compound IDs for lookup (only the ID column is needed for merging)
    compound_ids_for_lookup = lookup_data[['ID']].copy()
    
    # Perform inner join to get ground truth data for requested compounds
    # This ensures we only return data for compounds that exist in both datasets
    merged_ground_truth_data = pd.merge(
        left=compound_ids_for_lookup,
        right=ground_truth_dataframe,
        left_on='ID',
        right_on='ID', 
        how='inner'
    )
    
    # Validate that we found matching compounds
    if merged_ground_truth_data.empty:
        raise ValueError("No matching compounds found between compound dataset and ground truth data")
    
    # Select only the requested columns in a consistent order
    output_columns = ['ID'] + requested_columns
    final_result = merged_ground_truth_data[output_columns].copy()
    
    return final_result


def validate_ground_truth_file(ground_truth_csv_path: str) -> dict:
    """
    Validate and analyze the structure of a ground truth CSV file.
    
    Utility function to check the format and content of ground truth data files
    before using them in active learning experiments or data retrieval.
    
    Args:
        ground_truth_csv_path (str): Path to ground truth CSV file to validate
        
    Returns:
        dict: Validation report containing:
              - 'is_valid': Boolean indicating if file is valid
              - 'num_rows': Number of data rows in file
              - 'available_columns': List of all available columns
              - 'numeric_columns': List of columns containing numeric data
              - 'sample_ids': Sample of IDs from the file (up to 5)
              - 'issues': List of any validation issues found
              
    Example:
        >>> report = validate_ground_truth_file('data/ground_truth.csv')
        >>> print(f"Valid: {report['is_valid']}")
        >>> print(f"Available columns: {report['available_columns']}")
        >>> print(f"Numeric columns: {report['numeric_columns']}")
    """
    validation_report = {
        'is_valid': False,
        'num_rows': 0,
        'available_columns': [],
        'numeric_columns': [],
        'sample_ids': [],
        'issues': []
    }
    
    # Check if file exists
    if not os.path.exists(ground_truth_csv_path):
        validation_report['issues'].append(f"File not found: {ground_truth_csv_path}")
        return validation_report
    
    try:
        # Load and analyze the file
        dataframe = pd.read_csv(ground_truth_csv_path)
        
        # Check for empty file
        if dataframe.empty:
            validation_report['issues'].append("File is empty")
            return validation_report
        
        # Check for required ID column
        if 'ID' not in dataframe.columns:
            validation_report['issues'].append("Missing required 'ID' column")
        
        # Get all available columns
        all_columns = dataframe.columns.tolist()
        validation_report['available_columns'] = all_columns
        
        # Identify numeric columns (useful for data analysis)
        numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        validation_report['numeric_columns'] = numeric_columns
        
        # Update validation report
        validation_report['num_rows'] = len(dataframe)
        
        if 'ID' in dataframe.columns:
            sample_ids = dataframe['ID'].head(5).tolist()
            validation_report['sample_ids'] = sample_ids
        
        # Check for duplicate IDs
        if 'ID' in dataframe.columns:
            duplicate_ids = dataframe['ID'].duplicated().sum()
            if duplicate_ids > 0:
                validation_report['issues'].append(f"Found {duplicate_ids} duplicate IDs")
        
        # File is valid if no critical issues found
        critical_issues = [issue for issue in validation_report['issues'] 
                         if 'Missing required' in issue or 'File not found' in issue or 'empty' in issue]
        validation_report['is_valid'] = len(critical_issues) == 0
        
    except Exception as e:
        validation_report['issues'].append(f"Error reading file: {e}")
    
    return validation_report
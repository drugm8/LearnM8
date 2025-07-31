"""CSV-based oracle for looking up pre-computed compound properties."""

import pandas as pd
from pathlib import Path
from learnm8.core.interfaces import Oracle


class CSVOracle(Oracle):
    """Oracle that retrieves compound properties from a CSV file."""
    
    def __init__(self, data_path: str = None, csv_path: str = None, **kwargs):
        """
        Initialize the CSV oracle.
        
        Args:
            data_path: Path to CSV file containing ground truth data (new parameter name)
            csv_path: Path to CSV file (legacy parameter for backward compatibility)
            **kwargs: Additional parameters for compatibility
        """
        # Handle both new and legacy parameter names
        if data_path is not None:
            self.csv_path = Path(data_path)
        elif csv_path is not None:
            self.csv_path = Path(csv_path)
        else:
            raise ValueError("Either data_path or csv_path must be provided")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Load the ground truth data
        self.ground_truth = pd.read_csv(self.csv_path)
        
        # Validate required columns
        if 'ID' not in self.ground_truth.columns:
            raise ValueError("CSV must contain an 'ID' column")
    
    def measure(self, compounds: pd.DataFrame, properties: list[str]) -> pd.DataFrame:
        """
        Look up property values for given compounds.
        
        Args:
            compounds: DataFrame with 'ID' column (and optionally 'SMILES')
            properties: List of column names to retrieve
            
        Returns:
            DataFrame with 'ID', 'SMILES' (if available), and requested property columns
        """
        # Validate requested properties exist
        missing_props = [p for p in properties if p not in self.ground_truth.columns]
        if missing_props:
            available = list(self.ground_truth.columns)
            raise ValueError(f"Properties not found: {missing_props}. Available: {available}")
        
        # Determine which columns to preserve from input compounds
        preserve_cols = ['ID']
        if 'SMILES' in compounds.columns:
            preserve_cols.append('SMILES')
        
        # Extract compound data to preserve
        compound_data = compounds[preserve_cols].copy()
        
        # Determine which columns to get from ground truth
        ground_truth_cols = ['ID'] + properties
        # If SMILES is not in compounds but is in ground truth, get it from ground truth
        if 'SMILES' not in compounds.columns and 'SMILES' in self.ground_truth.columns:
            ground_truth_cols.append('SMILES')
        
        # Merge with ground truth to get requested properties
        result = pd.merge(
            compound_data,
            self.ground_truth[ground_truth_cols],
            on='ID',
            how='inner'
        )
        
        # Check if any compounds were not found
        if len(result) < len(compounds):
            missing_count = len(compounds) - len(result)
            print(f"Warning: {missing_count} compounds not found in ground truth")
        
        return result
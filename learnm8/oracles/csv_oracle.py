"""CSV-based oracle for looking up pre-computed compound properties."""

import pandas as pd
from pathlib import Path
from core.interfaces import Oracle


class CSVOracle(Oracle):
    """Oracle that retrieves compound properties from a CSV file."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the CSV oracle.
        
        Args:
            csv_path: Path to CSV file containing ground truth data
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load the ground truth data
        self.ground_truth = pd.read_csv(csv_path)
        
        # Validate required columns
        if 'ID' not in self.ground_truth.columns:
            raise ValueError("CSV must contain an 'ID' column")
    
    def measure(self, compounds: pd.DataFrame, properties: list[str]) -> pd.DataFrame:
        """
        Look up property values for given compounds.
        
        Args:
            compounds: DataFrame with 'ID' column
            properties: List of column names to retrieve
            
        Returns:
            DataFrame with 'ID' and requested property columns
        """
        # Validate requested properties exist
        missing_props = [p for p in properties if p not in self.ground_truth.columns]
        if missing_props:
            available = list(self.ground_truth.columns)
            raise ValueError(f"Properties not found: {missing_props}. Available: {available}")
        
        # Extract compound IDs
        compound_ids = compounds[['ID']].copy()
        
        # Merge with ground truth to get requested properties
        result = pd.merge(
            compound_ids,
            self.ground_truth[['ID'] + properties],
            on='ID',
            how='inner'
        )
        
        # Check if any compounds were not found
        if len(result) < len(compounds):
            missing_count = len(compounds) - len(result)
            print(f"Warning: {missing_count} compounds not found in ground truth")
        
        return result
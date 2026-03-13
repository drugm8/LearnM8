"""CSV-based oracle for looking up pre-computed compound properties."""

import logging
import polars as pl
from pathlib import Path
from learnm8.core.interfaces import Oracle

logger = logging.getLogger(__name__)


class CSVOracle(Oracle):
    """Oracle that retrieves compound properties from a CSV file."""

    def __init__(self, data_path: str, id_column: str = 'ID'):
        """
        Initialize the CSV oracle.

        Args:
            data_path: Path to CSV file containing ground truth data
            id_column: Name of the column to rename to 'ID' (default: 'ID')
        """
        self.csv_path = Path(data_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Oracle CSV file not found: {self.csv_path}. "
                f"Verify the file path is correct and the file exists."
            )

        # Load the ground truth data
        self.ground_truth = pl.read_csv(self.csv_path)

        # Convert numeric columns to proper data types
        # This handles mixed data like numeric values + 'no_score' strings
        for column in self.ground_truth.columns:
            if column not in ['ID', id_column]:
                col_dtype = self.ground_truth[column].dtype
                # Check if column is string type (Utf8 in Polars)
                if col_dtype == pl.Utf8:
                    # Try to convert to numeric, replacing invalid values with null
                    try:
                        numeric_col = self.ground_truth[column].cast(pl.Float64, strict=False)
                        # Only convert if we successfully converted some values to numeric
                        if not numeric_col.is_null().all():
                            self.ground_truth = self.ground_truth.with_columns(
                                numeric_col.alias(column)
                            )
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        pass

        # Filter out rows with null values in numeric columns
        # This removes compounds with invalid scores like 'no_score' that were converted to null
        initial_height = self.ground_truth.height
        for column in self.ground_truth.columns:
            if column not in ['ID', id_column, 'SMILES']:
                col_dtype = self.ground_truth[column].dtype
                if col_dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    null_count = self.ground_truth[column].is_null().sum()
                    if null_count > 0:
                        logger.warning(
                            f"CSVOracle: Filtering {null_count} compounds with null values in '{column}'"
                        )
                        self.ground_truth = self.ground_truth.filter(pl.col(column).is_not_null())

        filtered_count = initial_height - self.ground_truth.height
        if filtered_count > 0:
            logger.info(
                f"CSVOracle: Filtered {filtered_count} compounds with invalid/null property values"
            )

        # Validate the ID column exists
        if id_column not in self.ground_truth.columns:
            available = list(self.ground_truth.columns)
            raise ValueError(
                f"ID column '{id_column}' not found in oracle CSV. "
                f"Available columns: {available}. "
                f"Specify the correct column name with id_column parameter."
            )

        # Rename the specified column to 'ID' if it's not already 'ID'
        if id_column != 'ID':
            # If there's already an 'ID' column, drop it first to avoid duplicates
            if 'ID' in self.ground_truth.columns:
                self.ground_truth = self.ground_truth.drop('ID')
            self.ground_truth = self.ground_truth.rename({id_column: 'ID'})
    
    def measure(self, compounds: pl.DataFrame, properties: list[str]) -> pl.DataFrame:
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
            available = [c for c in self.ground_truth.columns if c not in ['ID', 'SMILES']]
            raise ValueError(
                f"Requested properties not found in oracle CSV: {missing_props}. "
                f"Available property columns: {available}. "
                f"Check that target_col matches a column name in your CSV file."
            )

        # Determine which columns to preserve from input compounds
        preserve_cols = ['ID']
        if 'SMILES' in compounds.columns:
            preserve_cols.append('SMILES')

        # Extract compound data to preserve with row order tracking
        compound_data = compounds.select(preserve_cols).with_row_index('_input_order')

        # Determine which columns to get from ground truth
        ground_truth_cols = ['ID'] + properties
        # If SMILES is not in compounds but is in ground truth, get it from ground truth
        if 'SMILES' not in compounds.columns and 'SMILES' in self.ground_truth.columns:
            ground_truth_cols.append('SMILES')

        # Merge with ground truth to get requested properties
        result = compound_data.join(
            self.ground_truth.select(ground_truth_cols),
            on='ID',
            how='inner'
        )

        # Validate that no null values exist in requested properties
        for prop in properties:
            null_mask = result[prop].is_null()
            if null_mask.any():
                null_count = null_mask.sum()
                null_ids = result.filter(null_mask)['ID'].to_list()
                raise ValueError(
                    f"CSVOracle cannot measure property '{prop}' for {null_count} compounds "
                    f"(IDs: {null_ids[:5]}{'...' if len(null_ids) > 5 else ''}). "
                    f"These compounds have invalid/null values in the ground truth data."
                )

        # Preserve input order by sorting on temporary order column, then remove it
        result = result.sort('_input_order').drop('_input_order')

        # Check if any compounds were not found
        if result.height < compounds.height:
            missing_count = compounds.height - result.height
            print(f"Warning: {missing_count} compounds not found in ground truth")

        return result
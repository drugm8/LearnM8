import logging
from pathlib import Path
from typing import Optional, Union

import polars as pl
import datamol as dm
from rdkit import Chem

logger = logging.getLogger(__name__)

SUPPORTED_CSV_FORMATS = {'.csv', '.txt', '.tsv'}
SUPPORTED_SDF_FORMATS = {'.sdf', '.sd'}
SUPPORTED_SMI_FORMATS = {'.smi', '.smiles'}


def load_compound_file(
    file_path: Union[str, Path],
    smiles_column: Optional[str] = None,
    id_column: Optional[str] = None,
    name_column: Optional[str] = None,
    n_jobs: int = -1,
    progress: bool = True
) -> pl.DataFrame:
    """
    Load compounds from CSV, SDF, or SMI files using optimal tool per format.

    Uses hybrid approach:
    - SDF files: datamol (parallel loading, progress bars, automatic handling)
    - CSV files: Polars (fastest, most compatible)
    - SMI files: RDKit (only option, datamol doesn't support)

    Args:
        file_path: Path to compound file
        smiles_column: CSV column name for SMILES (CSV only, default: auto-detect)
        id_column: CSV/SDF column/property name for ID (default: auto-detect)
        name_column: CSV column name for molecule name (CSV only, default: auto-detect)
        n_jobs: Number of parallel jobs for SDF loading (-1 = auto, default: -1)
        progress: Show progress bar for SDF loading (default: True)

    Returns:
        Polars DataFrame with standardized 'ID', 'SMILES', and additional columns

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: Unsupported file format or invalid file structure

    Examples:
        >>> # Load SDF file with automatic SMILES generation
        >>> compounds = load_compound_file('chembl_compounds.sdf')

        >>> # Load CSV with custom column names
        >>> compounds = load_compound_file(
        ...     'compounds.csv',
        ...     smiles_column='Canonical_SMILES',
        ...     id_column='Compound_ID'
        ... )

        >>> # Load SMILES file
        >>> compounds = load_compound_file('zinc_library.smi')
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Compound pool file not found: {file_path}")

    format_type = _detect_file_format(file_path)

    if format_type == 'sdf':
        logger.info(f"Loading SDF file with datamol (n_jobs={n_jobs})")
        return _load_sdf_with_datamol(file_path, id_column, n_jobs, progress)
    elif format_type == 'csv':
        logger.info("Loading CSV file with Polars")
        return _load_csv_with_polars(file_path, smiles_column, id_column, name_column)
    elif format_type == 'smi':
        logger.info("Loading SMILES file with RDKit")
        return _load_smi_with_rdkit(file_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def _detect_file_format(file_path: Path) -> str:
    """
    Detect file format from extension.

    Args:
        file_path: Path to file

    Returns:
        Format type: 'csv', 'sdf', or 'smi'

    Raises:
        ValueError: Unsupported file format
    """
    suffix = file_path.suffix.lower()

    if suffix == '.gz' and file_path.stem.endswith('.sdf'):
        return 'sdf'

    if suffix in SUPPORTED_CSV_FORMATS:
        return 'csv'
    elif suffix in SUPPORTED_SDF_FORMATS:
        return 'sdf'
    elif suffix in SUPPORTED_SMI_FORMATS:
        return 'smi'
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: "
            f"CSV ({', '.join(SUPPORTED_CSV_FORMATS)}), "
            f"SDF ({', '.join(SUPPORTED_SDF_FORMATS)}), "
            f"SMILES ({', '.join(SUPPORTED_SMI_FORMATS)})"
        )


def _load_csv_with_polars(
    file_path: Path,
    smiles_column: Optional[str],
    id_column: Optional[str],
    name_column: Optional[str]
) -> pl.DataFrame:
    """
    Load CSV/TSV files using Polars for optimal performance.

    Args:
        file_path: Path to CSV file
        smiles_column: Custom SMILES column name
        id_column: Custom ID column name
        name_column: Custom name column name

    Returns:
        Polars DataFrame with standardized 'ID' and 'SMILES' columns
    """
    df = pl.read_csv(file_path)

    if smiles_column and smiles_column in df.columns:
        df = df.rename({smiles_column: 'SMILES'})

    if id_column and id_column in df.columns:
        df = df.rename({id_column: 'ID'})
    elif 'ID' not in df.columns:
        if name_column and name_column in df.columns:
            df = df.rename({name_column: 'ID'})
        else:
            df = df.with_columns(
                pl.lit('compound_').add(pl.int_range(pl.len()).cast(pl.Utf8)).alias('ID')
            )

    if 'SMILES' not in df.columns:
        raise ValueError(
            f"SMILES column not found in CSV file. "
            f"Available columns: {df.columns}. "
            f"Use smiles_column parameter to specify the correct column name."
        )

    logger.info(f"Loaded {len(df)} compounds from CSV file")
    return df


def _load_sdf_with_datamol(
    file_path: Path,
    id_column: Optional[str],
    n_jobs: int,
    progress: bool
) -> pl.DataFrame:
    """
    Load SDF files using datamol with parallel processing.

    Args:
        file_path: Path to SDF file
        id_column: Custom ID column/property name
        n_jobs: Number of parallel jobs (-1 = auto)
        progress: Show progress bar

    Returns:
        Polars DataFrame with standardized 'ID' and 'SMILES' columns
    """
    mols = dm.read_sdf(str(file_path), sanitize=True, as_df=False, n_jobs=n_jobs)

    records = []
    for idx, mol in enumerate(mols):
        if mol is None:
            continue

        smiles = dm.to_smiles(mol)
        if not smiles:
            continue

        mol_id = None
        if id_column and mol.HasProp(id_column):
            mol_id = mol.GetProp(id_column)

        if not mol_id and mol.HasProp('_Name'):
            mol_id = mol.GetProp('_Name')

        if not mol_id:
            mol_id = f'mol_{idx}'

        record = {'ID': mol_id, 'SMILES': smiles}

        for prop_name in mol.GetPropNames():
            if prop_name != '_Name':
                record[prop_name] = mol.GetProp(prop_name)

        records.append(record)

    if not records:
        df = pl.DataFrame({'ID': [], 'SMILES': []})
    else:
        df = pl.DataFrame(records)
        other_cols = [c for c in df.columns if c not in ['ID', 'SMILES']]
        df = df.select(['ID', 'SMILES'] + other_cols)

    logger.info(f"Loaded {len(df)} compounds from SDF file")
    return df


def _load_smi_with_rdkit(file_path: Path) -> pl.DataFrame:
    """
    Load SMILES files using RDKit SmilesMolSupplier.

    Standard format: SMILES<tab>Name<tab>Properties...

    Args:
        file_path: Path to SMILES file

    Returns:
        Polars DataFrame with standardized 'ID' and 'SMILES' columns
    """
    compounds = []

    try:
        suppl = Chem.SmilesMolSupplier(
            str(file_path),
            delimiter='\t',
            smilesColumn=0,
            nameColumn=1,
            titleLine=True
        )

        for idx, mol in enumerate(suppl):
            if mol is None:
                logger.debug(f"Skipping invalid molecule at index {idx}")
                continue

            smiles = Chem.MolToSmiles(mol, canonical=True)
            mol_id = mol.GetProp('_Name') if mol.HasProp('_Name') else f'mol_{idx}'

            record = {'ID': mol_id, 'SMILES': smiles}

            for prop_name in mol.GetPropNames():
                if prop_name not in ['SMILES', '_Name']:
                    record[prop_name] = mol.GetProp(prop_name)

            compounds.append(record)

    except (RuntimeError, Exception) as e:
        if not compounds:
            raise ValueError(
                f"Failed to load SMILES file: {e}. "
                f"Expected format: SMILES<tab>Name with optional header line."
            )

    if not compounds:
        raise ValueError(
            f"No valid molecules found in SMILES file: {file_path}"
        )

    df = pl.DataFrame(compounds)
    logger.info(f"Loaded {len(df)} compounds from SMILES file")
    return df

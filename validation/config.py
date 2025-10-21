"""
Centralized configuration for validation notebooks.

This module provides standardized dataset paths and metadata for all validation notebooks.
Use these configurations to ensure consistency across validations.
"""

from pathlib import Path

# Root paths
VALIDATION_ROOT = Path(__file__).parent
PROJECT_ROOT = VALIDATION_ROOT.parent
DATASETS_ROOT = Path('/home/tony/Compound_Libraries/LearnM8_datasets')
ESSENCE_ROOT = PROJECT_ROOT / 'ESSENCE_benchmark_input'

# Standard validation datasets
STANDARD_DATASETS = {
    'ampc_30k': {
        'path': VALIDATION_ROOT / 'ampc_30k_subsample.csv',
        'target_column': 'dockscore',
        'id_column': 'ID',
        'smiles_column': 'SMILES',
        'expected_size': 30000,
        'description': 'AmpC β-lactamase 30K screening dataset',
        'score_direction': 'lower',
        'note': 'Lower docking scores indicate better binding affinity'
    },
    'ampc_500k': {
        'path': DATASETS_ROOT / 'AmpC/subsampled_data/AmpC_screen_500K.csv',
        'target_column': 'dockscore',
        'id_column': 'zincid',
        'smiles_column': 'smiles',
        'expected_size': 500000,
        'description': 'AmpC β-lactamase 500K screening dataset',
        'score_direction': 'lower',
        'note': 'Larger dataset for scalability tests'
    },
    'ampc_1000k': {
        'path': DATASETS_ROOT / 'AmpC/subsampled_data/AmpC_screen_1000K.csv',
        'target_column': 'dockscore',
        'id_column': 'zincid',
        'smiles_column': 'smiles',
        'expected_size': 1000000,
        'description': 'AmpC β-lactamase 1000K screening dataset',
        'score_direction': 'lower',
        'note': 'Large dataset for performance benchmarking'
    },
}

# Default dataset for validation
DEFAULT_DATASET = 'ampc_30k'

# Recommended datasets by validation type
RECOMMENDED_DATASETS = {
    'clustering': 'ampc_30k',
    'acquisition': 'ampc_30k',
    'pruning': 'ampc_30k',
    'uncertainty': 'ampc_30k',
    'scalability': 'ampc_1000k',
    'quick_test': 'ampc_30k'
}

def get_dataset_info(dataset_name: str = DEFAULT_DATASET) -> dict:
    """
    Get dataset configuration by name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'ampc_100k', 'hivpr')

    Returns
    -------
    dict
        Dataset configuration with path, columns, and metadata

    Raises
    ------
    KeyError
        If dataset_name is not found in STANDARD_DATASETS
    """
    if dataset_name not in STANDARD_DATASETS:
        available = ', '.join(STANDARD_DATASETS.keys())
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")

    return STANDARD_DATASETS[dataset_name].copy()

def get_dataset_path(dataset_name: str = DEFAULT_DATASET) -> Path:
    """
    Get the file path for a standard dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    Path
        Path to the dataset CSV file
    """
    return get_dataset_info(dataset_name)['path']

def validate_dataset_exists(dataset_name: str = DEFAULT_DATASET) -> bool:
    """
    Check if a dataset file exists on disk.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    bool
        True if dataset file exists, False otherwise
    """
    path = get_dataset_path(dataset_name)
    return path.exists()

def list_available_datasets() -> list:
    """
    Get list of all available dataset names.

    Returns
    -------
    list
        List of dataset names
    """
    return list(STANDARD_DATASETS.keys())

def get_recommended_dataset(validation_type: str) -> str:
    """
    Get recommended dataset for a specific validation type.

    Parameters
    ----------
    validation_type : str
        Type of validation (e.g., 'clustering', 'acquisition', 'pruning')

    Returns
    -------
    str
        Recommended dataset name, defaults to DEFAULT_DATASET if type not found
    """
    return RECOMMENDED_DATASETS.get(validation_type, DEFAULT_DATASET)

print("✅ Validation configuration module loaded")
print(f"   Available datasets: {', '.join(list_available_datasets())}")
print(f"   Default dataset: {DEFAULT_DATASET}")

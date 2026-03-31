from pathlib import Path

VALIDATION_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = VALIDATION_ROOT.parent
DATASETS_ROOT = Path('/home/tony/Compound_Libraries/LearnM8_datasets')
ESSENCE_ROOT = PROJECT_ROOT / 'ESSENCE_benchmark_input'

STANDARD_DATASETS = {
    'ampc_100k': {
        'path': VALIDATION_ROOT / 'AmpC_screen_100K.csv',
        'target_column': 'dockscore',
        'id_column': 'zincid',
        'smiles_column': 'smiles',
        'expected_size': 100000,
        'description': 'AmpC β-lactamase 100K screening dataset',
        'score_direction': 'lower',
        'note': 'Medium dataset for GPU learner benchmarking'
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
    'ampc_100k': {
        'path': DATASETS_ROOT / 'AmpC/subsampled_data/AmpC_screen_100K.csv',
        'target_column': 'dockscore',
        'id_column': 'zincid',
        'smiles_column': 'smiles',
        'expected_size': 100000,
        'description': 'AmpC β-lactamase 100K screening dataset',
        'score_direction': 'lower',
        'note': 'GPU benchmark dataset for GPyTorch GP evaluation'
    },
}

DEFAULT_DATASET = 'ampc_100k'

RECOMMENDED_DATASETS = {
    'clustering': 'ampc_100k',
    'acquisition': 'ampc_100k',
    'pruning': 'ampc_100k',
    'uncertainty': 'ampc_100k',
    'scalability': 'ampc_1000k',
    'quick_test': 'ampc_100k',
    'gpu_benchmark': 'ampc_100k',
}


def get_dataset_info(dataset_name: str = DEFAULT_DATASET) -> dict:
    if dataset_name not in STANDARD_DATASETS:
        available = ', '.join(STANDARD_DATASETS.keys())
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")

    return STANDARD_DATASETS[dataset_name].copy()


def get_dataset_path(dataset_name: str = DEFAULT_DATASET) -> Path:
    return get_dataset_info(dataset_name)['path']


def validate_dataset_exists(dataset_name: str = DEFAULT_DATASET) -> bool:
    path = get_dataset_path(dataset_name)
    return path.exists()


def list_available_datasets() -> list:
    return list(STANDARD_DATASETS.keys())


def get_recommended_dataset(validation_type: str) -> str:
    return RECOMMENDED_DATASETS.get(validation_type, DEFAULT_DATASET)

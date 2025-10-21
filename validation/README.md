# LearnM8 Validation Notebooks

This directory contains validation notebooks for various LearnM8 components, ensuring correctness and demonstrating best practices for molecular active learning workflows.

## Overview

The validation notebooks use **standardized datasets** and **consistent data loading patterns** to ensure reproducible results across all validations. This standardization was implemented to:

- Ensure consistent results across different validation types
- Reduce disk usage by eliminating duplicate datasets
- Simplify maintenance (single source of truth for dataset configurations)
- Facilitate reproducibility and comparison of results

## Standard Datasets

All validation notebooks use centralized dataset configurations defined in `validation/config.py`. The primary validation dataset is:

### Primary Dataset: AmpC_screen_100K.csv

**Location**: `/home/tony/Compound_Libraries/LearnM8_datasets/AmpC/subsampled_data/AmpC_screen_100K.csv`

**Details**:
- **Size**: ~96,807 compounds (after filtering invalid scores)
- **Target**: AmpC β-lactamase enzyme
- **Score Type**: Docking scores (`dockscore` column)
- **Score Direction**: Lower is better (more negative = stronger binding)
- **Use Case**: Primary validation dataset for acquisition strategies, uncertainty quantification, and clustering

**Why AmpC?**
- Real molecular docking data from a well-studied target
- Good size for comprehensive validation (not too small, not too large)
- Wide score distribution suitable for testing different acquisition strategies
- Well-understood biochemistry for interpreting results

### Secondary Dataset: HIVPR.csv

**Location**: `/home/tony/LearnM8/ESSENCE_benchmark_input/HIVPR.csv`

**Details**:
- **Size**: 36,286 compounds
- **Target**: HIV-1 protease
- **Score Type**: ESSENCE-Dock_Score
- **Score Direction**: Lower is better
- **Use Case**: Large-scale validation, diversity testing, ESSENCE benchmarks

**When to Use HIVPR:**
- Score-based pruning validation (demonstrates full-scale performance)
- ESSENCE benchmark comparisons
- When testing with different molecular targets for diversity

## Standardized Data Loading

### Using the Standardized Loader

All notebooks should use the centralized data loading function:

```python
from validation.data_loading import load_validation_dataset

# Load standard dataset
compound_pool, metadata = load_validation_dataset(
    dataset_name='ampc_100k',  # or 'hivpr'
    clean_invalid_scores=True
)

# Access metadata
print(f"Loaded {metadata['final_size']:,} compounds")
print(f"Target column: {metadata['target_column']}")
print(f"Score direction: {metadata['score_direction']}")
```

### With Subsampling

```python
# Load with subsampling for faster testing
compound_pool, metadata = load_validation_dataset(
    dataset_name='ampc_100k',
    subsample_size=10000,
    random_state=42
)
```

### Available Datasets

You can list all available datasets programmatically:

```python
from validation.config import list_available_datasets, get_dataset_info

# List all datasets
datasets = list_available_datasets()
print(f"Available datasets: {datasets}")

# Get dataset details
info = get_dataset_info('ampc_100k')
print(f"Path: {info['path']}")
print(f"Description: {info['description']}")
print(f"Expected size: {info['expected_size']:,} compounds")
```

## Validation Notebooks

### Clustering Validation

#### `bitbirch_clustering/bitbirch_validation_interactive.ipynb`
**Purpose**: Validate BitBIRCH clustering acquisition strategy

**Dataset**: AmpC_screen_100K.csv (ampc_100k)

**What it tests**:
- BitBIRCH clustering parameter sweep (threshold, branching_factor)
- Chemical space visualization with UMAP embeddings
- Cluster quality metrics and selection strategies
- Performance across different parameter combinations

**Key outputs**:
- Parameter sweep results with timing and cluster statistics
- PCA/UMAP embeddings showing cluster distributions
- Selection visualization highlighting diversity
- Recommended parameter configurations

### Acquisition Strategy Validation

#### `uncertainty_acquisition/ucb_acquisition/ucb_validation.ipynb`
**Purpose**: Validate Upper Confidence Bound (UCB) acquisition function

**Dataset**: AmpC_screen_100K.csv (ampc_100k)

**What it tests**:
- UCB mathematical correctness (LCB formula for minimization)
- Beta parameter sensitivity (exploration vs exploitation)
- Active learning progression over 10 cycles
- Performance metrics (R², RMSE, Top-10% overlap)

**Key outputs**:
- UCB landscape evolution visualizations
- Selection trajectory across chemical space
- Distribution evolution over cycles
- Multi-view dashboard showing comprehensive progress

#### `uncertainty_acquisition/ei_acquisition/ei_validation.ipynb`
**Purpose**: Validate Expected Improvement (EI) acquisition function

**Dataset**: AmpC_screen_100K.csv (ampc_100k)

**What it tests**:
- EI mathematical correctness (minimization formulation)
- Xi parameter sensitivity (exploration threshold)
- Adaptation to different data quality scenarios
- Active learning performance over 10 cycles

**Key outputs**:
- EI landscape with contour plots
- Biased stratified sampling scenarios (good/medium/poor start)
- Selection behavior analysis
- Performance comparison across xi values

#### `uncertainty_acquisition/visualization_demo.ipynb`
**Purpose**: Demonstrate 4-panel dashboard visualization for active learning

**Dataset**: AmpC_screen_100K.csv (ampc_100k) with dynamic subsampling

**What it demonstrates**:
- Multi-strategy comparison (greedy, UCB, EI, Thompson)
- Real-time visualization of active learning progress
- Animated GIF generation for presentations
- Dashboard interpretation and analysis

**Note**: This notebook uses dynamic subsampling for flexibility and demonstration purposes.

### Pruning Validation

#### `score_based_pruning/score_based_pruning_validation.ipynb`
**Purpose**: Validate score-based compound pruning strategy

**Dataset**: HIVPR.csv (hivpr) - **Uses HIVPR for large-scale demonstration**

**What it tests**:
- Score-based pruning with different fractions (0-90%)
- Chemical space visualization of progressive pruning
- Pruning impact on compound distribution
- Integration with active learning workflows

**Key outputs**:
- Pruning progression visualizations in chemical space
- Score distribution evolution
- Retention rate analysis
- Performance metrics across pruning fractions

**Why HIVPR for pruning?**
- Larger dataset (36K compounds) demonstrates scalability
- Different molecular target provides diversity testing
- ESSENCE benchmark compatibility

## Utility Modules

### `validation/config.py`
Centralized configuration for standard datasets. Defines:
- Dataset paths and metadata
- Column name mappings (ID, SMILES, target)
- Score direction specifications
- Recommended datasets by validation type

### `validation/data_loading.py`
Standardized data loading functions:
- `load_validation_dataset()`: Load standard datasets with preprocessing
- `setup_data_with_error_handling()`: Validate SMILES and filter invalid compounds

### `validation/create_embedding_plots.py`
Visualization utilities for clustering and chemical space analysis:
- `create_embedding_plots()`: Generate 3-panel embedding visualizations
- Clustering results, selected compounds, cluster size distributions

## Dataset Guidelines

### When to Use Each Dataset

| Validation Type | Recommended Dataset | Rationale |
|----------------|---------------------|-----------|
| **Acquisition strategies** | AmpC_100K | Standard size, good score distribution |
| **Uncertainty quantification** | AmpC_100K | Established benchmark for UCB/EI/PI |
| **Clustering** | AmpC_100K | Sufficient size for meaningful clusters |
| **Pruning** | HIVPR | Larger dataset demonstrates scalability |
| **Quick tests** | AmpC_100K (subsampled) | Fast iteration during development |
| **Scalability** | AmpC_1000K | Large-scale performance testing |
| **Diversity testing** | Both AmpC + HIVPR | Different targets for generalization |

### Dataset Selection Best Practices

1. **Default to AmpC_100K**: Unless you have a specific reason, use AmpC as the standard validation dataset
2. **Subsample for Development**: Use `subsample_size` parameter during development for faster iteration
3. **Use HIVPR for Diversity**: When you need to validate across different molecular targets
4. **Document Deviations**: If using a custom dataset, clearly document why in the notebook
5. **Maintain Reproducibility**: Always set `random_state` for subsampling

## Migration Guide

### Updating Old Notebooks

If you have notebooks using hard-coded dataset paths, migrate them to the standardized approach:

**Before (Hard-coded path)**:
```python
dataset_path = '/home/tony/LearnM8/validation/bitbirch_clustering/AmpC_screen_100K.csv'
ampc_data = pd.read_csv(dataset_path)
ampc_data = ampc_data.rename(columns={'zincid': 'ID', 'smiles': 'SMILES'})
ampc_data['dockscore'] = pd.to_numeric(ampc_data['dockscore'], errors='coerce')
ampc_data = ampc_data.dropna(subset=['dockscore'])
```

**After (Standardized loader)**:
```python
from validation.data_loading import load_validation_dataset

compound_pool, metadata = load_validation_dataset(
    dataset_name='ampc_100k',
    clean_invalid_scores=True
)
target_column = metadata['target_column']
```

### Benefits of Migration
- ✅ Automatic column renaming to standard format
- ✅ Consistent error handling for invalid scores
- ✅ Metadata access (size, path, score direction)
- ✅ No duplicate dataset files
- ✅ Easy switching between datasets
- ✅ Reproducible results across notebooks

## Adding New Validation Notebooks

When creating a new validation notebook:

1. **Use Standardized Data Loading**:
   ```python
   from validation.data_loading import load_validation_dataset
   compound_pool, metadata = load_validation_dataset('ampc_100k')
   ```

2. **Reference Canonical Paths**:
   ```python
   from validation.config import get_dataset_path
   dataset_path = get_dataset_path('ampc_100k')
   oracle = CSVOracle(str(dataset_path), id_column='zincid')
   ```

3. **Document Dataset Choice**:
   - Add markdown cell explaining which dataset and why
   - Reference this README for standard datasets

4. **Test with Both Datasets**:
   - Verify notebook works with both AmpC and HIVPR
   - Document any dataset-specific assumptions

5. **Use Metadata**:
   ```python
   # Get target column and score direction from metadata
   target_column = metadata['target_column']
   score_direction = metadata['score_direction']
   ```

## Troubleshooting

### Dataset Not Found Error

```python
FileNotFoundError: Dataset file not found: /path/to/dataset.csv
```

**Solution**: Verify the dataset exists and the path in `validation/config.py` is correct:

```python
from validation.config import validate_dataset_exists
if not validate_dataset_exists('ampc_100k'):
    print("Dataset not found! Check validation/config.py")
```

### Invalid SMILES Warnings

The standardized loader automatically handles invalid SMILES:
- Converts scores to numeric with `errors='coerce'`
- Removes compounds with NaN scores
- Logs the number of removed compounds

To preserve invalid compounds:
```python
compound_pool, metadata = load_validation_dataset(
    dataset_name='ampc_100k',
    clean_invalid_scores=False  # Keep all compounds
)
```

### Column Name Mismatches

The loader automatically renames columns to standard format:
- `zincid` → `ID`
- `smiles` → `SMILES`
- `dockscore` → `dockscore` (preserved)

Always use standardized column names in notebook code:
```python
# Always use 'ID' and 'SMILES' after loading
assert 'ID' in compound_pool.columns
assert 'SMILES' in compound_pool.columns
```

## Future Enhancements

Planned improvements to validation infrastructure:

1. **Dataset Versioning**: Track dataset versions and validate checksums
2. **Additional Datasets**: Add more molecular targets for diversity testing
3. **Automated Validation**: CI/CD pipeline for notebook regression testing
4. **Performance Benchmarks**: Standardized timing and memory profiling
5. **Visualization Templates**: Reusable plotting functions for all notebooks

## Contributing

When adding new validation notebooks:
1. Follow the standardized data loading patterns
2. Update this README with notebook description
3. Add dataset recommendations to the guidelines
4. Ensure notebooks are self-documenting with markdown cells
5. Test with multiple datasets to verify portability

## Questions?

For questions about validation datasets or standardization:
1. Check `validation/config.py` for available datasets
2. Review existing notebooks for examples
3. Consult CLAUDE.md in project root for development guidelines

# Quickstart

Get up and running with LearnM8 in under 5 minutes. This guide covers the essentials for your first active learning experiment.

## Your First Experiment (Python API)

The recommended way to use LearnM8 is through the Python API, which provides full programmatic control:

### Basic Benchmark Experiment

For a benchmark experiment with a CSV file containing known activities:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='compounds.csv',
    learner='gp',
    target_col='Activity',
    featurizer_type='morgan',
    n_cycles=10,
    batch_fraction=0.01
)
```

This code:
- Uses `compounds.csv` as both compound pool and oracle (benchmark mode)
- Predicts the `Activity` column
- Uses Gaussian Process regression with Morgan fingerprints
- Runs 10 active learning cycles

### Understanding the Parameters

**Required Parameters:**
- `compound_pool='compounds.csv'`: CSV file with `ID`, `SMILES`, and target columns
- `oracle='compounds.csv'`: Oracle for measurements (same file = benchmark mode)
- `learner='gp'`: Machine learning model (Gaussian Process)
- `target_col='Activity'`: Column name containing property values
- `featurizer_type='morgan'`: Molecular representation (Morgan fingerprints)

**Optional Parameters:**
- `n_cycles=10`: Number of active learning cycles (default: 10)
- `batch_fraction=0.01`: Fraction of pool to select each cycle (default: 0.01)

### Accessing Results

The `results` dictionary contains comprehensive experiment data:

```python
# Output directory path
output_dir = results['output_dir']

# Master DataFrame with all data (Polars DataFrame)
compounds_df = results['compounds_df']

# Cycle-by-cycle metrics
cycle_metrics = results['cycle_metrics']
for i, metrics in enumerate(cycle_metrics):
    print(f"Cycle {i}: EF1% = {metrics.get('ef_1pct', 'N/A'):.2f}")

# Labeled and unlabeled data
labeled_data = results['labeled_data']
unlabeled_data = results['unlabeled_data']

# Validation results
validation_result = results['validation_result']
print(f"Valid compounds: {len(validation_result.valid_compounds)}")
```

### Converting Polars to Pandas

LearnM8 uses Polars internally for performance. Convert to Pandas if needed:

```python
import pandas as pd

# Convert results to Pandas
compounds_pd = results['compounds_df'].to_pandas()
labeled_pd = results['labeled_data'].to_pandas()
```

### Where Are My Results?

Results are saved to a timestamped directory:

```
learnm8_output_20251118_143022/
├── compounds_final.csv          # Final dataset with predictions
├── cycle_metrics.csv            # Performance metrics per cycle
├── selection_history.csv        # Compounds selected each cycle
├── validation_report.csv        # SMILES validation results
└── experiment_config.json       # Full configuration for reproducibility
```

## Your First Experiment (CLI Alternative)

You can also run LearnM8 through the command-line interface for quick experiments:

### Basic Benchmark Experiment

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10
```

This command:
- Uses `compounds.csv` as both compound pool and oracle (benchmark mode)
- Predicts the `Activity` column
- Uses Gaussian Process regression with Morgan fingerprints
- Runs 10 active learning cycles

### CLI Parameters

**Required:**
- `compounds.csv`: CSV file with `ID`, `SMILES`, and target columns
- `--target Activity`: Column name containing property values
- `--learner gp`: Machine learning model
- `--featurizer morgan`: Molecular representation

**Optional:**
- `--n-cycles 10`: Number of cycles (default: 10)
- `--batch-fraction 0.01`: Fraction to select (default: 0.01)

## Validating Compounds

Before running experiments, validate your compound pool to catch errors early:

### Python API Validation

```python
from learnm8 import validate_compound_pool
import polars as pl

compounds = pl.read_csv('compounds.csv')
result = validate_compound_pool(compounds, n_jobs=-1, progress=True)

print(f"Valid: {len(result.valid_compounds)}")
print(f"Invalid: {len(result.invalid_compounds)}")
print(f"Success rate: {result.success_rate:.1%}")

# Access error details
for compound_id, error in result.validation_errors.items():
    print(f"{compound_id}: {error}")
```

### CLI Validation

```bash
learnm8 validate compounds.csv
```

This checks all SMILES strings for validity using datamol. Invalid compounds are reported with error messages.

**Save validation report:**

```bash
learnm8 validate compounds.csv -o validation_results/
```

Creates `validation_results/validation_report.csv` with detailed error information.

## Exploring Available Components

Discover what models, strategies, and featurizers are available:

### List Learners

```bash
learnm8 list learners
```

Shows all available machine learning models. Availability depends on installed dependencies (e.g., `chemprop` requires separate installation).

### List Acquisition Strategies

```bash
learnm8 list acquisition
```

Displays acquisition functions for compound selection (e.g., `greedy`, `ucb`, `random`, `bitbirch`).

### List Featurizers

```bash
learnm8 list featurizers
```

Shows molecular representation methods: `morgan`, `maccs`, `ecfp6`, `descriptors`, `morgan_feat`.

### List Predefined Schedules

```bash
learnm8 list schedules
```

Displays built-in cycle schedules for different screening scenarios:
- `quick`: 5 cycles for fast exploration
- `standard`: 10 cycles for balanced screening
- `intensive`: 20 cycles for thorough optimization
- `diverse`: 10 cycles with mixed diversity strategies

## Next Steps

### Try Different Models

Experiment with different learners to find the best for your dataset:

```python
from learnm8 import run_active_learning

# Random Forest (fast baseline)
results_rf = run_active_learning(
    compound_pool='compounds.csv', oracle='oracle.csv',
    learner='rf', target_col='Activity', featurizer_type='morgan', n_cycles=10
)

# Ensemble (robust performance)
results_ensemble = run_active_learning(
    compound_pool='compounds.csv', oracle='oracle.csv',
    learner='ensemble', target_col='Activity', featurizer_type='morgan', n_cycles=10
)

# XGBoost (high performance)
results_xgb = run_active_learning(
    compound_pool='compounds.csv', oracle='oracle.csv',
    learner='xgb', target_col='Activity', featurizer_type='morgan', n_cycles=10
)
```

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan --n-cycles 10
learnm8 run compounds.csv --target Activity --learner ensemble --featurizer morgan --n-cycles 10
```

### Use Custom Acquisition Strategies

Move beyond greedy selection with uncertainty-based strategies:

```python
# Upper Confidence Bound (exploration/exploitation balance)
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer_type='morgan',
    strategy='ucb',
    acquisition_params={'exploration_weight': 2.0},
    n_cycles=10
)
```

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --cycles "random:0.01 ucb:0.005*9"
```

### Production Screening with Custom Oracle

For real screening campaigns, provide a custom scoring function:

```python
# Python API
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='scoring_module.py:calculate_affinity',
    learner='mc_dropout',
    target_col='binding_score',
    featurizer_type='morgan',
    n_cycles=20
)
```

Where `scoring_module.py` contains:

```python
def calculate_affinity(smiles_list):
    """Calculate binding affinity for SMILES strings.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of scores (floats)
    """
    scores = []
    for smiles in smiles_list:
        # Your scoring logic (docking, ML prediction, etc.)
        score = your_scoring_function(smiles)
        scores.append(score)
    return scores
```

**CLI alternative:**
```bash
learnm8 run compounds.csv scoring_module.py:calculate_affinity \
  --target binding_score --learner mc_dropout --featurizer morgan --n-cycles 20
```

## Learn More

Now that you've run your first experiment:

- **[Core Concepts](concepts.md)** - Understand active learning fundamentals
- **[Running Experiments Tutorial](../tutorials/running-experiments.md)** - Detailed workflow guide
- **[Building Custom Cycles](../tutorials/building-custom-cycles.md)** - Advanced cycle specifications
- **[CLI Reference](../user-guide/cli-reference.md)** - Complete command documentation
- **[API Reference](../user-guide/api-reference.md)** - Python API documentation

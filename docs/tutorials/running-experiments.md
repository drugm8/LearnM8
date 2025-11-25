# Running Experiments

This tutorial walks through running active learning experiments using the Python API and CLI.

## Basic Python API Workflow

Run a simple benchmark experiment with 10 cycles using the Python API:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01
)
```

**Parameter Explanations:**
- `compound_pool='compounds.csv'`: Compound pool (must have `ID` and `SMILES` columns)
- `oracle=None`: Auto-detect oracle from compound pool (benchmark mode)
- `learner='gp'`: Gaussian Process model (provides uncertainty)
- `target_col='Activity'`: Target property column name
- `featurizer='morgan'`: Morgan fingerprints (2048-bit)
- `n_cycles=10`: Total number of active learning cycles

**Oracle Auto-Detection:**
When `oracle=None`, LearnM8 uses the compound pool CSV as the oracle (benchmark mode). If your CSV contains ground truth values, the experiment measures discovery/ranking metrics.

### Accessing Results Dictionary

The `results` dictionary contains:

```python
compounds_df = results['compounds_df']
cycle_metrics = results['cycle_metrics']
validation_result = results['validation_result']
output_dir = results['output_dir']

labeled_data = results['labeled_data']
unlabeled_data = results['unlabeled_data']
```

**Results Structure:**
- `compounds_df`: Final Polars DataFrame with all compounds
- `cycle_metrics`: List of dictionaries (one per cycle)
- `validation_result`: ValidationResult object with valid/invalid compounds
- `labeled_data`: Polars DataFrame filtered to labeled compounds
- `unlabeled_data`: Polars DataFrame filtered to unlabeled compounds

### Converting Polars to Pandas

LearnM8 uses Polars internally for performance. Convert to Pandas if needed:

```python
import polars as pl

compounds_pandas = results['compounds_df'].to_pandas()
labeled_pandas = results['labeled_data'].to_pandas()
```

### Understanding Output Directory Structure

Results are saved to a timestamped directory:

```
learnm8_results_20250118_143022/
├── compounds_final.csv          # Final master DataFrame
├── cycle_metrics.csv            # Metrics per cycle
├── selection_history.csv        # Compounds selected each cycle
├── validation_report.csv        # Validation results
├── config.json                  # Experiment configuration
└── learnm8.log                  # Execution log
```

**Key Files:**
- `compounds_final.csv`: All compounds with status (labeled/unlabeled), predictions, cycle information
- `cycle_metrics.csv`: Per-cycle metrics (enrichment factor, top-10 discovery, timing)
- `selection_history.csv`: Which compounds were selected in each cycle

## Basic CLI Workflow

You can also run experiments via the command-line interface:

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10
```

**Flag Explanations:**
- `compounds.csv`: Compound pool (must have `ID` and `SMILES` columns)
- `--target Activity`: Target property column name
- `--learner gp`: Gaussian Process model (provides uncertainty)
- `--featurizer morgan`: Morgan fingerprints (2048-bit)
- `--n-cycles 10`: Total number of active learning cycles

## Using Different Models

### Random Forest (Fast Baseline)

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01
)
```

**When to use:**
- Fast training and prediction
- Good general-purpose baseline
- Works well with 100-10,000 compounds
- Provides uncertainty via out-of-bag scoring

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan --n-cycles 10
```

### Gaussian Process (Gold Standard Uncertainty)

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01
)
```

**When to use:**
- Need principled uncertainty estimates
- Smaller datasets (<5,000 compounds)
- Research/benchmarking scenarios
- When uncertainty-based acquisition is critical

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10
```

### XGBoost (High Performance)

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='xgb',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01
)
```

**When to use:**
- Large datasets (10,000+ compounds)
- Need fast predictions
- Gradient boosting advantages (handles complex patterns)
- Production screening where speed matters

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner xgb --featurizer morgan --n-cycles 10
```

### Custom Model Parameters

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01,
    random_state=42
)
```

## Choosing Acquisition Strategies

### Random (Baseline)

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    strategy='random'
)
```

**When to use:**
- Baseline comparison
- Unbiased exploration
- Initial screening cycles

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --n-cycles 10 --strategy random
```

### Greedy (Pure Exploitation)

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    strategy='greedy'
)
```

**When to use:**
- Find best compounds quickly
- After initial exploration phase
- When model is confident
- Production screening focused on top hits

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --n-cycles 10 --strategy greedy
```

### UCB (Balanced Exploration/Exploitation)

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01,
    strategy='ucb',
    acquisition_params={'exploration_weight': 2.0}
)
```

**When to use:**
- Balance finding good compounds with reducing uncertainty
- Gaussian Process or ensemble models (provide uncertainty)
- Exploration in early cycles
- When both top hits and coverage matter

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --n-cycles 10 --strategy ucb
```

## Feature Caching for Speed

Enable persistent HDF5 caching (100x speedup on reuse):

```python
from pathlib import Path
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    cache_dir=Path('.shared_cache')
)
```

**Benefits:**
- First run: Features computed and cached
- Subsequent runs: Features loaded from cache (100x faster)
- Cache persists across experiments
- Share cache directory across multiple experiments

**Performance Impact:**

| Dataset Size | First Run | Cached Run | Speedup |
|--------------|-----------|------------|---------|
| 1,000 compounds | 2.5s | 0.03s | ~80x |
| 10,000 compounds | 25s | 0.15s | ~150x |
| 100,000 compounds | 250s | 1.2s | ~200x |

**Cache Location:**
- Default: `{output_dir}/.cache` (per-experiment)
- Shared: Specify absolute path to share across experiments
- Cache is keyed by SMILES + featurizer type
- Thread-safe for parallel experiments

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --n-cycles 10 --cache-dir .shared_cache
```

## Experiment Variations

### Production Screening with Custom Oracle

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compound_library.csv',
    oracle='scoring_module.py:calculate_affinity',
    learner='ensemble',
    target_col='binding_score',
    featurizer='morgan',
    n_cycles=20
)
```

**CLI alternative:**
```bash
learnm8 run compound_library.csv scoring_module.py:calculate_affinity \
  --target binding_score --learner ensemble --featurizer morgan --n-cycles 20
```

### Benchmark with Explicit Oracle

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle_ground_truth.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=15
)
```

**CLI alternative:**
```bash
learnm8 run compounds.csv oracle_ground_truth.csv --target Activity \
  --learner gp --featurizer morgan --n-cycles 15
```

### Complete Example with All Options

```python
from learnm8 import run_active_learning
from pathlib import Path

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01,
    strategy='greedy',
    initial_strategy='random',
    score_direction='higher',
    output_dir=Path('results/experiment_001'),
    cache_dir=Path('.shared_cache'),
    random_state=42
)

print(f"Labeled {len(results['labeled_data'])} compounds")
print(f"Results saved to {results['output_dir']}")
```

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --n-cycles 10 -o my_experiment_results/
```

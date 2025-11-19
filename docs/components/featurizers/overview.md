# Featurizers Overview

Featurizers convert molecular SMILES strings into numerical vectors that machine learning models can process. This transformation is essential for most learners, as neural networks and traditional ML algorithms require fixed-length numerical inputs rather than variable-length string representations.

## What Are Featurizers?

Featurizers encode molecular structure and chemical properties into fixed-dimensional feature vectors. Each SMILES string is transformed into a numerical array where similar molecules produce similar feature vectors, enabling ML models to learn structure-activity relationships.

**Transformation:**
```
SMILES: "CCO"  →  Featurizer  →  [0, 1, 0, 1, ..., 0]  (2048-dimensional vector)
```

## When Featurizers Are Required

### Required for Most Learners

All scikit-learn, PyTorch, and XGBoost learners require featurizers:

- RandomForest, AdvancedRandomForest
- GaussianProcess
- XGBoost, DecisionTree, LinearRegression
- MLP, MCDropout, Fastprop
- All ensemble variants (Mixed, RF, LR, XGB, DT, Fastprop)

**Python API:**
```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer_type='morgan'
)
```

**CLI:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan
```

### Optional for Graph Neural Networks

Chemprop works directly with SMILES strings and does not require featurizers. However, featurizers can be used in **hybrid mode** to combine graph features with molecular descriptors.

**Standard Chemprop (no featurizer needed):**
```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',
    target_col='Activity'
    # No featurizer_type needed
)
```

**Hybrid Chemprop (combines graph + descriptors):**
```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',
    target_col='Activity',
    featurizer_type='descriptors'  # Optional: enables hybrid mode
)
```

**CLI alternatives:**
```bash
# Standard
learnm8 run compounds.csv --target Activity --learner chemprop

# Hybrid
learnm8 run compounds.csv --target Activity --learner chemprop --featurizer descriptors
```

## Feature Extraction API

### Core Function

The primary feature extraction function is `extract_features()`, which handles caching, parallelization, and error handling automatically:

```python
from learnm8 import extract_features
from pathlib import Path

smiles_list = ['CCO', 'CCC', 'CCCO', 'c1ccccc1']

features = extract_features(
    smiles_list=smiles_list,
    featurizer_type='morgan',
    cache_dir=Path('.cache'),
    n_jobs=-1,
    show_progress=True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smiles_list` | List[str] | Required | List of SMILES strings |
| `featurizer_type` | str | Required | Featurizer type (`morgan`, `maccs`, `ecfp6`, `morgan_feat`, `descriptors`) |
| `cache_dir` | Path | `.cache` | Directory for HDF5 cache files |
| `n_jobs` | int | `-1` | Number of parallel jobs (-1 for auto-optimization) |
| `show_progress` | bool | `False` | Display progress bar (requires tqdm) |

**Returns:**
- `np.ndarray`: Feature matrix with shape `(n_compounds, n_features)`

### Integration with Active Learning

Feature extraction is handled automatically by `run_active_learning()`:

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer_type='morgan',
    cache_dir=Path('.shared_cache')  # Optional: specify cache location
)
```

## HDF5 Caching System

### Performance Benefits

LearnM8 uses HDF5-based caching to avoid recomputing molecular features, providing dramatic speedups for repeated experiments:

- **First computation:** Full feature extraction time
- **Subsequent runs:** **100x faster** (cache hit)
- **Partial caching:** Mixed workloads benefit from cached molecules

**Example performance:**
```
First run:  1000 molecules × 0.5s = 500s (8.3 minutes)
Second run: 1000 molecules cached = 5s (0.08 minutes)
Speedup: 100x
```

### How Caching Works

#### SMILES Hash-Based Keys

Each SMILES string is hashed (MD5) to create a unique cache key:

```python
import hashlib

smiles = "CCO"
cache_key = hashlib.md5(smiles.encode('utf-8')).hexdigest()
# cache_key: "f89cbecc5e46dc5a4b84bb94d75c4bbd"
```

Features are stored in HDF5 with these hash keys, enabling:
- **Deduplication:** Same SMILES always maps to same features
- **Partial loading:** Only uncached molecules are computed
- **Cache sharing:** Multiple experiments share same cache

#### Cache Structure

```
.cache/
├── morgan_features.h5         # Morgan fingerprints
├── maccs_features.h5          # MACCS keys
├── ecfp6_features.h5          # ECFP6 fingerprints
├── morgan_feat_features.h5    # Morgan feature fingerprints
└── descriptors_features.h5    # Mordred descriptors
```

Each HDF5 file contains a `features` group with datasets keyed by SMILES hash:

```
morgan_features.h5
└── features/
    ├── f89cbecc5e46dc5a4b84bb94d75c4bbd  # "CCO"
    ├── a3d5e19... # Another SMILES
    └── ...
```

### Cache Directory Management

#### Default Cache Location

If no cache directory is specified, features are cached in the experiment output directory:

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan
# Cache: ./learnm8_results_TIMESTAMP/.cache/
```

#### Explicit Cache Directory

Specify a cache directory for reuse across experiments:

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --cache-dir .shared_cache
```

**API:**
```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer_type='morgan',
    cache_dir=Path('.shared_cache')
)
```

### Sharing Cache Across Experiments

Use a shared cache directory to benefit from caching across multiple experiments:

```bash
mkdir .shared_cache

learnm8 run experiment1.csv --target Activity --learner rf --featurizer morgan --cache-dir .shared_cache
learnm8 run experiment2.csv --target Activity --learner gp --featurizer morgan --cache-dir .shared_cache
learnm8 run experiment3.csv --target Activity --learner xgb --featurizer morgan --cache-dir .shared_cache
```

**Benefits:**
- First experiment computes features
- Subsequent experiments reuse cached features (100x speedup)
- Works across different learners, strategies, and datasets
- Cache is valid as long as SMILES strings match

### Cache Invalidation

Cache invalidation is **automatic** and handled by SMILES hashing:

- **New SMILES:** Automatically computed and cached
- **Existing SMILES:** Retrieved from cache
- **Modified SMILES:** Treated as new (different hash)
- **Manual invalidation:** Delete cache files to force recomputation

**Manual cache clearing:**
```bash
rm -rf .cache/morgan_features.h5  # Clear Morgan cache
rm -rf .cache/*                    # Clear all caches
```

## Automatic Parallelization

Feature extraction is automatically parallelized based on dataset size, with no configuration required.

### Parallelization Strategy

| Dataset Size | Strategy | Reasoning |
|--------------|----------|-----------|
| < 100 compounds | Sequential (n_jobs=1) | Parallelization overhead exceeds benefit |
| 100-10,000 compounds | All cores (n_jobs=all) | Maximum parallelization benefit |
| > 10,000 compounds | Capped at 32 cores | Diminishing returns, memory considerations |

### Performance Characteristics

**Small datasets (< 100):**
```python
features = extract_features(smiles_list[:50], 'morgan')
# Sequential execution: ~25s
```

**Medium datasets (100-10k):**
```python
features = extract_features(smiles_list[:1000], 'morgan')
# Parallel (16 cores): ~30s vs ~480s sequential = 16x speedup
```

**Large datasets (> 10k):**
```python
features = extract_features(smiles_list[:100000], 'morgan')
# Parallel (32 cores): ~15 minutes vs ~8 hours sequential = 32x speedup
```

### Manual Parallelization Control

Override automatic optimization when needed:

**Force sequential execution:**
```python
features = extract_features(
    smiles_list=smiles_list,
    featurizer_type='morgan',
    n_jobs=1  # Disable parallelization
)
```

**Specify core count:**
```python
features = extract_features(
    smiles_list=smiles_list,
    featurizer_type='morgan',
    n_jobs=8  # Use exactly 8 cores
)
```

**Maximum parallelization:**
```python
features = extract_features(
    smiles_list=smiles_list,
    featurizer_type='morgan',
    n_jobs=-1  # Auto-optimize (default)
)
```

## Choosing a Featurizer

### Quick Selection Guide

| Goal | Recommended Featurizer | Reasoning |
|------|------------------------|-----------|
| General-purpose, balanced | `morgan` | Most common, good performance across tasks |
| Fast computation, small data | `maccs` | 167 bits, fastest computation |
| Larger molecular neighborhoods | `ecfp6` | Radius 3, captures more context |
| Pharmacophore features | `morgan_feat` | Feature-based, not atom-based |
| Maximum information | `descriptors` | 1613 features, richest representation |
| Chemprop hybrid mode | `descriptors` | Complements graph features |

### Performance Considerations

**Computation Speed:**
```
maccs > morgan ≈ ecfp6 ≈ morgan_feat >> descriptors
```

**Feature Dimensionality:**
```
maccs (167) < morgan (2048) = ecfp6 (2048) = morgan_feat (2048) < descriptors (1613)
```

**Memory Usage:**
```
maccs (167 bytes) < fingerprints (2048 bytes) < descriptors (6452 bytes float32)
```

## Common Use Cases

### Standard Workflow

Most experiments use Morgan fingerprints as the default:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer_type='morgan'
)
```

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan
```

### High-Throughput Screening

Use MACCS for fastest feature extraction:

```python
from pathlib import Path

results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer_type='maccs',
    cache_dir=Path('.cache')
)
```

**CLI alternative:**
```bash
learnm8 run large_library.csv --target Activity --learner rf --featurizer maccs --cache-dir .cache
```

### Descriptor-Rich Models

Use Mordred descriptors for maximum chemical information:

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer_type='descriptors'
)
```

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer descriptors
```

### Chemprop Hybrid

Combine graph neural network with molecular descriptors:

```python
from pathlib import Path

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',
    target_col='Activity',
    featurizer_type='descriptors',
    cache_dir=Path('.cache')
)
```

**CLI alternative:**
```bash
learnm8 run compounds.csv --target Activity --learner chemprop --featurizer descriptors
```

### Shared Cache Setup

Create shared cache for multiple experiments:

```python
from pathlib import Path
from learnm8 import run_active_learning

shared_cache = Path('.shared_cache')
shared_cache.mkdir(exist_ok=True)

# First experiment
results1 = run_active_learning(
    compound_pool='dataset1.csv', oracle='oracle1.csv',
    learner='rf', target_col='Activity', featurizer_type='morgan',
    cache_dir=shared_cache
)

# Second experiment (reuses cached features)
results2 = run_active_learning(
    compound_pool='dataset2.csv', oracle='oracle2.csv',
    learner='gp', target_col='Activity', featurizer_type='morgan',
    cache_dir=shared_cache
)
```

**CLI alternative:**
```bash
mkdir .shared_cache

learnm8 run dataset1.csv --target Activity --learner rf --featurizer morgan --cache-dir .shared_cache
learnm8 run dataset2.csv --target Activity --learner gp --featurizer morgan --cache-dir .shared_cache
learnm8 run dataset3.csv --target Activity --learner xgb --featurizer morgan --cache-dir .shared_cache
```

## Next Steps

- [Available Featurizers](available-featurizers.md) - Detailed documentation for each featurizer
- [Running Experiments](../../tutorials/running-experiments.md) - Complete experimental workflows
- [Learner Overview](../learners/overview.md) - Understanding learner-featurizer compatibility

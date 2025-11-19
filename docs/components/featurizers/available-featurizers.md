# Available Featurizers

LearnM8 provides five molecular featurizers, each encoding different aspects of molecular structure and properties. This guide covers the characteristics, use cases, and examples for each featurizer.

## Featurizer Comparison

| Featurizer | Dimensions | Speed | Best For | Notes |
|------------|------------|-------|----------|-------|
| `morgan` | 2048 | Fast | General-purpose screening | Default choice, circular fingerprints radius=2 |
| `maccs` | 167 | Fastest | Large libraries, rapid exploration | Structural keys, smallest representation |
| `ecfp6` | 2048 | Fast | Larger molecular context | Extended radius=3, more distant features |
| `morgan_feat` | 2048 | Fast | Pharmacophore-based tasks | Feature-based encoding (not atom types) |
| `descriptors` | 1613 | Slowest | Rich feature sets, hybrid Chemprop | Mordred descriptors, maximum information |

## Morgan Fingerprints

### Overview

Morgan fingerprints (also known as circular fingerprints) encode molecular structure based on atom neighborhoods up to a specified radius. This is the most commonly used featurizer in molecular machine learning.

**Algorithm:**
- Circular fingerprints with radius=2
- 2048-bit hashed representation
- Captures local chemical environments

**Dimensions:** 2048

### When to Use

- **Default choice** for most active learning tasks
- General-purpose molecular property prediction
- Good balance between information content and dimensionality
- Well-validated across diverse chemical spaces
- Fast computation suitable for large libraries

### Example Usage

**CLI:**
```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan
```

**API:**
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

**Direct Feature Extraction:**
```python
from learnm8 import extract_features

smiles_list = ['CCO', 'CCC', 'c1ccccc1']
features = extract_features(smiles_list, featurizer_type='morgan')
print(features.shape)  # (3, 2048)
```

## MACCS Keys

### Overview

MACCS (Molecular ACCess System) keys are a set of 167 predefined structural features based on common chemical substructures. This is the smallest and fastest featurizer.

**Algorithm:**
- 167 binary structural keys
- Each bit represents presence/absence of specific substructure
- Fixed patterns defined by MACCS system

**Dimensions:** 167

### When to Use

- **Large compound libraries** (>100k molecules) where speed matters
- Rapid exploration and initial screening
- Memory-constrained environments
- Fast baseline models
- When interpretability is important (each bit has defined meaning)

**Performance advantage:**
- Fastest computation (3-5x faster than Morgan)
- Smallest memory footprint (12x smaller than Morgan)
- Excellent for high-throughput virtual screening

### Example Usage

**CLI:**
```bash
learnm8 run large_library.csv --target Activity --learner rf --featurizer maccs
```

**API:**
```python
results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer_type='maccs',
    cache_dir=Path('.cache')
)
```

**Direct Feature Extraction:**
```python
from learnm8 import extract_features

features = extract_features(
    smiles_list=['CCO', 'CCC'],
    featurizer_type='maccs'
)
print(features.shape)  # (2, 167)
```

## ECFP6

### Overview

Extended-Connectivity Fingerprints with radius 3 (ECFP6 refers to diameter=6, radius=3). Captures larger molecular neighborhoods compared to standard Morgan fingerprints.

**Algorithm:**
- Circular fingerprints with radius=3
- 2048-bit hashed representation
- Encodes more distant atom relationships

**Dimensions:** 2048

### When to Use

- **Larger molecular context** important for activity
- Molecules with extended functional groups
- When radius=2 Morgan fingerprints insufficient
- Structure-activity relationships depend on distant features
- Complex molecular scaffolds

**When ECFP6 outperforms Morgan:**
- Large molecules (>30 heavy atoms)
- Long-range electronic effects
- Multi-ring systems with distant interactions

### Example Usage

**CLI:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer ecfp6
```

**API:**
```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer_type='ecfp6'
)
```

**Comparison with Morgan:**
```python
from learnm8 import extract_features

smiles_list = ['CCCCOc1ccc(cc1)C(=O)c1ccccc1']  # Large molecule

morgan_features = extract_features(smiles_list, featurizer_type='morgan')
ecfp6_features = extract_features(smiles_list, featurizer_type='ecfp6')

print(f"Morgan: {morgan_features.shape}")  # (1, 2048)
print(f"ECFP6:  {ecfp6_features.shape}")   # (1, 2048)
```

## Morgan Feature Fingerprints

### Overview

Morgan feature fingerprints encode pharmacophore features rather than exact atom types. This representation focuses on chemical properties (donor, acceptor, aromatic, etc.) instead of atomic identity.

**Algorithm:**
- Circular fingerprints with radius=2
- Feature-based encoding (pharmacophore properties)
- 2048-bit hashed representation
- Uses RDKit's `useFeatures=True` parameter

**Dimensions:** 2048

### When to Use

- **Pharmacophore-based screening** where functional properties matter more than exact atoms
- Drug-like molecule optimization
- Scaffold hopping (finding different scaffolds with similar properties)
- When activity depends on chemical features, not specific atom types
- Hit-to-lead optimization

**Feature types encoded:**
- Hydrogen bond donors
- Hydrogen bond acceptors
- Aromatic rings
- Aliphatic chains
- Positive/negative ionizable groups

### Example Usage

**CLI:**
```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan_feat
```

**API:**
```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer_type='morgan_feat'
)
```

**Comparison with Standard Morgan:**
```python
from learnm8 import extract_features

smiles = 'CCO'

morgan_standard = extract_features([smiles], featurizer_type='morgan')
morgan_feature = extract_features([smiles], featurizer_type='morgan_feat')

print(f"Standard Morgan: {morgan_standard.shape}")  # (1, 2048)
print(f"Feature Morgan:  {morgan_feature.shape}")   # (1, 2048)
```

## Mordred Descriptors

### Overview

Mordred is a comprehensive molecular descriptor calculator that generates 1613 physicochemical descriptors covering diverse molecular properties. This provides the richest molecular representation but at the cost of computation time.

**Algorithm:**
- 1613 molecular descriptors
- Physicochemical properties (MW, LogP, PSA, etc.)
- Topological indices
- Constitutional descriptors
- Geometric descriptors
- Electronic properties

**Dimensions:** 1613

### When to Use

- **Maximum molecular information** needed
- Small to medium datasets where computation time acceptable
- QSAR/QSPR modeling with interpretable features
- **Chemprop hybrid mode** (combining graph + descriptors)
- When fingerprints underperform
- Explainable AI applications (descriptors have physical meaning)

**Performance considerations:**
- Slowest computation (5-10x slower than Morgan)
- Higher dimensional feature space
- Best with feature selection or dimensionality reduction
- Excellent for hybrid Chemprop models

### Example Usage

**CLI:**
```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer descriptors
```

**API:**
```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer_type='descriptors'
)
```

**Chemprop Hybrid Mode:**
```bash
learnm8 run compounds.csv --target Activity --learner chemprop --featurizer descriptors
```

**API (Chemprop Hybrid):**
```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',
    target_col='Activity',
    featurizer_type='descriptors',
    cache_dir=Path('.cache')
)
```

**Direct Feature Extraction:**
```python
from learnm8 import extract_features

features = extract_features(
    smiles_list=['CCO', 'CCC'],
    featurizer_type='descriptors',
    cache_dir=Path('.cache'),
    show_progress=True
)
print(features.shape)  # (2, 1613)
```

## Detailed Comparison

### Computational Performance

**Timing benchmarks (1000 molecules, 16 cores):**

| Featurizer | First Run (no cache) | Cached Run | Speedup |
|------------|---------------------|------------|---------|
| `maccs` | 15s | 0.2s | 75x |
| `morgan` | 30s | 0.3s | 100x |
| `ecfp6` | 32s | 0.3s | 107x |
| `morgan_feat` | 31s | 0.3s | 103x |
| `descriptors` | 180s | 1.8s | 100x |

### Memory Usage

**Memory per compound (approximate):**

| Featurizer | Per Compound | 100k Compounds |
|------------|--------------|----------------|
| `maccs` | 167 bytes | 16 MB |
| `morgan` | 2048 bytes | 195 MB |
| `ecfp6` | 2048 bytes | 195 MB |
| `morgan_feat` | 2048 bytes | 195 MB |
| `descriptors` | 6452 bytes (float32) | 615 MB |

### Information Content

**Molecular aspects captured:**

| Featurizer | Structural | Topological | Physicochemical | Geometric |
|------------|-----------|-------------|-----------------|-----------|
| `maccs` | ✓✓ | ✓ | - | - |
| `morgan` | ✓✓✓ | ✓✓✓ | ✓ | - |
| `ecfp6` | ✓✓✓ | ✓✓✓ | ✓ | - |
| `morgan_feat` | ✓✓ | ✓✓ | ✓✓ | - |
| `descriptors` | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |

### Learner Compatibility

**All featurizers compatible with:**

- RandomForest, AdvancedRandomForest
- GaussianProcess
- XGBoost, DecisionTree, LinearRegression
- MLP, MCDropout, Fastprop
- All ensemble variants

**Special case:**
- Chemprop: Works without featurizers, but accepts `descriptors` for hybrid mode

## Practical Recommendations

### Starting Point

Begin with Morgan fingerprints for most tasks:

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan
```

### Large-Scale Screening

Use MACCS for maximum speed with large libraries:

```bash
learnm8 run large_library.csv --target Activity --learner rf --featurizer maccs \
  --cache-dir .shared_cache
```

### Complex Molecules

Use ECFP6 for large molecules with extended features:

```bash
learnm8 run peptides.csv --target Activity --learner gp --featurizer ecfp6
```

### Pharmacophore Tasks

Use Morgan feature fingerprints for functional property focus:

```bash
learnm8 run drug_candidates.csv --target Binding --learner ensemble --featurizer morgan_feat
```

### Maximum Performance

Use Mordred descriptors when feature richness critical:

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer descriptors \
  --cache-dir .cache
```

### Chemprop Hybrid

Combine graph neural network with descriptors:

```bash
learnm8 run compounds.csv --target Activity --learner chemprop --featurizer descriptors
```

## Feature Extraction Best Practices

### Cache Management

Always specify cache directory for reusable features:

```python
from learnm8 import extract_features
from pathlib import Path

features = extract_features(
    smiles_list=smiles_list,
    featurizer_type='morgan',
    cache_dir=Path('.shared_cache')  # Reuse across experiments
)
```

### Parallel Processing

Let automatic optimization handle parallelization:

```python
features = extract_features(
    smiles_list=large_smiles_list,
    featurizer_type='morgan',
    n_jobs=-1  # Auto-optimize based on dataset size
)
```

### Progress Tracking

Enable progress bars for long computations:

```python
features = extract_features(
    smiles_list=very_large_smiles_list,
    featurizer_type='descriptors',
    show_progress=True  # Requires tqdm
)
```

### Batch Processing

Process large datasets in batches to manage memory:

```python
batch_size = 10000
all_features = []

for i in range(0, len(smiles_list), batch_size):
    batch = smiles_list[i:i+batch_size]
    features = extract_features(
        smiles_list=batch,
        featurizer_type='morgan',
        cache_dir=Path('.cache')
    )
    all_features.append(features)
```

## Troubleshooting

### Invalid SMILES

Use validation before feature extraction:

```bash
learnm8 validate compounds.csv -o validation_results/
```

**API:**
```python
from learnm8 import validate_compound_pool
import polars as pl

compounds = pl.read_csv('compounds.csv')
result = validate_compound_pool(compounds, n_jobs=-1, progress=True)

print(f"Valid: {len(result.valid_compounds)}")
print(f"Invalid: {len(result.invalid_compounds)}")
```

### Memory Issues

Use MACCS for memory-constrained environments:

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer maccs
```

### Slow Computation

Enable caching and parallelization:

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --cache-dir .cache
```

### Cache Corruption

Clear corrupted cache files:

```bash
rm -rf .cache/morgan_features.h5
```

## Next Steps

- [Featurizers Overview](overview.md) - Caching, parallelization, and API details
- [Running Experiments](../../tutorials/running-experiments.md) - Complete experimental workflows
- [Learner Overview](../learners/overview.md) - Choosing compatible learners
- [Custom Featurizers](../../customization/extending-framework.md) - Implementing custom featurizers

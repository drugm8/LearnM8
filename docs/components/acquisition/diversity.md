# Diversity-Based Acquisition Strategies

Diversity-based acquisition strategies select compounds that are structurally or chemically diverse, avoiding redundant measurements in similar regions of chemical space. These methods are particularly valuable for broad exploration, scaffold hopping, and avoiding oversampling of similar compounds.

Unlike uncertainty-based strategies, diversity methods typically do not require model uncertainty estimates and focus on molecular structure and feature similarity.

## Why Diversity Matters

### Redundancy Avoidance

Standard acquisition strategies (greedy, UCB, EI) can oversample similar high-value compounds, leading to redundant measurements that provide diminishing returns. Diversity strategies explicitly avoid this by:

- Selecting compounds from different regions of chemical space
- Ensuring structural variety in selected batches
- Preventing model overfitting to narrow structural classes

### Exploration Benefits

Diversity-based selection provides systematic exploration advantages:

- **Scaffold hopping:** Discover active compounds in different structural classes
- **Mechanism diversity:** Sample compounds with varied interaction modes
- **Robustness:** Build models that generalize across chemical space
- **Discovery:** Find unexpected actives in unexplored regions

### When to Use Diversity

Diversity strategies are particularly effective:

- **Early exploration:** First few cycles after random initialization
- **Large libraries:** When compound pool contains many similar structures
- **Multi-objective screening:** When structural novelty is valued alongside activity
- **Model building:** When goal is robust models rather than pure optimization

## BitBIRCH Clustering

### Overview

BitBIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is a specialized clustering algorithm designed for binary molecular fingerprints with native Tanimoto similarity support. It provides exceptional scalability for molecular libraries with millions of compounds.

**Algorithm:**
1. Cluster molecular fingerprints using Tanimoto distance
2. Build hierarchical clustering tree with configurable branching
3. Select representatives evenly from clusters for diversity

BitBIRCH is optimized for molecular data and significantly faster than general-purpose clustering methods.

### Molecular Fingerprints

BitBIRCH works with binary molecular fingerprints:

- **Morgan fingerprints** (radius=2, 2048 bits) - Default, general-purpose
- **ECFP6** (radius=3, 2048 bits) - Larger radius for extended neighborhoods
- **MACCS keys** (167 bits) - Structural keys, faster clustering

The fingerprint type should match the featurizer used for model training.

### Tanimoto Similarity

BitBIRCH uses Tanimoto (Jaccard) similarity, the standard metric for molecular fingerprints:

**Formula:** `Tanimoto(A, B) = (A ∩ B) / (A ∪ B)`

Tanimoto values range from 0 (completely dissimilar) to 1 (identical). BitBIRCH clusters molecules based on Tanimoto distance (1 - similarity).

### Scalability

BitBIRCH is designed for large-scale molecular screening:

- **100K compounds:** Clusters in seconds
- **1M compounds:** Clusters in minutes
- **10M+ compounds:** Feasible with appropriate hardware

Memory usage scales linearly with library size, making it suitable for production screening campaigns.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | ndarray | Required | Pre-computed molecular fingerprints (n_compounds, n_features). |
| `compound_ids` | List[str] | Required | List of compound IDs corresponding to feature rows. |
| `featurizer` | str | 'morgan' | Type of molecular features ('morgan', 'ecfp6', 'maccs'). |
| `threshold` | float | 0.5 | Tanimoto similarity threshold for clustering (0.0-1.0). |
| `branching_factor` | int | 50 | Maximum number of subclusters in each node. |
| `random_state` | int | 42 | Random seed for reproducible selection within clusters. |

**Tuning Guidance:**

**threshold:**
- Lower values (0.3-0.4): More clusters, higher diversity, smaller batches per cluster
- Default (0.5): Balanced clustering for typical molecular libraries
- Higher values (0.6-0.7): Fewer clusters, less strict diversity, faster clustering

**branching_factor:**
- Lower values (20-30): Deeper trees, slower clustering, more precise clusters
- Default (50): Balanced performance for most use cases
- Higher values (80-100): Shallower trees, faster clustering, less precise clusters

### Installation

BitBIRCH is an optional dependency:

```bash
pip install git+https://github.com/mqcomplab/bitbirch.git
```

LearnM8 will raise an informative error if BitBIRCH is used without installation.

### Examples

**CLI with BitBIRCH diversity:**
```bash
learnm8 run compounds.csv --target Activity \
  --learner rf --featurizer morgan \
  --cycles "random:0.02 greedy:0.005*5 bitbirch:0.01*3"
```

**CLI with custom threshold (higher diversity):**
```bash
learnm8 run compounds.csv --target Activity \
  --learner rf --featurizer morgan \
  --cycles "random:0.02 bitbirch:0.01*5" \
  --acquisition-params '{"threshold": 0.4}'
```

**Python API with pre-computed features:**
```python
from learnm8 import run_active_learning, extract_features
from learnm8.acquisition import BitBIRCHAcquisition
import polars as pl

compounds = pl.read_csv('compounds.csv')
smiles_list = compounds['SMILES'].to_list()
compound_ids = compounds['ID'].to_list()

features = extract_features(
    smiles_list=smiles_list,
    featurizer='morgan',
    cache_dir='.cache',
    n_jobs=-1
)

bitbirch_acq = BitBIRCHAcquisition(
    features=features,
    compound_ids=compound_ids,
    featurizer='morgan',
    threshold=0.5,
    branching_factor=50
)

results = run_active_learning(
    compound_pool=compounds,
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.02),
        ('greedy', 0.005),
    ],
    custom_acquisition=bitbirch_acq
)
```

**Multi-stage cycle with diversity phase:**
```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='ensemble',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', batch_fraction=0.02),
        CycleConfig('ucb', n_cycles=3, batch_fraction=0.005),
        CycleConfig('bitbirch', n_cycles=2, batch_fraction=0.01,
                    acquisition_params={'threshold': 0.45}),
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.005)
    ]
)
```

## Simulated Annealing

### Overview

Simulated Annealing is an optimization-based acquisition strategy that uses temperature-based probabilistic selection to balance exploration and exploitation. Inspired by the physical annealing process, it starts with high temperature (random exploration) and gradually cools (increasingly greedy) following a cooling schedule.

**Algorithm:**
1. Initialize at high temperature with random compound
2. Iteratively propose candidate compounds
3. Accept candidates using Metropolis criterion: `P(accept) = exp(-ΔE/T)`
4. Gradually reduce temperature following cooling schedule
5. Return best compounds found during annealing

The energy function is based on model predictions (higher predictions = lower energy for maximization).

### Temperature-Based Selection

Temperature controls exploration vs exploitation:

- **High temperature (T ≈ 1.0):** Accept most candidates, random exploration
- **Medium temperature (T ≈ 0.1-0.5):** Balanced acceptance, guided exploration
- **Low temperature (T ≈ 0.01):** Accept only improvements, greedy exploitation

The cooling schedule determines how quickly temperature decreases.

### Cooling Schedules

LearnM8 supports two cooling schedules:

**Exponential Cooling (Default):**
```
T(t) = T_initial × (T_final / T_initial)^progress
```

- Rapid initial cooling, slow final cooling
- Good for most active learning scenarios
- Explores broadly early, exploits later

**Linear Cooling:**
```
T(t) = T_initial × (1 - progress) + T_final × progress
```

- Constant cooling rate
- More predictable behavior
- Useful when linear exploration-exploitation transition is desired

### When to Use

Simulated Annealing is effective when:

- **No uncertainty available:** Works with any learner (no uncertainty required)
- **Complex energy landscapes:** Avoids local optima through stochastic acceptance
- **Alternative to greedy:** Provides exploration without requiring uncertainty
- **Benchmark comparisons:** Standard optimization baseline

Avoid Simulated Annealing when:
- Uncertainty-based methods are available (UCB, EI typically better)
- Computational budget is limited (many iterations required)
- Deterministic selection is required

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_temp` | float | 1.0 | Starting temperature for annealing process. |
| `final_temp` | float | 0.01 | Final temperature (must be < initial_temp). |
| `max_iterations` | int | 1000 | Maximum number of annealing iterations. |
| `cooling_schedule` | str | 'exponential' | Cooling schedule ('exponential' or 'linear'). |
| `score_direction` | str | 'higher' | Optimization direction ('higher' or 'lower'). |
| `random_state` | int | 42 | Random seed for reproducible annealing. |

**Tuning Guidance:**

**initial_temp:**
- Higher values (2.0-5.0): More initial exploration, slower convergence
- Default (1.0): Balanced for typical molecular screening
- Lower values (0.5): Less exploration, faster convergence to local optima

**final_temp:**
- Keep low (0.01-0.05) to ensure final greedy behavior
- Very low (0.001): Pure greedy at end
- Higher (0.1): Maintains some stochasticity throughout

**max_iterations:**
- Fewer iterations (500): Faster, less thorough exploration
- Default (1000): Balanced for batch selection
- More iterations (2000-5000): Thorough exploration, slower

**cooling_schedule:**
- Exponential (default): Recommended for most cases
- Linear: When uniform exploration-exploitation transition is desired

### Examples

**CLI with default parameters:**
```bash
learnm8 run compounds.csv --target Activity \
  --learner rf --featurizer morgan \
  --cycles "random:0.02 simulated_annealing:0.005*10"
```

**CLI with custom cooling schedule:**
```bash
learnm8 run compounds.csv --target Activity \
  --learner xgb --featurizer morgan \
  --cycles "random:0.02 simulated_annealing:0.01*5" \
  --acquisition-params '{"cooling_schedule": "linear", "max_iterations": 2000}'
```

**Python API with aggressive exploration:**
```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.02),
        ('simulated_annealing', 0.005)
    ],
    acquisition_params={
        'initial_temp': 2.0,
        'final_temp': 0.01,
        'max_iterations': 1500,
        'cooling_schedule': 'exponential'
    }
)
```

**Multi-stage with annealing exploration phase:**
```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='xgb',
    target_col='Activity',
    featurizer='ecfp6',
    cycles=[
        CycleConfig('random', batch_fraction=0.02),
        CycleConfig('simulated_annealing', n_cycles=3, batch_fraction=0.01,
                    acquisition_params={
                        'initial_temp': 1.5,
                        'final_temp': 0.01,
                        'cooling_schedule': 'exponential'
                    }),
        CycleConfig('greedy', n_cycles=7, batch_fraction=0.005)
    ]
)
```

## Comparison: BitBIRCH vs Simulated Annealing

| Aspect | BitBIRCH | Simulated Annealing |
|--------|----------|---------------------|
| **Basis** | Molecular similarity (Tanimoto) | Prediction-based energy |
| **Diversity Type** | Structural clustering | Stochastic optimization |
| **Requires Uncertainty** | No | No |
| **Scalability** | Excellent (1M+ compounds) | Good (limited by iterations) |
| **Determinism** | Deterministic clustering + random selection | Stochastic (Metropolis criterion) |
| **Parameter Tuning** | threshold, branching_factor | initial_temp, cooling_schedule |
| **Best Use Case** | Large libraries needing scaffold diversity | No uncertainty available, complex landscapes |
| **Computational Cost** | O(n) fingerprint clustering | O(iterations × n) evaluation |

## Combining Diversity with Other Strategies

Diversity strategies are most effective in multi-stage workflows:

**Exploration → Exploitation:**
```python
cycles=[
    ('random', 0.02),
    ('bitbirch', 0.01),
    ('ucb', 0.005),
    ('greedy', 0.005)
]
```

**Alternating diversity and exploitation:**
```python
cycles=[
    ('random', 0.02),
    ('greedy', 0.005),
    ('bitbirch', 0.01),
    ('greedy', 0.005),
    ('bitbirch', 0.01)
]
```

**Optimization with periodic diversity:**
```python
cycles=[
    ('random', 0.02),
    ('ei', 0.005),
    ('ei', 0.005),
    ('simulated_annealing', 0.01),
    ('greedy', 0.005)
]
```

## Performance Considerations

### BitBIRCH Performance

- **Clustering time:** O(n) with low constant factor
- **Memory:** O(n × fingerprint_size), manageable for binary fingerprints
- **Selection time:** O(n_clusters) for representative selection
- **Optimization:** Native Tanimoto, highly optimized for binary data

**Tips:**
- Pre-compute features once, reuse across cycles
- Use HDF5 caching (`--cache-dir`) for feature persistence
- Increase `branching_factor` for faster clustering at slight precision cost
- Use morgan fingerprints (default) for best balance of speed and diversity

### Simulated Annealing Performance

- **Time complexity:** O(max_iterations × n) for scoring
- **Memory:** Minimal (only current/candidate states)
- **Parallelization:** Not currently parallelized (sequential by design)

**Tips:**
- Reduce `max_iterations` for faster selection (1000 is usually sufficient)
- Exponential cooling converges faster than linear
- Batch evaluation not yet optimized (opportunities for speedup)

## Choosing Between Diversity Methods

**Use BitBIRCH when:**
- Large compound libraries (>100K compounds)
- Structural diversity is priority
- Tanimoto similarity is appropriate metric
- Scalability and speed are important

**Use Simulated Annealing when:**
- Moderate-sized libraries (<100K compounds)
- No uncertainty estimates available
- Want prediction-guided exploration
- Prefer optimization-based approach

**Use both in sequence:**
```python
cycles=[
    ('random', 0.02),
    ('bitbirch', 0.01),
    ('simulated_annealing', 0.005),
    ('greedy', 0.005)
]
```

## Integration with Uncertainty-Based Strategies

Diversity methods complement uncertainty-based strategies:

1. **Early diversity + late uncertainty:**
   - Start with BitBIRCH to establish diverse foundation
   - Switch to UCB/EI for optimization

2. **Alternating phases:**
   - Alternate between diversity (exploration) and UCB (balanced)
   - Prevents over-exploitation while maintaining progress

3. **Hybrid scoring:**
   - Combine diversity with uncertainty (not yet implemented)
   - Future enhancement: diversity-weighted acquisition scores

Example multi-stage workflow:
```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='oracle.csv',
    learner='ensemble',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', batch_fraction=0.01),
        CycleConfig('bitbirch', n_cycles=2, batch_fraction=0.01,
                    acquisition_params={'threshold': 0.45}),
        CycleConfig('ucb', n_cycles=3, batch_fraction=0.005,
                    acquisition_params={'beta': 2.5}),
        CycleConfig('bitbirch', n_cycles=1, batch_fraction=0.01),
        CycleConfig('greedy', n_cycles=4, batch_fraction=0.005)
    ]
)
```

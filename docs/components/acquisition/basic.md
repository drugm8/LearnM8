# Basic Acquisition Strategies

Basic acquisition strategies provide fundamental compound selection methods that require only model predictions (no uncertainty estimates). These strategies serve as strong baselines and are essential components of multi-stage active learning workflows.

## Greedy Acquisition

Greedy acquisition selects compounds with the highest (or lowest) predicted values, representing pure exploitation without exploration.

### Overview

Greedy selection sorts compounds by model prediction and selects the top-K based on score direction. This strategy maximizes immediate expected value but may miss superior compounds in unexplored regions.

**Algorithm**:

1. Sort unlabeled compounds by prediction
2. Select top n_select compounds (highest for 'higher', lowest for 'lower')
3. Return selected compounds

**Characteristics**:

- **Exploitation-focused**: Prioritizes immediate predicted value
- **Fast**: Simple sorting operation, O(n log n)
- **Deterministic**: Same predictions always yield same selection
- **No exploration**: Never samples uncertain or diverse compounds

### When to Use

**Recommended scenarios**:

- Final active learning cycles when shifting from exploration to exploitation
- Benchmarking uncertainty-based methods (baseline comparison)
- Known smooth response surfaces with limited local optima
- Computational constraints preventing uncertainty estimation

**Not recommended**:

- Early cycles (insufficient exploration leads to poor coverage)
- Highly multimodal landscapes (risk of local optima)
- When discovery of diverse hits is prioritized over single best compound

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_direction` | str | `'higher'` | Optimization direction: `'higher'` (maximize) or `'lower'` (minimize) |

**Performance notes**:

- No additional computational cost beyond model prediction
- Scales efficiently to millions of compounds

### Examples

**CLI - Maximize binding affinity**:
```bash
learnm8 run compounds.csv oracle.py:score --target binding_affinity \
  --learner gp --featurizer morgan \
  --strategy greedy --score-direction higher \
  --n-cycles 10
```

**CLI - Minimize toxicity**:
```bash
learnm8 run compounds.csv oracle.py:score --target toxicity \
  --learner rf --featurizer morgan \
  --strategy greedy --score-direction lower \
  --n-cycles 10
```

**API - Pure exploitation after exploration**:
```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=2, batch_fraction=0.01),
        CycleConfig('ucb', n_cycles=5, batch_fraction=0.005,
                    acquisition_params={'beta': 2.0}),
        CycleConfig('greedy', n_cycles=3, batch_fraction=0.005,
                    acquisition_params={'score_direction': 'higher'})
    ]
)
```

**API - Custom greedy instance**:
```python
from learnm8.acquisition import GreedyAcquisition

greedy = GreedyAcquisition(score_direction='higher')

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    strategy=greedy,
    n_cycles=10
)
```

## Random Acquisition

Random acquisition selects compounds uniformly at random from the unlabeled pool, providing an unbiased baseline for evaluating more sophisticated strategies.

### Overview

Random selection samples compounds without considering predictions or structure, ensuring unbiased exploration of chemical space. This strategy establishes baseline performance that any intelligent acquisition function should exceed.

**Algorithm**:

1. Randomly sample n_select compounds from unlabeled pool
2. Return selected compounds

**Characteristics**:

- **Unbiased exploration**: No preference for predictions or structure
- **Fast**: O(n_select) sampling operation
- **Stochastic**: Different selections each run (unless random_state fixed)
- **Baseline**: Minimum expected performance for intelligent strategies

### When to Use

**Recommended scenarios**:

- Cycle 0 initialization (random warm-start is standard practice)
- Baseline comparison for evaluating acquisition strategies
- Completely unknown chemical spaces with no prior knowledge
- Negative control in method benchmarking studies

**Not recommended**:

- Production screening (inefficient use of experimental budget)
- Later cycles (intelligent strategies should outperform random)
- When any prior knowledge or structure-activity relationships exist

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | int | `42` | Random seed for reproducible selection |

**Performance notes**:

- Minimal computational overhead
- Scales to arbitrarily large compound pools

### Examples

**CLI - Random initialization**:
```bash
learnm8 run compounds.csv oracle.py:score --target Activity \
  --learner gp --featurizer morgan \
  --cycles "random:0.02 ucb:0.005*9" \
  --random-state 42
```

**CLI - Pure random baseline**:
```bash
learnm8 run compounds.csv oracle.py:score --target Activity \
  --learner rf --featurizer morgan \
  --strategy random --n-cycles 10 \
  --random-state 42
```

**API - Random warm-start then intelligent selection**:
```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='ensemble',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('thompson', n_cycles=9, batch_fraction=0.005)
    ],
    random_state=42
)
```

**API - Custom random instance with different seed**:
```python
from learnm8.acquisition import RandomAcquisition

random_acq = RandomAcquisition(random_state=123)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    strategy=random_acq,
    n_cycles=10
)
```

## Top-K Acquisition

Top-K acquisition provides flexible greedy selection by randomly sampling from the top-K highest (or lowest) predicted compounds, introducing controlled stochasticity while maintaining exploitation focus.

### Overview

Top-K selection first identifies the top K% of compounds by prediction, then randomly selects the acquisition batch from this subset. This balances exploitation (considering only high predictions) with diversity (random sampling within top-K).

**Algorithm**:

1. Calculate K = max(n_select, k_fraction × pool_size)
2. Select top-K compounds by prediction (based on score_direction)
3. Randomly sample n_select compounds from top-K subset
4. Return selected compounds

**Characteristics**:

- **Soft exploitation**: Focuses on high predictions but allows diversity
- **Configurable balance**: k_fraction controls exploration/exploitation
- **Stochastic**: Introduces randomness within top-K candidates
- **Flexible**: Can range from pure greedy (k_fraction→0) to random (k_fraction=1.0)

### When to Use

**Recommended scenarios**:

- Middle active learning cycles balancing exploitation and diversity
- When greedy selection is too deterministic (want exploration within good compounds)
- Avoiding repeated selection of identical high-prediction clusters
- Uncertain prediction quality (hedge bets within top predictions)

**Not recommended**:

- When pure exploitation is desired (use Greedy instead)
- When uncertainty-guided exploration is available (use UCB, EI instead)
- Very small compound pools (limited benefit over greedy)

### Parameters

| Parameter | Type | Default | Description | Performance Impact |
|-----------|------|---------|-------------|-------------------|
| `k_fraction` | float | `0.1` | Fraction of pool to consider (0 < k_fraction ≤ 1.0) | Higher values increase diversity, lower increase exploitation |
| `score_direction` | str | `'higher'` | Optimization direction: `'higher'` or `'lower'` | N/A |

**Parameter tuning**:

- `k_fraction=0.01`: Near-greedy (1% top compounds)
- `k_fraction=0.1`: Balanced (10% top compounds, default)
- `k_fraction=0.5`: High diversity (50% top compounds)
- `k_fraction=1.0`: Equivalent to random selection

**Performance notes**:

- Sorting: O(n log n)
- Random sampling: O(k) where k = k_fraction × n

### Examples

**CLI - Balanced top-10% selection**:
```bash
learnm8 run compounds.csv oracle.py:score --target Activity \
  --learner rf --featurizer morgan \
  --strategy topk --acquisition-params '{"k_fraction": 0.1}' \
  --n-cycles 10
```

**CLI - Multi-stage with Top-K middle phase**:
```bash
learnm8 run compounds.csv oracle.py:score --target binding \
  --learner gp --featurizer morgan \
  --cycles "random:0.02 topk:0.005*5 greedy:0.005*4" \
  --acquisition-params '{"k_fraction": 0.15}' \
  --score-direction higher
```

**API - Top-K for diversity within good predictions**:
```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='xgb',
    target_col='Activity',
    featurizer='morgan',
    strategy='topk',
    acquisition_params={
        'k_fraction': 0.1,
        'score_direction': 'higher'
    },
    n_cycles=10
)
```

**API - Custom Top-K instance with tight focus**:
```python
from learnm8.acquisition import TopKAcquisition
from learnm8 import run_active_learning, CycleConfig

# Tight top-5% focus for exploitation with some diversity
topk_tight = TopKAcquisition(k_fraction=0.05, score_direction='higher')

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='ensemble',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=2, batch_fraction=0.01),
        CycleConfig('ucb', n_cycles=4, batch_fraction=0.005),
        topk_tight,  # Custom instance for final cycles
    ]
)
```

**API - Minimize toxicity with Top-K**:
```python
from learnm8.acquisition import TopKAcquisition

topk_minimize = TopKAcquisition(
    k_fraction=0.2,
    score_direction='lower'  # Select from lowest predictions
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='toxicity',
    featurizer='descriptors',
    strategy=topk_minimize,
    n_cycles=10
)
```

## Strategy Comparison

| Strategy | Deterministic | Exploration | Exploitation | Computational Cost | Typical Use Case |
|----------|--------------|-------------|--------------|-------------------|------------------|
| Greedy | Yes | None | Maximum | Minimal | Final cycles, baseline |
| Random | No | Maximum | None | Minimal | Initialization, baseline |
| Top-K | No | Moderate | High | Minimal | Middle cycles, diversity in top predictions |

## See Also

- [Uncertainty-Based Strategies](uncertainty-based.md) - UCB, EI, PI, Thompson, Entropy
- [Diversity Methods](diversity.md) - Simulated Annealing
- [Acquisition Overview](overview.md) - Strategy selection guide
- [Building Custom Cycles](../../tutorials/building-custom-cycles.md) - Combining multiple strategies

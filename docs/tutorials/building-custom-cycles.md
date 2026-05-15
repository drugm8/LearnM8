# Building Custom Cycles

This tutorial explains how to design custom active learning cycles for different screening strategies.

## Understanding Cycle Specifications

Active learning cycles define **what strategy** to use and **how many compounds** to select.

### Tuple Format (Python API)

Cycles are specified as tuples:

```python
('strategy_name', batch_fraction)
```

**Example:**
```python
cycles = [
    ('random', 0.01),   # Cycle 0: Random 1% of pool
    ('greedy', 0.005),  # Cycle 1: Greedy 0.5% of pool
    ('greedy', 0.005)   # Cycle 2: Greedy 0.5% of pool
]
```

### Why Batch Fraction is Relative to Pool Size

Batch fractions are **always relative to the original pool size**, not the remaining unlabeled compounds.

**Example with 10,000 compound pool:**

- `batch_fraction=0.01` → 100 compounds per cycle
- `batch_fraction=0.005` → 50 compounds per cycle
- Consistent batch sizes across all cycles

**Rationale:**

- Predictable total labeling budget
- Easy to calculate total compounds: `sum(fractions) * pool_size`
- Fair comparison across different strategies

## Simple Cycle Patterns

### Initial Random Sampling

Always start with random sampling to get unbiased initial training data:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    initial_strategy='random',
    strategy='greedy',
    n_cycles=10
)
```

**CLI alternative:**

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --initial-strategy random --strategy greedy --n-cycles 10
```

### Pure Exploitation (Greedy)

Select compounds with best predicted scores:

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    strategy='greedy',
    n_cycles=10,
    batch_fraction=0.01
)
```

**When to use:**

- Find top hits quickly
- After sufficient exploration
- Production screening focused on best compounds

### Pure Exploration (Random)

Continue random sampling throughout:

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    strategy='random',
    n_cycles=10,
    batch_fraction=0.01
)
```

**When to use:**

- Baseline comparison
- Building representative training sets
- Avoiding model bias

## Multi-Strategy Cycles

Combine different strategies for sophisticated screening:

### Random → Greedy → Simulated Annealing

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.02),      # Phase 1: Initial exploration
        ('greedy', 0.01),      # Phase 2: Exploit best regions
        ('greedy', 0.01),
        ('greedy', 0.01),
        ('greedy', 0.01),
        ('greedy', 0.01),
        ('simulated_annealing', 0.01),  # Phase 3: Diversify coverage
        ('simulated_annealing', 0.01),
        ('greedy', 0.01),              # Phase 4: Final exploitation
        ('greedy', 0.01)
    ]
)
```

**Rationale:**

1. **Random (2%)**: Get unbiased initial training data
2. **Greedy (5 cycles × 1%)**: Rapidly find top hits
3. **Diverse (2 cycles × 1%)**: Explore uncovered regions
4. **Greedy (2 cycles × 1%)**: Refine top hits

### Exploration → Exploitation with UCB

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.01),                # Initial random
        ('ucb', 0.01),                   # High exploration
        ('ucb', 0.01),
        ('ucb', 0.01),
        ('greedy', 0.005),               # Pure exploitation
        ('greedy', 0.005),
        ('greedy', 0.005)
    ],
    acquisition_params={'beta': 2.0}
)
```

**Rationale:**

- UCB balances exploration (uncertainty) and exploitation (prediction)
- Transition to pure greedy after sufficient exploration
- `beta=2.0` emphasizes exploration early

## CycleConfig Dataclass (Advanced)

For **full parameter control** per cycle, use `CycleConfig`:

```python
from learnm8 import run_active_learning
from learnm8.core.config import CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),

        CycleConfig(
            'greedy',
            n_cycles=5,
            batch_fraction=0.01,
            pruning_strategy='score',
            pruning_params={'pruning_fraction': 0.3}
        ),

        CycleConfig(
            'ucb',
            n_cycles=4,
            batch_fraction=0.005,
            acquisition_params={'beta': 1.5}
        )
    ]
)
```

**CycleConfig Parameters:**

- `strategy`: Acquisition function name
- `n_cycles`: Number of cycles with this configuration
- `batch_fraction`: Fraction of pool to select per cycle
- `pruning_strategy`: Optional pruning method
- `pruning_params`: Optional pruning parameters
- `acquisition_params`: Optional acquisition function parameters

**When to use CycleConfig:**

- Cycle-specific pruning
- Different acquisition parameters per phase
- Complex multi-phase strategies
- Reproducible experiment definitions

### Multi-Phase with Different Parameters

```python
cycles = [
    CycleConfig('random', n_cycles=1, batch_fraction=0.01),

    CycleConfig(
        'ucb',
        n_cycles=3,
        batch_fraction=0.01,
        acquisition_params={'beta': 3.0}
    ),

    CycleConfig(
        'ucb',
        n_cycles=3,
        batch_fraction=0.01,
        acquisition_params={'beta': 1.0}
    ),

    CycleConfig(
        'greedy',
        n_cycles=3,
        batch_fraction=0.005,
        pruning_strategy='score',
        pruning_params={'pruning_fraction': 0.2}
    )
]

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=cycles
)
```

**Rationale:**

- UCB with high exploration weight → UCB with lower weight → Greedy
- Gradual transition from exploration to exploitation
- Pruning only in final greedy phase

## CLI Cycle Specifications

Use string format with `*n` syntax for repeating strategies:

### Basic String Format

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --cycles "random:0.01 greedy:0.01*5"
```

**Format:**

- `strategy:fraction` - Single cycle
- `strategy:fraction*n` - Repeat n times
- Space-separated for multiple phases

**Equivalent to:**
```python
cycles = [
    ('random', 0.01),
    ('greedy', 0.01),
    ('greedy', 0.01),
    ('greedy', 0.01),
    ('greedy', 0.01),
    ('greedy', 0.01)
]
```

### Complex Multi-Phase CLI

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --cycles "random:0.02 ucb:0.01*3 greedy:0.005*4 simulated_annealing:0.01*2"
```

**Expanded cycles:**

1. random: 2%
2. ucb: 1% (repeated 3 times)
3. greedy: 0.5% (repeated 4 times)
4. simulated_annealing: 1% (repeated 2 times)

**Total: 10 cycles**

### CLI with Acquisition Parameters

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --cycles "random:0.01 ucb:0.01*5" \
  --acquisition-params '{"beta": 2.0}'  # UCB beta parameter
```

**Note:** Acquisition parameters apply to all cycles that use them.

> **Note:** Predefined schedules (`--schedule quick/standard/intensive`) are not available. Use the [cycles specification](#cli-cycle-specifications) section above to define custom cycle schedules instead.

## Design Principles

**Early Exploration:**

- Start with random or simulated_annealing sampling
- Build unbiased initial training set
- Typical: 1-2% of pool

**Mid-Phase Exploitation:**

- Use greedy or low-weight UCB
- Focus on finding best compounds
- Typical: 5-10 cycles × 0.5-1% of pool

**Late Diversification:**

- Optional diversity methods (simulated_annealing)
- Cover unexplored regions
- Typical: 1-2 cycles × 1% of pool

**Final Refinement:**

- Return to greedy
- Verify top hits
- Typical: 1-2 cycles × 0.5% of pool

## Calculating Total Labeling Budget

**Formula:**
```
total_compounds = pool_size * sum(batch_fractions)
```

**Example with 10,000 compound pool:**

```python
cycles = [
    ('random', 0.01),      # 100 compounds
    ('greedy', 0.01),      # 100 compounds
    ('greedy', 0.01),      # 100 compounds
    ('greedy', 0.01),      # 100 compounds
    ('greedy', 0.01),      # 100 compounds
    ('simulated_annealing', 0.01)  # 100 compounds
]

total_fraction = 0.01 + 0.01 + 0.01 + 0.01 + 0.01 + 0.01
total_compounds = 10000 * 0.06
```

**Result: 600 compounds labeled (6% of pool)**

# Diversity-Based Acquisition

Diversity-based acquisition strategies select compounds that are structurally or chemically varied, avoiding redundant measurements in similar regions of chemical space. These methods are particularly valuable for broad exploration, scaffold hopping, and avoiding oversampling of similar compounds.

Unlike uncertainty-based strategies, diversity methods do not require model uncertainty estimates and work with any learner.

## Simulated Annealing

### Overview

Simulated Annealing is an optimization-based acquisition strategy that uses temperature-based probabilistic selection to balance exploration and exploitation. Inspired by the physical annealing process, it starts with high temperature (random exploration) and gradually cools (increasingly greedy) following a cooling schedule.

**Algorithm:**

1. Initialize at high temperature with a random starting compound
2. Iteratively propose candidate compounds
3. Accept candidates using Metropolis criterion: `P(accept) = exp(-ΔE/T)`
4. Gradually reduce temperature following the cooling schedule
5. Return the best compounds found during annealing

The energy function is based on model predictions — higher predictions = lower energy for maximization problems.

### Temperature-Based Selection

Temperature controls the exploration–exploitation balance:

| Temperature | Behavior |
|-------------|----------|
| High (T ≈ 1.0) | Accept most candidates — random exploration |
| Medium (T ≈ 0.1–0.5) | Guided exploration |
| Low (T ≈ 0.01) | Accept only improvements — greedy exploitation |

### Cooling Schedules

**Exponential (default):**
```
T(t) = T_initial × (T_final / T_initial)^progress
```
Rapid initial cooling, slow final cooling. Recommended for most scenarios.

**Linear:**
```
T(t) = T_initial × (1 - progress) + T_final × progress
```
Constant cooling rate. Use when a uniform exploration–exploitation transition is preferred.

### When to Use

Simulated Annealing works well when:

- No uncertainty estimate is available (works with any learner)
- You want to avoid local optima through stochastic acceptance
- You need a diversity alternative to pure greedy without requiring an ensemble

Prefer UCB, EI, or entropy-based strategies when uncertainty estimates are available from the learner — they are more principled for exploitation–exploration balance.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_temp` | `float` | `1.0` | Starting temperature |
| `final_temp` | `float` | `0.01` | Final temperature (must be < `initial_temp`) |
| `max_iterations` | `int` | `1000` | Maximum annealing iterations |
| `cooling_schedule` | `str` | `'exponential'` | `'exponential'` or `'linear'` |
| `random_state` | `int` | `42` | Random seed |

**Tuning notes:**

- `initial_temp` 2.0–5.0 → more initial exploration; 0.5 → faster convergence
- `final_temp` 0.001–0.05 → effectively greedy at end; 0.1 → retains some randomness
- `max_iterations` 500 → fast; 2000–5000 → thorough, slower

### Examples

**Minimal example:**
```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    strategy='simulated_annealing'
)
```

**Custom cooling schedule:**
```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    learner='xgb',
    target_col='Activity',
    featurizer='morgan',
    strategy='simulated_annealing',
    acquisition_params={
        'initial_temp': 2.0,
        'final_temp': 0.01,
        'max_iterations': 1500,
        'cooling_schedule': 'exponential'
    }
)
```

**Multi-stage: annealing for exploration, then greedy exploitation:**
```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
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

## Combining Diversity with Other Strategies

Diversity strategies are most effective in multi-stage workflows where initial exploration gives way to targeted exploitation:

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    learner='rf_ensemble',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),        # unbiased init
        CycleConfig('simulated_annealing', n_cycles=3, batch_fraction=0.01),  # explore
        CycleConfig('ucb', n_cycles=3, batch_fraction=0.01,
                    acquisition_params={'beta': 2.0}),                  # balance
        CycleConfig('greedy', n_cycles=3, batch_fraction=0.005)         # exploit
    ]
)
```

## See Also

- [Acquisition Overview](overview.md)
- [Uncertainty-Based Strategies](uncertainty-based.md)
- [Basic Strategies](basic.md)

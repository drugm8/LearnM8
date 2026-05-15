# Acquisition Functions Overview

Acquisition functions (also called selection strategies) determine which compounds to select for measurement in each active learning cycle. They form the strategic decision layer that balances exploration of uncertain regions against exploitation of promising predictions.

## What are Acquisition Functions?

An acquisition function receives model predictions and uncertainties for unlabeled compounds, then strategically selects a subset for experimental measurement. This selection critically determines how efficiently the active learning process discovers high-value compounds.

**Core responsibility**: Given predicted values and optional uncertainties, select the most informative compounds to measure next.

## Exploration vs Exploitation Tradeoff

Effective acquisition strategies balance two competing objectives:

**Exploitation**: Select compounds with highest predicted values

- Immediately maximize expected value
- Risk missing better compounds in unexplored regions
- Example: Greedy selection (pure exploitation)

**Exploration**: Select compounds with high uncertainty or diversity

- Gather information about poorly characterized regions
- May sacrifice short-term gains for long-term discovery
- Example: Entropy-based selection (pure exploration)

**Balanced Strategies**: Combine both objectives with tunable parameters

- Upper Confidence Bound (UCB): prediction + β × uncertainty
- Expected Improvement (EI): probabilistic improvement calculation
- Thompson Sampling: stochastic balance

## AcquisitionFunction Protocol

All acquisition strategies implement the `AcquisitionFunction` protocol defined in `learnm8.core.interfaces`:

```python
from abc import ABC, abstractmethod
import polars as pl

class AcquisitionFunction(ABC):
    @abstractmethod
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        """Select compounds for labeling.

        Args:
            compounds: DataFrame with 'ID', 'SMILES', 'prediction' columns
                      May also contain 'uncertainty' column if available
            n_select: Number of compounds to select

        Returns:
            DataFrame subset with selected compounds
        """
        pass

    def requires_uncertainty(self) -> bool:
        """Return True if this acquisition function requires uncertainty."""
        return False

    def get_name(self) -> str:
        """Return descriptive name for this acquisition function."""
        return self.__class__.__name__
```

**Key methods**:

- `select()`: Core selection logic (required)
- `requires_uncertainty()`: Whether uncertainty column is needed (optional, defaults to False)
- `get_name()`: Identifier for logging and reporting (optional)

## Choosing an Acquisition Strategy

Use this decision tree to select an appropriate strategy:

```
Does your learner provide uncertainty estimates?
├─ Yes → Use uncertainty-based strategies
│   ├─ Want balanced exploration/exploitation? → UCB (beta=2.0)
│   ├─ Want probabilistic guarantees? → Expected Improvement (EI)
│   ├─ Want stochastic exploration? → Thompson Sampling
│   └─ Want maximum information gain? → Entropy
│
└─ No → Use basic or diversity strategies
    ├─ Want pure exploitation? → Greedy
    ├─ Want baseline comparison? → Random
    ├─ Want flexible top selection? → Top-K
    └─ Want optimization-based? → Simulated Annealing
```

**Additional considerations**:

- **Early cycles**: Random initialization recommended for unbiased initial sampling
- **Later cycles**: Switch to exploitation (Greedy) or balanced strategies (UCB)
- **No uncertainty available**: Use greedy, random, topk, or simulated_annealing

## Strategy Registry

LearnM8 provides 9 acquisition strategies:

| Strategy | Category | Requires Uncertainty | When to Use | Key Parameters |
|----------|----------|---------------------|-------------|----------------|
| `greedy` | Basic | No | Pure exploitation, final cycles | `score_direction` |
| `random` | Basic | No | Baseline, initial exploration | `random_state` |
| `topk` | Basic | No | Flexible greedy selection | `k_fraction`, `score_direction` |
| `ucb` | Uncertainty | Yes | Balanced exploration/exploitation | `beta` |
| `ei` | Uncertainty | Yes | Expected improvement over best | `xi` |
| `pi` | Uncertainty | Yes | Probability of improvement | `xi` |
| `thompson` | Uncertainty | Yes | Stochastic sampling | `random_state` |
| `entropy` | Uncertainty | Yes | Maximum information gain | `entropy_type` |
| `simulated_annealing` | Optimization | No | Temperature-based optimization | `initial_temp`, `cooling_schedule` |

## Score Direction

The `score_direction` parameter controls optimization direction:

**'higher' (default)**: Select compounds with highest scores

- Use when higher target values are better (e.g., binding affinity, activity)
- Greedy selects maximum predictions
- UCB uses upper bound: prediction + β × uncertainty

**'lower'**: Select compounds with lowest scores

- Use when lower target values are better (e.g., toxicity, synthesis cost)
- Greedy selects minimum predictions
- UCB uses lower bound: prediction - β × uncertainty

```python
from learnm8 import run_active_learning

# Maximize binding affinity (higher is better)
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='binding_affinity',
    featurizer='morgan',
    strategy='greedy',
    score_direction='higher',  # Default
    n_cycles=10
)

# Minimize toxicity (lower is better)
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='toxicity',
    featurizer='morgan',
    strategy='greedy',
    score_direction='lower',
    n_cycles=10
)
```

## Using Acquisition Functions

### Python API

Use strategy name (string) or custom instance:

```python
from learnm8 import run_active_learning

# String name (recommended)
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    strategy='ucb',
    acquisition_params={'beta': 3.0},
    n_cycles=10
)

# Custom instance (advanced)
from learnm8.acquisition import UCBAcquisition

custom_ucb = UCBAcquisition(beta=3.0, score_direction='higher')
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    strategy=custom_ucb,
    n_cycles=10
)
```

### CLI Alternative

Specify strategy with `--strategy` or in `--cycles` specification:

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --strategy ucb --n-cycles 10
```

## Multi-Strategy Cycles

Combine different strategies across cycles for sophisticated exploration:

```python
# Python API: Use CycleConfig for fine-grained control
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='ensemble',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.01),
        CycleConfig('ucb', n_cycles=5, batch_fraction=0.005,
                    acquisition_params={'beta': 2.0}),
        CycleConfig('greedy', n_cycles=4, batch_fraction=0.005)
    ]
)
```

**CLI alternative:**
```bash
# Random initialization → UCB exploration → Greedy exploitation
learnm8 run compounds.csv oracle.py:score --target binding \
  --cycles "random:0.01 ucb:0.005*5 greedy:0.005*4" \
  --learner ensemble --featurizer morgan
```

## See Also

- [Basic Strategies](basic.md) - Greedy, Random, Top-K
- [Uncertainty-Based Strategies](uncertainty-based.md) - UCB, EI, PI, Thompson, Entropy
- [Simulated Annealing](diversity.md) - Simulated Annealing
- [Custom Acquisition Functions](../../customization/custom-acquisition.md) - Implementing your own strategies

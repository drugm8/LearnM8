# Uncertainty-Based Acquisition Strategies

Uncertainty-based acquisition strategies leverage model confidence estimates to balance exploration (sampling uncertain regions) with exploitation (sampling predicted high-value compounds). These strategies are particularly powerful when paired with learners that provide well-calibrated uncertainty quantification.

All strategies in this section require learners that support uncertainty estimation (Gaussian Process, ensembles, MC Dropout, or Advanced Random Forest).

## Upper Confidence Bound (UCB)

### Overview

Upper Confidence Bound balances exploitation and exploration by selecting compounds with high upper confidence bounds. The acquisition score is computed as:

**Formula:** `score = prediction + β × uncertainty`

For maximization problems, UCB selects compounds with high predicted values plus a bonus for high uncertainty. The parameter β controls the exploration-exploitation tradeoff.

### Exploration/Exploitation Balance

- **β = 0:** Pure exploitation (greedy selection on predictions only)
- **β = 1-2:** Balanced exploration and exploitation (recommended starting point)
- **β > 3:** Aggressive exploration (prioritizes uncertain regions)

Higher β values encourage sampling in uncertain regions of chemical space, which can be valuable early in campaigns when building model confidence.

### When to Use

- **Early-stage screening:** When you want to explore diverse chemical space
- **Model uncertainty:** When predictions are unreliable and exploration is needed
- **Balanced campaigns:** Default choice for general-purpose active learning
- **Large compound pools:** Scales well to millions of compounds

Avoid UCB when:

- Pure exploitation is desired (use greedy instead)
- Uncertainty estimates are poorly calibrated
- Computational budget is extremely limited (fewer cycles mean less time for exploration)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | 2.0 | Exploration parameter. Higher values favor exploration over exploitation. |
| `score_direction` | str | 'higher' | Optimization direction ('higher' or 'lower'). |

**Tuning Guidance:**

- Start with β = 2.0 for balanced behavior
- Increase to β = 3-5 for more exploration (useful in first few cycles)
- Decrease to β = 1.0 for more exploitation (useful in later cycles)
- Can vary β across cycles using `CycleConfig` with different `acquisition_params`

### Requires Uncertainty

UCB requires uncertainty estimates. Compatible learners:

- **GaussianProcessLearner** - Best calibrated uncertainty
- **Ensemble learners** - Uncertainty from model disagreement (MixedEnsemble, RFEnsemble, etc.)
- **MCDropoutLearner** - Uncertainty from dropout sampling
- **AdvancedRandomForestLearner** - Uncertainty from OOB scoring

### Examples

**CLI with default β:**
```bash
learnm8 run compounds.csv --target Activity \
  --learner gp --featurizer morgan \
  --cycles "random:0.02 ucb:0.005*10"
```

**CLI with custom β (more exploration):**
```bash
learnm8 run compounds.csv --target Activity \
  --learner ensemble --featurizer morgan \
  --cycles "random:0.02 ucb:0.005*5" \
  --acquisition-params '{"beta": 3.0}'
```

**Python API with varying β across cycles:**
```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', batch_fraction=0.02),
        CycleConfig('ucb', n_cycles=5, batch_fraction=0.005,
                    acquisition_params={'beta': 3.0}),
        CycleConfig('ucb', n_cycles=5, batch_fraction=0.005,
                    acquisition_params={'beta': 1.5})
    ]
)
```

## Expected Improvement (EI)

### Overview

Expected Improvement calculates the expected improvement over the current best observed value from labeled training data. It provides a principled probabilistic framework for balancing exploration and exploitation.

**Formula:** `EI = E[max(0, improvement)]`

Where improvement is calculated relative to the best value seen in training data. EI uses the predictive distribution (mean and uncertainty) to compute the probability-weighted expected gain.

**Requires:** SciPy for normal distribution calculations (`pip install scipy`)

### Exploration/Exploitation Balance

- **ξ = 0:** Exploitation-focused (only sample if improvement is expected)
- **ξ = 0.01:** Balanced (default, small exploration bonus)
- **ξ = 0.1:** Exploration-focused (accepts candidates with lower expected improvement)

The ξ parameter adds a small buffer to encourage exploration even when improvement probability is modest.

### When to Use

- **Iterative optimization:** When you want to systematically improve beyond current best
- **Well-calibrated uncertainty:** When learner uncertainty is reliable (GP recommended)
- **Sequential screening:** Works well with greedy follow-up phases
- **Bayesian optimization:** Standard choice in Bayesian optimization workflows

Avoid EI when:

- Current best value is unknown or unreliable
- Uncertainty calibration is poor
- Goal is exploration rather than optimization

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xi` | float | 0.01 | Exploration parameter. Small positive values encourage exploration. |
| `score_direction` | str | 'higher' | Optimization direction ('higher' or 'lower'). |
| `current_best` | float | Required | Best observed value from labeled training data. |

**Tuning Guidance:**

- Keep ξ small (0.001-0.1) for optimization-focused campaigns
- Increase ξ to 0.1-0.5 for more exploration
- `current_best` is automatically computed from training data in each cycle
- Pass via `acquisition_params` at cycle level: `{'current_best': 0.95}`

### Requires Uncertainty

EI requires well-calibrated uncertainty estimates. Compatible learners:

- **GaussianProcessLearner** - Gold standard for EI (best calibration)
- **Ensemble learners** - Reasonable uncertainty estimates
- **MCDropoutLearner** - Requires sufficient dropout samples (100+)

### Examples

**CLI with default ξ:**
```bash
learnm8 run compounds.csv --target Activity \
  --learner gp --featurizer morgan \
  --cycles "random:0.02 ei:0.005*10"
```

**Python API with current best:**
```python
from learnm8 import run_active_learning
import polars as pl

df = pl.read_csv('compounds.csv')
current_best = df['Activity'].max()

results = run_active_learning(
    compound_pool=df,
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.02),
        ('ei', 0.005)
    ],
    acquisition_params={'xi': 0.01, 'current_best': current_best}
)
```

## Probability of Improvement (PI)

### Overview

Probability of Improvement calculates the probability that a compound will improve over the current best observed value. Unlike EI, PI focuses solely on the probability of improvement rather than the magnitude.

**Formula:** `PI = P(f(x) > current_best + ξ)`

PI is simpler and more conservative than EI, selecting compounds with high probability of any improvement.

**Requires:** SciPy for normal distribution calculations (`pip install scipy`)

### Exploration/Exploitation Balance

- **ξ = 0:** Pure optimization (select if any improvement probability)
- **ξ = 0.01:** Balanced (default, requires slight improvement)
- **ξ = 0.1:** Conservative (requires more substantial improvement)

Similar to EI, ξ controls the exploration bonus but with different interpretation.

### When to Use

- **Conservative optimization:** When you prefer high-probability improvements
- **Risk-averse screening:** When measurement costs are high
- **Complement to EI:** Alternative probabilistic approach
- **Simple optimization:** When EI's complexity is unnecessary

Avoid PI when:

- Magnitude of improvement matters (use EI instead)
- Goal is broad exploration (use UCB or entropy instead)
- Uncertainty estimates are unreliable

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xi` | float | 0.01 | Exploration parameter. Controls improvement threshold. |
| `score_direction` | str | 'higher' | Optimization direction ('higher' or 'lower'). |
| `current_best` | float | Required | Best observed value from labeled training data. |

**Tuning Guidance:**

- Keep ξ = 0.01 for balanced probability of improvement
- Increase ξ to 0.1-0.5 for more conservative selection (higher improvement required)
- Decrease ξ to 0.001 for more aggressive selection
- Like EI, `current_best` should be computed from training data each cycle

### Requires Uncertainty

PI requires uncertainty estimates. Compatible learners:

- **GaussianProcessLearner** - Best choice for calibrated probabilities
- **Ensemble learners** - Reasonable probability estimates
- **MCDropoutLearner** - Requires sufficient samples

### Examples

**CLI:**
```bash
learnm8 run compounds.csv --target Activity \
  --learner gp --featurizer morgan \
  --cycles "random:0.02 pi:0.005*10"
```

**Python API with conservative threshold:**
```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.02),
        ('pi', 0.005)
    ],
    acquisition_params={'xi': 0.1}
)
```

## Thompson Sampling

### Overview

Thompson Sampling is a stochastic exploration strategy that samples from the posterior predictive distribution and selects compounds based on sampled values. This Bayesian approach naturally balances exploration and exploitation through the posterior uncertainty.

**Algorithm:**

1. For each compound, sample a value from its predictive distribution: `sample ~ N(prediction, uncertainty)`
2. Select compounds with highest (or lowest) sampled values
3. Stochasticity ensures exploration in uncertain regions

Thompson Sampling provides probability matching to the optimal strategy in the long run.

### Exploration/Exploitation Balance

Thompson Sampling automatically balances exploration and exploitation without requiring parameter tuning:

- Compounds with high uncertainty have wider sampling distributions (more exploration)
- Compounds with high predicted values are more likely to be sampled (exploitation)
- Stochasticity provides natural diversity across cycles

The only parameter is `random_state` for reproducibility, not exploration control.

### When to Use

- **Parameter-free exploration:** When you want automatic exploration without tuning
- **Stochastic sampling:** When diversity across runs is acceptable or desired
- **Bayesian workflows:** Natural choice in Bayesian optimization contexts
- **Long campaigns:** Provably optimal in the long run (multi-armed bandit theory)

Avoid Thompson Sampling when:

- Deterministic selection is required
- Short campaigns where stochasticity introduces too much variance
- Interpretability is important (behavior is less intuitive than UCB/EI)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | int | 42 | Random seed for reproducible sampling. |
| `score_direction` | str | 'higher' | Optimization direction ('higher' or 'lower'). |

**Tuning Guidance:**

- Set `random_state` for reproducibility in benchmark experiments
- Vary `random_state` across independent runs for exploration diversity
- No other tuning required (automatic exploration)

### Requires Uncertainty

Thompson Sampling requires uncertainty estimates. Compatible learners:

- **GaussianProcessLearner** - Best calibrated sampling
- **Ensemble learners** - Samples from model disagreement
- **MCDropoutLearner** - Samples from dropout distribution
- **AdvancedRandomForestLearner** - Samples from OOB estimates

### Examples

**CLI:**
```bash
learnm8 run compounds.csv --target Activity \
  --learner gp --featurizer morgan \
  --cycles "random:0.02 thompson:0.005*10" \
  --random-state 42
```

**Python API with custom random state:**
```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='ensemble',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        ('random', 0.02),
        ('thompson', 0.005)
    ],
    acquisition_params={'random_state': 123}
)
```

**Multiple independent runs with different seeds:**
```python
from learnm8 import run_active_learning

for seed in [42, 43, 44, 45, 46]:
    results = run_active_learning(
        compound_pool='compounds.csv',
        oracle='oracle.csv',
        learner='gp',
        target_col='Activity',
        featurizer='morgan',
        cycles=[('random', 0.02), ('thompson', 0.005)],
        acquisition_params={'random_state': seed},
        output_dir=f'results/thompson_seed_{seed}'
    )
```

## Entropy

### Overview

Entropy-based acquisition selects compounds that maximize information gain by choosing samples with highest predictive entropy (uncertainty). This is a pure exploration strategy focused on learning model parameters rather than finding high-value compounds.

**Formula:** `Entropy = uncertainty` or `Entropy = variance`

Entropy acquisition prioritizes reducing model uncertainty uniformly across chemical space.

### Exploration/Exploitation Balance

Entropy is a pure exploration strategy with no exploitation component:

- Selects compounds with maximum uncertainty regardless of predicted value
- Useful for building robust models before exploitation
- Often combined with greedy phases in multi-stage cycles

The `entropy_type` parameter controls whether raw uncertainty (standard deviation) or variance is used.

### When to Use

- **Model building:** Early cycles to establish model confidence
- **Exploration phase:** Before exploitation in multi-stage workflows
- **Uncertain landscapes:** When predictions are unreliable everywhere
- **Diversity complement:** Combines well with diversity-based strategies

Avoid Entropy when:

- Goal is optimization rather than exploration
- Later cycles when exploitation is needed
- Uncertainty estimates are poorly calibrated

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_type` | str | 'uncertainty' | Type of entropy measure ('uncertainty' or 'variance'). |
| `score_direction` | str | 'higher' | Optimization direction (ignored by entropy, but required for consistency). |

**Tuning Guidance:**

- Use `entropy_type='uncertainty'` (default) for linear uncertainty scoring
- Use `entropy_type='variance'` to emphasize high-uncertainty compounds more strongly
- Variance heavily penalizes low-uncertainty compounds (quadratic relationship)

### Requires Uncertainty

Entropy requires uncertainty estimates. Compatible learners:

- **GaussianProcessLearner** - Best calibrated entropy
- **Ensemble learners** - Entropy from model disagreement
- **MCDropoutLearner** - Entropy from dropout variability
- **AdvancedRandomForestLearner** - Entropy from OOB estimates

### Examples

**CLI with uncertainty-based entropy:**
```bash
learnm8 run compounds.csv --target Activity \
  --learner gp --featurizer morgan \
  --cycles "random:0.02 entropy:0.01*3 greedy:0.005*7"
```

**CLI with variance-based entropy:**
```bash
learnm8 run compounds.csv --target Activity \
  --learner ensemble --featurizer morgan \
  --cycles "random:0.02 entropy:0.01*5" \
  --acquisition-params '{"entropy_type": "variance"}'
```

**Python API with exploration-exploitation sequence:**
```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', batch_fraction=0.02),
        CycleConfig('entropy', n_cycles=3, batch_fraction=0.01,
                    acquisition_params={'entropy_type': 'uncertainty'}),
        CycleConfig('greedy', n_cycles=7, batch_fraction=0.005)
    ]
)
```

## Comparison Table

| Strategy | Exploration | Exploitation | Requires | Key Parameter | Best Use Case |
|----------|-------------|--------------|----------|---------------|---------------|
| **UCB** | High (tunable) | High (tunable) | Uncertainty | β (2.0) | General-purpose balanced |
| **EI** | Medium | High | Uncertainty + current_best | ξ (0.01) | Iterative optimization |
| **PI** | Low | High | Uncertainty + current_best | ξ (0.01) | Conservative optimization |
| **Thompson** | Medium | Medium | Uncertainty | random_state | Parameter-free exploration |
| **Entropy** | Very High | None | Uncertainty | entropy_type | Pure exploration phase |

## Performance Considerations

All uncertainty-based strategies have similar computational complexity:

1. **Training time:** Dominated by learner (GP is slowest, ensembles 3x cost)
2. **Selection time:** O(n) scoring + O(n log k) selection where n = pool size, k = batch size
3. **Memory:** Minimal additional memory beyond predictions/uncertainties

**Optimization tips:**

- EI and PI require SciPy (normal distribution CDF/PDF)
- Thompson Sampling uses numpy random sampling (faster than EI/PI)
- UCB is fastest (simple arithmetic)
- All scale well to millions of compounds (vectorized operations)

## Combining with Other Strategies

Uncertainty-based strategies work well in multi-stage cycles:

**Exploration → Exploitation:**
```python
cycles=[
    ('random', 0.02),
    ('entropy', 0.01),
    ('ucb', 0.005),
    ('greedy', 0.005)
]
```

**Balanced throughout:**
```python
cycles=[
    ('random', 0.02),
    ('ucb', 0.005),
    ('ucb', 0.005)
]
```

**Optimization-focused:**
```python
cycles=[
    ('random', 0.02),
    ('ei', 0.005),
    ('greedy', 0.005)
]
```

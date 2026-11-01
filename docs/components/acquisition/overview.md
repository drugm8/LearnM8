# Acquisition Functions

An acquisition function decides which unlabeled compounds are measured in each cycle, given the model's predictions and, for some strategies, its uncertainty estimates.

## Exploration vs exploitation

Exploitation picks compounds with the best predicted values; exploration picks compounds the model is unsure about. [Greedy](greedy.md) is pure exploitation, [entropy](entropy.md) is pure exploration, and [UCB](ucb.md), [EI](expected-improvement.md), [PI](probability-improvement.md) and [Thompson sampling](thompson-sampling.md) trade the two off.

## Strategies

| `strategy=` | Page | Needs uncertainty | Parameters |
|-------------|------|-------------------|------------|
| `greedy` | [Greedy](greedy.md) | no | none |
| `random` | [Random](random.md) | no | `random_state` |
| `topk` | [Top-K](topk.md) | no | `k_fraction`, `random_state` |
| `ucb` | [UCB](ucb.md) | yes | `beta` |
| `ei` | [Expected Improvement](expected-improvement.md) | yes | `xi`, `current_best` |
| `pi` | [Probability of Improvement](probability-improvement.md) | yes | `xi`, `current_best` |
| `thompson` | [Thompson Sampling](thompson-sampling.md) | yes | `random_state` |
| `entropy` | [Entropy](entropy.md) | yes | `entropy_type` |

Strategies that need uncertainty require a learner that provides it (for example `rf`, `gp`, `mc_dropout` or an ensemble); otherwise the cycle raises an error. See the [learners overview](../learners/overview.md). The names above are the keys of `learnm8.acquisition.ACQUISITION_REGISTRY`.

## Score direction

`score_direction` (`'higher'` by default, or `'lower'`) says whether larger or smaller target values are better. It is a top-level option, not an acquisition parameter, and the value-ranking strategies respect it:

```bash
learnm8 run compounds.csv --target Toxicity --learner rf --featurizer morgan \
  --strategy greedy --score-direction lower
```

```python
results = run_active_learning(..., strategy='greedy', score_direction='lower')
```

[Random](random.md) and [entropy](entropy.md) ignore the direction.

## Passing parameters

Strategy parameters are passed as a dict (`acquisition_params`) in the API, or as a JSON string with `--acquisition-params` on the CLI. The strategy is chosen with `strategy=` / `--strategy` (default `greedy`); the first cycle uses `initial_strategy` (default `random`).

## Multi-strategy cycles

To use different strategies in different stages (for example random, then UCB, then greedy), see [Building custom cycles](../../tutorials/building-custom-cycles.md).

## Custom strategies

To add your own strategy, see [Extending LearnM8](../../customization/extending.md).

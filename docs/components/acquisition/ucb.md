# Upper Confidence Bound (UCB)

Scores each compound by its prediction plus a multiple of its uncertainty, so uncertain compounds are favoured as `beta` grows.

- Registry name: `ucb` (`UCBAcquisition`)
- Requires uncertainty: yes (use a learner that supports it, e.g. `rf`, `gp`, `mc_dropout`, or an ensemble)

## Formula

With prediction `mu` and uncertainty (standard deviation) `sigma`:

- `score_direction='higher'`: select the highest `mu + beta * sigma`
- `score_direction='lower'`: select the lowest `mu - beta * sigma`

`beta = 0` reduces to [greedy](greedy.md) selection. Infinite `sigma` values raise an error.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | `2.0` | Weight on uncertainty; must be `>= 0`. Larger values favour exploration |

## Passing parameters

- CLI: `--strategy ucb --acquisition-params '{"beta": 1.0}'`
- API: `strategy='ucb'`, `acquisition_params={'beta': 1.0}`

## CLI example

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --strategy ucb --acquisition-params '{"beta": 1.0}' --n-cycles 10
```

## API example

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    target_col='Activity',
    learner='gp',
    featurizer='morgan',
    strategy='ucb',
    acquisition_params={'beta': 1.0},
    score_direction='higher',
    n_cycles=10,
)
```

## See also

- [Acquisition overview](overview.md)
- [Learners overview](../learners/overview.md) for which learners provide uncertainty

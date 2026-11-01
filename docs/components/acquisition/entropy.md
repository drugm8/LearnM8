# Entropy

Selects the compounds whose predictive distribution has the highest differential entropy, which ranks compounds by uncertainty alone. Pure exploration: predictions do not affect the ranking, and `score_direction` has no effect on it.

- Registry name: `entropy` (`EntropyAcquisition`)
- Requires uncertainty: yes

## Formula

For a Gaussian with standard deviation `sigma`, the score is the differential entropy in nats:

- `entropy_type='uncertainty'`: `0.5 * log(2*pi*e) + log(sigma)`
- `entropy_type='variance'`: `0.5 * log(2*pi*e) + 2 * log(sigma)`

Both are monotone in `sigma`, so the selection is the same as ranking by `sigma` descending. Entropy is not equivalent to [UCB](ucb.md) with `beta=0`, which ranks by prediction. Very small `sigma` values are clamped at `1e-9`, and a warning is issued when at least 1% of the values are clamped.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_type` | str | `'uncertainty'` | `'uncertainty'` or `'variance'`; any other value raises `ValueError` |

## Passing parameters

- CLI: `--strategy entropy --acquisition-params '{"entropy_type": "variance"}'`
- API: `strategy='entropy'`, `acquisition_params={'entropy_type': 'variance'}`

## CLI example

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --strategy entropy --n-cycles 10
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
    strategy='entropy',
    n_cycles=10,
)
```

## See also

- [Acquisition overview](overview.md)

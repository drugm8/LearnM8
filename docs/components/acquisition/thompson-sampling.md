# Thompson Sampling

Draws one sample per compound from a normal distribution with the predicted mean and uncertainty, then selects the compounds with the best sampled values. Exploration arises from the randomness of the draw.

- Registry name: `thompson` (`ThompsonSamplingAcquisition`)
- Requires uncertainty: yes

## Formula

`sample = Normal(mu, sigma)` per compound, where `mu` is the prediction and `sigma` the uncertainty (standard deviation). `score_direction='higher'` selects the largest samples, `'lower'` the smallest.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | int | `42` | Seed for the posterior draw |

When `random_state` is not given in `acquisition_params`, the cycle derives a per-cycle seed from the run's `random_state`.

## Passing parameters

- CLI: `--strategy thompson`; optional `--acquisition-params '{"random_state": 7}'`
- API: `strategy='thompson'`, `acquisition_params={'random_state': 7}`

## CLI example

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --strategy thompson --n-cycles 10
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
    strategy='thompson',
    n_cycles=10,
)
```

## See also

- [Acquisition overview](overview.md)

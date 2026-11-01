# Random

Selects compounds uniformly at random from the unlabeled pool, ignoring predictions. Used for the initial cycle and as a baseline to compare other strategies against.

- Registry name: `random` (`RandomAcquisition`)
- Requires uncertainty: no
- `score_direction` has no effect on which compounds are drawn

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | int | `42` | Seed for the draw |

When `random_state` is not given in `acquisition_params`, the cycle derives a per-cycle seed from the run's `random_state`, so repeated cycles do not draw the same way.

## Passing parameters

- CLI: `--strategy random`; extra parameters go in `--acquisition-params '{"random_state": 7}'`
- API: `strategy='random'`, `acquisition_params={'random_state': 7}`

## CLI example

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --strategy random --n-cycles 10
```

## API example

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    target_col='Activity',
    learner='rf',
    featurizer='morgan',
    strategy='random',
    n_cycles=10,
)
```

The first cycle (cycle 0) is always a random draw controlled by `initial_strategy` (default `'random'`); see [Running experiments](../../tutorials/running-experiments.md).

## See also

- [Acquisition overview](overview.md)

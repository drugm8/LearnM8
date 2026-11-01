# Top-K

Takes the best `k_fraction` of the pool by prediction and draws the batch at random from that candidate set. Less greedy than [greedy](greedy.md): it spreads picks over a wider set of high-ranked compounds.

- Registry name: `topk` (`TopKAcquisition`)
- Requires uncertainty: no
- Candidate set size: `max(n_select, int(pool_size * k_fraction))`, capped at the pool size. If the candidate set equals the batch size, the batch is the candidate set.
- Direction: `score_direction='higher'` ranks from the top, `'lower'` from the bottom

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_fraction` | float | `0.1` | Fraction of the pool to treat as candidates; must be in `(0, 1]` |
| `random_state` | int | `42` | Seed for the random draw from the candidates |
| `score_direction` | str | `'higher'` | Set via the top-level `score_direction` option |

When `random_state` is not given in `acquisition_params`, the cycle derives a per-cycle seed from the run's `random_state`.

## Passing parameters

- CLI: `--strategy topk --acquisition-params '{"k_fraction": 0.05}'`
- API: `strategy='topk'`, `acquisition_params={'k_fraction': 0.05}`

## CLI example

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --strategy topk --acquisition-params '{"k_fraction": 0.05}' --n-cycles 10
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
    strategy='topk',
    acquisition_params={'k_fraction': 0.05},
    n_cycles=10,
)
```

## See also

- [Acquisition overview](overview.md)

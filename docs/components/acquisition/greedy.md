# Greedy

Selects the compounds with the best predicted values. Pure exploitation: no exploration, no randomness.

- Registry name: `greedy` (`GreedyAcquisition`)
- Requires uncertainty: no (works with any learner)
- Score: the prediction, negated internally when `score_direction='lower'`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_direction` | str | `'higher'` | `'higher'` picks the largest predictions, `'lower'` the smallest |

`score_direction` is a top-level run option (see below), not something you normally put in `acquisition_params`.

## Passing parameters

- CLI: `--strategy greedy --score-direction lower`
- API: `strategy='greedy', score_direction='lower'`

## CLI example

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --strategy greedy --score-direction higher --n-cycles 10
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
    strategy='greedy',
    score_direction='higher',
    n_cycles=10,
)
```

## See also

- [Acquisition overview](overview.md)
- [Top-K](topk.md) for greedy selection with randomness among the best candidates

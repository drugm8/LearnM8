# Probability of Improvement (PI)

Scores each compound by the probability that it improves on the best value observed so far. It ignores the size of the improvement, which makes it more exploitative than [EI](expected-improvement.md).

- Registry name: `pi` (`ProbabilityImprovementAcquisition`)
- Requires uncertainty: yes

## Formula

With prediction `mu`, uncertainty `sigma`, best labeled value `current_best` and margin `xi`:

- Maximizing: `improvement = mu - current_best - xi`
- Minimizing: `improvement = current_best - mu - xi`
- `PI = Phi(improvement / sigma)` (`Phi` is the standard normal CDF)

When `sigma = 0`, PI is 1 if `improvement > 0`, else 0. The compounds with the highest PI are selected for either direction. Infinite `sigma` values raise an error.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xi` | float | `0.01` | Improvement margin; must be `>= 0` |
| `current_best` | float or None | set automatically | Best labeled value |

`current_best` is filled in each cycle from the labeled data (the maximum for `'higher'`, the minimum for `'lower'`) unless you supply it in `acquisition_params`. The `minimize` argument is deprecated; use `score_direction`.

## Passing parameters

- CLI: `--strategy pi --acquisition-params '{"xi": 0.05}'`
- API: `strategy='pi'`, `acquisition_params={'xi': 0.05}`

## CLI example

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --strategy pi --acquisition-params '{"xi": 0.05}' --n-cycles 10
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
    strategy='pi',
    acquisition_params={'xi': 0.05},
    n_cycles=10,
)
```

## See also

- [Expected Improvement](expected-improvement.md)
- [Acquisition overview](overview.md)

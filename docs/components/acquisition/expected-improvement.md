# Expected Improvement (EI)

Scores each compound by the expected amount by which it improves on the best value observed so far.

- Registry name: `ei` (`ExpectedImprovementAcquisition`)
- Requires uncertainty: yes

## Formula

With prediction `mu`, uncertainty `sigma`, best labeled value `current_best` and margin `xi`:

- Maximizing: `improvement = mu - current_best - xi`
- Minimizing: `improvement = current_best - mu - xi`
- `z = improvement / sigma`
- `EI = improvement * Phi(z) + sigma * phi(z)` (`Phi`, `phi` are the standard normal CDF and PDF)

When `sigma = 0`, EI is `max(improvement, 0)`. The compounds with the highest EI are selected for either direction. Infinite `sigma` values raise an error.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xi` | float | `0.01` | Improvement margin; must be `>= 0` |
| `current_best` | float or None | set automatically | Best labeled value |

`current_best` is filled in each cycle from the labeled data (the maximum for `'higher'`, the minimum for `'lower'`) unless you supply it in `acquisition_params`. The `minimize` argument is deprecated; use `score_direction`.

## Passing parameters

- CLI: `--strategy ei --acquisition-params '{"xi": 0.05}' --score-direction lower`
- API: `strategy='ei'`, `acquisition_params={'xi': 0.05}`, `score_direction='lower'`

## CLI example

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan \
  --strategy ei --acquisition-params '{"xi": 0.05}' --n-cycles 10
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
    strategy='ei',
    acquisition_params={'xi': 0.05},
    n_cycles=10,
)
```

## See also

- [Probability of Improvement](probability-improvement.md)
- [Acquisition overview](overview.md)

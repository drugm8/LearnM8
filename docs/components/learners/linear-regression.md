# Linear Regression

Ridge regression (the default) or ordinary least squares, with an analytic leverage-based uncertainty.

- **Registry name:** `lr`
- **Class:** `learnm8.learners.LinearRegressionLearner`
- **Uncertainty:** yes. `sqrt(sigma_hat^2 * (1 + h))`, where `h = x^T (X^T X + alpha I)^-1 x` is the leverage of the compound and `sigma_hat^2 = RSS / max(n - p, 1)` is the residual variance. Ordinal, not a calibrated interval.
- **Requires SMILES:** no

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `alpha` | `0.1` | L2 strength. `None` switches to ordinary `LinearRegression` |
| `fit_intercept` | `True` | Fit an intercept |
| `n_jobs` | `-1` | Parallel jobs (only used when `alpha=None`) |
| `random_state` | `42` | Random seed |

Any other keyword argument goes to the `SklearnLearner` base class.

## Usage

```bash
learnm8 run data.csv --target Value --learner lr --featurizer rdkit_2d_descriptors --strategy ucb
```

```python
from learnm8 import run_active_learning
from learnm8.learners import LinearRegressionLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=LinearRegressionLearner(alpha=1.0),
    featurizer='morgan', strategy='ucb', n_cycles=10,
)
```

## Notes

- Features are standardized when `alpha` is set (Ridge); they are not with `alpha=None`.
- Training factorizes the `p x p` Gram matrix, so cost grows with the number of features.
- A GPU version is available as [`ridge_cuml`](ridge-cuml.md). `lr_ensemble` combines several Ridge strengths; see [Ensembles](ensembles.md).

See also: [Learners overview](overview.md).

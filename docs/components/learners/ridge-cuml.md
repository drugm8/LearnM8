# Ridge (cuML)

RAPIDS cuML Ridge regression on the GPU with the same leverage-based uncertainty as [Linear Regression](linear-regression.md).

- **Registry name:** `ridge_cuml`
- **Class:** `learnm8.learners.RidgeCumlLearner`
- **Uncertainty:** yes. `sqrt(sigma_hat^2 * (1 + h))` with leverage `h = x^T (X^T X + alpha I)^-1 x`, computed with CuPy on the GPU and falling back to CPU (SciPy Cholesky) on GPU out-of-memory. Ordinal, not a calibrated interval.
- **Requires SMILES:** no
- **Requirements:** a CUDA GPU and RAPIDS cuML >= 25.04. The learner can be constructed without them; `train()` raises `LearnerError` if cuML is missing.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `alpha` | `0.1` | L2 strength. Must not be `None` (use `lr` with `alpha=None` for unregularized regression) |
| `fit_intercept` | `True` | Fit an intercept |
| `solver` | `'eig'` | cuML Ridge solver: `'eig'`, `'svd'` or `'cd'` |
| `random_state` | `42` | Random seed |
| `remove_zero_variance` | `True` | Drop zero-variance features |

## Usage

```bash
learnm8 run data.csv --target Value --learner ridge_cuml --featurizer rdkit_2d_descriptors --strategy ucb
```

```python
from learnm8 import run_active_learning
from learnm8.learners import RidgeCumlLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=RidgeCumlLearner(alpha=1.0),
    featurizer='morgan', strategy='ucb', n_cycles=10,
)
```

## Notes

- Features are standardized before fitting.
- The residual variance and Cholesky factor used for uncertainty are computed on the CPU in float64 at training time.
- `lr_ensemble` and `mixed_ensemble` use this learner for their linear members when `device='cuda'`; see [Ensembles](ensembles.md).

See also: [Learners overview](overview.md).

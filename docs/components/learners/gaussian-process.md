# Gaussian Process

Scikit-learn `GaussianProcessRegressor` with a Tanimoto kernel for binary fingerprints or an RBF kernel for continuous features. Exact inference costs O(n^3) in the training-set size, so it suits small labeled sets.

- **Registry name:** `gp`
- **Class:** `learnm8.learners.GaussianProcessLearner`
- **Uncertainty:** yes. Posterior standard deviation (`predict(..., return_std=True)`).
- **Requires SMILES:** no

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `kernel` | `'auto'` | `'auto'`, `'tanimoto'`, `'rbf'`, `None` (same as `'auto'`), or a scikit-learn `Kernel` instance |
| `alpha` | `1e-3` | Noise added to the kernel diagonal |
| `n_restarts_optimizer` | `5` | Restarts of the kernel-hyperparameter optimizer |
| `normalize_y` | `True` | Normalize targets before fitting |
| `random_state` | `42` | Random seed |
| `max_train_size` | `5000` | Training sets larger than this raise `LearnerError` |

Any other keyword argument goes to the `SklearnLearner` base class.

## Usage

```bash
learnm8 run data.csv --target Value --learner gp --featurizer morgan --strategy ei
```

```python
from learnm8 import run_active_learning
from learnm8.learners import GaussianProcessLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=GaussianProcessLearner(kernel='tanimoto', max_train_size=3000),
    featurizer='morgan', strategy='ei', n_cycles=10,
)
```

## Notes

- `kernel='auto'` picks the Tanimoto kernel when every training feature is 0 or 1, otherwise a constant times RBF kernel.
- Features are standardized only when the resolved kernel is not Tanimoto.
- A warning is logged once the training set exceeds 40% of `max_train_size`. Prediction cost grows with the pool size; for larger labeled sets see [GPU GP](gpu-gp.md), [SVGP](svgp.md) or [Random Forest](random-forest.md).
- Features are loaded as float32.

See also: [Learners overview](overview.md), [Uncertainty-based acquisition](../acquisition/overview.md).

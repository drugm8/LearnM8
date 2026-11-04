# XGBoost

Gradient-boosted trees via `xgboost.XGBRegressor` (`tree_method='hist'`). Single-model XGBoost provides no uncertainty, so use it with `greedy`, `topk` or `random` acquisition, or use [`xgb_ensemble`](ensembles.md) for uncertainty.

- **Registry name:** `xgb`
- **Class:** `learnm8.learners.XGBoostLearner`
- **Uncertainty:** no
- **Requires SMILES:** no
- **GPU:** optional, `device='cuda'`

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_estimators` | `100` | Boosting rounds |
| `learning_rate` | `0.1` | Step-size shrinkage |
| `max_depth` | `6` | Maximum tree depth |
| `min_child_weight` | `1` | Minimum instance weight in a child |
| `subsample` | `0.8` | Row subsample ratio |
| `colsample_bytree` | `0.8` | Column subsample ratio per tree |
| `reg_alpha` | `0.0` | L1 regularization |
| `reg_lambda` | `1.0` | L2 regularization |
| `random_state` | `42` | Random seed |
| `n_jobs` | `-1` | Parallel jobs |
| `device` | `'cpu'` | `'cpu'`, `'cuda'`, `'cuda:N'` or `'auto'` (resolves to `cuda` if a GPU is visible) |
| `enable_aggressive_gc` | `True` | Delete the old model and run `gc.collect()` before refitting |

## Usage

```bash
learnm8 run data.csv --target Value --learner xgb --featurizer morgan --strategy greedy
```

```python
from learnm8 import run_active_learning
from learnm8.learners import XGBoostLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=XGBoostLearner(n_estimators=300, max_depth=8, device='cuda'),
    featurizer='morgan', strategy='greedy', n_cycles=10,
)
```

## Notes

- Binary featurizers are loaded as `uint8` (`preferred_feature_dtype()`).
- If CUDA training fails with `device='cuda'`, a `LearnerError` is raised rather than silently falling back to CPU.
- Uncertainty-based strategies (`ucb`, `ei`, `pi`, `thompson`, `entropy`) need uncertainty; with `xgb` they raise `ConfigurationError` before cycle 0, so use `xgb_ensemble` instead; see [Acquisition](../acquisition/overview.md).

See also: [Learners overview](overview.md), [Ensembles](ensembles.md).

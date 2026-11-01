# Random Forest

Scikit-learn `RandomForestRegressor` (bootstrap, out-of-bag scoring enabled). A good default learner for fingerprint features.

- **Registry name:** `rf`
- **Class:** `learnm8.learners.RandomForestLearner`
- **Uncertainty:** yes. Population standard deviation (`ddof=0`) across the individual tree predictions. It is useful for ranking compounds by relative confidence, not a calibrated prediction interval.
- **Requires SMILES:** no

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_estimators` | `100` | Number of trees |
| `max_depth` | `None` | Maximum tree depth (`None` = unlimited) |
| `min_samples_split` | `2` | Minimum samples to split an internal node |
| `min_samples_leaf` | `1` | Minimum samples at a leaf |
| `max_features` | `'sqrt'` | Features considered per split |
| `random_state` | `42` | Random seed |
| `n_jobs` | `-1` | Parallel jobs (`-1` = all cores) |

Extra keyword arguments (for example `remove_zero_variance`) go to the `SklearnLearner` base class.

## Usage

```bash
learnm8 run data.csv --target Value --learner rf --featurizer morgan --strategy ucb
```

```python
from learnm8 import run_active_learning
from learnm8.learners import RandomForestLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=RandomForestLearner(n_estimators=300, max_depth=20),
    featurizer='morgan', strategy='ucb', n_cycles=10,
)
```

## Notes

- Binary featurizers are loaded as `uint8` (`preferred_feature_dtype()`), which avoids a float32 copy of the feature matrix.
- When the acquisition strategy does not need uncertainty (for example `greedy`), the per-tree uncertainty pass is skipped. Pass `--force-uncertainty` / `force_uncertainty=True` to compute it anyway.
- `get_oob_score()` returns the out-of-bag R² of the trained forest.

See also: [Learners overview](overview.md), [RF FIL](rf-fil.md) (GPU inference).

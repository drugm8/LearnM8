# Decision Tree

Scikit-learn `DecisionTreeRegressor`. A single, fast, interpretable tree.

- **Registry name:** `dt`
- **Class:** `learnm8.learners.DecisionTreeLearner`
- **Uncertainty:** yes, when the tree uses `criterion='squared_error'` (the default). The uncertainty is `sqrt(impurity)` of the leaf each compound falls into, i.e. the training-target standard deviation within that leaf. It is ordinal, not a calibrated interval.
- **Requires SMILES:** no

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | `10` | Maximum tree depth |
| `min_samples_split` | `10` | Minimum samples to split an internal node |
| `min_samples_leaf` | `5` | Minimum samples at a leaf |
| `max_features` | `None` | Features considered per split |
| `random_state` | `42` | Random seed |

Any other keyword argument goes to the `SklearnLearner` base class.

## Usage

```bash
learnm8 run data.csv --target Value --learner dt --featurizer maccs --strategy ucb
```

```python
from learnm8 import run_active_learning
from learnm8.learners import DecisionTreeLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=DecisionTreeLearner(max_depth=8, min_samples_leaf=10),
    featurizer='maccs', strategy='ucb', n_cycles=10,
)
```

## Notes

- Compounds that land in a pure leaf get zero uncertainty. Raise `min_samples_leaf` or lower `max_depth` if many predictions have zero uncertainty.
- Binary featurizers are loaded as `uint8` (`preferred_feature_dtype()`).
- For a diversified set of trees see `dt_ensemble` in [Ensembles](ensembles.md).

See also: [Learners overview](overview.md), [Random Forest](random-forest.md).

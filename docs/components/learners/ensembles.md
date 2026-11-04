# Ensembles

An ensemble trains several member learners on the same data and combines their predictions. The disagreement between members is the uncertainty estimate, which makes ensembles a way to give uncertainty to models that have none (XGBoost, Fastprop, Chemprop) and to get a different kind of uncertainty from those that do.

## How ensemble uncertainty works

Every member predicts the full pool. For `n` members the prediction matrix has shape `(n, n_compounds)` and is reduced per compound:

| Quantity | Default | Alternatives (`EnsembleLearner` arguments) |
|---|---|---|
| Prediction | mean across members (`aggregation_method='mean'`, optionally weighted by `weights`) | `'median'` |
| Uncertainty | population standard deviation, `ddof=0` (`uncertainty_method='std'`) | `'mad'` (median absolute deviation from the median prediction), `'quantile'` (half the interquartile range) |

The ensemble always reports `supports_uncertainty() == True`, regardless of whether the members do. Member uncertainties are not used; only the member predictions are.

When the acquisition strategy does not need uncertainty, `compute_uncertainty=False` is forwarded to each member and the outer reduction is skipped.

### Member seeds

Unless you pass an explicit `random_states` list, member `i` is seeded with `random_state + offset[i]`, where the offsets are `(0, 81, 314)`. With the default `random_state=42` the three members use seeds 42, 123 and 356. Ensembles with more than three members extend the offsets by repeatedly adding 314 to the last one, so the same base seed always yields the same member seeds.

## Type-specific ensembles

| Registry name | Class | Members and how they differ |
|---|---|---|
| `lr_ensemble` | `LREnsemble` | One Ridge `LinearRegressionLearner` per `regularization_strengths` (default `[0.1, 1.0, 10.0]`) |
| `xgb_ensemble` | `XGBEnsemble` | One `XGBoostLearner` per `learning_rates` (default `[0.05, 0.1, 0.2]`) |
| `dt_ensemble` | `DTEnsemble` | One `DecisionTreeLearner` per `max_depths` (default `[5, 10, 15]`) |
| `fastprop_ensemble` | `FastpropEnsemble` | 3 `FastpropLearner`, different seeds; accepts the `FastpropLearner` hyperparameters (see [Fastprop](fastprop.md)) |
| `mixed_ensemble` | `MixedEnsemble` | RF + Ridge + XGBoost; see [Mixed Ensemble](mixed-ensemble.md) |
| `chemprop_ensemble` | `ChempropEnsemble` | 3 `ChempropLearner`; see [Chemprop Ensemble](chemprop-ensemble.md) |

All of them take `random_state` (default `42`) or an explicit `random_states` list (not on `MixedEnsemble`), and extra keyword arguments are forwarded to `EnsembleLearner`. `lr_ensemble` and `xgb_ensemble` also take `device`; with `device='cuda'` (or `'cuda:N'`), `lr_ensemble` uses [`ridge_cuml`](ridge-cuml.md) members and `xgb_ensemble` runs XGBoost on the GPU. `'auto'` selects the GPU members when CUDA is available (for `lr_ensemble` also only if RAPIDS cuML is importable) and CPU members otherwise; it never raises. `run_active_learning` passes its own `device` (default `'auto'`) to these ensembles when you select them by name.

## Usage

```bash
learnm8 run data.csv --target Value --learner xgb_ensemble --featurizer morgan --strategy ucb
```

```python
from learnm8 import run_active_learning
from learnm8.learners import XGBEnsemble

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=XGBEnsemble(learning_rates=[0.05, 0.1, 0.2, 0.3], random_state=7),
    featurizer='morgan', strategy='ucb', n_cycles=10,
)
```

## Custom ensembles

`EnsembleLearner` (importable from `learnm8.learners`) combines any list of `Learner` instances. It is not in the string registry, so pass the instance.

```python
from learnm8.learners import (
    EnsembleLearner, RandomForestLearner, XGBoostLearner, GaussianProcessLearner,
)

ensemble = EnsembleLearner(
    learners=[RandomForestLearner(), XGBoostLearner(), GaussianProcessLearner()],
    aggregation_method='median',
    uncertainty_method='mad',
)
```

| Parameter | Default | Description |
|---|---|---|
| `learners` | required | Member learners (at least one) |
| `aggregation_method` | `'mean'` | `'mean'` or `'median'`; unknown values log a warning and use the mean |
| `uncertainty_method` | `'std'` | `'std'`, `'mad'` or `'quantile'`; unknown values log a warning and use `std` |
| `weights` | `None` | One weight per member, must sum to 1; applied only with `aggregation_method='mean'` |
| `enable_parallel_training` | `False` | Accepted but not implemented; members train sequentially |

If any member fails to train or predict, the whole ensemble raises `LearnerError`. Training and prediction cost scale with the number of members.

See also: [Learners overview](overview.md), [Random Forest](random-forest.md), [Customization](../../customization/extending.md).

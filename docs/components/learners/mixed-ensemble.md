# Mixed Ensemble

An ensemble of three different model families: Random Forest, Ridge regression and XGBoost. The diversity of model types gives a different uncertainty signal than seed-only ensembles. Uncertainty is the standard deviation across the three members' predictions (see [Ensembles](ensembles.md)).

- **Registry name:** `mixed_ensemble`
- **Class:** `learnm8.learners.MixedEnsemble`
- **Uncertainty:** yes (std across members)
- **Requires SMILES:** no

## Members

| Device | Random forest | Linear | Boosting |
|---|---|---|---|
| `'cpu'` (default) and `'auto'` | `RandomForestLearner(n_estimators=100)` | `LinearRegressionLearner(alpha=1.0)` | `XGBoostLearner(learning_rate=0.1)` |
| `'cuda'` | [`RfFilLearner`](rf-fil.md)`(n_estimators=100)` | [`RidgeCumlLearner`](ridge-cuml.md)`(alpha=1.0)` | `XGBoostLearner(learning_rate=0.1, device='cuda')` |

Member seeds are `random_state + (0, 81, 314)` for RF, Ridge and XGBoost respectively.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `random_state` | `42` | Base seed for the member seeds |
| `device` | `'cpu'` | `'cpu'`, `'cuda'` or `'auto'` (`'auto'` selects the CPU members) |
| `**kwargs` | | Forwarded to `EnsembleLearner` (`aggregation_method`, `uncertainty_method`, `weights`) |

## Usage

```bash
learnm8 run data.csv --target Value --learner mixed_ensemble --featurizer morgan --strategy ucb
```

```python
from learnm8 import run_active_learning
from learnm8.learners import MixedEnsemble

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=MixedEnsemble(random_state=0),
    featurizer='morgan', strategy='ucb', n_cycles=10,
)
```

## Notes

- Member hyperparameters are fixed; to change them build an `EnsembleLearner` from your own members.
- The `'cuda'` members require RAPIDS cuML and Treelite; see [RF FIL](rf-fil.md) and [Ridge (cuML)](ridge-cuml.md).

See also: [Ensembles](ensembles.md), [Learners overview](overview.md).

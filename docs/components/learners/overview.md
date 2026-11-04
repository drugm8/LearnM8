# Learners Overview

A learner is the surrogate model of the active learning loop. Each cycle it is trained on the compounds labeled so far and predicts every unlabeled compound. It returns a prediction and, if it supports it, an uncertainty estimate; the [acquisition function](../acquisition/overview.md) turns these into the next batch.

Learners are either **feature-based** (they receive the matrix produced by a [featurizer](../featurizers/overview.md)) or **SMILES-native** (Chemprop reads the molecular graph and only optionally uses features). The interface every learner implements, and how to write your own, is documented in [Extending LearnM8](../../customization/extending.md).

## Available learners

| Registry name | Model | Uncertainty | SMILES-native | GPU |
|---|---|---|---|---|
| [`rf`](random-forest.md) | Random Forest | yes (std across trees) | no | no |
| [`gp`](gaussian-process.md) | Gaussian Process (scikit-learn) | yes (posterior std) | no | no |
| [`xgb`](xgboost.md) | XGBoost | no | no | optional |
| [`dt`](decision-tree.md) | Decision Tree | yes (leaf impurity) | no | no |
| [`lr`](linear-regression.md) | Ridge / linear regression | yes (leverage) | no | no |
| [`gpu_gp`](gpu-gp.md) | Exact GP (GPyTorch) | yes (posterior std) | no | optional |
| [`svgp`](svgp.md) | Sparse variational GP (GPyTorch) | yes (variational std) | no | optional |
| [`rf_fil`](rf-fil.md) | Random Forest with RAPIDS FIL inference | yes (std across trees) | no | required |
| [`ridge_cuml`](ridge-cuml.md) | Ridge on RAPIDS cuML | yes (leverage) | no | required |
| [`mlp`](mlp.md) | Feedforward network | no | no | optional |
| [`mc_dropout`](mc-dropout.md) | MLP with Monte Carlo dropout | yes (std over samples) | no | optional |
| [`fastprop`](fastprop.md) | Fastprop network | no | no | optional |
| [`chemprop`](chemprop.md) | Chemprop MPNN | no | yes | optional |
| [`lr_ensemble`](ensembles.md) | Ridge ensemble | yes (std across members) | no | optional |
| [`xgb_ensemble`](ensembles.md) | XGBoost ensemble | yes (std across members) | no | optional |
| [`dt_ensemble`](ensembles.md) | Decision Tree ensemble | yes (std across members) | no | no |
| [`fastprop_ensemble`](ensembles.md) | 3 Fastprop networks | yes (std across members) | no | optional |
| [`mixed_ensemble`](mixed-ensemble.md) | RF + Ridge + XGBoost | yes (std across members) | no | optional |
| [`chemprop_ensemble`](chemprop-ensemble.md) | 3 Chemprop models | yes (std across members) | yes | optional |

"Optional" means the learner uses a GPU when `device` selects one (`'auto'` picks CUDA if available; for `lr_ensemble` and `mixed_ensemble` that also requires RAPIDS cuML, otherwise they use CPU members). "Required" means the learner needs a CUDA GPU and RAPIDS cuML. All 19 learners are always listed; without GPyTorch, creating `gpu_gp` or `svgp` raises `ConfigurationError`, and without RAPIDS cuML, `rf_fil` and `ridge_cuml` raise `LearnerError` when trained.

Uncertainty-based acquisition strategies (`ucb`, `ei`, `pi`, `thompson`, `entropy`) need a learner with uncertainty. Learners without it (`xgb`, `mlp`, `fastprop`, `chemprop`) work with `greedy`, `topk` and `random`; pairing them with an uncertainty strategy raises `ConfigurationError` before cycle 0. If the strategy does not need uncertainty, most learners skip computing it (`mc_dropout` still runs its sampling passes). Override with `--force-uncertainty` / `force_uncertainty=True`.

## Choosing a learner by string or by instance

Pass a registry name to use default hyperparameters:

```bash
learnm8 run data.csv --target Value --learner rf --featurizer morgan
```

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner='rf', featurizer='morgan',
)
```

With a string, `run_active_learning` also passes its `random_state`, `n_jobs` and `device` to the learner when the learner's constructor accepts them. To set hyperparameters, pass an instance; its own constructor arguments are then used as given:

```python
from learnm8.learners import RandomForestLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=RandomForestLearner(n_estimators=500, max_depth=30),
    featurizer='morgan',
)
```

## Listing available learners

```python
from learnm8.api import list_available_learners

print(list_available_learners())
```

The CLI `--learner` choices come from the same list.

## Featurizer requirement

Every learner needs a `featurizer` except `chemprop` and `chemprop_ensemble`, which can run from SMILES alone. Giving them a featurizer switches them to hybrid mode (see [Chemprop](chemprop.md#hybrid-mode)). Tree learners (`rf`, `xgb`, `dt`) load binary fingerprints as `uint8` to save memory.

See also: [Acquisition](../acquisition/overview.md), [Featurizers](../featurizers/overview.md), [Extending LearnM8](../../customization/extending.md).

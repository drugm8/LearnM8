# Extending LearnM8

LearnM8 can be extended with your own learner, acquisition function, oracle or featurizer. Learners and oracles are passed to `run_active_learning()` as instances. Acquisition functions are added to a registry, and featurizers to a config table.

## Custom learner

Subclass `Learner` from `learnm8.core.interfaces`. Three methods are abstract:

| Method | Notes |
|--------|-------|
| `train(features, targets, smiles=None) -> None` | `features` is `(n_samples, n_features)`, `targets` is `(n_samples,)`. |
| `predict(features, smiles=None, *, compute_uncertainty=True) -> (predictions, uncertainties)` | `uncertainties` may be `None`. `compute_uncertainty` is keyword-only and must be in your signature. |
| `get_name() -> str` | Identifier used in logs and results. |

Two optional methods have defaults:

- `supports_uncertainty()` returns `False` by default. Override it to return `True` if `predict()` returns uncertainties.
- `requires_smiles()` returns `False` by default. Override it to return `True` for learners that work from SMILES (Chemprop, Fastprop). The SMILES are then passed in the `smiles` argument.

The cycle sets `compute_uncertainty = force_uncertainty OR acquisition.requires_uncertainty()` once per cycle and passes it to `predict()`. When it is `False` you may skip uncertainty work and return `None` as the second element. A `predict()` without the `compute_uncertainty` keyword raises `TypeError` on the first cycle.

Uncertainty-based acquisition functions (`ucb`, `ei`, `pi`, `thompson`, `entropy`) need a learner whose `supports_uncertainty()` is `True`.

```python
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from learnm8 import run_active_learning
from learnm8.core.interfaces import Learner


class ExtraTreesLearner(Learner):
    def __init__(self, n_estimators: int = 200, random_state: int = 0):
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )

    def train(self, features, targets, smiles=None):
        self.model.fit(features, targets)

    def predict(self, features, smiles=None, *, compute_uncertainty=True):
        if not compute_uncertainty:
            return self.model.predict(features), None
        per_tree = np.stack([t.predict(features) for t in self.model.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)

    def get_name(self) -> str:
        return 'extra_trees'

    def supports_uncertainty(self) -> bool:
        return True


results = run_active_learning(
    compound_pool='pool.csv',
    oracle=None,
    learner=ExtraTreesLearner(),
    target_col='Value',
    featurizer='morgan',
    strategy='ucb',
    n_cycles=5,
)
```

A learner instance is not registered by name, so it cannot be selected from the CLI. To make it available by name, add it to `LEARNER_REGISTRY` in `learnm8/api.py`.

## Custom acquisition function

Subclass `AcquisitionFunction` from `learnm8.acquisition.base`.

- `select(compounds, n_select)` is abstract. `compounds` is a Polars DataFrame with `ID`, `SMILES` and `prediction` columns, plus `uncertainty` when the learner provided it. Return the selected rows.
- `requires_uncertainty()` returns `False` by default. Return `True` if your score uses the `uncertainty` column; the cycle then asks the learner for uncertainties.
- `__init__` is called with `score_direction` (`'higher'` or `'lower'`, stored as `self.score_direction` and `self.maximize`), `featurizer_obj`, `cache_dir`, and the entries of `acquisition_params`. Accept `**kwargs` and call `super().__init__(score_direction=score_direction)`.
- `self._safe_select_top_k(compounds, scores, n_select, ascending=False)` returns the top rows by `scores` with an `acquisition_score` column added. Non-finite scores are ranked last.
- `self.validate_input(compounds, n_select)` checks required columns, NaN predictions, negative uncertainties and duplicate IDs. Call it at the start of `select()`.

```python
from learnm8 import run_active_learning
from learnm8.acquisition import ACQUISITION_REGISTRY
from learnm8.acquisition.base import AcquisitionFunction


class LowerBoundAcquisition(AcquisitionFunction):
    """Rank by prediction - kappa * uncertainty (a conservative estimate)."""

    def __init__(self, score_direction='higher', kappa=1.0, **kwargs):
        super().__init__(score_direction=score_direction)
        self.kappa = kappa

    def requires_uncertainty(self) -> bool:
        return True

    def select(self, compounds, n_select):
        self.validate_input(compounds, n_select)
        pred = compounds['prediction'].to_numpy()
        unc = compounds['uncertainty'].to_numpy()
        sign = 1.0 if self.maximize else -1.0
        scores = sign * pred - self.kappa * unc
        return self._safe_select_top_k(compounds, scores, n_select)


ACQUISITION_REGISTRY['lcb'] = LowerBoundAcquisition

results = run_active_learning(
    compound_pool='pool.csv', oracle=None, learner='rf',
    target_col='Value', featurizer='morgan',
    strategy='lcb', acquisition_params={'kappa': 2.0},
)
```

Strategies are looked up by name in `ACQUISITION_REGISTRY` (`learnm8/acquisition/__init__.py`), so register the class before calling `run_active_learning()`. The name is then valid in `strategy`, `initial_strategy` and `CycleConfig`.

### Optional streaming tier

The streaming path scores the pool chunk by chunk instead of holding every prediction in memory. A strategy opts in by overriding:

- `supports_streaming()` to return `True`
- `score_chunk(predictions, uncertainties, *, global_offset, n_total)` to return higher-is-better scores for one chunk, with `score_direction` already folded in
- `shortlist_size(n_select, pool_size)` (default `n_select`) and `finalize(shortlist, n_select)` (default: the first `n_select` rows), if you select from a wider candidate set than the batch

If you do not override `supports_streaming()`, it returns `False` and the strategy always runs through your in-memory `select()`. With `streaming='auto'` (the default) the cycle uses that path; `streaming='always'` raises `ConfigurationError` for such a strategy.

## Custom oracle

Subclass `Oracle` and implement `measure(compounds, properties)`, which must return the measured values in the input row order. See [Python oracle](../components/oracles/python-oracle.md).

## Custom featurizer

Featurizers are entries in `_FEATURIZER_CONFIG` in `learnm8/features/__init__.py`, each specifying the scikit-fingerprints `cls`, its `defaults`, `storage_dtype`, `feature_type` and `requires_conformers`. `FEATURIZER_REGISTRY` is derived from the config keys. There is no runtime registration, and plain functions cannot be used as featurizers. Contributing a featurizer is described in `CONTRIBUTING.md`.

## See also

- [Learners overview](../components/learners/overview.md)
- [Acquisition overview](../components/acquisition/overview.md)
- [Featurizers overview](../components/featurizers/overview.md)

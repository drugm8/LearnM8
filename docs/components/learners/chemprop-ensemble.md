# Chemprop Ensemble

Three [Chemprop](chemprop.md) models trained with different seeds. Uncertainty is the standard deviation of their predictions (see [Ensembles](ensembles.md)), which gives Chemprop the uncertainty a single model lacks.

- **Registry name:** `chemprop_ensemble`
- **Class:** `learnm8.learners.ensemble.ChempropEnsemble`
- **Uncertainty:** yes (std across the 3 members; `uncertainty_method` can be `'mad'` or `'quantile'`)
- **Requires SMILES:** yes
- **GPU:** optional

## Parameters

It accepts every [`ChempropLearner`](chemprop.md) architecture and training parameter (same names and defaults) and passes them to all members, plus:

| Parameter | Default | Description |
|---|---|---|
| `random_state` | `42` | Base seed; members use `random_state + (0, 81, 314)` (42, 123, 356 by default) |
| `random_states` | `None` | Explicit list of 3 member seeds (overrides `random_state`) |
| `enable_fine_tuning` | `False` | Per-member checkpoint fine-tuning |
| `checkpoint_dir` | `None` | Required when fine-tuning; member `i` writes to `checkpoint_dir/member_i/` |
| `**kwargs` | | Forwarded to `EnsembleLearner` (`aggregation_method`, `uncertainty_method`, `weights`) |

## Usage

```bash
learnm8 run data.csv --target Value --learner chemprop_ensemble --strategy ucb --device cuda
```

```python
from learnm8 import run_active_learning
from learnm8.learners.ensemble import ChempropEnsemble

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=ChempropEnsemble(depth=4, max_epochs=40),
    featurizer=None, strategy='ucb', n_cycles=10,
)
```

## Notes

- Like `chemprop`, a featurizer is optional. If given, its features are passed to every member as extra descriptors (hybrid mode, see [Chemprop](chemprop.md#hybrid-mode)).
- Training cost is three Chemprop trainings per cycle.
- Fine-tuning across cycles is described in [Chemprop fine-tuning](../../advanced/chemprop-fine-tuning.md).

See also: [Ensembles](ensembles.md), [Learners overview](overview.md).

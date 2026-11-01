# Chemprop Fine-Tuning

By default Chemprop trains a fresh model every cycle. With fine-tuning enabled, each Chemprop model saves a checkpoint after training and the next cycle starts from that checkpoint instead of from random weights, then trains on the enlarged labeled set (still bounded by `max_epochs` and early stopping). If a checkpoint is missing or incompatible with the current architecture, the model logs a warning and trains from scratch.

Fine-tuning applies to `ChempropLearner` (`learner='chemprop'`) and `ChempropEnsemble` (`learner='chemprop_ensemble'`). It is available only through the Python API; it has no CLI flag and is not read from configuration files.

## String shortcut

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.py:score',
    learner='chemprop_ensemble',
    target_col='affinity',
    n_cycles=20,
    batch_fraction=0.005,
    enable_chemprop_fine_tuning=True,
    output_dir='results/chemprop_ft',
)
```

`enable_chemprop_fine_tuning=True` creates the learner with `enable_fine_tuning=True` and stores checkpoints under `<output_dir>/.checkpoints/chemprop/`. For the ensemble, each member uses its own subdirectory (`member_0/`, `member_1/`, ...) containing `cycle_N.ckpt` files.

The flag works with both shortcuts, `'chemprop'` and `'chemprop_ensemble'`. It is ignored (with a warning) when `learner` is already an instance; configure fine-tuning on the instance instead, as shown below.

## Learner instance

Configure fine-tuning on the instance when you need a custom checkpoint location or other Chemprop parameters:

```python
from pathlib import Path
from learnm8 import run_active_learning
from learnm8.learners.torch import ChempropLearner

learner = ChempropLearner(
    enable_fine_tuning=True,
    checkpoint_dir=Path('checkpoints/chemprop'),
    max_epochs=50,
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.py:score',
    learner=learner,
    target_col='affinity',
    n_cycles=20,
    batch_fraction=0.005,
)
```

`ChempropEnsemble` (from `learnm8.learners.ensemble`) takes the same two arguments.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `enable_fine_tuning` | `False` | Save a checkpoint after each training and resume from it next cycle |
| `checkpoint_dir` | `None` | Where checkpoints are written; required when fine-tuning is enabled (created if missing) |

See [Chemprop](../components/learners/chemprop.md) and [Chemprop Ensemble](../components/learners/chemprop-ensemble.md) for the remaining parameters.

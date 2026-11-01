# Pruning

Pruning permanently removes poorly predicted compounds from the unlabeled pool, so later cycles predict for and select from a smaller design space. Pruned compounds get `status == 'pruned'` and a `pruned_cycle` in the results.

There is one strategy, `'score'`. Each cycle that has pruning enabled, after the model has predicted the unlabeled pool and before selection, it removes the `pruning_fraction` of the current pool with the worst predictions. "Worst" follows `score_direction`: the lowest predictions for `'higher'`, the highest for `'lower'`. Pruning therefore compounds geometrically over cycles: a fraction of what is *still unlabeled* is removed each time. Pruning uses only predictions, not uncertainty, and at least one compound is always kept.

Pruning is irreversible. A compound pruned because of an early, poor model is never selected later, so prune conservatively when the model is still trained on few compounds.

## Run-level pruning

`pruning_fraction` must be in `[0.0, 0.9]`. Setting `pruning_fraction` alone enables the `'score'` strategy.

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='oracle.py:score',
    learner='gp',
    target_col='affinity',
    featurizer='morgan',
    n_cycles=20,
    pruning_strategy='score',
    pruning_fraction=0.1,
    score_direction='higher',
)
```

```bash
learnm8 run large_library.csv oracle.py:score --target affinity --learner gp \
  --featurizer morgan --n-cycles 20 --pruning-strategy score --pruning-fraction 0.1
```

The top-level values apply to every cycle that has an acquisition step (cycles 1 and later).

## Per-phase pruning

With `CycleConfig`, `pruning_strategy` and `pruning_params` can differ per phase. A block that sets its own `pruning_strategy` overrides the top-level pruning settings; a block that does not inherits them.

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='oracle.py:score',
    learner='rf',
    target_col='affinity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('ucb', n_cycles=5, batch_fraction=0.01),             # no pruning
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01,
                    pruning_strategy='score',
                    pruning_params={'pruning_fraction': 0.1}),
        CycleConfig('greedy', n_cycles=10, batch_fraction=0.01,
                    pruning_strategy='score',
                    pruning_params={'pruning_fraction': 0.3}),
    ],
)
```

`pruning_params` takes `pruning_fraction` only. The optimization direction always comes from the run-level `score_direction`; a `score_direction` key inside `pruning_params` is overwritten.

## See also

- [Building Custom Cycles](../tutorials/building-custom-cycles.md)
- [Configuration Files](configuration-files.md)

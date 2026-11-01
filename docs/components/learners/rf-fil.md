# RF FIL

Random forest trained on CPU with scikit-learn (float32) and converted to a RAPIDS Forest Inference Library (FIL) model via Treelite, so prediction runs on the GPU. Training is identical in kind to [Random Forest](random-forest.md); only inference moves to the GPU.

- **Registry name:** `rf_fil`
- **Class:** `learnm8.learners.RfFilLearner`
- **Uncertainty:** yes. Population standard deviation (`ddof=0`) across per-tree predictions from `predict_per_tree()`; the prediction is their mean. Ordinal, not a calibrated interval.
- **Requires SMILES:** no
- **Requirements:** a CUDA GPU, RAPIDS cuML >= 25.04 and Treelite >= 4.6 (`< 5.0`). The learner can be constructed without them; `train()` raises `LearnerError` if they are missing.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_estimators` | `100` | Number of trees |
| `max_depth` | `None` | Maximum tree depth |
| `min_samples_split` | `2` | Minimum samples to split an internal node |
| `min_samples_leaf` | `1` | Minimum samples at a leaf |
| `max_features` | `'sqrt'` | Features considered per split |
| `random_state` | `42` | Random seed |
| `n_jobs` | `-1` | Parallel jobs for CPU training |

## Usage

```bash
learnm8 run data.csv --target Value --learner rf_fil --featurizer morgan --strategy ucb
```

```python
from learnm8 import run_active_learning
from learnm8.learners import RfFilLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=RfFilLearner(n_estimators=500),
    featurizer='morgan', strategy='ucb', n_cycles=10,
)
```

## Notes

- Training in float32 gives slightly different forests from the CPU `rf` learner at the same seed.
- Without uncertainty (for example `greedy`), prediction uses the fused FIL `predict` and avoids the `(n_samples, n_trees)` per-tree array.
- Prediction is retried in smaller chunks on GPU out-of-memory errors.

See also: [Learners overview](overview.md), [Random Forest](random-forest.md).

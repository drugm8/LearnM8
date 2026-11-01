# MC Dropout

An MLP whose dropout layers stay active at prediction time. Each prediction runs `n_dropout_samples` stochastic forward passes; the mean is the prediction and the standard deviation across passes is the uncertainty.

- **Registry name:** `mc_dropout`
- **Class:** `learnm8.learners.MCDropoutLearner`
- **Uncertainty:** yes. Standard deviation over the dropout samples (`np.std`, `ddof=0`).
- **Requires SMILES:** no
- **GPU:** optional (`device='auto'` uses CUDA when available)

## Parameters

Model-specific:

| Parameter | Default | Description |
|---|---|---|
| `hidden_sizes` | `(256, 128)` | Hidden layer widths |
| `dropout_rate` | `0.2` | Dropout probability (applied after every hidden layer) |
| `n_dropout_samples` | `100` | Stochastic forward passes per prediction |
| `activation` | `'relu'` | `'relu'`, `'tanh'`, `'gelu'` or `'leaky_relu'` |
| `batch_norm` | `False` | Batch normalization (kept in eval mode during MC sampling) |
| `predict_batch_size` | `None` | Prediction chunk size override |

It also accepts the shared `TorchLearner` keyword arguments (`device`, `batch_size=1024`, `max_epochs=100`, `learning_rate=0.001`, `early_stopping_patience=10`, `random_state=42`, `remove_zero_variance=True`).

## Usage

```bash
learnm8 run data.csv --target Value --learner mc_dropout --featurizer morgan --strategy ucb
```

```python
from learnm8 import run_active_learning
from learnm8.learners import MCDropoutLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=MCDropoutLearner(dropout_rate=0.3, n_dropout_samples=50),
    featurizer='morgan', strategy='ucb', n_cycles=10,
)
```

## Notes

- The stochastic passes always run, even when the acquisition strategy does not need uncertainty: the mean prediction is itself the average over dropout samples, so skipping them would change the prediction. Only the standard deviation is discarded.
- Prediction is chunked and retried with smaller chunks on GPU out-of-memory errors.
- Cost per prediction grows linearly with `n_dropout_samples`.

See also: [Learners overview](overview.md), [MLP](mlp.md).

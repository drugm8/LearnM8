# Fastprop

Feedforward network from the `fastprop` library, trained with PyTorch Lightning. It works on any precomputed feature matrix and standardizes features and targets internally. A single model provides no uncertainty; use `fastprop_ensemble` for that.

- **Registry name:** `fastprop` (ensemble: `fastprop_ensemble`, see [Ensembles](ensembles.md))
- **Class:** `learnm8.learners.FastpropLearner`
- **Uncertainty:** no
- **Requires SMILES:** no
- **GPU:** optional (`device='auto'` uses CUDA when available)

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `fnn_layers` | `2` | Hidden layers (must be >= 1) |
| `hidden_size` | `300` | Width of each hidden layer |
| `max_epochs` | `50` | Maximum epochs |
| `learning_rate` | `0.0001` | Learning rate |
| `batch_size` | `32` | Training batch size |
| `predict_batch_size` | `None` | Prediction batch size (`None` = 4 x `batch_size`) |
| `precision` | `'auto'` | `'auto'` (16-mixed on GPU, 32-true on CPU), `'16-mixed'`, `'32-true'`, `'bf16-mixed'`, `'32'`, `'16'` |
| `pin_memory` | `False` | Pinned host memory |
| `clamp_input` | `True` | Clamp (winsorize) inputs for robustness |
| `early_stopping_patience` | `5` | Epochs without improvement before stopping |
| `val_fraction` | `0.1` | Validation fraction for early stopping (`0` <= value < `1`) |
| `random_state` | `42` | Random seed |
| `device` | `'auto'` | `'auto'`, `'cpu'`, `'cuda'` |
| `enable_aggressive_gc` | `True` | Free GPU memory after training and prediction |

## Usage

```bash
learnm8 run data.csv --target Value --learner fastprop --featurizer rdkit_2d_descriptors --strategy greedy
```

```python
from learnm8 import run_active_learning
from learnm8.learners import FastpropLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=FastpropLearner(fnn_layers=3, hidden_size=512),
    featurizer='rdkit_2d_descriptors', strategy='greedy', n_cycles=10,
)
```

## Notes

- Early stopping uses a validation split only when there are at least `max(5, 5 / val_fraction)` training samples; below that, no validation split is made.
- Invalid `fnn_layers`, `val_fraction` or `precision` values raise `ConfigurationError`.

See also: [Learners overview](overview.md), [MLP](mlp.md), [Ensembles](ensembles.md).

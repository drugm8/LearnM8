# MLP

Feedforward neural network (PyTorch) with configurable hidden layers. A single network, so it provides no uncertainty; use [MC Dropout](mc-dropout.md) when you need uncertainty from a neural model.

- **Registry name:** `mlp`
- **Class:** `learnm8.learners.MLPLearner`
- **Uncertainty:** no
- **Requires SMILES:** no
- **GPU:** optional (`device='auto'` uses CUDA when available)

## Parameters

Model-specific:

| Parameter | Default | Description |
|---|---|---|
| `hidden_sizes` | `(512, 256, 128)` | Hidden layer widths |
| `activation` | `'relu'` | `'relu'`, `'tanh'`, `'gelu'` or `'leaky_relu'` (unknown names fall back to ReLU) |
| `dropout_rate` | `0.2` | Dropout after each hidden layer (`0` disables) |
| `batch_norm` | `True` | Batch normalization after each linear layer |

Shared with all `TorchLearner` models (passed as keyword arguments):

| Parameter | Default | Description |
|---|---|---|
| `device` | `'auto'` | `'auto'`, `'cpu'`, `'cuda'` |
| `batch_size` | `1024` | Training batch size |
| `max_epochs` | `100` | Maximum epochs |
| `learning_rate` | `0.001` | Optimizer learning rate |
| `early_stopping_patience` | `10` | Epochs without improvement before stopping |
| `random_state` | `42` | Random seed |
| `remove_zero_variance` | `True` | Drop zero-variance features |

## Usage

```bash
learnm8 run data.csv --target Value --learner mlp --featurizer morgan --strategy greedy --device cuda
```

```python
from learnm8 import run_active_learning
from learnm8.learners import MLPLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=MLPLearner(hidden_sizes=(256, 128), dropout_rate=0.3, max_epochs=200),
    featurizer='morgan', strategy='greedy', n_cycles=10,
)
```

## Notes

- Features are loaded as float32.
- Weights use He initialization for ReLU and Xavier otherwise.

See also: [Learners overview](overview.md), [MC Dropout](mc-dropout.md), [Fastprop](fastprop.md).

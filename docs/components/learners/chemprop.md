# Chemprop

Directed message-passing neural network (Chemprop v2) that learns from the molecular graph. It reads SMILES directly, so a featurizer is optional.

- **Registry name:** `chemprop` (3-member ensemble with uncertainty: `chemprop_ensemble`, see [Chemprop Ensemble](chemprop-ensemble.md))
- **Class:** `learnm8.learners.ChempropLearner`
- **Uncertainty:** no (single model)
- **Requires SMILES:** yes (`requires_smiles()` is `True`)
- **GPU:** optional; accelerator and device are handled by PyTorch Lightning

## Parameters

Architecture:

| Parameter | Default | Description |
|---|---|---|
| `message_hidden_dim` | `300` | Hidden size of the message-passing layers |
| `depth` | `3` | Message-passing steps |
| `aggregation` | `'mean'` | Atom-to-molecule aggregation: `mean`, `sum`, `norm` |
| `atom_messages` | `False` | Pass messages on atoms instead of bonds |
| `batch_norm` | `False` | Batch normalization |
| `message_bias` | `False` | Bias in the message-passing layers |
| `ffn_hidden_dim` | `300` | Hidden size of the feed-forward head |
| `ffn_num_layers` | `1` | Feed-forward layers |
| `dropout` | `0.0` | Dropout probability |

Training:

| Parameter | Default | Description |
|---|---|---|
| `max_epochs` | `50` | Maximum epochs |
| `batch_size` | `32` | Training batch size |
| `predict_batch_size` | `None` | Prediction batch size (`None` = 4 x `batch_size`) |
| `precision` | `'auto'` | `'auto'`, `'16-mixed'`, `'32-true'`, `'bf16-mixed'`, `'32'`, `'16'` |
| `pin_memory` | `True` | Pinned host memory for GPU transfers |
| `learning_rate` | `1e-4` | Learning rate |
| `early_stopping` | `True` | Stop on validation loss |
| `early_stopping_patience` | `10` | Epochs without improvement |
| `early_stopping_min_delta` | `0.0` | Minimum improvement |
| `val_fraction` | `0.1` | Validation fraction |
| `random_state` | `42` | Random seed |
| `accelerator` | `'auto'` | Lightning accelerator |
| `device` | `'auto'` | If not `'auto'`, overrides `accelerator` (for example `'cuda'`, `'cpu'`) |
| `enable_aggressive_gc` | `True` | Free GPU memory after training and prediction |
| `enable_fine_tuning`, `checkpoint_dir` | `False`, `None` | Checkpoint-based fine-tuning; `checkpoint_dir` is required when enabled |

## Usage

Graph only, no featurizer:

```bash
learnm8 run data.csv --target Value --learner chemprop --strategy greedy --device cuda
```

```python
from learnm8 import run_active_learning
from learnm8.learners import ChempropLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=ChempropLearner(depth=5, message_hidden_dim=500),
    featurizer=None, strategy='greedy', n_cycles=10,
)
```

## Hybrid mode

If you also pass a featurizer, its feature vectors are used as extra descriptors (`x_d`) and concatenated to the graph representation before the feed-forward head (`ffn_hidden_dim` input = `message_hidden_dim` + number of features).

```bash
learnm8 run data.csv --target Value --learner chemprop --featurizer rdkit_2d_descriptors --strategy greedy
```

```python
results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner='chemprop', featurizer='rdkit_2d_descriptors', n_cycles=10,
)
```

## Notes

- Fine-tuning across cycles is described in [Chemprop fine-tuning](../../advanced/chemprop-fine-tuning.md).
- Training and prediction without SMILES raise an error.

See also: [Learners overview](overview.md), [Chemprop Ensemble](chemprop-ensemble.md).

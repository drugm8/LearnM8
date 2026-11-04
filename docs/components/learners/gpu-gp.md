# GPU Gaussian Process

Exact Gaussian process regression in GPyTorch, with optional CUDA acceleration. Prediction variances use GPyTorch's fast predictive variance (LOVE) setting and are computed in chunks to bound GPU memory.

- **Registry name:** `gpu_gp`
- **Class:** `learnm8.learners.gpytorch.GPyTorchGPLearner`
- **Uncertainty:** yes. Predictive standard deviation of the GP posterior, rescaled to target units. Targets are standardized internally.
- **Requires SMILES:** no
- **Requirements:** GPyTorch; GAUCHE for the Tanimoto kernel. A GPU is optional (`device='auto'` uses CUDA when available, else CPU). The learner is always listed, but creating it without GPyTorch raises `ConfigurationError`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `kernel` | `'auto'` | `'auto'` (Tanimoto for binary features, else RBF), `'tanimoto'`, `'rbf'`, or a `gpytorch.kernels.Kernel` instance (wrapped in a `ScaleKernel` if needed) |
| `alpha` | `0.1` | Initial noise of the Gaussian likelihood |
| `max_train_size` | `10000` | Larger training sets raise `LearnerError` |
| `n_iterations` | `None` | Adam iterations; `None` = 100 if fewer than 1000 training samples, else 200 |
| `device` | `'auto'` | `'auto'`, `'cpu'`, `'cuda'`, `'cuda:N'` |
| `random_state` | `42` | Random seed |
| `remove_zero_variance` | `True` | Drop zero-variance features |
| `enable_aggressive_gc` | `True` | `gc.collect()` and CUDA cache cleanup after CUDA operations |
| `predict_chunk_size` | `10000` | Compounds per prediction chunk |

## Usage

```bash
learnm8 run data.csv --target Value --learner gpu_gp --featurizer morgan --strategy ei --device cuda
```

```python
from learnm8 import run_active_learning
from learnm8.learners.gpytorch import GPyTorchGPLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=GPyTorchGPLearner(kernel='tanimoto', n_iterations=150, device='cuda'),
    featurizer='morgan', strategy='ei', n_cycles=10,
)
```

## Notes

- Exact inference scales cubically in time and quadratically in memory with the training-set size; a warning is logged above 5000 training samples. For larger labeled sets use [SVGP](svgp.md).
- When the acquisition strategy does not need uncertainty, the variance computation is skipped.
- The CPU equivalent is [Gaussian Process](gaussian-process.md).

See also: [Learners overview](overview.md).

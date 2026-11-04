# SVGP

Stochastic variational Gaussian process (GPyTorch `ApproximateGP`). It approximates the GP with a fixed set of inducing points and trains with minibatches, so memory is independent of the training-set size.

- **Registry name:** `svgp`
- **Class:** `learnm8.learners.gpytorch.SVGPLearner`
- **Uncertainty:** yes. Predictive standard deviation of the variational posterior, rescaled to target units. The variational posterior systematically underestimates uncertainty.
- **Requires SMILES:** no
- **Requirements:** GPyTorch; GAUCHE for the Tanimoto kernel. A GPU is optional (`device='auto'`). The learner is always listed, but creating it without GPyTorch raises `ConfigurationError`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `kernel` | `'auto'` | `'auto'` (Tanimoto for binary features, else RBF), `'tanimoto'`, `'rbf'`, or a `gpytorch.kernels.Kernel` instance |
| `alpha` | `0.1` | Initial noise of the Gaussian likelihood |
| `n_inducing` | `512` | Number of inducing points (clamped to the training-set size) |
| `batch_size` | `256` | Minibatch size |
| `n_epochs` | `50` | Maximum epochs |
| `learning_rate` | `0.01` | Adam learning rate |
| `early_stopping_patience` | `10` | Stop after this many epochs without ELBO improvement |
| `device` | `'auto'` | `'auto'`, `'cpu'`, `'cuda'`, `'cuda:N'` |
| `random_state` | `42` | Random seed |
| `remove_zero_variance` | `True` | Drop zero-variance features |
| `enable_aggressive_gc` | `True` | `gc.collect()` and CUDA cache cleanup after CUDA operations |
| `predict_chunk_size` | `10000` | Compounds per prediction chunk |

## Usage

```bash
learnm8 run data.csv --target Value --learner svgp --featurizer morgan --strategy ucb --device cuda
```

```python
from learnm8 import run_active_learning
from learnm8.learners.gpytorch import SVGPLearner

results = run_active_learning(
    compound_pool='data.csv', oracle=None, target_col='Value',
    learner=SVGPLearner(n_inducing=1024, n_epochs=100, device='cuda'),
    featurizer='morgan', strategy='ucb', n_cycles=10,
)
```

## Notes

- Inducing points are a random subset of the training data and are not optimized by gradient descent, which keeps binary fingerprints valid inputs for the Tanimoto kernel.
- Training cost per minibatch is O(M^2 B) for M inducing points and batch size B.
- Targets are standardized internally; the kernel and likelihood are trained with Adam.

See also: [GPU Gaussian Process](gpu-gp.md) (exact), [Learners overview](overview.md).

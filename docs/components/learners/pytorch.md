# PyTorch Models

LearnM8 provides three PyTorch-based neural network learners for molecular property prediction. These models leverage GPU acceleration when available and offer various approaches to uncertainty quantification.

All PyTorch learners require the optional `torch` dependency:

```bash
pip install torch
```

## MLPLearner

Standard feedforward neural network with configurable architecture for molecular property prediction.

### Overview

MLPLearner provides a multi-layer perceptron with customizable hidden layers, activation functions, and regularization. The default architecture (512-256-128) works well for most molecular property prediction tasks with traditional fingerprints or descriptors.

**When to use:**

- Large datasets (>10,000 compounds) where neural networks excel
- Complex non-linear relationships in molecular properties
- When GPU acceleration is available
- Standard supervised learning without uncertainty requirements

**Key characteristics:**

- No uncertainty quantification (single forward pass)
- Fast inference after training
- Configurable architecture depth and width
- Batch normalization for training stability

### Parameters

All parameters with GPU considerations:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_sizes` | `Tuple[int, ...]` | `(512, 256, 128)` | Hidden layer sizes. Larger layers increase capacity but require more GPU memory. |
| `activation` | str | `'relu'` | Activation function: `'relu'`, `'tanh'`, `'gelu'`, `'leaky_relu'`. ReLU is fastest. |
| `dropout_rate` | float | `0.2` | Dropout rate for regularization. Higher values (0.3-0.5) reduce overfitting but may underfit. |
| `batch_norm` | bool | `True` | Enable batch normalization. Improves training stability, minimal performance cost. |
| `learning_rate` | float | `0.001` | Learning rate for optimizer. Decrease if training is unstable. |
| `max_epochs` | int | `100` | Maximum training epochs. Monitor loss curves to optimize. |
| `batch_size` | int | `32` | Batch size for training. Larger batches (128-256) utilize GPU better but need more memory. |
| `device` | str | `'auto'` | Device: `'auto'`, `'cpu'`, `'cuda'`. Auto-detects GPU availability. |
| `random_state` | int | `42` | Random seed for reproducibility. |

**GPU Performance:**

- Training speedup: 5-20x on GPU vs CPU for large datasets
- Inference speedup: 10-50x on GPU vs CPU
- Memory requirements scale with `hidden_sizes` and `batch_size`
- Use larger `batch_size` (128-512) on GPU for optimal throughput

### Uncertainty Support

MLPLearner does **not** provide uncertainty estimates. Each prediction is a single forward pass through the network.

For uncertainty quantification with neural networks, use **MCDropoutLearner** instead.

### Examples

**CLI Usage:**

```bash
learnm8 run compounds.csv --target Activity \
  --learner mlp \
  --featurizer morgan \
  --n-cycles 10
```

**Python API:**

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='mlp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)
```

**Custom Architecture:**

```python
from learnm8.learners.torch import MLPLearner
from learnm8 import run_active_learning

custom_mlp = MLPLearner(
    hidden_sizes=(1024, 512, 256, 128),
    activation='gelu',
    dropout_rate=0.3,
    batch_norm=True,
    learning_rate=0.0005,
    max_epochs=150,
    batch_size=128
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=custom_mlp,
    target_col='Activity',
    featurizer='descriptors',
    n_cycles=10
)
```

---

## MCDropoutLearner

Multi-layer perceptron with Monte Carlo Dropout for uncertainty estimation through stochastic forward passes.

### Overview

MCDropoutLearner extends standard MLP with uncertainty quantification using Monte Carlo Dropout. During prediction, dropout is kept enabled and multiple forward passes generate an ensemble of predictions. The variance across these predictions provides uncertainty estimates.

**When to use:**

- When uncertainty quantification is required for active learning
- Large datasets where standard Gaussian Process would be too slow
- GPU is available (100 forward passes benefit significantly from parallelization)
- Uncertainty-based acquisition strategies (UCB, Thompson Sampling, EI)

**Key characteristics:**

- Provides uncertainty estimates through stochastic sampling
- Prediction time is ~100x slower than standard MLP (100 forward passes)
- GPU acceleration critical for reasonable inference speed
- Quality of uncertainty depends on dropout rate and number of samples

### Parameters

All parameters with uncertainty considerations:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_sizes` | `Tuple[int, ...]` | `(256, 128)` | Hidden layer sizes. Smaller than MLP default for faster MC sampling. |
| `dropout_rate` | float | `0.2` | Dropout rate for uncertainty. Higher rates (0.3-0.5) increase uncertainty magnitude. Critical parameter. |
| `n_dropout_samples` | int | `100` | Number of forward passes for uncertainty. More samples = better estimates but slower. |
| `activation` | str | `'relu'` | Activation function: `'relu'`, `'tanh'`, `'gelu'`, `'leaky_relu'`. |
| `batch_norm` | bool | `True` | Enable batch normalization. Helps training stability. |
| `learning_rate` | float | `0.001` | Learning rate for optimizer. |
| `max_epochs` | int | `100` | Maximum training epochs. |
| `batch_size` | int | `32` | Batch size for training and prediction. |
| `device` | str | `'auto'` | Device: `'auto'`, `'cpu'`, `'cuda'`. GPU strongly recommended. |
| `random_state` | int | `42` | Random seed for reproducibility. |

**Uncertainty Tuning:**

- `dropout_rate` controls uncertainty magnitude: higher = more uncertain
- `n_dropout_samples` controls estimate quality: 50-200 typical range
- Too high dropout (>0.5) can hurt prediction accuracy
- Too few samples (<50) produce noisy uncertainty estimates

**GPU Performance:**

- Prediction with 100 samples: 50-100x faster on GPU
- CPU prediction can become bottleneck in active learning cycles
- Consider reducing `n_dropout_samples` to 50 on CPU

### Uncertainty Support

MCDropoutLearner **provides uncertainty estimates** via Monte Carlo sampling.

The returned uncertainty is the standard deviation across `n_dropout_samples` forward passes with dropout enabled.

### Examples

**CLI Usage:**

```bash
learnm8 run compounds.csv --target Activity \
  --learner mc_dropout \
  --featurizer morgan \
  --cycles "random:0.01 ucb:0.005*5" \
  --n-cycles 10
```

**Python API:**

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='mc_dropout',
    target_col='Activity',
    featurizer='morgan',
    cycles=[('random', 0.01), ('ucb', 0.005)]
)
```

**Custom Configuration with Tuned Uncertainty:**

```python
from learnm8.learners.torch import MCDropoutLearner
from learnm8 import run_active_learning

mc_learner = MCDropoutLearner(
    hidden_sizes=(512, 256, 128),
    dropout_rate=0.3,
    n_dropout_samples=100,
    activation='gelu',
    batch_norm=True,
    learning_rate=0.001,
    max_epochs=100,
    batch_size=64
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=mc_learner,
    target_col='Activity',
    featurizer='descriptors',
    cycles=[('random', 0.01), ('ucb', 0.005)],
    acquisition_params={'beta': 2.0}
)
```

---

## FastpropLearner

PyTorch Lightning-based feedforward neural network using the fastprop library with automatic feature scaling and early stopping.

### Overview

FastpropLearner provides a production-ready neural network implementation using PyTorch Lightning. It handles feature normalization automatically, includes early stopping to prevent overfitting, and supports input clamping (winsorization) for robustness. Works with any pre-computed molecular features.

**When to use:**

- Production workflows requiring robust training pipelines
- When automatic feature scaling is desired
- Large datasets with potential outliers (input clamping helps)
- PyTorch Lightning features needed (distributed training, logging)

**Key characteristics:**

- Automatic feature and target standardization
- Early stopping prevents overfitting without manual tuning
- Input clamping (winsorization) for outlier robustness
- No uncertainty estimates (single model)
- Aggressive GPU memory cleanup for active learning compatibility

### Parameters

All parameters with performance notes:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fnn_layers` | int | `2` | Number of hidden layers: 0=linear, 2=standard, 3+=deep. More layers = more capacity. |
| `hidden_size` | int | `1800` | Hidden layer size. Fastprop's recommended default. Scales GPU memory linearly. |
| `max_epochs` | int | `50` | Maximum training epochs. Early stopping typically stops before max. |
| `learning_rate` | float | `0.0001` | Learning rate. Fastprop uses conservative default. |
| `batch_size` | int | `32` | Batch size for training and prediction. Larger batches (128-256) improve GPU utilization. |
| `clamp_input` | bool | `True` | Apply winsorization to inputs. Recommended for robustness. Minimal performance cost. |
| `early_stopping_patience` | int | `5` | Epochs to wait for improvement. Lower = faster training, risk of early stopping. |
| `random_state` | int | `42` | Random seed for reproducibility. |
| `device` | str | `'auto'` | Device: `'auto'`, `'cpu'`, `'cuda'`. Auto-detects GPU. |
| `enable_aggressive_gc` | bool | `True` | Enable GPU memory cleanup after training/prediction. Recommended for active learning. |

**Performance Considerations:**

- `hidden_size=1800` is fastprop's optimized default for molecular descriptors
- Early stopping reduces total training time (typically stops at 20-30 epochs)
- `enable_aggressive_gc=True` prevents GPU memory accumulation in multi-cycle active learning
- `clamp_input=True` adds minimal overhead but improves robustness

**GPU Memory Management:**

- Aggressive garbage collection runs after train() and predict()
- Critical for active learning with many cycles (prevents memory leaks)
- Disable with `enable_aggressive_gc=False` if GPU memory is abundant

### Uncertainty Support

FastpropLearner does **not** provide uncertainty estimates.

For uncertainty with fastprop architecture, use **FastpropEnsemble** (see [Ensembles documentation](ensembles.md)).

### Examples

**CLI Usage:**

```bash
learnm8 run compounds.csv --target Activity \
  --learner fastprop \
  --featurizer descriptors \
  --n-cycles 10
```

**Python API:**

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='fastprop',
    target_col='Activity',
    featurizer='descriptors',
    n_cycles=10
)
```

**Custom Configuration:**

```python
from learnm8.learners.torch import FastpropLearner
from learnm8 import run_active_learning

fastprop = FastpropLearner(
    fnn_layers=3,
    hidden_size=2048,
    max_epochs=100,
    learning_rate=0.0001,
    batch_size=128,
    clamp_input=True,
    early_stopping_patience=10,
    enable_aggressive_gc=True
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=fastprop,
    target_col='Activity',
    featurizer='descriptors',
    n_cycles=15
)
```

---

## Comparison Table

| Learner | Uncertainty | GPU Benefit | Best Featurizer | Typical Use Case |
|---------|-------------|-------------|-----------------|------------------|
| **MLPLearner** | No | 5-20x training, 10-50x inference | morgan, descriptors | Large datasets, pure exploitation |
| **MCDropoutLearner** | Yes | 50-100x inference critical | morgan, descriptors | Active learning with uncertainty |
| **FastpropLearner** | No | 5-15x training | descriptors | Production pipelines, robustness |

**Choosing between PyTorch learners:**

- **Need uncertainty?** → MCDropoutLearner
- **Production robustness?** → FastpropLearner (automatic scaling, early stopping)
- **Maximum speed with GPU?** → MLPLearner
- **CPU-only system?** → Consider Random Forest or XGBoost instead (better CPU performance)

**GPU Requirements:**

- All PyTorch learners benefit significantly from GPU acceleration
- MCDropoutLearner especially needs GPU for reasonable inference speed
- Minimum 4GB GPU memory for typical molecular datasets
- Larger datasets (>100k compounds) benefit from 8GB+ GPU memory

# Graph Neural Networks

LearnM8 provides state-of-the-art graph neural network learners through the Chemprop library. These models work directly with SMILES strings using message-passing neural networks (MPNNs) to learn from molecular structure, eliminating the need for traditional featurizers in most cases.

## Chemprop Overview

**What is Chemprop?**

Chemprop implements message-passing neural networks (MPNNs) that learn molecular representations directly from graph structure. Instead of converting molecules to fixed fingerprints, Chemprop learns optimal molecular embeddings during training.

**Key advantages:**
- Works directly with SMILES strings (no featurizer needed)
- Learns task-specific molecular representations
- State-of-the-art performance on molecular property prediction
- Can combine graph features with traditional descriptors (hybrid mode)

**How it works:**
1. Converts SMILES to molecular graph (atoms as nodes, bonds as edges)
2. Passes messages along bonds for multiple steps (depth parameter)
3. Aggregates node representations into molecular embedding
4. Feeds embedding through feedforward network for prediction

**Optional dependency:**

```bash
pip install chemprop
```

Chemprop requires PyTorch and PyTorch Lightning. GPU recommended but not required.

---

## ChempropLearner

Single Chemprop message-passing neural network for molecular property prediction.

### Overview

ChempropLearner provides a single MPNN model that works directly with SMILES strings. It offers fast training and inference but does not provide uncertainty estimates. For uncertainty quantification, use **ChempropEnsemble** instead.

**When to use:**
- When graph-based learning is desired over fingerprints
- Fast training/inference is priority over uncertainty
- Dataset size >1000 compounds (Chemprop needs data to learn representations)
- Pure exploitation strategies (greedy, topk)

**Key characteristics:**
- No uncertainty quantification (single model)
- Works directly with SMILES (no featurizer needed)
- Optional hybrid mode (graph + descriptors)
- Early stopping with automatic validation split
- Fine-tuning support for active learning

### Parameters

#### Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message_hidden_dim` | int | `300` | Hidden dimension of message vectors. Larger = more capacity. 300-500 typical. |
| `depth` | int | `3` | Number of message passing steps. Larger graphs may benefit from depth 4-5. |
| `aggregation` | str | `'mean'` | Aggregation mode: `'mean'`, `'sum'`, `'norm'`. Mean is most common. |
| `atom_messages` | bool | `False` | Pass messages on atoms vs bonds. Bond messages (False) typically better. |
| `batch_norm` | bool | `False` | Enable batch normalization. Can help training stability. |
| `message_bias` | bool | `False` | Add bias to message passing layers. |
| `ffn_hidden_dim` | int | `300` | Feedforward network hidden dimension after aggregation. |
| `ffn_num_layers` | int | `1` | Number of FFN layers. More layers = more capacity. |
| `dropout` | float | `0.0` | Dropout probability. Increase (0.1-0.3) for regularization on small datasets. |

**Architecture tuning:**
- Increase `depth` to 4-5 for larger, more complex molecules
- Increase `message_hidden_dim` to 500 for richer datasets
- Use `dropout=0.2` if overfitting on small datasets (<5000 compounds)

#### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epochs` | int | `50` | Maximum training epochs. Early stopping typically stops before max. |
| `batch_size` | int | `32` | Batch size. Larger (64-128) improves GPU utilization. |
| `learning_rate` | float | `1e-4` | Learning rate. Chemprop uses conservative default. |
| `random_state` | int | `42` | Random seed for reproducibility. |
| `accelerator` | str | `'auto'` | PyTorch Lightning accelerator. Auto-detects GPU. |
| `early_stopping` | bool | `True` | Enable early stopping to prevent overfitting. |
| `early_stopping_patience` | int | `3` | Epochs to wait for improvement. Lower = faster training. |
| `early_stopping_min_delta` | float | `0.0` | Minimum improvement threshold. |
| `val_fraction` | float | `0.1` | Fraction of training data for validation (for early stopping). |

**Training considerations:**
- Early stopping reduces training time (typically stops at 20-40 epochs)
- Validation split requires minimum dataset size (20 compounds)
- GPU training is 5-20x faster than CPU

#### Fine-Tuning Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_fine_tuning` | bool | `False` | Enable checkpoint-based fine-tuning for active learning. |
| `checkpoint_dir` | Path | `None` | Directory for checkpoint storage (required if fine-tuning enabled). |
| `enable_aggressive_gc` | bool | `True` | Enable GPU memory cleanup after training/prediction. |

**Fine-tuning details:** See [Fine-Tuning for Active Learning](#fine-tuning-for-active-learning) section below.

### Uncertainty Support

ChempropLearner does **not** provide uncertainty estimates.

For uncertainty quantification with Chemprop, use **ChempropEnsemble** (3 models with uncertainty from disagreement).

### GPU Support

Chemprop benefits significantly from GPU acceleration:
- Training speedup: 5-20x on GPU vs CPU
- Inference speedup: 10-50x on GPU vs CPU
- Recommended: 4GB+ GPU memory for typical molecular datasets

GPU memory management:
- `enable_aggressive_gc=True` (default) prevents memory accumulation in active learning
- Important for multi-cycle experiments (prevents GPU OOM errors)

### Examples

**CLI Usage (Pure Graph-Based):**

```bash
learnm8 run compounds.csv --target Activity \
  --learner chemprop \
  --n-cycles 10
```

No `--featurizer` needed - Chemprop works directly with SMILES.

**Python API (Pure Graph-Based):**

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',
    target_col='Activity',
    n_cycles=10
)
```

**Custom Architecture:**

```python
from learnm8.learners.torch import ChempropLearner
from learnm8 import run_active_learning

chemprop = ChempropLearner(
    message_hidden_dim=500,
    depth=5,
    ffn_hidden_dim=400,
    ffn_num_layers=2,
    dropout=0.2,
    max_epochs=100,
    batch_size=64,
    early_stopping=True,
    early_stopping_patience=5
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=chemprop,
    target_col='Activity',
    n_cycles=15
)
```

---

## ChempropEnsemble

Ensemble of 3 Chemprop learners providing uncertainty quantification through model disagreement.

### Overview

ChempropEnsemble creates 3 ChempropLearner instances with different random seeds. Uncertainty is estimated from the variance across ensemble predictions. This provides robust uncertainty estimates for active learning without requiring Monte Carlo sampling.

**When to use:**
- When uncertainty quantification is critical for active learning
- Uncertainty-based acquisition strategies (UCB, EI, Thompson Sampling)
- Graph-based learning with uncertainty needs
- Sufficient computational budget (3x training/inference time)

**Key characteristics:**
- Uncertainty from ensemble disagreement (no dropout needed)
- Same parameters as ChempropLearner (applied to all 3 members)
- 3x computational cost vs single model
- More stable uncertainty than single model dropout
- Fine-tuning support for all ensemble members

### Parameters

ChempropEnsemble accepts **all ChempropLearner parameters** (applied to each ensemble member) plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_states` | List[int] | `[42, 123, 456]` | Random seeds for ensemble diversity. 3 seeds create 3 models. |
| `aggregation_method` | str | `'mean'` | How to combine predictions: `'mean'`, `'median'`. Mean typical. |
| `uncertainty_method` | str | `'std'` | Uncertainty calculation: `'std'`, `'mad'`, `'quantile'`. Std typical. |

**Ensemble considerations:**
- 3 models provide good uncertainty estimates (more models = diminishing returns)
- Training time: 3x single model (models trained sequentially)
- Inference time: 3x single model
- Different `random_states` ensure model diversity

### Uncertainty Support

ChempropEnsemble **provides uncertainty estimates** via ensemble disagreement.

The returned uncertainty is calculated from the variance across 3 model predictions using the specified `uncertainty_method`:
- `'std'`: Standard deviation (default, most common)
- `'mad'`: Median absolute deviation (robust to outliers)
- `'quantile'`: Interquartile range

### Examples

**CLI Usage:**

```bash
learnm8 run compounds.csv --target Activity \
  --learner chemprop_ensemble \
  --cycles "random:0.01 ucb:0.005*5" \
  --n-cycles 10
```

**Python API:**

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop_ensemble',
    target_col='Activity',
    cycles=[('random', 0.01), ('ucb', 0.005)],
    acquisition_params={'beta': 2.0}
)
```

**Custom Configuration:**

```python
from learnm8.learners.ensemble import ChempropEnsemble
from learnm8 import run_active_learning

ensemble = ChempropEnsemble(
    message_hidden_dim=500,
    depth=5,
    ffn_hidden_dim=400,
    dropout=0.2,
    max_epochs=100,
    batch_size=64,
    random_states=[42, 123, 456],
    uncertainty_method='std'
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=ensemble,
    target_col='Activity',
    cycles=[('random', 0.01), ('ucb', 0.005)]
)
```

---

## Hybrid Mode

Chemprop can combine learned graph features with traditional molecular descriptors for improved performance on some tasks.

### When to Use Hybrid Mode

**Use hybrid mode when:**
- Task benefits from explicit physicochemical properties (solubility, lipophilicity)
- Dataset is small (<5000 compounds) and needs feature engineering help
- Combining structural learning with domain knowledge improves performance

**Use pure graph mode when:**
- Dataset is large (>10,000 compounds) - Chemprop learns sufficient representations
- Task is structure-activity relationship focused
- Simplicity and speed are priorities

### How Hybrid Mode Works

1. Chemprop learns graph embedding from SMILES (as usual)
2. Pre-computed molecular descriptors passed as `x_d` (extra descriptors)
3. After aggregation, descriptors concatenated with graph embedding
4. Combined features fed through feedforward network

**Descriptor concatenation:**
- FFN input dimension = `message_hidden_dim` + descriptor dimension
- Example: 300 (graph) + 1613 (Mordred descriptors) = 1913 total

### Examples

**CLI Usage with Descriptors:**

```bash
learnm8 run compounds.csv --target Activity \
  --learner chemprop \
  --featurizer descriptors \
  --n-cycles 10
```

Specifying `--featurizer descriptors` enables hybrid mode automatically.

**Python API with Morgan Fingerprints:**

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop_ensemble',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)
```

**Supported Featurizers for Hybrid Mode:**
- `descriptors` (Mordred, 1613 features) - most comprehensive
- `morgan` (2048 bits) - fast, good coverage
- `ecfp6` (2048 bits) - larger radius
- `maccs` (167 bits) - structural keys

---

## Fine-Tuning for Active Learning

Chemprop fine-tuning allows incremental training from previous checkpoints, significantly improving active learning performance.

### What is Fine-Tuning?

In standard active learning, each cycle trains a model from scratch on the growing labeled dataset. With fine-tuning:

1. **Cycle 1:** Train from scratch, save checkpoint
2. **Cycle 2:** Load checkpoint, fine-tune on expanded dataset
3. **Cycle 3:** Load checkpoint from cycle 2, fine-tune further
4. **Continue...**

**Benefits:**
- Faster convergence (model retains knowledge from previous cycles)
- Better uncertainty calibration (model smoothly adapts to new data)
- Reduced computational cost (fewer epochs needed per cycle)
- More stable active learning trajectories

**Costs:**
- Checkpoint storage (one checkpoint per cycle, ~50-200 MB each)
- Slightly more complex configuration

### When to Use Fine-Tuning

**Recommended for:**
- Active learning campaigns with many cycles (>10 cycles)
- Production screening workflows (faster cycle times matter)
- Limited computational budget (reduces total training time)
- Tasks where uncertainty calibration is critical

**Not needed for:**
- Benchmarking experiments (full retraining is more rigorous)
- Few cycles (<5 cycles) where benefit is minimal
- When checkpoint storage is constrained

### Enabling Fine-Tuning

**CLI:**

Currently not exposed via CLI. Use Python API for fine-tuning.

**Python API (ChempropLearner):**

```python
from learnm8.learners.torch import ChempropLearner
from learnm8 import run_active_learning
from pathlib import Path

chemprop = ChempropLearner(
    message_hidden_dim=300,
    depth=3,
    max_epochs=50,
    enable_fine_tuning=True,
    checkpoint_dir=Path('./chemprop_checkpoints')
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=chemprop,
    target_col='Activity',
    n_cycles=20
)
```

**Python API (ChempropEnsemble):**

```python
from learnm8.learners.ensemble import ChempropEnsemble
from learnm8 import run_active_learning
from pathlib import Path

ensemble = ChempropEnsemble(
    message_hidden_dim=300,
    depth=3,
    max_epochs=50,
    enable_fine_tuning=True,
    checkpoint_dir=Path('./ensemble_checkpoints')
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=ensemble,
    target_col='Activity',
    cycles=[('random', 0.01), ('ucb', 0.005)]
)
```

### Checkpoint Management

**Checkpoint Directory Structure:**

For **ChempropLearner**:
```
chemprop_checkpoints/
├── cycle_1.ckpt
├── cycle_2.ckpt
├── cycle_3.ckpt
└── ...
```

For **ChempropEnsemble**:
```
ensemble_checkpoints/
├── member_0/
│   ├── cycle_1.ckpt
│   ├── cycle_2.ckpt
│   └── ...
├── member_1/
│   ├── cycle_1.ckpt
│   ├── cycle_2.ckpt
│   └── ...
└── member_2/
    ├── cycle_1.ckpt
    ├── cycle_2.ckpt
    └── ...
```

**Storage Requirements:**
- Single model checkpoint: 50-200 MB (depends on architecture)
- Ensemble checkpoint: 150-600 MB (3x single model)
- 20 cycles of ensemble: 3-12 GB total

**Checkpoint Safety:**
- Architecture mismatch handled gracefully (trains fresh model if incompatible)
- Missing checkpoints handled (trains fresh model)
- Checkpoints not required for prediction (only for training)

### Fine-Tuning Workflow

```python
from learnm8.learners.torch import ChempropLearner
from learnm8 import run_active_learning
from pathlib import Path

checkpoint_dir = Path('./production_checkpoints')

learner = ChempropLearner(
    message_hidden_dim=300,
    depth=3,
    max_epochs=50,
    early_stopping=True,
    early_stopping_patience=5,
    enable_fine_tuning=True,
    checkpoint_dir=checkpoint_dir
)

results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='scoring_function.py:calculate_binding',
    learner=learner,
    target_col='binding_affinity',
    cycles=[
        ('random', 0.01),
        ('ucb', 0.005),
        ('ucb', 0.005),
        ('greedy', 0.005)
    ]
)
```

**What happens:**
1. Cycle 1: Trains from scratch, saves `cycle_1.ckpt`
2. Cycle 2: Loads `cycle_1.ckpt`, fine-tunes on expanded dataset, saves `cycle_2.ckpt`
3. Cycle 3: Loads `cycle_2.ckpt`, fine-tunes further, saves `cycle_3.ckpt`
4. Cycle 4: Loads `cycle_3.ckpt`, fine-tunes further, saves `cycle_4.ckpt`

---

## Installation

Chemprop is an optional dependency. Install with:

```bash
pip install chemprop
```

**Requirements:**
- PyTorch (installed automatically with chemprop)
- PyTorch Lightning (installed automatically with chemprop)
- RDKit (already required by LearnM8)

**GPU Requirements:**
- Optional but strongly recommended
- Minimum 4GB GPU memory for typical datasets
- 8GB+ recommended for large datasets (>100k compounds)
- CPU inference works but is 10-50x slower

**Testing Installation:**

```bash
python -c "from learnm8.learners.torch import ChempropLearner; print('Chemprop available')"
```

---

## Comparison Table

| Model | Uncertainty | Training Time | Inference Time | Best Use Case |
|-------|-------------|---------------|----------------|---------------|
| **ChempropLearner** | No | Fast | Fast | Pure exploitation, speed priority |
| **ChempropEnsemble** | Yes (ensemble) | 3x single | 3x single | Active learning with uncertainty |

**Chemprop vs Traditional Learners:**

| Aspect | Chemprop | Random Forest | Gaussian Process |
|--------|----------|---------------|------------------|
| **Input** | SMILES (graph) | Fingerprints | Descriptors |
| **Dataset Size** | >1000 compounds | Any size | <5000 compounds |
| **Training Speed** | Medium | Fast | Slow |
| **Inference Speed** | Fast (GPU) | Fast | Medium |
| **Uncertainty** | Ensemble only | Via trees | Natural |
| **Featurizer Needed** | No (optional hybrid) | Yes | Yes |

**When to choose Chemprop:**
- Dataset >1000 compounds (Chemprop needs data to learn representations)
- State-of-the-art performance desired
- GPU available for reasonable training times
- Graph-based learning preferred over fixed fingerprints
- Fine-tuning for active learning is beneficial

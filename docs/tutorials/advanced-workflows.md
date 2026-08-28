# Advanced Workflows

This guide covers advanced LearnM8 features for scaling to large compound libraries and maximizing active learning performance.

## Pruning Large Compound Libraries

### What is Pruning

Pruning removes low-value compounds from the unlabeled pool based on model predictions and uncertainties. This reduces computational costs and focuses active learning on promising regions.

**Benefits:**

- 10-50% computational savings
- Faster cycle times
- Focused search in chemical space
- Memory efficiency for large libraries

**When to use:**

- Compound pools > 100,000 compounds
- Limited computational budget
- Clear exploitation phase (later cycles)
- Low-scoring regions confirmed uninteresting

### Score-Based Pruning

Remove compounds with poor predicted scores.

**Python API Example:**

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
    score_direction='higher'
)
```

Removes bottom 10% of compounds by predicted score each cycle.

**Parameters:**

- `pruning_strategy`: `'score'` for prediction-based pruning
- `pruning_fraction`: Fraction to remove (0.0-0.9), typically 0.1-0.3
- `score_direction`: `'higher'` or `'lower'` based on optimization goal

**CLI Alternative:**

```bash
learnm8 run large_library.csv oracle.py:score \
  --target affinity \
  --learner gp \
  --featurizer morgan \
  --n-cycles 20 \
  --pruning-strategy score \
  --pruning-fraction 0.1 \
  --score-direction higher
```

### Uncertainty-Based Pruning

Remove compounds with low uncertainty (model is confident they are uninteresting).

**Example:**

```python
from learnm8.core.config import CycleConfig

results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='oracle.py:score',
    learner='gp',
    target_col='affinity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.01),
        CycleConfig('ucb', n_cycles=5, batch_fraction=0.005),
        CycleConfig(
            'greedy',
            n_cycles=10,
            batch_fraction=0.005,
            pruning_strategy='score',
            pruning_fraction=0.2
        )
    ]
)
```

**When uncertainty-based pruning works:**

- Learner provides reliable uncertainties (GP, ensembles, MC Dropout)
- Later cycles when model is confident
- Exploration phase complete

### Per-Cycle Pruning Strategy

Apply different pruning strategies per cycle using CycleConfig.

**Example: Progressive Pruning**

```python
from learnm8.core.config import CycleConfig

cycles = [
    # Cycle 0-1: No pruning (exploration)
    CycleConfig('random', n_cycles=1, batch_fraction=0.02),
    CycleConfig('ucb', n_cycles=5, batch_fraction=0.01),

    # Cycle 6-10: Light pruning (10%)
    CycleConfig(
        'greedy',
        n_cycles=5,
        batch_fraction=0.01,
        pruning_strategy='score',
        pruning_fraction=0.1
    ),

    # Cycle 11-20: Aggressive pruning (30%)
    CycleConfig(
        'greedy',
        n_cycles=10,
        batch_fraction=0.01,
        pruning_strategy='score',
        pruning_fraction=0.3
    )
]

results = run_active_learning(
    compound_pool='large_library.csv',
    oracle='oracle.py:score',
    learner='ensemble',
    target_col='affinity',
    featurizer='morgan',
    cycles=cycles
)
```

### Pruning Performance Example

**Dataset: 500,000 compounds, 20 cycles, score-based pruning (20%)**

| Cycle | Pool Size | Pruned  | Predictions | Time Saved |
| ----- | --------- | ------- | ----------- | ---------- |
| 0     | 500,000   | 0       | 500,000     | -          |
| 1     | 500,000   | 100,000 | 400,000     | 20%        |
| 5     | 400,000   | 80,000  | 320,000     | 36%        |
| 10    | 300,000   | 60,000  | 240,000     | 52%        |
| 20    | 150,000   | 30,000  | 120,000     | 76%        |

Cumulative computational savings: ~40%

## Ensemble Learning

### Why Ensembles

Ensemble learning combines multiple models to improve predictions and uncertainty quantification.

**Advantages:**

- Better uncertainty estimates through model disagreement
- Robustness to overfitting
- No hyperparameter tuning for uncertainty
- Captures different aspects of data

**Uncertainty calculation:**

```
ensemble_uncertainty = sqrt(prediction_variance + model_disagreement)
```

Where:

- `prediction_variance`: Average uncertainty from individual models
- `model_disagreement`: Variance across model predictions

### Mixed Ensemble Example

Combines RandomForest, LinearRegression, and XGBoost for maximum diversity.

**Python API Example:**

```python
from learnm8 import run_active_learning
from learnm8.learners.ensemble import MixedEnsemble

learner = MixedEnsemble(random_state=42)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.py:score',
    learner=learner,
    target_col='affinity',
    featurizer='morgan',
    n_cycles=15,
    batch_fraction=0.01
)
```

**CLI Alternative:**

```bash
learnm8 run compounds.csv oracle.py:score \
  --target affinity \
  --learner mixed_ensemble \
  --featurizer morgan \
  --n-cycles 15 \
  --batch-fraction 0.01
```

### Custom Ensemble

Create ensembles with specific model combinations.

```python
from learnm8.learners.ensemble import EnsembleLearner
from learnm8.learners.sklearn import RandomForestLearner, GaussianProcessLearner
from learnm8.learners.sklearn import XGBoostLearner

learners = [
    RandomForestLearner(n_estimators=100, random_state=42),
    GaussianProcessLearner(random_state=42),
    XGBoostLearner(learning_rate=0.1, random_state=42)
]

ensemble = EnsembleLearner(
    learners=learners,
    aggregation_method='mean',
    uncertainty_method='std'
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.py:score',
    learner=ensemble,
    target_col='affinity',
    featurizer='morgan',
    n_cycles=15
)
```

### Type-Specific Ensembles

LearnM8 provides pre-configured ensembles for common model types.

**Available ensembles:**

- `rf_ensemble`: 3 RandomForest variants
- `xgb_ensemble`: 3 XGBoost variants
- `lr_ensemble`: 3 LinearRegression variants
- `dt_ensemble`: 3 DecisionTree variants
- `fastprop_ensemble`: 3 FastProp neural networks
- `chemprop_ensemble`: 3 Chemprop graph neural networks

**Python API Example:**

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.py:score',
    learner='rf_ensemble',
    target_col='affinity',
    featurizer='morgan',
    n_cycles=10
)
```

**CLI Alternative:**

```bash
learnm8 run compounds.csv oracle.py:score \
  --target affinity \
  --learner rf_ensemble \
  --featurizer morgan \
  --n-cycles 10
```

### Ensemble Performance Considerations

**Computational cost:**

- 3x training time (3 models)
- 3x prediction time
- 3x memory usage
- No parallelization (sequential training)

**When worth the cost:**

- Uncertainty-based acquisition (UCB, EI, Thompson)
- Small to medium datasets (< 50,000 compounds)
- High-value measurements (expensive oracles)
- Research and benchmarking

**When to use single models:**

- Very large datasets (> 100,000 compounds)
- Greedy acquisition (no uncertainty needed)
- Computational budget constraints
- Production with time limits

## Chemprop Fine-Tuning

### What is Fine-Tuning

Fine-tuning for active learning reuses trained model weights from previous cycles as initialization for subsequent training, instead of training from scratch each cycle.

**Benefits:**

- Faster convergence (fewer epochs needed)
- Better uncertainty calibration
- Improved performance with small batches
- Preserved learned representations

**Computational cost:**

- 20-40% faster training per cycle
- Additional checkpoint storage (minimal)
- No prediction cost increase

### Enabling Fine-Tuning

**Python API Example:**

```python
from learnm8 import run_active_learning
from learnm8.learners.torch import ChempropLearner
from pathlib import Path

learner = ChempropLearner(
    enable_fine_tuning=True,
    checkpoint_dir=Path('./checkpoints'),
    max_epochs=50,
    learning_rate=1e-4
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.py:score',
    learner=learner,
    target_col='affinity',
    n_cycles=20,
    batch_fraction=0.005
)
```

Chemprop fine-tuning is only available via the Python API (no CLI equivalent).

### Fine-Tuning Parameters

**Learner initialization:**

```python
learner = ChempropLearner(
    enable_fine_tuning=True,
    checkpoint_dir=Path('./checkpoints'),
    max_epochs=50,
    early_stopping=True,
    early_stopping_patience=3
)
```

**Parameters:**

- `enable_fine_tuning`: Enable checkpoint-based fine-tuning (default: False)
- `checkpoint_dir`: Directory for checkpoint storage (required if fine-tuning)
- `max_epochs`: Maximum epochs per training cycle
- `early_stopping`: Stop when validation loss plateaus
- `early_stopping_patience`: Epochs to wait before stopping

### Example Workflow

**Complete active learning campaign with fine-tuning:**

```python
from learnm8 import run_active_learning
from learnm8.learners.ensemble import ChempropEnsemble
from pathlib import Path

learner = ChempropEnsemble(
    n_models=3,
    enable_fine_tuning=True,
    checkpoint_dir=Path('./checkpoints/chemprop_ensemble'),
    max_epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    early_stopping=True
)

results = run_active_learning(
    compound_pool='screening_library.csv',
    oracle='docking_oracle.py:dock_compounds',
    learner=learner,
    target_col='docking_score',
    score_direction='lower',
    cycles=[
        ('random', 0.02),
        ('ucb', 0.01),
        ('ucb', 0.01),
        ('ucb', 0.01),
        ('greedy', 0.005),
        ('greedy', 0.005),
        ('greedy', 0.005),
        CycleConfig('simulated_annealing', n_cycles=1, batch_fraction=0.01),
    ],
    output_format='csv',
    output_dir='results/chemprop_fine_tuned'
)
```

**Checkpoint management:**

```
checkpoints/chemprop_ensemble/
├── model_0_cycle_1.ckpt
├── model_0_cycle_2.ckpt
├── model_1_cycle_1.ckpt
├── model_1_cycle_2.ckpt
├── model_2_cycle_1.ckpt
└── model_2_cycle_2.ckpt
```

### Fine-Tuning vs From-Scratch Comparison

**Small batch scenario (0.5% per cycle):**

| Cycle | From Scratch Epochs | Fine-Tuning Epochs | Speedup |
| ----- | ------------------- | ------------------ | ------- |
| 1     | 50                  | 50                 | 1.0x    |
| 2     | 50                  | 20                 | 2.5x    |
| 5     | 50                  | 15                 | 3.3x    |
| 10    | 50                  | 12                 | 4.2x    |
| 20    | 50                  | 10                 | 5.0x    |

Average training time reduction: 60%

### When Fine-Tuning Helps Most

**Ideal scenarios:**

- Small batch sizes (< 1% per cycle)
- Many cycles (> 10 cycles)
- Chemprop or deep learning models
- Limited computational budget
- Incremental learning tasks

**Less beneficial:**

- Large batch sizes (> 5% per cycle)
- Few cycles (< 5 cycles)
- Traditional ML models (RF, XGB)
- Abundant GPU resources

## Configuration Files for Reproducibility

### Creating YAML Configuration

Store complete experiment parameters in version-controlled configuration files.

**Example: `experiment_config.yaml`**

```yaml
compound_pool: screening_library.csv
oracle: docking_oracle.py:dock_compounds
learner: chemprop_ensemble
target_col: docking_score
featurizer: morgan

score_direction: lower
random_state: 42
output_dir: results/docking_campaign_001

cycles:
  - strategy: random
    batch_fraction: 0.02
    n_cycles: 1
  - strategy: ucb
    batch_fraction: 0.01
    n_cycles: 5
    acquisition_params:
      beta: 2.0
  - strategy: greedy
    batch_fraction: 0.005
    n_cycles: 10
    pruning_strategy: score
    pruning_fraction: 0.15
  - strategy: simulated_annealing
    batch_fraction: 0.01
    n_cycles: 4

output_format: csv
cache_dir: .cache/docking_features
```

### Loading Configuration

**Python API:**

```python
import yaml
from learnm8 import run_active_learning

with open('experiment_config.yaml') as f:
    config = yaml.safe_load(f)

results = run_active_learning(**config)
```

**CLI:**

```bash
learnm8 run --config experiment_config.yaml
```

**Override specific parameters:**

```bash
learnm8 run --config experiment_config.yaml \
  --random-state 123 \
  --output-dir results/docking_campaign_002
```

### Version Control for Experiments

**Directory structure:**

```
project/
├── configs/
│   ├── baseline_rf.yaml
│   ├── ensemble_ucb.yaml
│   ├── chemprop_finetuned.yaml
│   └── production_campaign.yaml
├── oracles/
│   ├── docking_oracle.py
│   ├── qsar_oracle.py
│   └── experimental_oracle.py
├── data/
│   ├── screening_library.csv
│   └── validation_set.csv
├── results/
│   ├── exp_001/
│   ├── exp_002/
│   └── exp_003/
└── checkpoints/
    ├── exp_001/
    └── exp_002/
```

**Git workflow:**

```bash
# Version control configurations
git add configs/production_campaign.yaml
git commit -m "Add production campaign configuration"

# Run experiment
learnm8 run --config configs/production_campaign.yaml

# Version control results metadata
git add results/exp_001/metadata.json
git commit -m "Record experiment 001 results"
```

### Configuration Templates

**Quick exploration:**

```yaml
# configs/quick_explore.yaml
compound_pool: compounds.csv
oracle: oracle.py:score
learner: rf
target_col: activity
featurizer: morgan
n_cycles: 5
batch_fraction: 0.02
random_state: 42
```

**Intensive campaign:**

```yaml
# configs/intensive_campaign.yaml
compound_pool: large_library.csv
oracle: oracle.py:score
learner: chemprop_ensemble
target_col: affinity
featurizer: morgan
score_direction: higher
random_state: 42

cycles:
  - strategy: random
    batch_fraction: 0.01
    n_cycles: 1
  - strategy: ucb
    batch_fraction: 0.005
    n_cycles: 10
    acquisition_params:
      beta: 2.0
  - strategy: greedy
    batch_fraction: 0.005
    n_cycles: 15
    pruning_strategy: score
    pruning_fraction: 0.2

output_format: csv
cache_dir: .cache/intensive
```

### Best Practices

**Configuration management:**

1. Use descriptive config names (`chemprop_ucb_beta2.yaml`)
2. Include `random_state` for reproducibility
3. Document parameter choices in comments
4. Version control all configs
5. Store oracle code with configs

**Experiment tracking:**

1. Unique output directories per experiment
2. Record git commit hash in results
3. Save config copy to output directory
4. Document hypothesis and conclusions
5. Archive checkpoints for successful runs

**Parameter search:**

```bash
# Test different beta values
for beta in 1.0 1.5 2.0 2.5; do
  learnm8 run --config base_config.yaml \
    --acquisition-params "beta=$beta" \
    --output-dir "results/beta_sweep_$beta"
done
```

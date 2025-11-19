# Configuration Files

Configuration files provide a convenient way to specify all experiment parameters in a structured format, making experiments reproducible and version-controllable. LearnM8 supports both YAML and JSON configuration formats.

## Overview

Configuration files are useful when:

- **Running multiple experiments** with different parameter combinations
- **Ensuring reproducibility** by version controlling experiment configurations
- **Managing complex parameter sets** that would be unwieldy as CLI flags
- **Sharing experiments** with collaborators
- **Documenting experimental protocols** for publications

Configuration files support all parameters available in the `run_active_learning()` API, including advanced features like custom cycle specifications and acquisition parameters.

## YAML Configuration Format

YAML is the recommended format for configuration files due to its readability and support for comments.

### Complete Schema

```yaml
# Input Data
compound_pool: "path/to/compounds.csv"
oracle: "path/to/oracle.csv"  # or "module.py:function"
target_col: "Activity"

# Model Configuration
learner: "rf"  # Learner shortcut string
featurizer_type: "morgan"  # Required for non-GNN learners

# Cycle Configuration (choose one of these approaches)

# Option 1: Simple API (consistent batch sizes)
n_cycles: 10
batch_fraction: 0.01
strategy: "greedy"
initial_strategy: "random"

# Option 2: Explicit cycle list (advanced)
cycles:
  - strategy: "random"
    n_cycles: 1
    batch_fraction: 0.02
  - strategy: "greedy"
    n_cycles: 5
    batch_fraction: 0.01
  - strategy: "ucb"
    n_cycles: 4
    batch_fraction: 0.01
    acquisition_params:
      beta: 2.0

# Optimization Settings
score_direction: "higher"  # or "lower"
mode: "benchmark"  # or "run" (usually auto-detected)

# Output Configuration
output_dir: "results/experiment_001"
cache_dir: ".cache/shared"  # Optional, defaults to output_dir/.cache

# Advanced Features
random_state: 42

# Chemprop Fine-Tuning (for Chemprop learners only)
enable_chemprop_fine_tuning: false

# Pruning Configuration
pruning_fraction: 0.3  # Optional, enables pruning if specified
pruning_strategy: "score_based"  # Only supported strategy
pruning_params:
  pruning_threshold: 0.5  # Optional: absolute score threshold

# Acquisition Parameters (optional, strategy-specific)
acquisition_params:
  beta: 1.5  # for UCB
  temperature: 1.0  # for Thompson sampling
```

### Simple Configuration Example

Minimal configuration for a standard experiment:

```yaml
# experiment.yaml
compound_pool: "compounds.csv"
target_col: "Activity"
learner: "gp"
featurizer_type: "morgan"
n_cycles: 10
batch_fraction: 0.01
```

### Advanced Configuration with Custom Cycles

Multi-strategy experiment with per-cycle control:

```yaml
# advanced_experiment.yaml
compound_pool: "compound_library.csv"
oracle: "scoring_module.py:calculate_affinity"
target_col: "binding_score"
learner: "ensemble"
featurizer_type: "morgan"
score_direction: "lower"
random_state: 123

cycles:
  - strategy: "random"
    n_cycles: 1
    batch_fraction: 0.02
  - strategy: "ucb"
    n_cycles: 5
    batch_fraction: 0.01
    acquisition_params:
      beta: 2.0
  - strategy: "greedy"
    n_cycles: 4
    batch_fraction: 0.01

pruning_fraction: 0.2
pruning_strategy: "score_based"

output_dir: "results/ensemble_screening"
cache_dir: ".cache/shared"
```

### Configuration with Pruning and Ensemble

Example using advanced features:

```yaml
# pruning_ensemble_experiment.yaml
compound_pool: "large_library.csv"
target_col: "Activity"
learner: "mixed_ensemble"
featurizer_type: "descriptors"
score_direction: "higher"

cycles:
  - strategy: "random"
    n_cycles: 1
    batch_fraction: 0.01
  - strategy: "ucb"
    n_cycles: 8
    batch_fraction: 0.005
    acquisition_params:
      beta: 1.5
  - strategy: "diverse"
    n_cycles: 3
    batch_fraction: 0.01

pruning_fraction: 0.3
pruning_strategy: "score_based"
pruning_params:
  pruning_threshold: 0.5

cache_dir: ".cache/global"
random_state: 42
```

## JSON Configuration Format

JSON format is also supported, with identical schema to YAML but different syntax.

### Complete JSON Example

```json
{
  "compound_pool": "compounds.csv",
  "oracle": "oracle.csv",
  "target_col": "Activity",
  "learner": "gp",
  "featurizer_type": "morgan",
  "n_cycles": 10,
  "batch_fraction": 0.01,
  "strategy": "greedy",
  "initial_strategy": "random",
  "score_direction": "higher",
  "mode": "benchmark",
  "output_dir": "results/experiment_001",
  "cache_dir": ".cache/shared",
  "random_state": 42
}
```

### JSON with Custom Cycles

```json
{
  "compound_pool": "compound_library.csv",
  "oracle": "scoring_module.py:calculate_affinity",
  "target_col": "binding_score",
  "learner": "ensemble",
  "featurizer_type": "morgan",
  "score_direction": "lower",
  "cycles": [
    {
      "strategy": "random",
      "n_cycles": 1,
      "batch_fraction": 0.02
    },
    {
      "strategy": "ucb",
      "n_cycles": 5,
      "batch_fraction": 0.01,
      "acquisition_params": {
        "beta": 2.0
      }
    },
    {
      "strategy": "greedy",
      "n_cycles": 4,
      "batch_fraction": 0.01
    }
  ],
  "pruning_fraction": 0.2,
  "pruning_strategy": "score_based",
  "output_dir": "results/ensemble_screening",
  "cache_dir": ".cache/shared",
  "random_state": 123
}
```

### JSON with All Features

```json
{
  "compound_pool": "large_library.csv",
  "oracle": "docking_oracle.py:dock_compounds",
  "target_col": "docking_score",
  "learner": "chemprop_ensemble",
  "featurizer_type": "descriptors",
  "score_direction": "lower",
  "enable_chemprop_fine_tuning": true,
  "cycles": [
    {
      "strategy": "random",
      "n_cycles": 1,
      "batch_fraction": 0.01
    },
    {
      "strategy": "ucb",
      "n_cycles": 8,
      "batch_fraction": 0.005,
      "acquisition_params": {
        "beta": 1.5
      }
    },
    {
      "strategy": "diverse",
      "n_cycles": 3,
      "batch_fraction": 0.01
    }
  ],
  "pruning_fraction": 0.3,
  "pruning_strategy": "score_based",
  "pruning_params": {
    "pruning_threshold": 0.5
  },
  "acquisition_params": {
    "beta": 1.5,
    "temperature": 1.0
  },
  "output_dir": "results/chemprop_screening",
  "cache_dir": ".cache/global",
  "random_state": 42
}
```

## Loading Configurations

### Using the CLI

Load a configuration file using the `--config` flag:

```bash
# Load YAML configuration
learnm8 run --config experiment.yaml

# Load JSON configuration
learnm8 run --config experiment.json
```

The configuration file provides default values for all parameters. When loaded, LearnM8 will display the configuration and begin the experiment.

### Combining Config Files with CLI Flags

CLI flags override configuration file values when explicitly provided:

```bash
# Use config but override learner and output directory
learnm8 run --config experiment.yaml \
  --learner ensemble \
  -o results/modified_experiment
```

### Precedence Rules

Parameter values are determined in the following order (later overrides earlier):

1. **Configuration file defaults** (if `--config` is provided)
2. **CLI argument defaults** (defined in argparse)
3. **Explicit CLI flags** (user-provided values)

**Example:**

```yaml
# config.yaml
learner: "rf"
n_cycles: 10
output_dir: "results/default"
```

```bash
# Override learner and output_dir, keep n_cycles from config
learnm8 run --config config.yaml \
  --learner gp \
  -o results/custom
```

Result:
- `learner`: "gp" (CLI override)
- `n_cycles`: 10 (from config)
- `output_dir`: "results/custom" (CLI override)

### Python API with Config Files

Load and use configuration dictionaries programmatically:

```python
import yaml
from pathlib import Path
from learnm8 import run_active_learning

# Load configuration
config_path = Path("experiment.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Run experiment with config
results = run_active_learning(**config)
```

Override specific parameters:

```python
import yaml
from learnm8 import run_active_learning

# Load base configuration
with open("base_config.yaml") as f:
    config = yaml.safe_load(f)

# Override specific parameters
config['learner'] = 'ensemble'
config['output_dir'] = 'results/modified'

# Run experiment
results = run_active_learning(**config)
```

## Configuration Best Practices

### Version Control

Configuration files are ideal for version control:

```bash
# Track experiment configurations
git add experiments/config_*.yaml
git commit -m "Add configuration for ensemble comparison"
```

Benefits:
- **Reproducibility:** Exact experiment parameters preserved
- **Collaboration:** Share configurations with team members
- **Tracking:** Document experimental protocol evolution
- **Comparison:** Diff configs to understand parameter changes

### Naming Conventions

Use descriptive, systematic naming for configuration files:

```
experiments/
├── baseline_rf_morgan.yaml
├── ensemble_ucb_beta2.yaml
├── chemprop_finetune_intensive.yaml
├── pruning_30pct_uncertainty.yaml
└── production_screening_gp.yaml
```

Recommended naming pattern:
```
{experiment_type}_{key_feature}_{variation}.yaml
```

### Comments in YAML

YAML supports comments (JSON does not), making it ideal for documenting parameter choices:

```yaml
# Baseline Random Forest experiment with Morgan fingerprints
compound_pool: "compounds.csv"
target_col: "Activity"

# Using RF for fast baseline comparison
learner: "rf"
featurizer_type: "morgan"  # 2048-bit Morgan fingerprints

# Conservative exploration-exploitation balance
cycles:
  - strategy: "random"
    n_cycles: 1
    batch_fraction: 0.02  # Larger initial batch for diversity
  - strategy: "ucb"
    n_cycles: 5
    batch_fraction: 0.01  # Standard batch size
    acquisition_params:
      beta: 1.5  # Moderate exploration weight
  - strategy: "greedy"
    n_cycles: 4
    batch_fraction: 0.01  # Exploitation phase

# Reproducibility seed
random_state: 42
```

### When to Use Config Files vs CLI

**Use configuration files when:**

- Running **parameter sweeps** (create multiple config files)
- **Complex experiments** with many parameters
- **Reproducibility is critical** (publications, collaborations)
- **Sharing experiments** with collaborators
- **Long-term projects** requiring documentation

**Use CLI directly when:**

- **Quick exploratory experiments** with few parameters
- **One-off tests** not requiring documentation
- **Interactive development** and debugging
- **Simple experiments** with default parameters

### Configuration File Organization

Organize configurations by experiment type:

```
experiments/
├── benchmarking/
│   ├── ada_rf.yaml
│   ├── ada_gp.yaml
│   └── ada_ensemble.yaml
├── production/
│   ├── screening_campaign_01.yaml
│   └── screening_campaign_02.yaml
├── parameter_sweeps/
│   ├── ucb_beta_0.5.yaml
│   ├── ucb_beta_1.0.yaml
│   ├── ucb_beta_2.0.yaml
│   └── ucb_beta_3.0.yaml
└── ablation_studies/
    ├── no_pruning.yaml
    ├── pruning_20pct.yaml
    └── pruning_40pct.yaml
```

## Parameter Reference

All parameters from `run_active_learning()` are supported in configuration files.

### Core Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `compound_pool` | str/Path | CSV file with compound library (ID, SMILES columns) |
| `oracle` | str/Path | Oracle specification (CSV file or module.py:function) |
| `target_col` | str | Target property column name |
| `learner` | str | Learner shortcut (rf, gp, ensemble, etc.) |
| `featurizer_type` | str | Molecular featurizer (morgan, maccs, ecfp6, descriptors) |

### Cycle Configuration (Simple API)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_cycles` | int | 10 | Number of active learning cycles |
| `batch_fraction` | float | 0.01 | Fraction of pool to label per cycle |
| `strategy` | str | "greedy" | Acquisition strategy for cycles 1+ |
| `initial_strategy` | str | "random" | Acquisition strategy for cycle 0 |

### Cycle Configuration (Advanced API)

| Parameter | Type | Description |
|-----------|------|-------------|
| `cycles` | list[dict] | List of cycle configurations with strategy, n_cycles, batch_fraction |

Cycle dictionary structure:
```yaml
- strategy: "ucb"
  n_cycles: 5
  batch_fraction: 0.01
  acquisition_params:  # Optional
    beta: 2.0
  pruning_strategy: "score_based"  # Optional
  pruning_params:  # Optional
    pruning_fraction: 0.3
```

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_direction` | str | "higher" | Optimization direction (higher or lower) |
| `mode` | str | None | Execution mode (run or benchmark, auto-detected) |
| `output_dir` | str/Path | None | Output directory (auto-generated if None) |
| `cache_dir` | str/Path | None | Feature cache directory (output_dir/.cache if None) |
| `random_state` | int | 42 | Random seed for reproducibility |

### Advanced Features

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_chemprop_fine_tuning` | bool | False | Enable fine-tuning for Chemprop models |
| `pruning_fraction` | float | None | Fraction to prune per cycle (0.0-0.9) |
| `pruning_strategy` | str | None | Pruning method (only score_based is supported) |
| `pruning_params` | dict | None | Additional pruning parameters |
| `acquisition_params` | dict | None | Acquisition strategy parameters (beta, temperature, etc.) |

## Complete Example Configurations

### Benchmark Comparison Study

```yaml
# benchmark_rf_vs_gp.yaml
compound_pool: "benchmark_datasets/ADA.csv"
target_col: "Activity"
featurizer_type: "morgan"
score_direction: "higher"
mode: "benchmark"

# Will run with both learners in separate experiments
learner: "rf"  # or "gp"

cycles:
  - strategy: "random"
    n_cycles: 1
    batch_fraction: 0.02
  - strategy: "greedy"
    n_cycles: 9
    batch_fraction: 0.01

cache_dir: ".cache/benchmark"
random_state: 42
```

### Production Screening Campaign

```yaml
# production_screening.yaml
compound_pool: "screening_library_100k.csv"
oracle: "docking_pipeline.py:dock_and_score"
target_col: "docking_score"
learner: "mc_dropout"
featurizer_type: "descriptors"
score_direction: "lower"

cycles:
  - strategy: "random"
    n_cycles: 1
    batch_fraction: 0.001  # 100 compounds from 100k
  - strategy: "ucb"
    n_cycles: 10
    batch_fraction: 0.0005  # 50 compounds per cycle
    acquisition_params:
      beta: 2.0
  - strategy: "greedy"
    n_cycles: 5
    batch_fraction: 0.0005

pruning_fraction: 0.4
pruning_strategy: "score_based"
pruning_params:
  pruning_threshold: 0.5

cache_dir: ".cache/production"
output_dir: "results/screening_campaign_001"
random_state: 123
```

### Chemprop with Fine-Tuning

```yaml
# chemprop_finetuning.yaml
compound_pool: "compounds.csv"
target_col: "Activity"
learner: "chemprop_ensemble"
featurizer_type: "descriptors"  # Hybrid mode
enable_chemprop_fine_tuning: true

cycles:
  - strategy: "random"
    n_cycles: 1
    batch_fraction: 0.02
  - strategy: "ucb"
    n_cycles: 8
    batch_fraction: 0.01
    acquisition_params:
      beta: 1.5
  - strategy: "greedy"
    n_cycles: 3
    batch_fraction: 0.01

cache_dir: ".cache/chemprop"
random_state: 42
```

## Validation and Debugging

### Validating Configuration Files

Test configuration file syntax before running:

```bash
# YAML validation
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# JSON validation
python -c "import json; json.load(open('config.json'))"
```

### Common Configuration Errors

**Missing required parameters:**
```yaml
# ERROR: Missing target_col
compound_pool: "compounds.csv"
learner: "rf"
```

**Conflicting cycle specifications:**
```yaml
# ERROR: Cannot specify both simple and advanced API
n_cycles: 10
cycles:  # Conflicts with n_cycles
  - strategy: "random"
    n_cycles: 1
```

**Invalid parameter types:**
```yaml
# ERROR: batch_fraction must be float
batch_fraction: "0.01"  # Should be: 0.01
```

### Testing Configurations

Test configurations with small datasets before production runs:

```bash
# Test with subset of data
learnm8 run --config production_config.yaml \
  --compound-pool test_subset.csv \
  -o results/config_test
```

## Related Documentation

- **[CLI Reference](cli-reference.md)** - Complete CLI documentation
- **[API Reference](api-reference.md)** - Python API documentation
- **[Building Custom Cycles](../tutorials/building-custom-cycles.md)** - Cycle specification tutorial
- **[Advanced Workflows](../tutorials/advanced-workflows.md)** - Pruning and ensemble examples

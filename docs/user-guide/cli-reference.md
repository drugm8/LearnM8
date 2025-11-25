# CLI Reference

LearnM8 provides a comprehensive command-line interface for running active learning experiments, validating compounds, and exploring available components. The CLI uses a subcommand-based architecture with three main commands: `run`, `list`, and `validate`.

## Getting Help

Access help information at any level:

```bash
learnm8 --help
learnm8 run --help
learnm8 list --help
learnm8 validate --help
```

## Command Overview

LearnM8 CLI has three subcommands:

| Subcommand | Purpose | Common Usage |
|------------|---------|--------------|
| `run` | Execute active learning experiment | Main workflow for screening |
| `list` | List available components | Discover learners, acquisition strategies |
| `validate` | Validate compound pool | Pre-flight check before experiments |

## learnm8 run

Run an active learning experiment with specified configuration.

### Basic Syntax

```bash
learnm8 run COMPOUND_POOL [ORACLE] --target COLUMN --featurizer TYPE --learner MODEL [OPTIONS]
```

### Positional Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `COMPOUND_POOL` | Yes | CSV file with compound library (must have ID and SMILES columns) |
| `ORACLE` | No | Oracle specification (auto-detected if omitted) |

**Oracle Formats:**
- **CSV file**: `oracle.csv` (benchmark mode with ground truth)
- **Python module**: `module.py:function` (production mode with custom scoring)
- **Auto-detect**: Omit to use `COMPOUND_POOL` as oracle (benchmark mode)

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--target COLUMN` | string | Target property column name in CSV |
| `--featurizer TYPE` | choice | Molecular featurizer (see choices below) |

**Featurizer Choices:**
- `morgan` - Morgan fingerprints (2048 bits, radius=2)
- `maccs` - MACCS keys (167 bits)
- `ecfp6` - Extended connectivity (2048 bits, radius=3)
- `descriptors` - Mordred descriptors (1613 features)
- `morgan_feat` - Morgan feature fingerprints (2048 bits)

### Core Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--learner MODEL` | choice | `rf` | Machine learning model (see learner choices) |
| `--score-direction DIR` | choice | `higher` | Optimization direction: `higher` or `lower` |

**Learner Choices** (availability depends on installed dependencies):
- Scikit-learn: `rf`, `gp`, `xgb`, `dt`, `lr`, `advanced_rf`
- PyTorch: `mlp`, `mc_dropout`, `fastprop`
- Graph Neural Networks: `chemprop`, `chemprop_ensemble`
- Ensembles: `ensemble`, `mixed_ensemble`, `rf_ensemble`, `xgb_ensemble`, etc.

Use `learnm8 list learners` to see all available learners in your environment.

### Cycle Control (Mutually Exclusive)

Only one of these options can be used:

#### Option 1: Custom Cycle Specification (Recommended)

| Parameter | Type | Description |
|-----------|------|-------------|
| `--cycles SPEC` | string | Explicit cycle specification with strategy and batch fractions |

**Cycle Specification Format:**
```
"strategy1:fraction1 strategy2:fraction2*n ..."
```

**Examples:**
```bash
--cycles "random:0.02 greedy:0.01*5"
--cycles "random:0.01 ucb:0.005*8 diverse:0.01*2"
--cycles "random:0.02 greedy:0.01 ucb:0.01 greedy:0.01"
```

**Strategy Multiplier:** Use `*n` to repeat a strategy n times:
- `greedy:0.01*5` = 5 cycles of greedy with 0.01 batch fraction
- Equivalent to: `greedy:0.01 greedy:0.01 greedy:0.01 greedy:0.01 greedy:0.01`

#### Option 2: Predefined Schedule

| Parameter | Type | Description |
|-----------|------|-------------|
| `--schedule NAME` | choice | Use predefined cycle schedule |

**Available Schedules:**

| Schedule | Cycles | Description | Use Case |
|----------|--------|-------------|----------|
| `quick` | 5 | Fast exploration (1 random + 4 greedy) | Quick testing |
| `standard` | 10 | Balanced approach (1 random + 9 greedy) | Default screening |
| `intensive` | 20 | Thorough screening (1 random + 19 greedy) | Large-scale campaigns |
| `diverse` | 10 | Mixed strategies (random + bitbirch + greedy + ucb) | Diverse exploration |

**Examples:**
```bash
learnm8 run compounds.csv --target Activity --featurizer morgan --learner rf --schedule intensive
```

#### Option 3: Configuration File

| Parameter | Type | Description |
|-----------|------|-------------|
| `--config PATH` | file | YAML or JSON configuration file |

**Example:**
```bash
learnm8 run --config experiment.yaml
```

Configuration files can specify all parameters including cycles. See [Configuration Files](configuration-files.md) for details.

#### Option 4: Simple Mode (Default)

If no cycle control option is specified, simple mode is used with these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n-cycles N` | integer | 10 | Number of active learning cycles |
| `--batch-fraction F` | float | 0.01 | Fraction of pool to select per cycle (1%) |
| `--strategy NAME` | string | `greedy` | Main acquisition strategy |
| `--initial-strategy NAME` | string | `random` | First cycle strategy |

**Example:**
```bash
learnm8 run compounds.csv --target Activity --featurizer morgan --learner gp \
  --n-cycles 15 --batch-fraction 0.005 --strategy greedy --initial-strategy random
```

### Pruning Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pruning-fraction F` | float | None | Fraction of pool to prune each cycle (0.0-0.9) |
| `--pruning-strategy NAME` | string | `score_based` | Pruning strategy |

**Pruning Strategy:**
- `score_based` - Remove lowest-scoring compounds based on model predictions

**Pruning Parameters:**
- `pruning_fraction` (float, 0.0-0.9): Fraction of pool to prune each cycle
- `pruning_threshold` (float, optional): Absolute score threshold for pruning (alternative to fraction)

**Example:**
```bash
learnm8 run large_library.csv --target Activity --featurizer morgan --learner rf \
  --pruning-fraction 0.3 --pruning-strategy score_based
```

### Acquisition Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--acquisition-params JSON` | string | JSON string of acquisition function parameters |

**Example (UCB with custom beta):**
```bash
learnm8 run compounds.csv --target Activity --featurizer morgan --learner gp \
  --cycles "ucb:0.01*10" \
  --acquisition-params '{"beta": 2.5}'
```

**Example (TopK with fraction):**
```bash
learnm8 run compounds.csv --target Activity --featurizer morgan --learner rf \
  --cycles "topk:0.01*8" \
  --acquisition-params '{"k_fraction": 0.1}'
```

### Output Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-o PATH` / `--output PATH` | directory | Auto-generated | Output directory for results |
| `--cache-dir PATH` | directory | `{output}/.cache` | HDF5 feature cache directory |
| `--quiet` | flag | False | Suppress progress output |

**Output Directory Structure:**
```
output_dir/
├── compounds_final.csv          # Final compound status
├── cycle_metrics.csv            # Per-cycle metrics
├── selection_history.csv        # Compound selection history
├── validation_report.csv        # SMILES validation results
├── experiment_config.json       # Complete configuration
└── .cache/                      # Feature cache (if cache-dir not specified)
    └── features_morgan.h5       # Cached molecular features
```

**Cache Directory:** The feature cache persists across runs, providing 100x speedup when reusing the same compounds and featurizer. Share cache across experiments by specifying a common `--cache-dir`.

### Advanced Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--random-state SEED` | integer | 42 | Random seed for reproducibility |
| `--mode MODE` | choice | Auto-detect | Execution mode: `run` or `benchmark` |

**Mode Auto-Detection:**
- CSV oracle → `benchmark` mode (full dataset prediction, comprehensive metrics)
- Python oracle → `run` mode (unlabeled pool prediction, basic metrics)
- Override auto-detection with `--mode` flag if needed

## learnm8 list

List available components in your environment.

### Syntax

```bash
learnm8 list COMPONENT
```

### Component Types

| Component | Description | Example Output |
|-----------|-------------|----------------|
| `learners` | ML models available | `rf`, `gp`, `xgb`, `chemprop`, ... |
| `acquisition` | Acquisition strategies | `greedy`, `random`, `ucb`, `ei`, ... |
| `featurizers` | Molecular featurizers | `morgan`, `maccs`, `descriptors`, ... |
| `schedules` | Predefined cycle schedules | `quick`, `standard`, `intensive`, `diverse` |

### Examples

```bash
learnm8 list learners
```
Output:
```
Available Learners:
  • rf
  • gp
  • xgb
  • mlp
  • mc_dropout
  • ensemble
  • chemprop
```

```bash
learnm8 list acquisition
```
Output:
```
Available Acquisition Strategies:
  • greedy
  • random
  • topk
  • ucb
  • ei
  • pi
  • thompson
  • entropy
  • simulated_annealing
  • bitbirch
```

```bash
learnm8 list schedules
```
Output:
```
Available Schedules:
  • quick: 5 cycles - fast exploration
  • standard: 10 cycles - balanced
  • intensive: 20 cycles - thorough
  • diverse: 10 cycles - mixed diversity methods
```

## learnm8 validate

Validate compound pool using datamol SMILES validation before running experiments.

### Syntax

```bash
learnm8 validate COMPOUND_POOL [-o OUTPUT_DIR]
```

### Parameters

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `COMPOUND_POOL` | Yes | file | CSV file with compounds (must have ID and SMILES columns) |
| `-o PATH` / `--output PATH` | No | directory | Output directory for validation report |

### Validation Process

The validation command:
1. Reads compound CSV file
2. Validates each SMILES string using datamol
3. Identifies valid and invalid compounds
4. Reports success rate and sample errors
5. Optionally saves detailed validation report

### Output

**Console Output:**
```
Validating: compounds.csv
Total compounds: 1000

✓ Validation complete
Valid compounds: 987 (98.7%)
Invalid compounds: 13

Sample errors:
  comp_123: Invalid SMILES syntax
  comp_456: Aromatic bonds on sp3 atoms
  comp_789: Valence error for atom C
  ... and 10 more
```

**Validation Report (if `-o` specified):**

File: `{output_dir}/validation_report.csv`

| Column | Description |
|--------|-------------|
| ID | Compound identifier |
| SMILES | Invalid SMILES string |
| error | Validation error message |

### Examples

**Basic Validation:**
```bash
learnm8 validate compounds.csv
```

**Save Validation Report:**
```bash
learnm8 validate compounds.csv -o validation_results/
```

**Recommended Workflow:**
```bash
learnm8 validate compounds.csv -o validation/

learnm8 run compounds.csv --target Activity --featurizer morgan --learner rf --n-cycles 10
```

## Complete Examples

### Example 1: Simple Benchmark Experiment

```bash
learnm8 run ESSENCE_benchmark_input/ADA.csv \
  --target Activity \
  --featurizer morgan \
  --learner rf \
  --n-cycles 15
```

**What this does:**
- Uses CSV file as both compound pool and oracle (benchmark mode)
- Random Forest model with Morgan fingerprints
- 15 cycles (1 random + 14 greedy by default)
- Auto-generated output directory

### Example 2: Production Screening with Custom Oracle

```bash
learnm8 run compound_library.csv docking_module.py:calculate_binding \
  --target binding_score \
  --featurizer morgan \
  --learner mc_dropout \
  --cycles "random:0.01 ucb:0.005*10 greedy:0.01*5" \
  --cache-dir .shared_cache \
  -o screening_results/
```

**What this does:**
- Custom Python oracle for docking calculations
- MC Dropout for uncertainty quantification
- Multi-strategy cycles: random → UCB (10 cycles) → greedy (5 cycles)
- Shared feature cache for speed
- Explicit output directory

### Example 3: Large Library with Pruning

```bash
learnm8 run large_library.csv oracle.py:score \
  --target Activity \
  --featurizer descriptors \
  --learner ensemble \
  --schedule intensive \
  --pruning-fraction 0.3 \
  --pruning-strategy score_based \
  -o large_screen/
```

**What this does:**
- Ensemble learner for robustness
- Intensive schedule (20 cycles)
- Prune bottom 30% each cycle to reduce library size
- Mordred descriptors for rich chemical features

### Example 4: Uncertainty-Based Exploration

```bash
learnm8 run compounds.csv --target Activity \
  --featurizer morgan \
  --learner gp \
  --cycles "random:0.02 ucb:0.01*5 ei:0.01*3 greedy:0.01*2" \
  --acquisition-params '{"beta": 3.0}' \
  --random-state 12345
```

**What this does:**
- Gaussian Process for uncertainty estimation
- Mixed acquisition: random → UCB → Expected Improvement → greedy
- Custom UCB beta parameter (more exploration)
- Explicit random seed for reproducibility

### Example 5: Graph Neural Network Screening

```bash
learnm8 run compounds.csv --target Activity \
  --learner chemprop \
  --schedule standard \
  -o chemprop_results/
```

**What this does:**
- Chemprop (no featurizer needed, works with SMILES directly)
- Standard 10-cycle schedule
- Graph neural network for state-of-the-art predictions

### Example 6: Configuration File Workflow

```bash
learnm8 validate compounds.csv -o validation/

learnm8 run --config experiment.yaml
```

**experiment.yaml:**
```yaml
compound_pool: compounds.csv
oracle: oracle.csv
target_col: Activity
featurizer: morgan
learner: ensemble
cycles:
  - strategy: random
    batch_fraction: 0.02
    n_cycles: 1
  - strategy: greedy
    batch_fraction: 0.01
    n_cycles: 9
output_dir: results/
random_state: 42
```

## Global Behavior

### Reproducibility

All experiments use a random seed for reproducibility. Set explicitly with:
```bash
--random-state 42
```

Default seed is 42. Change for independent runs.

### Progress Output

By default, LearnM8 shows rich console output with progress bars and result tables. Suppress with:
```bash
--quiet
```

Useful for automated workflows or logging to files.

### Error Handling

LearnM8 provides clear error messages with suggestions:
- Missing files → Check path
- Invalid parameters → Show valid choices
- Dependency errors → Installation instructions
- SMILES validation errors → Use `validate` subcommand first

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (file not found, invalid parameter, etc.) |

## Best Practices

1. **Validate compounds first**: Always run `learnm8 validate` before experiments
2. **Use explicit cycles**: Prefer `--cycles` over simple mode for precise control
3. **Share cache**: Specify `--cache-dir` for experiments using same compounds
4. **Name outputs**: Use `-o` with descriptive directory names
5. **Set random state**: Specify `--random-state` for reproducible experiments
6. **Check available components**: Use `learnm8 list` to discover options
7. **Start with schedules**: Use predefined schedules before custom cycles
8. **Configuration files**: Use YAML configs for complex experiments and reproducibility

## Performance Tips

1. **Feature caching**: First run computes features, subsequent runs are 100x faster with cache
2. **Shared cache**: Reuse cache across experiments: `--cache-dir .shared_cache`
3. **Batch fractions**: Smaller batches (0.005-0.01) balance cost vs performance
4. **Pruning**: Use `--pruning-fraction` for libraries >100k compounds
5. **Fast models**: Use `rf` or `xgb` for rapid iteration
6. **GPU acceleration**: Use `chemprop` or `mlp` with GPU for large datasets

## Troubleshooting

**Problem:** "Compound pool file not found"
```bash
learnm8 run nonexistent.csv --target Activity --featurizer morgan --learner rf
```
**Solution:** Check file path is correct and file exists

**Problem:** "Invalid SMILES" errors during experiment
```bash
learnm8 validate compounds.csv -o validation/
```
**Solution:** Fix invalid SMILES before running experiment

**Problem:** "Learner 'chemprop' not available"
```bash
pip install chemprop
learnm8 list learners
```
**Solution:** Install optional dependency

**Problem:** "BitBIRCH not available" when using diverse schedule
```bash
pip install git+https://github.com/mqcomplab/bitbirch.git
```
**Solution:** Install optional BitBIRCH dependency or use different schedule

## See Also

- [API Reference](api-reference.md) - Python API documentation
- [Configuration Files](configuration-files.md) - YAML/JSON configuration format
- [Running Experiments Tutorial](../tutorials/running-experiments.md) - Step-by-step guide
- [Building Custom Cycles](../tutorials/building-custom-cycles.md) - Cycle specification guide

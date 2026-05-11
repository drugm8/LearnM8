# Python API Reference

Complete reference documentation for LearnM8's Python API.

## Table of Contents

1. [Core API](#core-api)
   - [run_active_learning()](#run_active_learning)
2. [Helper Functions](#helper-functions)
   - [list_available_learners()](#list_available_learners)
   - [validate_compound_pool()](#validate_compound_pool)
   - [extract_features()](#extract_features)
3. [Configuration Classes](#configuration-classes)
   - [CycleConfig](#cycleconfig)
   - [ValidationResult](#validationresult)
4. [Usage Examples](#usage-examples)

---

## Core API

### run_active_learning()

Main entry point for executing active learning experiments.

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01
)
```

#### Complete Function Signature

```python
def run_active_learning(
    compound_pool: Union[str, Path, pl.DataFrame],
    oracle: Union[str, Path, Oracle],
    learner: Union[str, Learner],
    target_col: str,
    featurizer: Optional[str] = None,
    smiles_column: Optional[str] = None,
    id_column: Optional[str] = None,
    # Advanced API
    cycles: Optional[List[CycleConfig]] = None,
    # Simple API
    n_cycles: int = 10,
    batch_fraction: float = 0.01,
    strategy: str = 'greedy',
    initial_strategy: str = 'random',
    # Common parameters
    score_direction: str = 'higher',
    mode: Optional[Literal['run', 'benchmark']] = None,
    output_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    # Chemprop fine-tuning
    enable_chemprop_fine_tuning: bool = False,
    # Pruning
    pruning_fraction: Optional[float] = None,
    pruning_strategy: Optional[str] = None,
    pruning_params: Optional[Dict] = None,
    # Acquisition
    acquisition_params: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]
```

#### Parameters

##### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| **compound_pool** | `str`, `Path`, or `pl.DataFrame` | Compound pool specification. Can be: path to CSV file (must have 'ID' and 'SMILES' columns), or Polars DataFrame with 'ID' and 'SMILES' columns. |
| **oracle** | `str`, `Path`, `Oracle`, or `None` | Oracle specification. Can be: `None` (auto-detect from compound_pool in benchmark mode), CSV file path (benchmark mode with ground truth), `'module.py:function'` (production mode with custom scoring), or Oracle instance. |
| **learner** | `str` or `Learner` | Learner specification. Can be: string shortcut (`'rf'`, `'gp'`, `'xgb'`, `'mlp'`, `'mc_dropout'`, `'fastprop'`, `'chemprop'`, `'ensemble'`, `'rf_ensemble'`, `'lr_ensemble'`, `'xgb_ensemble'`, `'dt_ensemble'`, `'mixed_ensemble'`, `'fastprop_ensemble'`, `'chemprop_ensemble'`), or custom Learner instance. |
| **target_col** | `str` | Target property column name in the compound pool. |

**Performance Notes:**
- **compound_pool**: CSV files are read with Polars for fast loading. Large files (>1M compounds) benefit from batch processing.
- **oracle**: CSVOracle loads entire dataset into memory for fast lookups. For large datasets (>10M compounds), consider custom Oracle with database backend.
- **learner**: Chemprop learners work directly with SMILES (no featurization overhead). Feature-based learners benefit from HDF5 caching for 100x speedup on repeated runs.

##### Feature Extraction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **smiles_column** | `str` or `None` | `None` | Column name for SMILES in the input file (CSV only). Auto-detects from common names (`'SMILES'`, `'smiles'`, `'Smiles'`) when `None`. Ignored for DataFrames and non-CSV files. |
| **id_column** | `str` or `None` | `None` | Column name for compound ID in the input file (CSV/SDF). Auto-detects from `'ID'` or generates synthetic IDs when `None`. Ignored when `compound_pool` is a DataFrame. |
| **featurizer** | `str` or `None` | `None` | Molecular featurizer type. Optional for SMILES-aware learners (e.g., `'chemprop'`). Required for feature-based learners. Valid options: `'morgan'` (2048-bit circular fingerprints, radius=2), `'maccs'` (167-bit structural keys), `'ecfp6'` (2048-bit extended-connectivity, radius=3), `'descriptors'` (1613 Mordred descriptors), `'morgan_feat'` (2048-bit feature fingerprints). |

**Performance Notes:**
- Morgan fingerprints: Fast computation (~1000 compounds/sec), recommended for most applications
- MACCS keys: Fastest (~5000 compounds/sec), good for similarity-based approaches
- ECFP6: Similar speed to Morgan, larger radius for complex patterns
- Descriptors: Slowest (~100 compounds/sec), best for linear models and interpretability
- All featurizers use HDF5 caching for 100x speedup on cache hits

**When to Use:**
- `morgan`: Default choice for random forests, neural networks, and general-purpose ML
- `maccs`: When speed is critical and structural keys suffice (e.g., quick similarity screening)
- `ecfp6`: For capturing larger structural motifs in complex molecules
- `descriptors`: For linear models (GPs, linear regression) and when interpretability matters
- `None`: Only with SMILES-aware learners like Chemprop (pure graph-based learning)

##### Cycle Control

**Advanced API** (full control):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **cycles** | `List[CycleConfig]` or `None` | `None` | List of CycleConfig objects for full control. If provided, overrides simple API parameters. Each CycleConfig can specify: strategy, n_cycles, batch_fraction, pruning_strategy, pruning_params, acquisition_params. See [CycleConfig](#cycleconfig) for details. |

**Simple API** (quick setup):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **n_cycles** | `int` | `10` | Total number of active learning cycles. Cycle 0 (initialization) uses `initial_strategy`, cycles 1+ use `strategy`. |
| **batch_fraction** | `float` | `0.01` | Fraction of original pool to select per cycle (applies to ALL cycles including cycle 0). Value must be in range (0, 1]. Calculated from original pool size for consistency across cycles. |
| **strategy** | `str` | `'greedy'` | Acquisition strategy for cycles 1+. Valid options: `'greedy'` (exploit best predictions), `'random'` (random selection), `'topk'` (top-k selection), `'ucb'` (upper confidence bound), `'ei'` (expected improvement), `'pi'` (probability of improvement), `'thompson'` (Thompson sampling), `'entropy'` (maximum entropy), `'simulated_annealing'`, `'bitbirch'` (diversity-based, requires optional dependency). |
| **initial_strategy** | `str` | `'random'` | Acquisition strategy for cycle 0 (initialization). Typically `'random'` for unbiased initial sampling. |

**Performance Notes:**
- Smaller batch_fraction (e.g., 0.005) → more cycles, better exploration, longer runtime
- Larger batch_fraction (e.g., 0.05) → fewer cycles, faster runtime, may miss optimal compounds
- Greedy strategy: Fastest acquisition (~0.1s per cycle), pure exploitation
- UCB/EI/PI: Slightly slower (~0.2s per cycle), balance exploration/exploitation
- BitBIRCH: Slowest (~1-10s per cycle depending on pool size), maximum diversity

**When to Use:**
- `greedy`: When exploitation is priority, fast feedback needed
- `random`: For baseline comparisons, unbiased sampling
- `ucb`: Good default for exploration-exploitation balance
- `ei`/`pi`: When statistical improvement guarantees matter
- `thompson`: For Bayesian approaches, natural exploration
- `entropy`: When uncertainty reduction is the goal
- `bitbirch`: When diversity is critical (avoid redundant compounds)

##### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **score_direction** | `str` | `'higher'` | Optimization direction. `'higher'` for maximization (e.g., activity, binding affinity), `'lower'` for minimization (e.g., toxicity, cost). |
| **mode** | `'run'`, `'benchmark'`, or `None` | `None` | Execution mode. `None` (auto-detect from oracle type), `'run'` (production screening, basic metrics only), `'benchmark'` (evaluation with ground truth, includes discovery/ranking metrics). |
| **output_dir** | `str`, `Path`, or `None` | `None` | Output directory path. If `None`, auto-generates timestamped directory: `learnm8_results_YYYYMMDD_HHMMSS`. |
| **cache_dir** | `str`, `Path`, or `None` | `None` | Cache directory for HDF5 feature storage. If `None`, uses `{output_dir}/.cache`. Can be shared across experiments for maximum speedup. |
| **random_state** | `int` | `42` | Random seed for reproducibility. Affects: initial batch selection, learner initialization, acquisition sampling. |

**Performance Notes:**
- **mode='benchmark'**: Computes full dataset predictions each cycle for accurate metrics (slower but comprehensive)
- **mode='run'**: Predicts only on unlabeled compounds (faster, suitable for production)
- **cache_dir**: Shared cache directory across experiments provides 100x speedup for features
- **random_state**: Ensures reproducible results across runs with same parameters

**When to Use:**
- `mode='benchmark'`: When evaluating active learning performance, comparing strategies, or publishing results
- `mode='run'`: For production screening where speed matters and ground truth unavailable
- Shared `cache_dir`: When running parameter sweeps or multiple experiments on same compound pool

##### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **enable_chemprop_fine_tuning** | `bool` | `False` | Enable incremental fine-tuning for Chemprop models. When enabled with `learner='chemprop'`, models are saved after each cycle and loaded for fine-tuning in subsequent cycles, potentially reducing training time by 50-90%. Checkpoints saved to: `{output_dir}/.checkpoints/chemprop/`. Ignored if learner is pre-instantiated (configure on learner instance instead). |
| **pruning_fraction** | `float` or `None` | `None` | Fraction of compounds to prune per cycle (range: 0.0-0.9). If provided, enables pruning with score-based strategy. Pruned compounds are removed from future consideration, reducing computational cost. |
| **pruning_strategy** | `str` or `None` | `None` | Pruning strategy name. Default: `'score'` (prune lowest predicted scores). If `pruning_fraction` is provided but `pruning_strategy` is `None`, defaults to `'score'`. |
| **pruning_params** | `dict` or `None` | `None` | Additional pruning parameters. If `pruning_fraction` provided, automatically added to this dict as `{'pruning_fraction': value}`. |
| **acquisition_params** | `dict` or `None` | `None` | Additional acquisition parameters. Strategy-specific parameters: UCB (`{'exploration_weight': 2.0}`), BitBIRCH (`{'n_clusters': 100}`), etc. |
| **prediction_batch_size** | `int` or `None` | `None` | Batch size for memory-efficient prediction. Uses unified always-batch approach: for small datasets (≤100k), batch_size equals dataset length (single iteration, zero overhead). For large datasets (>100k), uses auto-calculated batch size based on memory and featurizer type. Set to a specific integer to override auto-calculation. Minimum: 100, recommended range: 1000-50000. |

**Performance Notes:**
- **enable_chemprop_fine_tuning**: Dramatically reduces training time in later cycles (50-90% faster) but requires disk space for checkpoints
- **pruning_fraction**: Reduces prediction time in later cycles (e.g., 0.3 pruning → 30% fewer compounds to predict)
- **prediction_batch_size**: Auto-calculated batch sizes prevent out-of-memory errors for large libraries (100k+ compounds). Manual override useful for specific hardware constraints (e.g., limited RAM or GPU memory)
- Aggressive pruning (>0.5) risks pruning potentially good compounds, use conservatively

**When to Use:**
- **enable_chemprop_fine_tuning**: For long experiments (>10 cycles) with Chemprop where training time dominates
- **pruning_fraction**: For large compound pools (>100k compounds) where prediction time is bottleneck
- **prediction_batch_size**: For very large libraries (>100k compounds) or when working with limited memory. Auto-calculation works well in most cases, only override if experiencing memory issues
- **acquisition_params**: When fine-tuning acquisition behavior (e.g., more/less exploration in UCB)

#### Return Value

Returns a dictionary with the following structure:

```python
{
    'compounds_df': pl.DataFrame,        # Final master DataFrame with all cycle data
    'cycle_metrics': List[Dict],         # Metrics for each cycle (including cycle 0)
    'aggregate_metrics': Dict,           # Aggregate statistics across all cycles
    'validation_result': ValidationResult, # Compound validation results
    'output_dir': Path,                  # Path to output directory
    'saved_files': Dict[str, Path],      # Paths to saved CSV files
    'labeled_data': pl.DataFrame,        # Convenience accessor (compounds_df.filter(status='labeled'))
    'unlabeled_data': pl.DataFrame       # Convenience accessor (compounds_df.filter(status='unlabeled'))
}
```

##### compounds_df (Polars DataFrame)

Master DataFrame containing all compounds with cycle tracking:

| Column | Type | Description |
|--------|------|-------------|
| `ID` | `str` | Compound identifier |
| `SMILES` | `str` | SMILES string |
| `status` | `str` | Compound status: `'unlabeled'`, `'labeled'`, or `'pruned'` |
| `selected_cycle` | `int` or `null` | Cycle when compound was selected (null if never selected) |
| `labeled_cycle` | `int` or `null` | Cycle when compound was labeled (null if unlabeled) |
| `pruned_cycle` | `int` or `null` | Cycle when compound was pruned (null if not pruned) |
| `{target_col}` | `float` or `null` | Measured property value (null if unlabeled) |

##### cycle_metrics (List of Dictionaries)

Per-cycle metrics. Each dictionary contains:

**Common Metrics** (all modes):
| Metric | Description |
|--------|-------------|
| `cycle` | Cycle number (0 = initialization) |
| `n_selected` | Number of compounds selected this cycle |
| `n_labeled` | Total labeled compounds after this cycle |
| `remaining_unlabeled` | Unlabeled compounds remaining |
| `avg_score_selected` | Average oracle score of selected compounds |
| `best_so_far` | Best oracle score found so far |
| `total_time` | Total cycle execution time (seconds) |
| `training_time` | Model training time (seconds) |
| `prediction_time` | Prediction time (seconds) |
| `acquisition_time` | Acquisition function time (seconds) |
| `oracle_time` | Oracle measurement time (seconds) |
| `evaluation_time` | Metric computation time (seconds) |

**Benchmark Mode Metrics** (mode='benchmark' only):
| Metric | Description |
|--------|-------------|
| `top_10_discovery` | Fraction of top 10 compounds discovered |
| `top_100_discovery` | Fraction of top 100 compounds discovered |
| `cumulative_ef` | Cumulative enrichment factor |
| `batch_score_improvement_ratio` | Sign-aware improvement of batch mean over population mean (`>0` = better; see feature 019 CHANGELOG) |
| `unlabeled_spearman_correlation` | Spearman correlation on unlabeled set |
| `unlabeled_top_100_overlap` | Top 100 overlap on unlabeled set |
| `unlabeled_ef_1_0` | Enrichment factor (1%) on unlabeled set |

##### aggregate_metrics (Dictionary)

Summary statistics across all cycles:

**Common Aggregate Metrics**:
| Metric | Description |
|--------|-------------|
| `total_cycles` | Total number of cycles executed |
| `total_labeled` | Total compounds labeled |
| `total_pruned` | Total compounds pruned |
| `avg_selection_quality` | Mean of `avg_score_selected` across cycles |
| `std_selection_quality` | Std dev of `avg_score_selected` |
| `best_compound_value` | Best oracle score found |
| `best_compound_found_cycle` | Cycle when best compound was found |

**Benchmark Mode Aggregate Metrics**:
| Metric | Description |
|--------|-------------|
| `final_top_10_discovery` | Final top 10 discovery rate |
| `avg_top_10_discovery` | Average top 10 discovery across cycles |
| `final_top_100_discovery` | Final top 100 discovery rate |
| `avg_top_100_discovery` | Average top 100 discovery across cycles |
| `final_cumulative_ef` | Final cumulative enrichment factor |
| `avg_cumulative_ef` | Average enrichment factor across cycles |
| `avg_batch_score_ratio` | Average batch score ratio |
| `final_unlabeled_spearman` | Final Spearman correlation on unlabeled |
| `avg_unlabeled_spearman` | Average Spearman correlation |
| `final_unlabeled_top_100_overlap` | Final top 100 overlap on unlabeled |
| `avg_unlabeled_top_100_overlap` | Average top 100 overlap |
| `final_unlabeled_ef_1_0` | Final enrichment factor on unlabeled |
| `avg_unlabeled_ef_1_0` | Average enrichment factor on unlabeled |

##### validation_result (ValidationResult)

Compound validation results (see [ValidationResult](#validationresult) for details).

##### saved_files (Dictionary)

Paths to saved CSV files:

```python
{
    'compounds': Path('output_dir/compounds.csv'),
    'cycle_metrics': Path('output_dir/cycle_metrics.csv'),
    'validation_report': Path('output_dir/validation_report.csv'),
    'config': Path('output_dir/config.json')
}
```

#### Raises

| Exception | When |
|-----------|------|
| `FileNotFoundError` | If compound_pool CSV file not found |
| `ValueError` | If validation fails, invalid parameters, or missing required columns |
| `TypeError` | If invalid input types |
| `RuntimeError` | If cycle execution fails |

#### Notes

**Oracle Auto-Detection:**
- When `oracle=None` and `compound_pool` is CSV path → uses that CSV as benchmark oracle
- When `oracle=None` and `compound_pool` is DataFrame → raises error (oracle required)
- CSV oracle files automatically trigger `mode='benchmark'`
- Python function oracles (`module.py:function`) automatically trigger `mode='run'`

**Feature Extraction:**
- All learners use the same featurizer if specified
- Chemprop can use features as extra descriptors (x_d) or work purely with graphs
- Features are cached in HDF5 format for 100x speedup on repeated access

**Reproducibility:**
- Set `random_state` for deterministic results
- Same `random_state` produces identical results with same parameters
- Different learners may have additional randomness sources

**Performance Tips:**
- Use shared `cache_dir` across experiments for maximum speedup
- Enable Chemprop fine-tuning for long experiments (>10 cycles)
- Use pruning for large compound pools (>100k compounds)
- Benchmark mode is slower but provides comprehensive metrics

---

## Helper Functions

### list_available_learners()

Return list of available learner shortcuts.

```python
from learnm8.api import list_available_learners

learners = list_available_learners()
print(learners)
# ['chemprop', 'chemprop_ensemble', 'dt_ensemble', 'ensemble', 'fastprop', ...]
```

#### Signature

```python
def list_available_learners() -> List[str]
```

#### Returns

List of learner shortcut strings. Only includes learners whose imports succeeded (dependencies available).

**Common Learners:**
- `'rf'`: Random Forest (always available)
- `'gp'`: Gaussian Process (always available)
- `'xgb'`: XGBoost (requires xgboost)
- `'mlp'`: Multi-Layer Perceptron (requires torch)
- `'mc_dropout'`: MC Dropout (requires torch)
- `'fastprop'`: FastProp (requires torch, pytorch-lightning)
- `'chemprop'`: Chemprop (requires chemprop, torch)
- `'ensemble'`: Mixed Ensemble (requires multiple backends)
- `'rf_ensemble'`, `'lr_ensemble'`, `'xgb_ensemble'`, `'dt_ensemble'`: Single-model ensembles
- `'fastprop_ensemble'`, `'chemprop_ensemble'`: Neural ensemble variants

#### Example

```python
import sys
from learnm8.api import list_available_learners

available = list_available_learners()

if 'chemprop' in available:
    print("Chemprop is available!")
else:
    print("Chemprop not available. Install with: pip install chemprop")
    sys.exit(1)
```

---

### validate_compound_pool()

Validate compound pool with parallel datamol-based validation.

```python
from learnm8 import validate_compound_pool
import polars as pl

compounds = pl.read_csv('compounds.csv')
result = validate_compound_pool(compounds, n_jobs=-1, progress=True)

print(f"Valid: {len(result.valid_compounds)}")
print(f"Invalid: {len(result.invalid_compounds)}")
print(f"Success rate: {result.success_rate:.1%}")
```

#### Signature

```python
def validate_compound_pool(
    compound_pool: pl.DataFrame,
    n_jobs: int = -1,
    progress: bool = True
) -> ValidationResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **compound_pool** | `pl.DataFrame` | required | DataFrame with 'ID' and 'SMILES' columns |
| **n_jobs** | `int` | `-1` | Number of parallel jobs. `-1` uses all CPU cores. `1` for sequential processing. |
| **progress** | `bool` | `True` | Show progress bar (uses tqdm if available) |

**Performance Notes:**
- Parallel processing provides 50x speedup over sequential validation
- Uses datamol's process-based parallelization for true multi-core scaling
- Memory usage scales with n_jobs (each process loads RDKit)
- For small datasets (<100 compounds), sequential may be faster due to overhead

**When to Use:**
- Always validate before running experiments to catch errors early
- Use `n_jobs=-1` for maximum speed on large datasets
- Set `progress=True` for long validations (>10k compounds)

#### Returns

[ValidationResult](#validationresult) object with:
- `valid_compounds`: DataFrame with validated compounds
- `invalid_compounds`: DataFrame with failed compounds
- `validation_errors`: Dict mapping compound IDs to error messages
- `success_rate`: Validation success rate (0.0-1.0)

#### Example

```python
from learnm8 import validate_compound_pool
import polars as pl

compounds = pl.DataFrame({
    'ID': ['cmp1', 'cmp2', 'cmp3'],
    'SMILES': ['CCO', 'invalid', 'CCC']
})

result = validate_compound_pool(compounds)

print(f"Success rate: {result.success_rate:.1%}")

for compound_id, error in result.validation_errors.items():
    print(f"{compound_id}: {error}")

if len(result.invalid_compounds) > 0:
    result.invalid_compounds.write_csv('invalid_compounds.csv')
```

---

### extract_features()

Extract molecular features with caching and parallel processing.

```python
from learnm8 import extract_features
from pathlib import Path

smiles_list = ['CCO', 'CCC', 'CCCC']
features = extract_features(
    smiles_list,
    featurizer='morgan',
    cache_dir=Path('.cache'),
    n_jobs=-1,
    show_progress=True
)

print(features.shape)  # (3, 2048)
```

#### Signature

```python
def extract_features(
    smiles_list: List[str],
    featurizer: str,
    cache_dir: Optional[Path] = None,
    n_jobs: int = -1,
    show_progress: bool = False
) -> np.ndarray
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **smiles_list** | `List[str]` | required | List of SMILES strings |
| **featurizer** | `str` | required | Type of featurizer. Valid options: `'morgan'` (2048-bit, radius=2), `'maccs'` (167-bit), `'ecfp6'` (2048-bit, radius=3), `'morgan_feat'` (2048-bit feature fingerprints), `'descriptors'` (1613 Mordred descriptors) |
| **cache_dir** | `Path` or `None` | `None` | Directory for HDF5 cache files. If `None`, uses `.cache` in current directory. Cache persists across runs for 100x speedup. |
| **n_jobs** | `int` | `-1` | Number of parallel jobs. `-1` auto-detects optimal parallelization based on dataset size. `1` for sequential processing. |
| **show_progress** | `bool` | `False` | Show progress bar for long operations (requires tqdm) |

**Performance Notes:**
- **Automatic Parallelization**:
  - <100 compounds: Sequential (overhead > benefit)
  - 100-10k: All CPU cores
  - >10k: Cap at 32 cores (diminishing returns)
- **HDF5 Caching**:
  - First run: ~1000 compounds/sec (Morgan)
  - Cache hit: 100x faster (~100k compounds/sec)
  - Cache indexed by SMILES hash for fast lookup
- **Featurizer Speed** (approximate, single-threaded):
  - MACCS: ~5000 compounds/sec
  - Morgan/ECFP6: ~1000 compounds/sec
  - Descriptors: ~100 compounds/sec

**When to Use:**
- Use `cache_dir` for any repeated computation (parameter sweeps, multiple experiments)
- Share `cache_dir` across experiments on same compound pool
- Set `n_jobs=-1` for automatic optimization
- Use `show_progress=True` for datasets >10k compounds

#### Returns

NumPy array of features with shape `(n_compounds, n_features)`:
- Morgan/ECFP6/Morgan_feat: (n_compounds, 2048)
- MACCS: (n_compounds, 167)
- Descriptors: (n_compounds, 1613)

#### Raises

| Exception | When |
|-----------|------|
| `ValueError` | If featurizer is unknown or SMILES is invalid |

#### Example

```python
import polars as pl
from learnm8 import extract_features
from pathlib import Path

compounds = pl.read_csv('compounds.csv')
smiles_list = compounds['SMILES'].to_list()

cache_dir = Path('.shared_cache')
cache_dir.mkdir(exist_ok=True)

features = extract_features(
    smiles_list,
    featurizer='morgan',
    cache_dir=cache_dir,
    n_jobs=-1,
    show_progress=True
)

print(f"Feature matrix: {features.shape}")
print(f"Cache hits on second run:")

features2 = extract_features(
    smiles_list,
    featurizer='morgan',
    cache_dir=cache_dir
)
```

---

## Configuration Classes

### CycleConfig

Configuration for a single cycle or group of cycles.

```python
from learnm8 import CycleConfig

config = CycleConfig(
    strategy='greedy',
    n_cycles=5,
    batch_fraction=0.01
)
```

#### Signature

```python
@dataclass
class CycleConfig:
    strategy: str
    n_cycles: int = 1
    batch_fraction: Optional[float] = None
    pruning_strategy: Optional[str] = None
    pruning_params: Optional[Dict] = None
    acquisition_params: Optional[Dict] = None
```

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| **strategy** | `str` | required | Acquisition strategy name (e.g., `'greedy'`, `'ucb'`, `'random'`, `'ei'`, `'bitbirch'`) |
| **n_cycles** | `int` | `1` | Number of cycles with this configuration |
| **batch_fraction** | `float` | required | Fraction of original pool to select per cycle (0 < value ≤ 1) |
| **pruning_strategy** | `str` or `None` | `None` | Pruning strategy name (e.g., `'score'`) |
| **pruning_params** | `dict` or `None` | `None` | Parameters for pruning strategy (e.g., `{'pruning_fraction': 0.3}`) |
| **acquisition_params** | `dict` or `None` | `None` | Parameters for acquisition strategy (e.g., `{'exploration_weight': 2.0}` for UCB) |

#### Example

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01),
        CycleConfig('ucb', n_cycles=4, batch_fraction=0.005,
                   acquisition_params={'exploration_weight': 2.0}),
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01,
                   pruning_strategy='score',
                   pruning_params={'pruning_fraction': 0.3})
    ]
)
```

---

### ValidationResult

Result of compound pool validation.

```python
from learnm8 import validate_compound_pool

result = validate_compound_pool(compounds)
print(f"Success rate: {result.success_rate:.1%}")
```

#### Signature

```python
@dataclass
class ValidationResult:
    valid_compounds: pl.DataFrame
    invalid_compounds: pl.DataFrame
    validation_errors: Dict[str, str]

    @property
    def success_rate(self) -> float:
        """Calculate validation success rate (0.0-1.0)."""
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| **valid_compounds** | `pl.DataFrame` | DataFrame containing compounds that passed validation |
| **invalid_compounds** | `pl.DataFrame` | DataFrame containing compounds that failed validation |
| **validation_errors** | `dict` | Dictionary mapping compound IDs to error messages |
| **success_rate** | `float` | Validation success rate as fraction (0.0-1.0) |

#### Example

```python
from learnm8 import validate_compound_pool
import polars as pl

compounds = pl.read_csv('compounds.csv')
result = validate_compound_pool(compounds)

print(f"Total compounds: {len(compounds)}")
print(f"Valid: {len(result.valid_compounds)}")
print(f"Invalid: {len(result.invalid_compounds)}")
print(f"Success rate: {result.success_rate:.1%}")

if result.success_rate < 0.9:
    print("\nValidation errors:")
    for compound_id, error in result.validation_errors.items():
        print(f"  {compound_id}: {error}")

    result.invalid_compounds.write_csv('invalid_compounds.csv')
```

---

## Usage Examples

### Example 1: Simple API - Quick Start

Basic active learning with default settings:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01
)

print(f"Labeled compounds: {len(results['labeled_data'])}")
print(f"Best compound: {results['aggregate_metrics']['best_compound_value']:.3f}")
```

### Example 2: Advanced API - CycleConfig

Multi-stage strategy with different batch sizes and pruning:

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),

        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01),

        CycleConfig('ucb', n_cycles=5, batch_fraction=0.01,
                   acquisition_params={'exploration_weight': 2.0}),

        CycleConfig('greedy', n_cycles=4, batch_fraction=0.01,
                   pruning_strategy='score',
                   pruning_params={'pruning_fraction': 0.3})
    ],
    output_dir='results_multistage',
    random_state=42
)

cycle_metrics = results['cycle_metrics']
for i, metrics in enumerate(cycle_metrics):
    print(f"Cycle {i}: avg_score={metrics['avg_score_selected']:.3f}, "
          f"best={metrics['best_so_far']:.3f}")
```

### Example 3: Custom Learner

Using custom learner with advanced configuration:

```python
from learnm8 import run_active_learning
from learnm8.learners import GaussianProcessLearner
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

custom_kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
learner = GaussianProcessLearner(
    kernel=custom_kernel,
    alpha=1e-6,
    normalize_y=True,
    random_state=42
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=learner,
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01,
    strategy='ucb',
    acquisition_params={'exploration_weight': 1.5}
)
```

### Example 4: Custom Oracle (Production Mode)

Using custom scoring function for production screening:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='unlabeled_library.csv',
    oracle='scoring_module.py:calculate_affinity',
    learner='ensemble',
    target_col='binding_score',
    featurizer='morgan',
    n_cycles=20,
    batch_fraction=0.005,
    strategy='ucb',
    score_direction='higher',
    mode='run',
    output_dir='screening_results'
)

labeled = results['labeled_data']
top_compounds = labeled.sort('binding_score', descending=True).head(100)
top_compounds.write_csv('top_100_candidates.csv')
```

### Example 5: Chemprop with Fine-Tuning

Graph neural network with incremental fine-tuning:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',
    target_col='Activity',
    featurizer=None,
    n_cycles=15,
    batch_fraction=0.01,
    strategy='greedy',
    enable_chemprop_fine_tuning=True,
    output_dir='chemprop_finetuned',
    random_state=42
)

print(f"Training time saved: ~{50*(len(results['cycle_metrics'])-1):.0f}% in later cycles")
```

### Example 6: Parameter Sweep

Comparing different learners and strategies:

```python
from learnm8 import run_active_learning
from pathlib import Path
import polars as pl

shared_cache = Path('.shared_cache')
shared_cache.mkdir(exist_ok=True)

learners = ['rf', 'gp', 'xgb', 'ensemble']
strategies = ['greedy', 'ucb', 'ei']

results_summary = []

for learner in learners:
    for strategy in strategies:
        print(f"Running: {learner} + {strategy}")

        results = run_active_learning(
            compound_pool='compounds.csv',
            oracle='oracle.csv',
            learner=learner,
            target_col='Activity',
            featurizer='morgan',
            n_cycles=10,
            batch_fraction=0.01,
            strategy=strategy,
            cache_dir=shared_cache,
            output_dir=f'results_{learner}_{strategy}',
            random_state=42
        )

        results_summary.append({
            'learner': learner,
            'strategy': strategy,
            'best_value': results['aggregate_metrics']['best_compound_value'],
            'avg_quality': results['aggregate_metrics']['avg_selection_quality'],
            'top_10_discovery': results['aggregate_metrics'].get('final_top_10_discovery', None)
        })

summary_df = pl.DataFrame(results_summary)
summary_df.write_csv('parameter_sweep_results.csv')
print(summary_df.sort('best_value', descending=True))
```

### Example 7: Validation Before Running

Pre-validate compounds to catch errors early:

```python
from learnm8 import validate_compound_pool, run_active_learning
import polars as pl

compounds = pl.read_csv('compounds.csv')

validation_result = validate_compound_pool(compounds, n_jobs=-1, progress=True)

if validation_result.success_rate < 0.95:
    print(f"Warning: Only {validation_result.success_rate:.1%} compounds valid")
    print("\nValidation errors:")
    for cid, error in list(validation_result.validation_errors.items())[:10]:
        print(f"  {cid}: {error}")

    validation_result.invalid_compounds.write_csv('invalid_compounds.csv')

if len(validation_result.valid_compounds) == 0:
    raise ValueError("No valid compounds!")

results = run_active_learning(
    compound_pool=validation_result.valid_compounds,
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)
```

### Example 8: Large-Scale Screening with Pruning

Efficient screening of large compound library:

```python
from learnm8 import run_active_learning
from pathlib import Path

results = run_active_learning(
    compound_pool='large_library_1M.csv',
    oracle='docking_oracle.py:dock_compound',
    learner='chemprop',
    target_col='docking_score',
    featurizer=None,
    n_cycles=30,
    batch_fraction=0.001,
    strategy='ucb',
    pruning_fraction=0.2,
    enable_chemprop_fine_tuning=True,
    cache_dir=Path('/shared/cache'),
    output_dir='large_scale_screening',
    random_state=42
)

print(f"Screened: {len(results['labeled_data'])} / 1,000,000 compounds")
print(f"Pruned: {len(results['compounds_df'].filter(pl.col('status')=='pruned'))} low-value compounds")
print(f"Best score: {results['aggregate_metrics']['best_compound_value']:.3f}")

top_hits = results['labeled_data'].sort('docking_score', descending=True).head(1000)
top_hits.write_csv('top_1000_hits.csv')
```

---

## See Also

- [Quickstart Guide](../getting-started/quickstart.md) - Get started with LearnM8
- [Core Concepts](../getting-started/concepts.md) - Understanding active learning in LearnM8
- [Learners Guide](../components/learners/overview.md) - Available machine learning models
- [Acquisition Strategies](../components/acquisition/overview.md) - Compound selection methods
- [Running Experiments](../tutorials/running-experiments.md) - Performance tips and optimization

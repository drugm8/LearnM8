# Python API Reference

Complete reference for LearnM8's public Python API.

**Imports:**

```python
from learnm8 import (
    run_active_learning,
    validate_compound_pool,
    extract_features,
    CycleConfig,
    ValidationResult,
)
```

---

## run_active_learning()

```python
def run_active_learning(
    compound_pool: str | Path | pl.DataFrame,
    oracle: str | Path | Oracle,
    learner: str | Learner,
    target_col: str,
    featurizer: str | Featurizer | None = None,
    smiles_column: str | None = None,
    id_column: str | None = None,
    cycles: list[CycleConfig] | None = None,
    n_cycles: int = 10,
    batch_fraction: float = 0.01,
    strategy: str = 'greedy',
    initial_strategy: str = 'random',
    score_direction: str = 'higher',
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    random_state: int = 42,
    enable_chemprop_fine_tuning: bool = False,
    pruning_fraction: float | None = None,
    pruning_strategy: str | None = None,
    pruning_params: dict | None = None,
    acquisition_params: dict | None = None,
    memory_safety_factor: float = 0.7,
    n_jobs: int = -1,
    device: str = 'auto',
    large_features_ack: bool = False,
    output_format: Literal['auto', 'csv', 'parquet'] = 'auto',
    disable_molecular_similarity: bool | Iterable[str] = False,
    force_uncertainty: bool = False,
) -> dict[str, Any]
```

### Parameters

#### Input data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compound_pool` | `str \| Path \| pl.DataFrame` | — | Compound pool. CSV path (must have `ID` and `SMILES` columns) or a Polars DataFrame with the same. |
| `oracle` | `str \| Path \| Oracle \| None` | — | Measurement oracle. `None` auto-detects from `compound_pool`. CSV path → benchmark mode. `'module.py:function'` → production mode. Pre-built `Oracle` instance. |
| `learner` | `str \| Learner` | — | Learner shortcut string (e.g. `'rf'`, `'gp'`) or a `Learner` instance. See [Learners](../components/learners/overview.md). |
| `target_col` | `str` | — | Column name for the target property. |
| `featurizer` | `str \| Featurizer \| None` | `None` | Featurizer shortcut or instance. Required for feature-based learners; omit for SMILES-native learners (Chemprop). See [Featurizers](../components/featurizers/available-featurizers.md) — 39 options (38 unique). |
| `smiles_column` | `str \| None` | `None` | SMILES column name. Auto-detects `'SMILES'`, `'smiles'`, `'Smiles'` when `None`. Ignored for DataFrame input. |
| `id_column` | `str \| None` | `None` | ID column name. Auto-detects `'ID'` or generates synthetic IDs when `None`. Ignored for DataFrame input. |

#### Cycle control — simple API

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_cycles` | `int` | `10` | Total cycles. Cycle 0 uses `initial_strategy`; cycles 1+ use `strategy`. |
| `batch_fraction` | `float` | `0.01` | Fraction of the **original** pool per cycle. Floor at 1 compound; capped at remaining pool size. Must be in `(0, 1]`. |
| `strategy` | `str` | `'greedy'` | Acquisition strategy for cycles 1+. One of: `greedy`, `random`, `topk`, `ucb`, `ei`, `pi`, `thompson`, `entropy`, `simulated_annealing`. |
| `initial_strategy` | `str` | `'random'` | Acquisition strategy for cycle 0. |

#### Cycle control — advanced API

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cycles` | `list[CycleConfig] \| None` | `None` | Explicit per-stage cycle config. When provided, overrides all simple API cycle parameters. See [`CycleConfig`](#cycleconfig). |

#### Scoring

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_direction` | `str` | `'higher'` | `'higher'` to maximise (e.g. activity), `'lower'` to minimise (e.g. toxicity). |

#### Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str \| Path \| None` | `None` | Output directory. Auto-generates `learnm8_results_YYYYMMDD_HHMMSS` when `None`. |
| `cache_dir` | `str \| Path \| None` | `None` | HDF5 feature cache directory. Defaults to `{output_dir}/.cache`. Sharing across experiments gives ~100× speedup. |
| `output_format` | `'auto' \| 'csv' \| 'parquet'` | `'auto'` | Output file format. `'auto'` selects parquet for outputs >1M rows (constant-RAM streaming write), CSV otherwise. `cycle_metrics` is always CSV. |

#### Reproducibility

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `42` | Global random seed. Controls initial batch, learner init, and stochastic acquisition. |

#### Resource control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_jobs` | `int` | `-1` | Parallel workers for feature extraction. `-1` = all CPU cores. `1` = sequential. |
| `device` | `str` | `'auto'` | Compute device for GPU learners. `'auto'` uses CUDA if available. Options: `'cpu'`, `'cuda'`, `'cuda:N'`. Ignored by CPU-only learners. |
| `memory_safety_factor` | `float` | `0.7` | Fraction of available memory to target when auto-sizing GPU batches. Range `(0, 1]`. |

#### Memory / large-pool guards

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `large_features_ack` | `bool` | `False` | Acknowledge large feature footprint on pools >10M compounds. Descriptor-class featurizers (>1 KB/molecule) raise `ConfigurationError` on >10M pools unless this is `True`. |

#### Advanced

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_chemprop_fine_tuning` | `bool` | `False` | Incremental fine-tuning for Chemprop learners. Saves checkpoints after each cycle; loads them on the next. Reduces training time 50–90% in later cycles. Checkpoints go to `{output_dir}/.checkpoints/chemprop/`. |
| `pruning_fraction` | `float \| None` | `None` | Fraction of compounds to prune per cycle (range `0.0–0.9`). Pruned compounds are excluded from future selection. |
| `pruning_strategy` | `str \| None` | `None` | Pruning strategy. Defaults to `'score'` when `pruning_fraction` is set. |
| `pruning_params` | `dict \| None` | `None` | Extra pruning parameters. `pruning_fraction` is merged in automatically. |
| `acquisition_params` | `dict \| None` | `None` | Extra acquisition parameters passed to the strategy constructor. E.g. `{'beta': 2.0}` for UCB, `{'initial_temp': 2.0, 'neighbor_strategy': 'score_band'}` for simulated annealing. |
| `disable_molecular_similarity` | `bool \| Iterable[str]` | `False` | Disable diversity metric computation. `True` skips all. Pass a list of metric names to skip selectively. |
| `force_uncertainty` | `bool` | `False` | Force uncertainty computation every cycle even when the active strategy does not require it. Cannot disable uncertainty — only forces it on. |

### Returns

`dict[str, Any]` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `compounds_df` | `pl.DataFrame` | Master 7-column DataFrame (constant width regardless of cycle count). |
| `cycle_metrics` | `list[dict]` | Per-cycle metric dicts (including cycle 0). |
| `aggregate_metrics` | `dict` | Summary statistics across all cycles. |
| `validation_result` | `ValidationResult` | Compound validation outcome. |
| `output_dir` | `Path` | Path to the output directory. |
| `saved_files` | `dict[str, Path]` | Paths to all saved output files. |
| `prediction_files` | `dict[str, Path]` | Paths to per-cycle prediction files (`prediction_cycle_N.parquet`). |
| `labeled_count` | `int` | Number of labeled compounds at experiment end. |
| `unlabeled_count` | `int` | Number of unlabeled compounds at experiment end. |

#### compounds_df columns

The master DataFrame always has exactly 7 columns:

| Column | Type | Description |
|--------|------|-------------|
| `ID` | `Utf8` | Compound identifier |
| `SMILES` | `Utf8` | SMILES string |
| `status` | `Utf8` | `'unlabeled'`, `'labeled'`, or `'pruned'` |
| `selected_cycle` | `Int32 \| null` | Cycle when compound was selected |
| `labeled_cycle` | `Int32 \| null` | Cycle when compound was labeled |
| `pruned_cycle` | `Int32 \| null` | Cycle when compound was pruned |
| `{target_col}` | `Float64 \| null` | Measured value; `null` if unlabeled |

Per-cycle predictions are stored as `prediction_cycle_N.parquet` files, not as columns on the master DataFrame.

#### cycle_metrics keys

**All modes:**

| Key | Type | Description |
|-----|------|-------------|
| `cycle` | `int` | Cycle number (0 = initialization) |
| `n_selected` | `int` | Compounds selected this cycle |
| `n_labeled` | `int` | Total labeled after this cycle |
| `remaining_unlabeled` | `int` | Unlabeled compounds remaining |
| `avg_score_selected` | `float` | Mean oracle score of selected compounds |
| `best_so_far` | `float` | Best oracle score found so far |
| `total_time` | `float` | Total cycle time (seconds) |
| `training_time` | `float` | Model training time (seconds) |
| `prediction_time` | `float` | Prediction time (seconds) |
| `acquisition_time` | `float` | Acquisition selection time (seconds) |
| `oracle_time` | `float` | Oracle measurement time (seconds) |
| `has_uncertainty` | `bool` | Whether uncertainty was computed this cycle |
| `uncertainty_mean` | `float \| None` | Mean uncertainty; `None` when skipped |
| `uncertainty_std` | `float \| None` | Std dev of uncertainty; `None` when skipped |
| `feature_extraction_time` | `float` | Feature extraction time (seconds) |

**Benchmark mode only** (CSV oracle):

| Key | Type | Description |
|-----|------|-------------|
| `top_10_discovery` | `float` | Fraction of true top-10 compounds found |
| `top_100_discovery` | `float` | Fraction of true top-100 compounds found |
| `cumulative_ef` | `float` | Cumulative enrichment factor |
| `batch_score_improvement_ratio` | `float` | Signed improvement of batch mean over population mean |
| `unlabeled_spearman_correlation` | `float \| None` | Spearman rank correlation on unlabeled set; `None` when undefined |
| `unlabeled_top_100_overlap` | `float` | Top-100 overlap on unlabeled set |
| `unlabeled_ef_1_0` | `float` | Enrichment factor at 1% on unlabeled set |

**Diversity metrics** (when not disabled):

| Key | Type | Description |
|-----|------|-------------|
| `mean_tanimoto_similarity_sampled` | `float` | Sampled mean pairwise Tanimoto (batch) |
| `scaffold_diversity_index` | `float` | Scaffold diversity index (batch) |
| `shannon_entropy_diversity` | `float` | Shannon entropy diversity (batch) |
| `mean_tanimoto_similarity_sampled_cumulative` | `float` | Same metrics computed cumulatively |
| `scaffold_diversity_index_cumulative` | `float` | — |
| `shannon_entropy_diversity_cumulative` | `float` | — |
| `fingerprint_used` | `str` | Featurizer used for diversity computation |

> Diversity metrics depend on the featurizer. Comparing across runs with different featurizers is invalid. Verify consistency via `fingerprint_used`.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ConfigurationError` | Invalid parameters; or descriptor featurizer on >10M pool without `large_features_ack=True` |
| `ValidationError` | Compound pool fails SMILES validation; or `target_col` is non-numeric |
| `FeatureExtractionError` | Feature extraction fails |
| `LearnerError` | Model training or prediction fails |
| `FileNotFoundError` | Input CSV not found |

### Oracle auto-detection

- `oracle=None` + CSV `compound_pool` → same file used as benchmark oracle
- `oracle=None` + DataFrame `compound_pool` → raises `ConfigurationError`
- CSV oracle → benchmark mode (discovery/ranking metrics available)
- Python-function oracle → production mode (basic metrics only)

---

## validate_compound_pool()

```python
def validate_compound_pool(
    compound_pool: pl.DataFrame,
    n_jobs: int = -1,
    progress: bool = True,
    target_col: str | None = None,
) -> ValidationResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compound_pool` | `pl.DataFrame` | — | DataFrame with `ID` and `SMILES` columns. |
| `n_jobs` | `int` | `-1` | Parallel workers. `-1` = all CPU cores. `1` = sequential. |
| `progress` | `bool` | `True` | Show progress bar (requires tqdm). |
| `target_col` | `str \| None` | `None` | If provided and the column exists in `compound_pool`, validates that it has a numeric dtype. Raises `ValidationError` if non-numeric. |

### Returns

[`ValidationResult`](#validationresult)

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValidationError` | `target_col` is present but not numeric |

---

## extract_features()

```python
def extract_features(
    smiles_list: list[str],
    featurizer: str | Featurizer,
    cache_dir: Path | None = None,
    n_jobs: int = -1,
    show_progress: bool = False,
    preferred_dtype: str = 'float32',
) -> np.ndarray
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smiles_list` | `list[str]` | — | SMILES strings to featurize. |
| `featurizer` | `str \| Featurizer` | — | Featurizer shortcut string or instance. See [Available Featurizers](../components/featurizers/available-featurizers.md). |
| `cache_dir` | `Path \| None` | `None` | HDF5 cache directory. Defaults to `.cache` in CWD. |
| `n_jobs` | `int` | `-1` | Parallel workers. Auto-scales: sequential <100 SMILES, all cores 100–10K, capped at 32 above 10K. |
| `show_progress` | `bool` | `False` | Show tqdm progress bar. |
| `preferred_dtype` | `str` | `'float32'` | Hint to the cache layer for output dtype. Tree learners pass `'uint8'` to skip the float32 inflation step for binary fingerprints (~4× working-set reduction on Morgan 2048-bit). |

### Returns

`np.ndarray` of shape `(n_compounds, n_features)`. Output dtype depends on the featurizer's storage type: binary fingerprints are stored as `packed_uint8` and inflated to `float32` (or `uint8` when `preferred_dtype='uint8'`); continuous descriptors return `float32`.

### Raises

| Exception | Condition |
|-----------|-----------|
| `FeatureExtractionError` | Featurizer unknown, SMILES invalid, or cache corruption |

---

## list_available_learners()

```python
def list_available_learners() -> list[str]
```

Returns all learner shortcut strings whose imports succeeded. GPU learners (`gpu_gp`, `svgp`) appear only when GPyTorch is importable. `rf_fil` and `ridge_cuml` appear only when RAPIDS cuML is importable.

---

## CycleConfig

```python
@dataclass
class CycleConfig:
    strategy: str
    n_cycles: int = 1
    batch_fraction: float | None = None
    pruning_strategy: str | None = None
    pruning_params: dict | None = None
    acquisition_params: dict | None = None
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | — | Acquisition strategy name. |
| `n_cycles` | `int` | `1` | Number of cycles sharing this config. |
| `batch_fraction` | `float \| None` | `None` | Fraction of original pool per cycle. Falls back to `run_active_learning`'s `batch_fraction` when `None`. |
| `pruning_strategy` | `str \| None` | `None` | Pruning strategy name (e.g. `'score'`). |
| `pruning_params` | `dict \| None` | `None` | Pruning parameters (e.g. `{'pruning_fraction': 0.3}`). |
| `acquisition_params` | `dict \| None` | `None` | Strategy-specific parameters passed to the acquisition constructor. |

---

## ValidationResult

```python
@dataclass
class ValidationResult:
    valid_compounds: pl.DataFrame
    invalid_compounds: pl.DataFrame
    validation_errors: dict[str, str]

    @property
    def success_rate(self) -> float: ...
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `valid_compounds` | `pl.DataFrame` | Compounds that passed SMILES validation. |
| `invalid_compounds` | `pl.DataFrame` | Compounds that failed validation. |
| `validation_errors` | `dict[str, str]` | Compound ID → error message. |
| `success_rate` | `float` | Fraction valid, range `[0.0, 1.0]`. |

---

## Acquisition Strategy Classes

All strategies are importable from `learnm8.acquisition`.

### GreedyAcquisition

```python
class GreedyAcquisition(AcquisitionFunction):
    def __init__(self, score_direction: str = 'higher', **kwargs)
    def requires_uncertainty(self) -> bool  # False
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame
```

Selects the `n_select` compounds with the highest (or lowest) predicted value. Pure exploitation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_direction` | `str` | `'higher'` | `'higher'` to select max predictions; `'lower'` to select min. |

---

### RandomAcquisition

```python
class RandomAcquisition(AcquisitionFunction):
    def __init__(self, random_state: int = 42, **kwargs)
    def requires_uncertainty(self) -> bool  # False
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame
```

Selects `n_select` compounds uniformly at random. Unbiased exploration baseline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `42` | Random seed. |

---

### TopKAcquisition

```python
class TopKAcquisition(AcquisitionFunction):
    def __init__(
        self,
        k_fraction: float = 0.1,
        score_direction: str = 'higher',
        random_state: int = 42,
        **kwargs,
    )
    def requires_uncertainty(self) -> bool  # False
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame
```

Restricts the candidate set to the top `k_fraction` of compounds by predicted score, then samples `n_select` uniformly from that subset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_fraction` | `float` | `0.1` | Fraction of pool to consider (e.g. `0.1` = top 10%). |
| `score_direction` | `str` | `'higher'` | `'higher'` or `'lower'`. |
| `random_state` | `int` | `42` | Random seed for sampling within top-K. |

---

### UCBAcquisition

```python
class UCBAcquisition(AcquisitionFunction):
    def __init__(self, beta: float = 2.0, **kwargs)
    def requires_uncertainty(self) -> bool  # True
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame
```

Selects by upper confidence bound: `score = mean + β × uncertainty`. Higher `β` favours exploration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `2.0` | Exploration weight. |

---

### ExpectedImprovementAcquisition

```python
class ExpectedImprovementAcquisition(AcquisitionFunction):
    def __init__(
        self,
        xi: float = 0.01,
        score_direction: str = 'higher',
        current_best: float | None = None,
        minimize: bool | None = None,  # deprecated — use score_direction
        **kwargs,
    )
    def requires_uncertainty(self) -> bool  # True
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame
```

Selects by expected improvement over `current_best`. Uses `scipy.special.ndtr` for the CDF (z-clipped to `[-37.0, 37.0]` for numerical stability).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xi` | `float` | `0.01` | Exploration–exploitation trade-off. Higher values encourage more exploration. |
| `score_direction` | `str` | `'higher'` | `'higher'` or `'lower'`. |
| `current_best` | `float \| None` | `None` | Reference value for improvement calculation. Inferred from labeled data when `None`. |
| `minimize` | `bool \| None` | `None` | **Deprecated.** Use `score_direction` instead. |

---

### ProbabilityImprovementAcquisition

```python
class ProbabilityImprovementAcquisition(AcquisitionFunction):
    def __init__(
        self,
        xi: float = 0.01,
        score_direction: str = 'higher',
        current_best: float | None = None,
        minimize: bool | None = None,  # deprecated — use score_direction
        **kwargs,
    )
    def requires_uncertainty(self) -> bool  # True
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame
```

Selects by probability of improving over `current_best`. More conservative than EI — selects more reliably near the current best.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xi` | `float` | `0.01` | Jitter; higher values encourage exploration. |
| `score_direction` | `str` | `'higher'` | `'higher'` or `'lower'`. |
| `current_best` | `float \| None` | `None` | Reference value. Inferred from labeled data when `None`. |
| `minimize` | `bool \| None` | `None` | **Deprecated.** Use `score_direction` instead. |

---

### ThompsonSamplingAcquisition

```python
class ThompsonSamplingAcquisition(AcquisitionFunction):
    def __init__(self, random_state: int = 42, **kwargs)
    def requires_uncertainty(self) -> bool  # True
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame
```

Draws samples from the posterior predictive distribution (`N(mean, uncertainty²)`) and selects compounds with the highest sampled values.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `42` | Random seed for posterior sampling. |

---

### EntropyAcquisition

```python
class EntropyAcquisition(AcquisitionFunction):
    def __init__(self, entropy_type: str = 'uncertainty', **kwargs)
    def requires_uncertainty(self) -> bool  # True
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame
```

Selects by differential entropy of the Gaussian predictive distribution. Rank-equivalent to σ-descending selection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_type` | `str` | `'uncertainty'` | `'uncertainty'`: `H = 0.5·log(2πe·σ²)`. `'variance'`: `H = 0.5·log(2πe·σ⁴)`. Both rank identically to σ-descending. Sigma is floored at `1e-9` for numerical robustness. |

---

### SimulatedAnnealingAcquisition

```python
class SimulatedAnnealingAcquisition(AcquisitionFunction):
    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.01,
        max_iterations: int = 1000,
        cooling_schedule: str = 'exponential',
        score_direction: str = 'higher',
        random_state: int = 42,
        neighbor_strategy: str = 'random',
        n_neighbors: int = 10,
        band_width: int = 50,
        **kwargs,
    )
    def requires_uncertainty(self) -> bool  # False
    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame
```

Temperature-based probabilistic selection. Starts with high-temperature random exploration and cools toward greedy exploitation following the chosen schedule. Accepts worse candidates with probability `exp(-ΔE / T)`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_temp` | `float` | `1.0` | Starting temperature. Higher values give more initial randomness. |
| `final_temp` | `float` | `0.01` | Ending temperature. Must be < `initial_temp`. Near-zero gives greedy behaviour at the end. |
| `max_iterations` | `int` | `1000` | Metropolis iterations per `select()` call. |
| `cooling_schedule` | `str` | `'exponential'` | `'exponential'`: rapid early cooling, slow late. `'linear'`: uniform cooling rate. |
| `score_direction` | `str` | `'higher'` | `'higher'` or `'lower'`. |
| `random_state` | `int` | `42` | Random seed. |
| `neighbor_strategy` | `str` | `'random'` | Neighbour generation strategy. `'random'`: uniform random draw from pool — O(1) per move, scales to 100M+ compounds. `'score_band'`: draw from a rank window of width `±band_width` around the current compound — O(n log n) sort once per call, then O(1) moves. `'knn_features'`: draw from `n_neighbors` nearest neighbours in feature space — builds a `NearestNeighbors` index over the whole pool; suitable only for small pools. |
| `n_neighbors` | `int` | `10` | Neighbours to index when `neighbor_strategy='knn_features'`. |
| `band_width` | `int` | `50` | Rank-window half-width when `neighbor_strategy='score_band'`. |

> **Note:** `'knn_features'` requires passing `featurizer_obj` and `cache_dir` via `acquisition_params` when using the string-shortcut API.

---

## See Also

- [Usage Examples](examples.md)
- [Learners Overview](../components/learners/overview.md)
- [Available Featurizers](../components/featurizers/available-featurizers.md)
- [Acquisition Overview](../components/acquisition/overview.md)
- [CLI Reference](cli-reference.md)

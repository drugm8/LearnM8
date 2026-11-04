# Python API Reference

Reference for LearnM8's public Python API.

**Imports:**

```python
from learnm8 import (
    run_active_learning,
    validate_compound_pool,
    extract_features,
    CycleConfig,
    ValidationResult,
)
from learnm8.suggest import suggest_next_batch
from learnm8.api import list_available_learners
```

---

## run_active_learning()

```python
def run_active_learning(
    compound_pool: str | Path | pl.DataFrame,
    oracle: str | Path | Oracle | None,
    learner: str | Learner,
    target_col: str,
    featurizer: str | Featurizer | None = None,
    smiles_column: str | None = None,
    id_column: str | None = None,
    cycles: list[CycleConfig] | None = None,
    n_cycles: int = 10,
    batch_fraction: float = 0.01,
    strategy: str = 'greedy',
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
    skip_validation: bool = False,
    streaming: Literal['auto', 'always', 'never'] = 'auto',
) -> dict[str, Any]
```

### Parameters

#### Input data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compound_pool` | `str \| Path \| pl.DataFrame` | — | Compound pool. CSV path (must have `ID` and `SMILES` columns) or a Polars DataFrame with the same. |
| `oracle` | `str \| Path \| Oracle \| None` | — (required, may be `None`) | Measurement oracle. `None` uses `compound_pool` as a CSV oracle (benchmark mode; requires a file path). A `.csv` path → benchmark mode. `'module.py:function'` → run mode. Or an `Oracle` instance. See [Oracles](../components/oracles/overview.md). |
| `learner` | `str \| Learner` | — | Learner shortcut string (e.g. `'rf'`, `'gp'`) or a `Learner` instance. See [Learners](../components/learners/overview.md). |
| `target_col` | `str` | — | Column name for the target property. |
| `featurizer` | `str \| Featurizer \| None` | `None` | Featurizer shortcut or instance. Required for feature-based learners; omit for SMILES-native learners (Chemprop). See [Featurizers](../components/featurizers/overview.md). |
| `smiles_column` | `str \| None` | `None` | SMILES column name. Auto-detects `'SMILES'`, `'smiles'`, `'Smiles'` when `None`. Ignored for DataFrame input. |
| `id_column` | `str \| None` | `None` | ID column name. Auto-detects `'ID'` or generates synthetic IDs when `None`. Ignored for DataFrame input. |

#### Cycle control — simple API

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_cycles` | `int` | `10` | Total cycles. Cycle 0 is always a random batch (no model); cycles 1+ use `strategy`. |
| `batch_fraction` | `float` | `0.01` | Fraction of the **original** pool per cycle. Floor at 1 compound; capped at remaining pool size. Must be in `(0, 1]`. |
| `strategy` | `str` | `'greedy'` | Acquisition strategy for cycles 1+. One of: `greedy`, `random`, `topk`, `ucb`, `ei`, `pi`, `thompson`, `entropy`, or a name added to `ACQUISITION_REGISTRY` (see [Extending](../customization/extending.md)). |

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
| `cache_dir` | `str \| Path \| None` | `None` | HDF5 feature cache directory. Defaults to `{output_dir}/.cache`. Share one directory across experiments on the same compounds. |
| `output_format` | `'auto' \| 'csv' \| 'parquet'` | `'auto'` | Output file format. `'auto'` selects parquet for outputs >1M rows (constant-RAM streaming write), CSV otherwise. `cycle_metrics` is always CSV. |

#### Reproducibility

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `42` | Global random seed. Controls initial batch, learner init, and stochastic acquisition. |

#### Resource control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_jobs` | `int` | `-1` | Parallel jobs for CPU-bound operations. `-1` = all CPU cores, or a positive integer (`0` and values below `-1` raise `ConfigurationError`). Ignored by GPU-accelerated learners (`rf_fil`, `ridge_cuml`). |
| `device` | `str` | `'auto'` | Compute device for GPU learners. `'auto'` uses CUDA if available. Options: `'cpu'`, `'cuda'`, `'cuda:N'`. Ignored by CPU-only learners. |
| `memory_safety_factor` | `float` | `0.7` | Fraction of available memory used for prediction batching. Range `(0, 1]`. |

#### Memory / large-pool guards

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `large_features_ack` | `bool` | `False` | Acknowledge a large feature footprint. Descriptor-class featurizers on pools above 10M compounds raise `ConfigurationError` unless this is `True`. CLI: `--allow-large-features`. |

#### Advanced

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_chemprop_fine_tuning` | `bool` | `False` | Incremental fine-tuning for Chemprop learners. Saves checkpoints after each cycle; loads them on the next. Checkpoints go to `{output_dir}/.checkpoints/chemprop/`. |
| `pruning_fraction` | `float \| None` | `None` | Fraction of compounds to prune per cycle (range `0.0–0.9`). Pruned compounds are excluded from future selection. See [Pruning](../advanced/pruning.md). |
| `pruning_strategy` | `str \| None` | `None` | Pruning strategy. Defaults to `'score'` when `pruning_fraction` is set. |
| `pruning_params` | `dict \| None` | `None` | Extra pruning parameters. `pruning_fraction` is merged in automatically. |
| `acquisition_params` | `dict \| None` | `None` | Extra acquisition parameters passed to the strategy constructor. E.g. `{'beta': 2.0}` for UCB, `{'xi': 0.05}` for EI/PI, `{'k_fraction': 0.05}` for Top-K. |
| `disable_molecular_similarity` | `bool \| Iterable[str]` | `False` | Disable diversity metric computation. `True` skips all. Pass a list of metric names to skip selectively. |
| `force_uncertainty` | `bool` | `False` | Force uncertainty computation every cycle even when the active strategy does not require it. Cannot disable uncertainty — only forces it on. |
| `skip_validation` | `bool` | `False` | Skip SMILES validation, duplicate-ID checks and target-column dtype checks; all compounds are treated as valid. Use when the dataset was already checked (for example with `learnm8 validate`). Oracle-vs-pool reconciliation still runs. CLI: `--skip-validation`. |
| `streaming` | `'auto' \| 'always' \| 'never'` | `'auto'` | Memory-efficient two-pass predict-then-select path for large pools. A cycle is eligible when the acquisition strategy is streamable (every built-in strategy is) and `pruning_strategy` is `None` or `'score'`; `run_active_learning` always writes to an output directory, which the streaming path needs. `'auto'` uses it when eligible and falls back to the in-memory path otherwise; `'always'` raises `ConfigurationError` if a cycle is not eligible; `'never'` forces the in-memory path. Stochastic strategies (`thompson`, `random`, `topk`) draw different but equivalent samples under streaming. CLI: `--streaming`. |

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
| `prediction_files` | `list[Path]` | Paths to per-cycle prediction files (`prediction_cycle_N.parquet`). |
| `labeled_count` | `int` | Number of labeled compounds at experiment end. |
| `unlabeled_count` | `int` | Number of unlabeled compounds at experiment end. |

#### compounds_df columns

The master DataFrame always has exactly 7 columns:

| Column | Type | Description |
|--------|------|-------------|
| `ID` | `Utf8` | Compound identifier |
| `SMILES` | `Utf8` | SMILES string |
| `status` | `Utf8` | `'unlabeled'`, `'labeled'`, `'pruned'`, or `'failed'` (oracle returned no finite value; never retrained on or reselected) |
| `selected_cycle` | `Int32 \| null` | Cycle when compound was selected |
| `labeled_cycle` | `Int32 \| null` | Cycle when compound was labeled |
| `pruned_cycle` | `Int32 \| null` | Cycle when compound was pruned |
| `{target_col}` | `Float64 \| null` | Measured value; `null` if unlabeled or failed |

Per-cycle predictions are stored as `prediction_cycle_N.parquet` files, not as columns on the master DataFrame.

#### cycle_metrics keys

The tables below list the main keys; the dicts contain additional keys (for example `strategy`, `selection_path`, `pool_size`, prediction statistics and pruning counts).

**Both modes:**

| Key | Type | Description |
|-----|------|-------------|
| `cycle` | `int` | Cycle number (0 = initialization) |
| `batch_size` | `int` | Compounds selected this cycle |
| `cumulative_labeled` | `int` | Total labeled after this cycle |
| `remaining_unlabeled` | `int` | Unlabeled compounds remaining |
| `n_failed` | `int` | Compounds selected this cycle whose oracle value was missing, null, NaN, infinite or non-numeric (status `failed`); `0` when none |
| `avg_score_selected` | `float \| None` | Mean oracle score of selected compounds |
| `best_so_far` | `float \| None` | Best oracle score found so far |
| `total_time` | `float` | Total cycle time (seconds) |
| `training_time` | `float` | Model training time (seconds) |
| `prediction_time` | `float` | Prediction time (seconds) |
| `acquisition_time` | `float` | Acquisition selection time (seconds) |
| `oracle_time` | `float` | Oracle measurement time (seconds) |
| `has_uncertainty` | `bool` | Whether uncertainty was computed this cycle |
| `uncertainty_mean` | `float \| None` | Mean uncertainty; `None` when skipped |
| `uncertainty_std` | `float \| None` | Std dev of uncertainty; `None` when skipped |
| `feature_extraction_time` | `float` | Feature extraction time (seconds) |

**Benchmark mode only** (CSV oracle; may be `None` when undefined):

| Key | Type | Description |
|-----|------|-------------|
| `top_10_discovery` | `float` | Percentage (0-100) of the true top-10 compounds found so far |
| `top_100_discovery` | `float` | Percentage (0-100) of the true top-100 compounds found so far |
| `cumulative_ef` | `float` | Cumulative enrichment factor |
| `batch_score_improvement_ratio` | `float` | Signed improvement of batch mean over population mean |
| `unlabeled_spearman_correlation` | `float \| None` | Spearman rank correlation on unlabeled set; `None` when undefined |
| `unlabeled_top_100_overlap` | `float` | Top-100 overlap on unlabeled set |
| `unlabeled_ef_1_0` | `float` | Enrichment factor at 1% on unlabeled set |

> The `unlabeled_*` metrics score the model on the **remaining pool**: every compound labeled so far, including the batch selected in this cycle, is excluded. They answer "how well does the model rank what is left?", not "were this cycle's picks good?" (use `top_*_discovery` and `batch_*` for that). Under greedy selection the model's own best predictions have just been removed, so `unlabeled_top_*_overlap` is effectively measured on its next-best predictions and is typically low even for a good model.

**Diversity metrics** (when not disabled):

| Key | Type | Description |
|-----|------|-------------|
| `mean_tanimoto_similarity_sampled_batch` | `float \| None` | Mean pairwise Tanimoto similarity from adaptive pair sampling, newly selected batch |
| `mean_tanimoto_similarity_sampled_cumulative` | `float \| None` | Same, all compounds selected so far |
| `scaffold_diversity_index_batch` | `float \| None` | Unique Bemis-Murcko scaffolds divided by valid compounds, newly selected batch |
| `scaffold_diversity_index_cumulative` | `float \| None` | Same, all compounds selected so far |
| `bit_position_uniformity_entropy_batch` | `float \| None` | Entropy (bits) of the normalised bit-position activation distribution, newly selected batch. Higher means fingerprint bits are activated more uniformly |
| `bit_position_uniformity_entropy_cumulative` | `float \| None` | Same, all compounds selected so far |
| `fingerprint_used` | `str` | Fingerprint used for diversity computation: always Morgan(2, 2048), e.g. `morgan_2_2048_uint8packed` (the suffix is the run featurizer's storage dtype; `_fallback` marks a non-binary run featurizer) |

> Diversity metrics are always computed on Morgan(2, 2048) fingerprints, independent of the run featurizer, so they are comparable across runs.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ConfigurationError` | Bad parameters or configuration: unknown learner, featurizer or strategy; a feature-based learner without a featurizer; an uncertainty strategy (`ucb`, `ei`, `pi`, `thompson`, `entropy`) with a learner that has no uncertainty (checked before cycle 0); `pruning_fraction` out of range; compound pool file not found or `compound_pool` of the wrong type; invalid `n_jobs` or `device`; invalid `streaming` or `output_format`; unknown `disable_molecular_similarity` names; `oracle=None` with a DataFrame pool; descriptor featurizer on a pool above 10M compounds without `large_features_ack=True`; `streaming='always'` with an ineligible cycle |
| `ValidationError` | Missing `ID`/`SMILES` columns; no valid compounds; `target_col` non-numeric; with a CSV oracle, a pool compound whose target is empty, NaN or infinite |
| `OracleError` | The oracle cannot measure the selected compounds |
| `FeatureExtractionError` | Feature extraction fails |
| `LearnerError` | Model training or prediction fails |
| `PersistenceError` | Results cannot be written |
| `TypeError` | `oracle`, `learner` or `featurizer` is of an unsupported type; `n_jobs` is not an integer; `device` is not a string |

Learners without uncertainty are `xgb`, `mlp`, `fastprop` and `chemprop`; the other 15 provide it.

### Oracle auto-detection

- `oracle=None` + CSV `compound_pool` → same file used as benchmark oracle
- `oracle=None` + DataFrame `compound_pool` → raises `ConfigurationError`
- CSV oracle → benchmark mode (discovery/ranking metrics available)
- Python oracle → run mode (basic metrics only)

See [Benchmark mode vs run mode](../tutorials/running-experiments.md#benchmark-mode-vs-run-mode).

---

## suggest_next_batch()

One-shot batch suggestion for slow lab assays. Reads the batch files in `batch_dir`, trains on the measured values, writes the next `batch_cycle_N.csv` and returns it. See the [Lab Campaigns tutorial](../tutorials/lab-campaigns.md).

```python
def suggest_next_batch(
    compound_pool: str | Path,
    batch_dir: str | Path,
    target_col: str,
    learner: str,
    featurizer: str | None = None,
    *,
    batch_size: int = 96,
    strategy: str = 'greedy',
    score_direction: str = 'higher',
    acquisition_params: dict | None = None,
    random_state: int = 42,
    smiles_column: str | None = None,
    id_column: str | None = None,
    cache_dir: str | Path | None = None,
    n_jobs: int = -1,
    device: str = 'auto',
    pruning_fraction: float = 0.0,
    pruning_start: int = 3,
) -> pl.DataFrame
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compound_pool` | `str \| Path` | — | Compound file (`.csv`, `.sdf`, `.smi`) with ID and SMILES. |
| `batch_dir` | `str \| Path` | — | Folder with `batch_cycle_N.csv` files (and `pruned_cycle_N.parquet` prune records). Created if missing; `.learnm8/` working data is kept inside it. |
| `target_col` | `str` | — | Column the lab fills in. |
| `learner` | `str` | — | Learner name (see `list_available_learners()`). |
| `featurizer` | `str \| None` | `None` | Featurizer name. Required unless the learner works on SMILES directly. |
| `batch_size` | `int` | `96` | Compounds in the new batch (>= 1), clipped to the remaining pool. |
| `strategy` | `str` | `'greedy'` | Acquisition strategy for cycles >= 1. Cycle 0 is always random. |
| `score_direction` | `str` | `'higher'` | `'higher'` or `'lower'`. |
| `acquisition_params` | `dict \| None` | `None` | Parameters for the acquisition strategy. |
| `random_state` | `int` | `42` | Random seed. Same inputs and seed give the same batch. |
| `smiles_column` | `str \| None` | `None` | SMILES column in the pool (auto-detect). |
| `id_column` | `str \| None` | `None` | ID column in the pool (auto-detect). |
| `cache_dir` | `str \| Path \| None` | `None` | Feature cache location. Default `batch_dir/.learnm8/cache`. |
| `n_jobs` | `int` | `-1` | Parallel workers. `-1` = all CPU cores. |
| `device` | `str` | `'auto'` | `'auto'`, `'cpu'`, `'cuda'`, `'cuda:N'`. |
| `pruning_fraction` | `float` | `0.0` | Fraction of the unlabeled pool removed for good in each pruning round (`0.0`–`0.9`; a real number, not a bool). `0.0` = off. Lowered automatically (or pruning skipped for that round) so that at least `batch_size` unlabeled compounds remain; this cap is specific to `suggest_next_batch`, `run_active_learning` pruning does not cap. See [Pruning](../tutorials/lab-campaigns.md#pruning). |
| `pruning_start` | `int` | `3` | First cycle that may prune (an integer >= 1). Rounds without a trained model never prune. |

### Returns

`pl.DataFrame` with the batch just written to `batch_dir/batch_cycle_N.csv`, in this column order: `rank`, `ID`, `<target_col>` (null), `SMILES`, then, in rounds that trained a model (not round 0, and not the random fallback after an all-failed batch), `prediction` and `uncertainty` (when computed). `N` is the highest existing batch number plus one, or 0 when the folder has no batch files. Batch files start with a `sep=,` spreadsheet hint line before the header; a spreadsheet may drop it on re-save, so scripts that read them with pandas or polars should skip that line when present.

Value cells in earlier batch files: empty is pending, a finite number is measured, anything else is failed. Pending and failed compounds, and compounds in `pruned_cycle_K.parquet` files (`ID`, `SMILES`) from earlier rounds, are never selected again; only measured compounds are used for training. A compound that is both pruned and measured is trained on (with a warning). When no compound is measured yet but the newest batch has results, a fresh random batch is drawn that excludes every compound already in a batch file.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ConfigurationError` | The newest batch has no results entered; no selectable compounds remain; `batch_size < 1`; unknown strategy; feature-based learner without a featurizer; target batch file already exists; compound pool file not found; `score_direction` not `higher`/`lower`; `acquisition_params` not a dict; `pruning_fraction` not a number or outside `[0.0, 0.9]`; `pruning_start` not an integer or `< 1` |
| `ValidationError` | Pool fails validation; batch file empty, unparsable or missing `ID`/`<target_col>`; row with a value but no ID; duplicate IDs; batch IDs not in the pool; changed SMILES cell; ambiguous number (possible thousands separator); unreadable `pruned_cycle_N.parquet`; prune-file IDs not in the pool or changed SMILES |
| `LearnerError` | Model training or prediction fails |

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
| `progress` | `bool` | `True` | Show a progress bar. |
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
| `featurizer` | `str \| Featurizer` | — | Featurizer shortcut string or instance. See [Featurizers](../components/featurizers/overview.md). |
| `cache_dir` | `Path \| None` | `None` | Feature cache directory (default: `.cache`). |
| `n_jobs` | `int` | `-1` | Parallel workers. `-1` = all CPU cores, `1` = sequential. The value is passed through unchanged. |
| `show_progress` | `bool` | `False` | Show tqdm progress bar. |
| `preferred_dtype` | `str` | `'float32'` | Hint to the cache layer for output dtype. Tree learners pass `'uint8'` to skip the float32 inflation step for binary fingerprints (a 4× smaller working set than float32). |

### Returns

`np.ndarray` of shape `(n_compounds, n_features)`. Output dtype depends on the featurizer's storage type: binary fingerprints are stored as `packed_uint8` and inflated to `float32` (or `uint8` when `preferred_dtype='uint8'`); continuous descriptors return `float32`.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ConfigurationError` | Unknown featurizer string |
| `TypeError` | `featurizer` is neither a string nor a `Featurizer` |

---

## list_available_learners()

```python
def list_available_learners() -> list[str]
```

Returns all 19 learner shortcut strings, sorted alphabetically. The list does not depend on which optional packages are installed. Creating `gpu_gp` or `svgp` without GPyTorch raises `ConfigurationError`; `rf_fil` and `ridge_cuml` without RAPIDS cuML raise `LearnerError` when they are trained.

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
| `batch_fraction` | `float \| None` | `None` | Fraction of original pool per cycle, in `(0, 1]`. Required: `None` raises `ValueError`; so does a value outside `(0, 1]`. |
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

All strategies are importable from `learnm8.acquisition` and selected by name through `strategy` or `CycleConfig`. Constructor arguments are passed through `acquisition_params`.

| Class | Name | Constructor parameters | Needs uncertainty |
|-------|------|------------------------|-------------------|
| `GreedyAcquisition` | `greedy` | `score_direction='higher'` | No |
| `RandomAcquisition` | `random` | `random_state=42` | No |
| `TopKAcquisition` | `topk` | `k_fraction=0.1`, `score_direction='higher'`, `random_state=42` | No |
| `UCBAcquisition` | `ucb` | `beta=2.0` | Yes |
| `ExpectedImprovementAcquisition` | `ei` | `xi=0.01`, `minimize=None` (deprecated, use `score_direction`), `score_direction='higher'`, `current_best=None` | Yes |
| `ProbabilityImprovementAcquisition` | `pi` | `xi=0.01`, `minimize=None` (deprecated, use `score_direction`), `score_direction='higher'`, `current_best=None` | Yes |
| `ThompsonSamplingAcquisition` | `thompson` | `random_state=42` | Yes |
| `EntropyAcquisition` | `entropy` | `entropy_type='uncertainty'` (or `'variance'`) | Yes |

All accept `**kwargs`. Details: [Greedy](../components/acquisition/greedy.md), [Random](../components/acquisition/random.md), [Top-K](../components/acquisition/topk.md), [UCB](../components/acquisition/ucb.md), [Expected Improvement](../components/acquisition/expected-improvement.md), [Probability of Improvement](../components/acquisition/probability-improvement.md), [Thompson Sampling](../components/acquisition/thompson-sampling.md), [Entropy](../components/acquisition/entropy.md). To add your own, see [Extending LearnM8](../customization/extending.md).

---

## See Also

- [Usage Examples](examples.md)
- [Extending LearnM8](../customization/extending.md)
- [Learners Overview](../components/learners/overview.md)
- [Featurizers](../components/featurizers/overview.md)
- [Acquisition Overview](../components/acquisition/overview.md)
- [CLI Reference](cli-reference.md)

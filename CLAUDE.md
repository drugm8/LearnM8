# CLAUDE.md

## Constitution

**All development MUST follow the project constitution:** [`.specify/memory/constitution.md`](.specify/memory/constitution.md)

The constitution defines: SDD/TDD workflow, agent autonomy tiers, red lines, code standards, error handling, git workflow, quality gates, and architectural governance. When in doubt, the constitution takes precedence.

Key governance files:

- **Constitution:** `.specify/memory/constitution.md`
- **Spec Template:** `specs/TEMPLATE.md`
- **ADR Template:** `docs/decisions/TEMPLATE.md`

## Project Overview

LearnM8 is a general-purpose active learning framework for large-scale screening (v0.10.0). Pure functional API with `run_active_learning()` as main entry point. Modular design with 7 core modules. Polars-first DataFrames (accepts pandas, auto-converts).

**Tech Stack:** Python 3.11.9, polars, numpy, scikit-learn, rdkit, pytorch, xgboost, chemprop, scikit-fingerprints, PyTorch Lightning, Rich console, HDF5 caching.

## Setup

```bash
conda env create -f environment.yml && conda activate learnm8
pip install -e ".[test]"
```

All dependencies required (fail-fast at import). Managed via `environment.yml`. Build config in `pyproject.toml`.

## Development Commands

```bash
pytest -m "not slow" tests/           # Default: skip slow tests (~1295 fast tests)
pytest tests/path/to/test.py -v       # Targeted run
pytest tests/ --cov=learnm8           # Coverage (min 90%)
ruff check .                          # Lint
ruff format --check .                 # Format check
mypy learnm8/                         # Type check
```

**When to include slow tests:** Changes to `learners/torch/`, `learners/ensemble/`, 3D featurizers in `features/`, `api.py`, `core/cycle.py`, or pre-commit/PR validation.

## CLI

```bash
learnm8 run data.csv --target Value --learner gp --featurizer morgan --n-cycles 10
learnm8 validate data.csv             # Check input data validity
python -c "from learnm8.api import list_available_learners; print(list_available_learners())"  # List available components
learnm8 run --config experiment.yaml  # Config file support
```

## Architecture

### Seven-Phase Execution Flow
1. **Normalize** inputs → 2. **Validate** inputs → 3. **Initialize** master DataFrame + cycle 0 → 4. **Configure** cycle schedule → 5. **Execute** cycles 1-N → 6. **Persist** CSV results → 7. **Return** results dict

**Cycle numbering:** Cycle 0 = random init (no model). Cycles 1-N = active learning. `n_cycles=10` → cycles 0-9.

### Package Structure

```
learnm8/
├── api.py                    # Entry point: run_active_learning()
├── exceptions.py             # Centralized exception hierarchy (LearnM8Error, etc.)
├── core/
│   ├── interfaces.py         # Learner/Oracle abstract interfaces
│   ├── cycle.py              # Unified cycle execution + batch prediction
│   ├── config.py             # CycleConfig dataclass
│   ├── validation.py         # Early input validation
│   ├── initialization.py     # Master DataFrame setup
│   ├── persistence.py        # CSV export
│   ├── dataframe_ops.py      # Vectorized Polars operations
│   ├── data_structures.py    # Shared data structures
│   └── resources.py          # CPU/GPU resource validation (n_jobs, device)
├── features/                 # HDF5-cached feature extraction
│   ├── __init__.py           # _FEATURIZER_CONFIG factory (39 featurizers)
│   ├── extraction.py         # extract_features() function
│   ├── cache.py              # HDF5 caching layer
│   └── base.py               # SkfpFeaturizer base class
├── learners/
│   ├── base.py               # Base classes + feature preprocessing
│   ├── sklearn/              # RF, GP, XGBoost, DT, LR
│   ├── torch/                # MLP, MC Dropout, Chemprop, Fastprop
│   └── ensemble/             # RF/LR/XGB/DT/Mixed/Chemprop/Fastprop ensembles
├── acquisition/              # Greedy, TopK, UCB, EI, PI, Thompson, Entropy, SA
├── oracles/                  # CSV (benchmark) and Python function oracles
├── evaluation/core.py        # evaluate_cycle()
├── pruning/                  # Score-based design space reduction
├── utils/                    # Logging, data loading, Polars helpers
├── visualization/            # Animation and visualization helpers
└── cli/main.py               # CLI with subcommands
```

## API Design

```python
from learnm8 import run_active_learning, CycleConfig, validate_compound_pool, extract_features

# Simple
results = run_active_learning(
    compound_pool='data.csv', target_col='Value',
    learner='rf', featurizer='morgan', n_cycles=10, batch_fraction=0.01
)

# Advanced with CycleConfig
results = run_active_learning(
    compound_pool=df, oracle=my_oracle, learner='ensemble',
    target_col='Value', featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01,
                    pruning_strategy='score', pruning_params={'pruning_fraction': 0.3})
    ]
)

# Results: dict with compounds_df, cycle_metrics, validation_result, output_dir, saved_files
```

### Core Interfaces

```python
class Learner:
    def train(features: np.ndarray, targets: np.ndarray, smiles: list[str] | None = None) -> None
    def predict(features: np.ndarray, smiles: list[str] | None = None) -> tuple[np.ndarray, np.ndarray | None]
    def supports_uncertainty() -> bool
    def requires_smiles() -> bool    # True for Chemprop/Fastprop

class Oracle:
    def measure(compounds: pl.DataFrame, properties: list[str]) -> pl.DataFrame
```

## Exception Hierarchy

All exceptions in `learnm8/exceptions.py`. Catch `LearnM8Error` for any library error:

```
LearnM8Error(Exception)
├── ConfigurationError    # Bad params/config
├── ValidationError       # Invalid input data (has .invalid_indices, .invalid_smiles)
├── FeatureExtractionError
├── LearnerError
├── AcquisitionError
├── OracleError
├── PersistenceError
├── PruningError
├── ConvergenceWarning
└── LearnM8Warning(UserWarning)
```

## Key Implementation Details

### Feature Preprocessing (all learners)

- NaN/Inf → 0.0 via `np.nan_to_num`
- Zero-variance column removal during training (mask stored, applied on predict)
- Controlled by `remove_zero_variance=True` (default) on learner constructors

### Batch Size

`min(max(1, math.ceil(original_pool_size * batch_fraction)), len(selection_pool))` — floor-at-1 ceiling clipped to remaining pool size. Retains original pool for consistent selection pressure.

### Two-Mode Architecture

- **Benchmark mode** (CSV oracle): discovery + ranking metrics with ground truth
- **Run mode** (Python oracle): basic metrics only. Auto-detected, overridable.

### Feature Caching
HDF5-based v3 schema (bit-packed, 128-bit BLAKE2b cache keys), 100x speedup on reuse. Persists across sessions. Use `cache_dir=Path('.shared_cache')`.

**Storage dtypes** (`Featurizer.get_storage_dtype()`):

- `float32` (default) — descriptors, continuous-valued fingerprints
- `packed_uint8` — binary fingerprints via `np.packbits` (~32× space saving vs float32)
- `uint8` — small-range integer counts (e.g. ERG, MQNs sub-ranges)
- `csr_uint16` — sparse integer count vectors stored as CSR

**v3 schema** (single 2-D `features` dataset per featurizer + side `hash_index`/`row_index`): Blosc-LZ4 level 5 + byte-shuffle, sorted-index reads, 16 MiB chunk cache, `fcntl.flock` concurrency, fail-fast on corruption. `schema_version=3` root attr. `/hash_index` dtype `S16` (128-bit BLAKE2b digests, sorted lex ascending). Strong-ref `OrderedDict` LRU (cap=`DEFAULT_HASH_INDEX_LRU_MAX=4`, `threading.Lock`-guarded) keyed on (path, mtime_ns, write_epoch). v2 caches auto-migrate to `<name>.h5.hash64.bak` on first open; refused with `PersistenceError` if `.bak` already exists. Unknown / future schemas renamed to `<name>.h5.v<N>.bak`. **NFS / Lustre / GlusterFS NOT supported** (`fcntl.flock` has undefined semantics on shared filesystems).

### Prediction Persistence

Per-cycle predictions are written to parquet files (`prediction_cycle_N.parquet`) under the output directory and joined transiently for selection/evaluation. The master DataFrame stays at a constant 7 columns (`ID`, `SMILES`, `status`, `labeled_cycle`, `selected_cycle`, `pruned_cycle`, `<target_col>`) regardless of cycle count (no `prediction_cycle_N` / `uncertainty_cycle_N` accumulation).

### GPU Memory Management
PyTorch learners (Chemprop, Fastprop, ensembles) have `enable_aggressive_gc=True` by default. Cleans GPU memory after train/predict. Safe, best-effort, negligible overhead.

## Component Registry

### Learners

`rf, gp, xgb, lr, dt, mlp, mc_dropout, fastprop, chemprop, chemprop_ensemble, ensemble, rf_ensemble, lr_ensemble, xgb_ensemble, dt_ensemble, mixed_ensemble, fastprop_ensemble`

**Uncertainty support:** gp, mc_dropout, rf, lr, dt, gpu_gp, svgp, rf_fil, ridge_cuml, all ensembles (chemprop_ensemble, rf_ensemble, lr_ensemble, xgb_ensemble, dt_ensemble, mixed_ensemble, fastprop_ensemble)

### Acquisition Strategies

- **Basic** (any model): greedy, random, topk
- **Uncertainty-based:** ucb, ei, pi, thompson, entropy
- **Optimization:** simulated_annealing

### Featurizers

- **2D Circular:** morgan, ecfp, ecfp6, morgan_feat
- **2D Keys:** maccs, pubchem, klekota_roth, laggner
- **2D Topological:** avalon, atom_pair, topological_torsion, rdkit, pattern, layered
- **2D Hashed:** map4, mhfp, lingo, erg, secfp
- **2D Descriptors:** mordred/descriptors, rdkit_2d_descriptors, estate, ghose_crippen, mqns, vsa, bcut2d, physiochemical, pharmacophore, functional_groups
- **3D** (conformer generation): whim, usr, usrcat, e3fp, getaway, morse, rdf, autocorr, electroshape

### Schedules

`quick` (5 cycles), `standard` (10), `intensive` (20)

## Extending LearnM8

### Adding Learners

1. Inherit from `Learner` in `learnm8.core.interfaces`
2. Implement `train()` and `predict()`. Override `requires_smiles()` for SMILES-native learners.
3. Register in `LEARNER_REGISTRY` in `learnm8/api.py`

### Adding Featurizers
1. Add an entry to `_FEATURIZER_CONFIG` in `learnm8/features/__init__.py`
   (keyed by name; specify `cls`, `defaults`, `storage_dtype`, `feature_type`,
   `requires_conformers`) — featurizers are config entries, not separate files
2. `FEATURIZER_REGISTRY` is derived from the config keys automatically
3. Create test in `tests/features/`

### Adding Acquisition Strategies
1. Inherit from base in `learnm8.acquisition.base`
2. Register in acquisition registry
3. Handle graceful fallback on errors

## Chemprop Integration

**ChempropLearner:** Single MPNN model, works directly with SMILES (`requires_smiles()=True`), no uncertainty. Registered as `'chemprop'`.

**ChempropEnsemble:** 3 ChempropLearner instances (seeds 42, 123, 356 — `base_seed + ENSEMBLE_SEED_OFFSETS` where offsets are `(0, 81, 314)`), uncertainty via std dev. Registered as `'chemprop_ensemble'`.

Both support optional `features` as extra descriptors (`x_d`). When `featurizer` is provided, features are concatenated with graph representations (hybrid mode).

**Key parameters:** `message_hidden_dim` (300), `depth` (3), `dropout` (0.0), `ffn_hidden_dim` (300), `max_epochs` (50), `batch_size` (32), `predict_batch_size` (4x batch_size), `precision` ('auto'), `pin_memory` (True), `enable_aggressive_gc` (True).

## Common Pitfalls

**Mock import paths:** Mock at the import location, not the definition location.
```python
# Wrong: @patch('learnm8.acquisition.base.get_acquisition_function')
# Right: @patch('learnm8.core.cycle.get_acquisition_function')
```

**DataFrames:** Polars is immutable by default. `with_columns()` returns new DataFrame. Never modify inputs.

**No print():** Use `logging.getLogger(__name__)` or Rich console. All output via structured logging.

## Important Files

- `learnm8/api.py` — Main entry point
- `learnm8/core/cycle.py` — Cycle execution + batch prediction
- `learnm8/exceptions.py` — Exception hierarchy
- `learnm8/core/resources.py` — CPU/GPU resource validation
- `learnm8/features/extraction.py` — Feature extraction
- `learnm8/learners/base.py` — Base learner + preprocessing
- `learnm8/cli/main.py` — CLI implementation
- `pyproject.toml` — Build config, ruff, mypy, pytest settings
- `.coveragerc` — Coverage configuration
- `environment.yml` — Conda environment
- Do not create extra .md or .txt or README files except when explicitly told to do so.

## Recent Changes

- **Streaming predict→select fusion** — memory-efficient two-pass cycle path (run mode at 100M+ scale). New `streaming: Literal['auto','always','never']='auto'` on `run_active_learning` + CLI `--streaming`. **Eligibility** (`core/streaming_cycle.py:is_streaming_eligible`): streamable acquisition (all but `simulated_annealing`) AND `output_dir` set AND `pruning_strategy in {None,'score'}`; `'auto'` falls back to legacy otherwise, `'always'` raises `ConfigurationError`, `'never'` forces legacy. Dispatch is one branch in `execute_cycle` after `compute_uncertainty` resolution; legacy path byte-for-byte unchanged. **Pass 1**: `iter_status_chunks` (row-index gather, no `prediction_pool` copy) → `_predict_chunk_with_oom_retry` → `core/persistence.py:CyclePredictionsWriter` (context-manager, atomic `.tmp`→`os.replace`, exact `SCORING_CHUNK=1_000_000`-row groups, schema fixed at open from `with_uncertainty = learner.supports_uncertainty() and compute_uncertainty` — 023 column-presence preserved). **Stats**: `core/prediction_stats.py:PredictionStats.from_lazy` (one parquet aggregation; `ddof=0`, linear-interpolation median to match numpy; `from_arrays` is the legacy-parity path) — passed to `_calculate_cycle_metrics(precomputed_stats=...)` so the parent heap never holds the `(N,)` arrays. **Pruning** (`'score'` only): `StreamingTopK` finds the exact `n_to_remove` worst predictions (ties→smaller pool index), `-inf`-masked in pass 2 so global indices stay aligned, marked via `dataframe_ops.py:mark_pruned_by_ids` (join, not `is_in(python_list)`); `pruned_ids=[]` but `pruned_count` exact (`_calculate_cycle_metrics(pruned_count=...)`). **Pass 2**: iterate parquet by row group → `acq_func.score_chunk(...)` (new pointwise tier on `AcquisitionFunction`: `supports_streaming`/`score_chunk`/`shortlist_size`/`finalize`, defaults keep third-party subclasses on legacy) → `acquisition/streaming.py:StreamingTopK` (index-only buffer; canonical order best-first, ties→smaller global index) → gather shortlist by index → `finalize`. `score_chunk` returns **higher-is-better** (direction folded in) so StreamingTopK is always `descending=True`; EI/PI/UCB/entropy share a module-level `_*_scores` helper with their `select()` (single source); greedy/topk = (signed) prediction. **Metrics**: new `metrics['selection_path']` ∈ {`streaming`,`legacy`}. **RNG break (intentional, vs v0.11)**: thompson/random/topk-draw use `utils/rng.py` counter-based RNG (SplitMix64 over `(seed,cycle)`+global pool index) — chunking/memory-invariant draws, same distributions, DIFFERENT samples than v0.11. **Determinism**: deterministic strategies bit-identical to legacy on tie-free score boundaries; at exact ties streaming defines the canonical order (legacy `argpartition` was pivot-arbitrary). **Benchmark mode** stays O(N) (full-pool ranking metrics, `original_pool`, CSV oracle) — win targets run mode. **Bug fix**: `batching.py:_predict_chunk_with_oom_retry` now forwards `compute_uncertainty` (was dropped on the OOM sub-chunk recursion). Prereq refactors: `_measure_and_label` split out of `_select_and_measure` (shared oracle/label tail). **Measured memory** (`scripts/bench_streaming_memory.py`, PSS-summed across worker tree + tracemalloc): at 1M the FULL-pipeline peak is unchanged (legacy ≈ streaming ≈ 14.4 GB RSS) because morgan featurization dominates and is identical in both paths, AND at 1M the pool fits one prediction chunk + one `SCORING_CHUNK` row group (no chunking benefit). Isolating the SELECTION phase (stub 16-dim featurizer, forced 50k chunks) shows the real win: legacy 1257 MB vs streaming 863 MB peak USS = **−394 MB (−31%) at 1M**, scaling ~linearly to tens of GB at 100M (the plan's `selection_pool`≈10 GB / `all_valid_ids`≈7 GB table). The win is a run-mode, ≥10M-scale phenomenon; the 1M full benchmark serves as a correctness/no-regression check. `hypothesis`/`psutil` added to `[test]` extra. Robustness: `CyclePredictionsWriter.__enter__` defensively `mkdir(parents=True)`s `output_dir`.
- **025 Featurization & caching optimization** — five-item speed/memory pass on the feature-extraction subsystem; no public API change, no cache schema bump (`schema_version` stays 3). **Item 1 (cache-hash denylist)**: module-level `_CACHE_IRRELEVANT_PARAMS` frozenset (`n_jobs`, `verbose`, `batch_size`) in `features/base.py`; `SkfpFeaturizer.get_config()` filters it from the `fingerprint.get_params()` sub-dict ONLY — the manually-assembled keys (`random_state`, `conformer_params`, `auto_generate_conformers`) and any unknown `get_params()` key are kept (fail-safe). A one-shot WARNING (`_warn_orphaned_cache_once`, reset hook `_reset_orphan_cache_warning`) fires on the first `get_config()` call noting pre-025 cache rows are orphaned. Because `verbose`/`batch_size` were previously hashed, all featurizers cold-rebuild once after upgrade. **Item 2 DROPPED** during review (the proposed index-merge change removed a no-op copy — zero gain). **Item 3**: deleted the dead `_get_optimal_n_jobs` (+ `import os`) from `extraction.py`; caller-supplied `n_jobs` flows through unchanged (no auto-cap). **Item 4 (idle worker teardown)**: new `features/worker_pool.py:shutdown_featurization_pool()` — best-effort loky reusable-executor teardown (`kill_workers=True, wait=True`), safe no-op when `reusable_executor._executor is None`, logs at WARNING and never raises. Called once per cycle from `api.py` at two sites: after `select_initial_batch` in `_initialize_active_learning` (cycle 0) and in a per-cycle `finally` in `_execute_loop` (cycles 1-N, fires on the pool-exhausted `break` and on re-raise). Reaps the ~22 GiB cross-cycle idle-worker plateau; does NOT reduce the in-flight `transform` peak. NOT placed inside `extract_features`/`predict_with_batching` (would defeat loky's intra-cycle reuse). **Item 5 (canonicalize once at validation)**: `core/validation.py:canonicalize_for_cache(smiles)` — bare `Chem.MolToSmiles(Chem.MolFromSmiles(smiles))`, verbatim fallback on parse failure; byte-identical to the deleted `cache._canonicalize_smiles` recipe, so cache keys are unchanged and Item 5 triggers NO cold rebuild. `_validate_smiles` returns the canonical RAW SMILES; `validate_compound_pool` replaces the master DataFrame `SMILES` column in place (stays at the same column count; `invalid_compounds` keep raw SMILES). `skip_validation=True` runs the same helper as a standalone `dm.parallelized` pass in `api.py`. `cache._cache_keys_bytes16` now keys SMILES verbatim. **Behavior change**: raw `extract_features(['OCC'])` vs `['CCO']` no longer auto-dedupe (2 cache rows) — equivalence is established at validation; persisted `compounds_final.csv`/parquet now report canonical SMILES. **Item 6 (memory-aware featurization batching)**: `core/batching.py:featurization_chunk_size(input_len)` — fixed-cap-with-floor clamp (`FEATURIZATION_CHUNK_FLOOR=1024`, `FEATURIZATION_CHUNK_CAP=50_000`) feeding scikit-fingerprints' native `batch_size`, set on a per-call `copy.deepcopy` of the `SkfpFeaturizer` so the shared instance is never mutated (race-free across concurrent `extract_features` callers). All featurizers in scope — T001 probe confirmed chunk-independence (binary FP byte-identical, float/3D within `np.allclose`). Item 6 depends on Item 1 (`batch_size` denylisted so chunk size never forks the cache).
- **023 Acquisition-aware uncertainty (v0.11.0 — BREAKING)** — `Learner.predict()` gains a keyword-only `compute_uncertainty: bool = True` parameter. The cycle resolves `compute_uncertainty = force_uncertainty OR acq_func.requires_uncertainty()` once at cycle start and threads it through `_predict_pool → predict_with_batching → learner.predict → EnsembleBase` cascade. 8 skip-eligible learners (gp, rf, dt, lr, rf_fil, ridge_cuml, gpu_gp, svgp) genuinely omit uncertainty compute; ensembles cascade to members AND short-circuit the outer std-reduction (D10 — saves ~4 GB at 100M × N-member scale). RF unifies its mean source across both branches for bit-identity under the force_uncertainty toggle (D9). mc_dropout stays uniform-contract: N stochastic passes still run (skipping them would change mean semantics per Gal & Ghahramani 2016). New API: `run_active_learning(..., force_uncertainty=False)` + CLI flag `--force-uncertainty`. Per-cycle parquet drops the `uncertainty` column entirely when skipped (column-presence semantics; no fabricated nulls); cycle metrics keep `uncertainty_*` keys present at None with `has_uncertainty=False`. **Breaking**: third-party `Learner` subclasses raise `TypeError: predict() got an unexpected keyword argument 'compute_uncertainty'` on first cycle predict; pinned by `tests/core/test_third_party_subclass_breakage.py`. 5-line migration diff in CHANGELOG 0.11.0. No runtime shim, no DeprecationWarning. External validation: sklearn#31374 reports ~96% saving on the matching GP optimisation.
- **020 Cache integrity at 100M scale** — bumped HDF5 cache `schema_version` from 2 to 3. `/hash_index` widened from `uint64` to `S16` (128-bit BLAKE2b digests), collision probability at N=10**8 drops from ~2.7e-4 to ~1.5e-23. Replaced the prior weakref hash_index cache with a strong-ref `OrderedDict` LRU (`DEFAULT_HASH_INDEX_LRU_MAX=4`, ~6.4 GB headroom at 100M) guarded by `threading.Lock`; entries auto-evict on `write_epoch` bump or mtime change. v2 caches auto-migrate to `<name>.h5.hash64.bak` on first open under `LOCK_EX`; an existing `.bak` raises `PersistenceError` (no data clobber). 3D featurizers (whim, usr, usrcat, e3fp, getaway, morse, rdf, autocorr, electroshape) gain `random_state: int = 0xf00d` (`DEFAULT_3D_RANDOM_STATE`, RDKit ETKDG convention) forwarded to `ConformerGenerator` with try/except fallback for scikit-fingerprints<1.18.0; recorded in `get_config()` only when `requires_3d()` is True so cache keys disambiguate different seeds without invalidating 2D caches. Single-node single-filesystem only — NFS/Lustre/GlusterFS NOT supported.
- **017 Count-FP storage routing + uint8 tree inputs** — fixed silent count-fingerprint corruption: `MorganFeaturizer`, `MACCSFeaturizer`, `AtomPairFeaturizer`, `TopologicalTorsionFeaturizer` now flip `feature_type` to `'continuous'` and route to `csr_uint16` (Morgan/AtomPair/TopTorsion) or `uint8` (MACCS) storage when `count=True`. **Severity callout**: any user previously running `count=True` had `np.packbits` truncate every nonzero count to a single bit; legacy `packed_uint8` cache files are auto-migrated to `<name>.h5.dim*.bak` on first open with the new code. Added `Learner.preferred_feature_dtype()` (default `'float32'`); tree learners (RF, XGB, DT) override to `'uint8'` for binary features and `extract_features(..., preferred_dtype='uint8')` skips the float32 inflation — 4× working-set reduction on a 2048-bit Morgan matrix. `_preprocess_features` now has a uint8 fast path that skips `np.isnan`/`np.nan_to_num` and preserves the compact dtype through the zero-variance mask. CSR / float32 storage transparently fall back to float32.
- **016 Storage dtype expansion** (merged) — added `uint8` and `csr_uint16` storage paths to v2 cache for small-range integer counts and sparse integer vectors. `fingerprint_used` label appends `_uint8` / `_csruint16` dtype tokens. ERG, MQNs, pharmacophore, physiochemical featurizers updated.
- **015 Bit-packed cache v2** (merged) — rewrote `learnm8/features/cache.py` from v1 (per-row float32 dataset) to v2 (single 2-D `features` dataset per featurizer + side `hash_index`/`row_index` + `np.packbits` for binary). Blosc-LZ4 level 5 + byte-shuffle, `fcntl.flock` concurrency, fail-fast on corruption with 1 transient retry, `schema_version=2` root attr, `os.replace` to `.bak` for non-v2 files. Added `Featurizer.get_storage_dtype()` on the abstract interface. Public `extract_features(...)` signature unchanged. Performance: 1M-row Morgan-2048 warm read <5 s, 100M-row open <1 s, ≤30 GB on-disk at 100M.
- **014 Diagnostic metrics** — added `feature_extraction_time` and additional diagnostic cycle metrics (`learnm8/evaluation/metrics/diagnostics.py`).
- **013 Diversity metrics overhaul** — replaced legacy `intra_batch_diversity`/`inter_cycle_similarity`/`batch_novelty_score` with three rigorously-defined diversity primitives (`mean_tanimoto_similarity_sampled`, `scaffold_diversity_index`, `shannon_entropy_diversity`), each emitted twice per cycle (batch + cumulative = 6 keys) plus a `fingerprint_used` provenance column. Adaptive Tanimoto pair sampling, online Shannon accumulator, explicit `RunCache` lifetime. < 1 s per cycle at 100k cumulative. Flag accepts `bool | Iterable[str]` for per-metric opt-out. Featurizer-dependence caveat: metric values depend on the fingerprint used; comparing the same column across runs with DIFFERENT featurizers is NOT valid — audit via `fingerprint_used` column.
- **008 Parquet predictions** — per-cycle predictions persist to `prediction_cycle_N.parquet` files (joined transiently); master DataFrame stays at constant 7 columns. Migration tooling: `scripts/migrate_to_parquet.py`, `scripts/migrate_validation_reports.py`.
- **Oracle.known_ids** — added to `Oracle` interface; implemented on `CSVOracle` for pool reconciliation.
- Centralized exception hierarchy in `learnm8/exceptions.py` (ERR-001)
- Migrated build config from `setup.py`/`pytest.ini` to `pyproject.toml`
- Added `ruff` and `mypy` to conda environment
- Added CPU/GPU resource control (`n_jobs`, `device` parameters)
- Replaced `print()` statements with structured logging
- Enforced `--strict-markers` for pytest
- Added `.coveragerc` for coverage configuration
- Fixed GP learner feature preprocessing on predict
- Fixed TorchLearner model recreation on input dimension change
- 023 math-correctness: chunked-predict uncertainty alignment, Spearman None-on-undefined signaling, float64 accumulators in eval reducers, activity-count dtype validation (ValidationError on non-numeric), 3D conformer cache determinism (random_state=None fallback), Inf-sigma guard on EI/PI, top-K tie stability (Polars sort convergence).

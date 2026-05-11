# CLAUDE.md

## Constitution

**All development MUST follow the project constitution:** [`.specify/memory/constitution.md`](.specify/memory/constitution.md)

The constitution defines: SDD/TDD workflow, agent autonomy tiers, red lines, code standards, error handling, git workflow, quality gates, and architectural governance. When in doubt, the constitution takes precedence.

Key governance files:

- **Constitution:** `.specify/memory/constitution.md`
- **Spec Template:** `specs/TEMPLATE.md`
- **ADR Template:** `docs/decisions/TEMPLATE.md`

## Project Overview

LearnM8 is an active learning framework for molecular screening (v0.10.0). Pure functional API with `run_active_learning()` as main entry point. Modular design with 7 core modules. Polars-first DataFrames (accepts pandas, auto-converts).

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

**When to include slow tests:** Changes to `learners/torch/`, `learners/ensemble/`, `features/skfp_3d/`, `api.py`, `core/cycle.py`, or pre-commit/PR validation.

## CLI

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10
learnm8 validate compounds.csv        # Check SMILES validity
learnm8 list learners                 # List available components
learnm8 run --config experiment.yaml  # Config file support
```

## Architecture

### Seven-Phase Execution Flow
1. **Normalize** inputs → 2. **Validate** compounds → 3. **Initialize** master DataFrame + cycle 0 → 4. **Configure** cycle schedule → 5. **Execute** cycles 1-N → 6. **Persist** CSV results → 7. **Return** results dict

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
│   ├── validation.py         # Early compound validation
│   ├── initialization.py     # Master DataFrame setup
│   ├── persistence.py        # CSV export
│   ├── dataframe_ops.py      # Vectorized Polars operations
│   ├── data_structures.py    # Shared data structures
│   └── resources.py          # CPU/GPU resource validation (n_jobs, device)
├── features/                 # HDF5-cached feature extraction
│   ├── extraction.py         # extract_features() function
│   ├── cache.py              # HDF5 caching layer
│   ├── base.py               # SkfpFeaturizer base class
│   ├── skfp_2d/              # 26 2D fingerprint featurizers
│   └── skfp_3d/              # 9 3D fingerprint featurizers
├── learners/
│   ├── base.py               # Base classes + feature preprocessing
│   ├── sklearn/              # RF, GP, XGBoost, DT, LR, AdvancedRF
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
    compound_pool='compounds.csv', target_col='Activity',
    learner='rf', featurizer='morgan', n_cycles=10, batch_fraction=0.01
)

# Advanced with CycleConfig
results = run_active_learning(
    compound_pool=df, oracle=my_oracle, learner='ensemble',
    target_col='Activity', featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=5, batch_fraction=0.01,
                    pruning_strategy='score_based', pruning_params={'pruning_fraction': 0.3})
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

Per-cycle predictions are written to parquet files (`prediction_cycle_N.parquet`) under the output directory and joined transiently for selection/evaluation. The master DataFrame stays at a constant 8 columns regardless of cycle count (no `prediction_cycle_N` / `uncertainty_cycle_N` accumulation).

### GPU Memory Management
PyTorch learners (Chemprop, Fastprop, ensembles) have `enable_aggressive_gc=True` by default. Cleans GPU memory after train/predict. Safe, best-effort, negligible overhead.

## Component Registry

### Learners

`rf, gp, xgb, lr, dt, mlp, mc_dropout, fastprop, chemprop, chemprop_ensemble, ensemble, rf_ensemble, lr_ensemble, xgb_ensemble, dt_ensemble, mixed_ensemble, fastprop_ensemble`

**Uncertainty support:** gp, mc_dropout, all ensembles (including chemprop_ensemble, fastprop_ensemble)

### Acquisition Strategies

- **Basic** (any model): greedy, random, topk
- **Uncertainty-based:** ucb, ei, pi, thompson, entropy
- **Optimization:** simulated_annealing

### Featurizers

- **2D Circular:** morgan, ecfp, ecfp6, morgan_feat, secfp
- **2D Keys:** maccs, pubchem, klekota_roth, laggner
- **2D Topological:** avalon, atom_pair, topological_torsion, rdkit, pattern, layered
- **2D Hashed:** map4, mhfp, lingo, erg
- **2D Descriptors:** mordred/descriptors, rdkit_2d_descriptors, estate, ghose_crippen, mqns, vsa, bcut2d, physiochemical, pharmacophore, functional_groups
- **3D** (conformer generation): whim, usr, usrcat, e3fp, getaway, morse, rdf, autocorr, electroshape

### Schedules

`quick` (5 cycles), `standard` (10), `intensive` (20), `diverse` (10 mixed)

## Extending LearnM8

### Adding Learners

1. Inherit from `Learner` in `learnm8.core.interfaces`
2. Implement `train()` and `predict()`. Override `requires_smiles()` for SMILES-native learners.
3. Register in `LEARNER_REGISTRY` in `learnm8/api.py`

### Adding Featurizers
1. Inherit from `SkfpFeaturizer` in `learnm8.features.base`
2. Create file in `learnm8/features/skfp_2d/` (or `skfp_3d/`)
3. Register in `learnm8/features/__init__.py` (import, registry dict, `__all__`)
4. Create test in `tests/features/skfp_2d/`

### Adding Acquisition Strategies
1. Inherit from base in `learnm8.acquisition.base`
2. Register in acquisition registry
3. Handle graceful fallback on errors

## Chemprop Integration

**ChempropLearner:** Single MPNN model, works directly with SMILES (`requires_smiles()=True`), no uncertainty. Registered as `'chemprop'`.

**ChempropEnsemble:** 3 ChempropLearner instances (seeds 42, 123, 456), uncertainty via std dev. Registered as `'chemprop_ensemble'`.

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

- **020 Cache integrity at 100M scale** — bumped HDF5 cache `schema_version` from 2 to 3. `/hash_index` widened from `uint64` to `S16` (128-bit BLAKE2b digests), collision probability at N=10**8 drops from ~2.7e-4 to ~1.5e-23. Replaced the prior weakref hash_index cache with a strong-ref `OrderedDict` LRU (`DEFAULT_HASH_INDEX_LRU_MAX=4`, ~6.4 GB headroom at 100M) guarded by `threading.Lock`; entries auto-evict on `write_epoch` bump or mtime change. v2 caches auto-migrate to `<name>.h5.hash64.bak` on first open under `LOCK_EX`; an existing `.bak` raises `PersistenceError` (no data clobber). 3D featurizers (whim, usr, usrcat, e3fp, getaway, morse, rdf, autocorr, electroshape) gain `random_state: int = 0xf00d` (`DEFAULT_3D_RANDOM_STATE`, RDKit ETKDG convention) forwarded to `ConformerGenerator` with try/except fallback for scikit-fingerprints<1.18.0; recorded in `get_config()` only when `requires_3d()` is True so cache keys disambiguate different seeds without invalidating 2D caches. Single-node single-filesystem only — NFS/Lustre/GlusterFS NOT supported.
- **017 Count-FP storage routing + uint8 tree inputs** — fixed silent count-fingerprint corruption: `MorganFeaturizer`, `MACCSFeaturizer`, `AtomPairFeaturizer`, `TopologicalTorsionFeaturizer` now flip `feature_type` to `'continuous'` and route to `csr_uint16` (Morgan/AtomPair/TopTorsion) or `uint8` (MACCS) storage when `count=True`. **Severity callout**: any user previously running `count=True` had `np.packbits` truncate every nonzero count to a single bit; legacy `packed_uint8` cache files are auto-migrated to `<name>.h5.dim*.bak` on first open with the new code. Added `Learner.preferred_feature_dtype()` (default `'float32'`); tree learners (RF, XGB, DT, AdvancedRF) override to `'uint8'` for binary features and `extract_features(..., preferred_dtype='uint8')` skips the float32 inflation — 4× working-set reduction on a 2048-bit Morgan matrix. `_preprocess_features` now has a uint8 fast path that skips `np.isnan`/`np.nan_to_num` and preserves the compact dtype through the zero-variance mask. CSR / float32 storage transparently fall back to float32.
- **016 Storage dtype expansion** (merged) — added `uint8` and `csr_uint16` storage paths to v2 cache for small-range integer counts and sparse integer vectors. `fingerprint_used` label appends `_uint8` / `_csruint16` dtype tokens. ERG, MQNs, pharmacophore, physiochemical featurizers updated.
- **015 Bit-packed cache v2** (merged) — rewrote `learnm8/features/cache.py` from v1 (per-molecule float32 dataset) to v2 (single 2-D `features` dataset per featurizer + side `hash_index`/`row_index` + `np.packbits` for binary). Blosc-LZ4 level 5 + byte-shuffle, `fcntl.flock` concurrency, fail-fast on corruption with 1 transient retry, `schema_version=2` root attr, `os.replace` to `.bak` for non-v2 files. Added `Featurizer.get_storage_dtype()` on the abstract interface. Public `extract_features(...)` signature unchanged. Performance: 1M-row Morgan-2048 warm read <5 s, 100M-row open <1 s, ≤30 GB on-disk at 100M.
- **014 Diagnostic metrics** — added `feature_extraction_time` and additional diagnostic cycle metrics (`learnm8/evaluation/metrics/diagnostics.py`).
- **013 Diversity metrics overhaul** — replaced legacy `intra_batch_diversity`/`inter_cycle_similarity`/`batch_novelty_score` with three rigorously-defined diversity primitives (`mean_tanimoto_similarity_sampled`, `scaffold_diversity_index`, `shannon_entropy_diversity`), each emitted twice per cycle (batch + cumulative = 6 keys) plus a `fingerprint_used` provenance column. Adaptive Tanimoto pair sampling, online Shannon accumulator, explicit `RunCache` lifetime. < 1 s per cycle at 100k cumulative. Flag accepts `bool | Iterable[str]` for per-metric opt-out. Featurizer-dependence caveat: metric values depend on the fingerprint used; comparing the same column across runs with DIFFERENT featurizers is NOT valid — audit via `fingerprint_used` column.
- **008 Parquet predictions** — per-cycle predictions persist to `prediction_cycle_N.parquet` files (joined transiently); master DataFrame stays at constant 8 columns. Migration tooling: `scripts/migrate_to_parquet.py`, `scripts/migrate_validation_reports.py`.
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

<!-- SPECKIT:START -->
## Current Feature: 022-metrics-scale
**Phase**: plan (complete)
**Spec**: specs/022-metrics-scale/spec.md
**Plan**: specs/022-metrics-scale/plan.md
**Tech Stack**: Python 3.11.9, numpy, scipy.special (`ndtr`, `erfc`), polars (Float32 parquet), rdkit (ExplicitBitVect)
**Key Decisions**:
- `scipy.special.ndtr(z)` replaces inlined `0.5*(1+erf(z/√2))` in EI/PI (research amendment to FR-006)
- Vitter Algorithm R reservoir for `cumulative_fp_buffer` with global stream counter — new `RunCache.cumulative_seen_count` field
- `np.argpartition(-pred, max_K-1)[:max_K]` then sort the K-slice; numpy chosen over Polars `top_k` (needs indices, not values)
- Float32 parquet schema with `schema_overrides`-based back-compat loader for Float64 fixtures
- `np.nanmean/nanstd/nansum(..., dtype=np.float64)` accumulators for cycle metrics (FR-010)
<!-- SPECKIT:END -->

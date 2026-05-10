# LearnM8 — Project Memory for Claude Code

This file is loaded into the AI assistant's context for every conversation in this repository. Keep it concise and load-bearing.

## Project context

- **Domain**: Active learning for molecular screening (small-molecule virtual screening pipeline; scales to 10⁸ compounds).
- **Conda env**: `learnm8` (defined in `environment.yml`). Always run tests, ruff, and mypy via `conda run -n learnm8 ...`.
- **Tests**: pytest with strict speed markers — `unit`, `integration`, `slow`. Unmarked tests cause `pytest_collection_modifyitems` to call `pytest.exit`. **Use `@pytest.mark.unit`, not `@pytest.mark.fast`.**
- **Lint**: ruff (config in `pyproject.toml`). Lint-on-stop hook auto-runs after Claude finishes responding; address findings before continuing.
- **Type checks**: mypy strict on new modules; Pyright in IDE may report `reportMissingImports` for `h5py`/`hdf5plugin`/`skfp.*` — these are env-resolution warnings, not code errors.

## Architecture invariants

- **Pure functional API**: `extract_features`, `run_active_learning` accept config and return data; no global state (except the `_HASH_INDEX_CACHE` LRU which is an explicit performance optimization).
- **HDF5 v3 feature cache** (`learnm8/features/cache.py`): 128-bit BLAKE2b cache keys stored as `S16`, OrderedDict LRU with `threading.Lock`, `fcntl.flock` sidecar lock for inter-process coordination, single-node single-filesystem only (NFS/Lustre NOT supported).
- **Featurizer ABC** (`learnm8/core/interfaces.py`): all featurizers expose `feature_type` (`'binary'` or `'continuous'`), `get_name()`, `get_config()`, `get_config_hash()`, `transform()`. 3D featurizers also accept `random_state: int = 0xf00d`.
- **`SchemaVersion = 3`**; v2 caches auto-migrate to `<name>.h5.hash64.bak` on first open. Refusal-to-clobber if `.bak` already exists.

## Working with this codebase

- **Cache changes are Tier-3** (constitution §3.3): coordinate via spec-driven workflow (`/speckit.*` commands).
- **3D featurizer subclasses** (`learnm8/features/skfp_3d/*.py`) are mechanical — same constructor pattern across all 9 (whim, usr, usrcat, e3fp, getaway, morse, rdf, autocorr, electroshape). When adding parameters, update all 9 + the base in `features/base.py`.
- **GETAWAY** is non-deterministic per [rdkit#7264](https://github.com/rdkit/rdkit/issues/7264); cache-key disambiguates but byte-identity across processes is NOT guaranteed.
- **AutocorrFeaturizer** has a pre-existing bug: doesn't pass `use_3D=True`, so `requires_3d()` returns False. The file lives in `skfp_3d/` but behaves as 2D-like. Out of scope for Feature 020; flag as a separate ticket.

## Recent decisions (Feature 020 — 0.10.0 release)

- **`DEFAULT_HASH_INDEX_LRU_MAX = 4`** (was implicitly 8 with weakref): preserves 6.4 GB hash_index headroom at 100M scale, fits 30 GB cache budget alongside Morgan-2048 features. Override via `LEARNM8_HASH_INDEX_LRU_MAX` env var.
- **`DEFAULT_3D_RANDOM_STATE = 0xf00d`** (61453, RDKit ETKDG convention) over skfp's default `0`. Aligns with broader RDKit ecosystem reproducibility practice.
- **128-bit BLAKE2b** chosen over xxhash to avoid new runtime dependency (stdlib `hashlib`).
- **`S16` HDF5 dtype** chosen over `uint64[2]` because structured-dtype `np.searchsorted` is materially slower than fixed-width-bytes searchsorted.

## Markers for tools

<!-- speckit:active-feature -->
- Feature 020 (cache integrity at 100M) — implementation phase complete; tests in progress.
<!-- /speckit:active-feature -->

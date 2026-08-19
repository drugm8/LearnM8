# Changelog

## [Unreleased]

### Changed — adaptive annealing schedule (BREAKING behaviour; SemVer **MINOR**, 0.11.0 → 0.12.0)

`SimulatedAnnealingAcquisition` degenerated to `GreedyAcquisition` in
production and said nothing about it. Energy is `-prediction`, so temperature
carries the target's units, but `initial_temp` defaulted to a bare `1.0`. On
the real AmpC cycle-9 predictions (880,541 rows) the mean uphill gap is
`E[ΔE⁺] = 3.63`, so mean acceptance at `T₀ = 1.0` was **0.178** — the chain
froze almost immediately. `max_iterations` defaulted to `1000` independent of
`n_select`, so a 10,000-compound batch could never be supplied by the walk, and
the shortfall was filled with `argsort(prediction)` with no warning. Measured
on that pool at `n_select=10000`: **99.8% of the batch came from greedy
backfill** and the result reproduced greedy's top-k at **99.9% overlap**.

**The three schedule parameters now default to `None`, meaning "derive from
this cycle's predictions":**

- `initial_temp` / `final_temp` are solved by Ben-Ameur fixed-point iteration
  so the *realized* mean uphill acceptance equals 0.8 / 0.01. The closed form
  `T = E[ΔE⁺] / −ln χ` only seeds the iteration: `exp` is convex, so by Jensen
  it overshoots χ by an amount that grows with the spread of ΔE (0.009 at
  CV 0.71, 0.039 at CV 2.22 — the tolerance is 0.01). Gaps are sampled through
  the *same proposal path* the walk uses, never from the global σ, because a
  `'score_band'` neighbourhood and a `'random'` one differ by orders of
  magnitude on identical predictions.
- `max_iterations` is derived from `n_select`, pool size, and a **measured**
  acceptance rate. On AmpC this gives `T₀ = 14.70` (14.7× the old default) and
  36,000 steps (36× the old default), yielding **0.0% backfill**.

Passing a number for any of the three still uses it verbatim and disables
derivation for that parameter alone. **Note the semantic change:** an explicit
`max_iterations` is now the *total* step budget across all chains
(`n_chains × chain_length ≤ max_iterations`), not a per-chain length.

**Also changed:**

- **Vectorized `R × L` chains.** The walk now runs `R` independent chains
  (started at `R` independent random positions, so they explore different
  basins) advanced together in numpy. The Python loop is `L ≤ 2000` iterations
  regardless of `n_select` or pool size. At 880k × 10k this is **13× faster**
  than the old single chain: 0.07 s vs 0.92 s.
- **Backfill is now visible and fatal past a threshold.** New
  `last_backfill_fraction` attribute and `acquisition_backfill_fraction` cycle
  metric (`None` for every non-SA strategy — no fabricated zeros). Above 10%
  logs a WARNING; above 50% raises `AcquisitionError`. Silent degradation to
  greedy is no longer possible.
- **`select()` returns rows best-first** by `acquisition_score`. Previously it
  returned pool order, so `core/cycle.py`'s `.head(batch_size)` truncation
  dropped arbitrary rather than worst compounds.
- **A position is recorded as visited only when the move was accepted.** The
  old code appended unconditionally, re-logging the same index once per
  rejected iteration.
- `supports_streaming()` **remains `False`.** Annealing is a path-dependent
  walk over the full pool and cannot be expressed as the pointwise
  `score_chunk` contract, so SA continues to run the legacy cycle path. This is
  unchanged behaviour, stated explicitly so it is not read as an oversight.

**RNG-stream break:** every seeded SA selection differs from v0.11 output. The
walk is a different algorithm (R chains vs one, a calibration sample and a
pilot walk drawn before it starts), so identical seeds no longer reproduce
identical batches. Determinism *within* 0.12.0 is preserved and tested.

**Migration:** delete `initial_temp`, `final_temp` and `max_iterations` from
existing `acquisition_params` unless you specifically know the target's energy
scale — hand-set values are what caused the defect. Archived benchmark runs of
the `simulated_annealing` arm are **not** recoverable: they are greedy runs
mislabelled as SA, and must be re-run against this implementation if SA is to
be reported.

### Fixed — silently dropped user parameters (`acquisition_params` and the config-file path)

Every `--acquisition-params` value passed alongside `--cycles` (or a config
file's `cycles:` list) was discarded. `parse_cycle_schedule` declared the
parameter and then used only the per-block value in its `cycles is not None`
branch — the branch the CLI always takes — so runs completed normally, reported
success, and used the acquisition defaults. A UCB `beta` sweep produced
byte-identical trajectories at every cycle. Fixed by mirroring the top-level
pruning fallback added in `851d572`: a block without its own
`acquisition_params` inherits the top-level dict, an explicit per-block dict
still wins.

An audit of the surrounding plumbing for the same defect class found four more,
all on the `--config <yaml>` path:

- **`pruning_params` never reached `run_active_learning`** — it was absent from
  `cli/main.py:_build_run_kwargs` entirely, so the documented YAML key was
  inert. Now forwarded.
- **Config keys documented under their API name landed on unread attributes** —
  `output_dir`, `smiles_column`, `id_column` and `large_features_ack` do not
  match their argparse dests (`output`, `smiles_col`, `id_col`,
  `allow_large_features`). `output_dir` appears in every documented config
  example, so results silently went to an auto-generated timestamped directory.
  A `CONFIG_KEY_ALIASES` map now resolves them.
- **A YAML `acquisition_params:` mapping crashed** with
  `TypeError: the JSON object must be str, bytes or bytearray, not dict` —
  `cmd_run` ran `json.loads` on a value the YAML loader had already decoded.
  Mappings are now passed through.
- **Config values overrode explicit CLI flags** — the reverse of the documented
  contract printed by the CLI itself. `main()` now records which dests the user
  actually typed (a second parse with defaults suppressed) and `cmd_run` skips
  those config keys, logging each one it ignores. `cmd_run(args)` keeps its
  old behaviour for direct programmatic callers.

### Fixed — GT-stats cache finalizer self-deadlock

`evaluation/core.py`'s `_GT_STATS_LOCK` is now a `threading.RLock` instead of
`Lock`. `_get_gt_stats` holds the lock while calling `_compute_gt_stats`, whose
logging can trigger GC and fire the `_evict_gt_cache` weakref finalizer on the
**same thread**; the non-reentrant `Lock` self-deadlocked there. GC-timing
dependent (surfaced under long benchmark-mode test runs). A finalizer firing on
a different thread still blocks normally.

### Added — Streaming predict→select fusion (memory-efficient large-pool path)

A memory-efficient two-pass cycle path that avoids materializing the full
``(N,)`` prediction arrays, the per-cycle ``all_valid_ids`` Python list, the
unlabeled⋈predictions join, and the full ``cycle_predictions`` DataFrame in the
parent heap. At 1M compounds these structures accounted for ~6.6 GB of
Python/numpy allocation (measured); the streaming path eliminates them.

- **New `streaming` parameter** on ``run_active_learning(...)`` and CLI
  ``learnm8 run --streaming {auto,always,never}`` (default ``'auto'``).
  ``'auto'`` uses streaming when the cycle is eligible and falls back to the
  legacy path otherwise; ``'always'`` raises ``ConfigurationError`` on an
  ineligible cycle; ``'never'`` forces the legacy in-memory path.
- **Eligibility**: a streamable acquisition strategy (greedy, ucb, ei, pi,
  entropy, thompson, random, topk — all but simulated_annealing), an
  ``output_dir`` (the per-cycle parquet is the dataflow medium), and a pruning
  strategy in ``{None, 'score'}``.
- **Pass 1** predicts the unlabeled pool chunk-by-chunk into
  ``prediction_cycle_N.parquet`` via the new ``CyclePredictionsWriter`` (atomic,
  exact 1M-row groups). **Pass 2** scores the parquet by row group with a new
  pointwise acquisition tier (``score_chunk``) and reduces to the batch with a
  bounded ``StreamingTopK`` index buffer.
- **New `metrics['selection_path']`** records ``'streaming'`` or ``'legacy'``
  per cycle (provenance, mirrors ``fingerprint_used``).
- **Determinism (re-scoped)**: deterministic strategies (greedy/ucb/ei/pi/
  entropy) select the **identical** batch as the legacy path whenever the
  k-th-place score boundary is tie-free. At exact score ties the streaming
  path defines a NEW canonical order — best first, ties broken by smaller pool
  index — deterministic and invariant to chunking (the legacy
  ``np.argpartition`` membership was pivot-arbitrary at ties).
- **RNG change (intentional)**: stochastic strategies (thompson, random,
  topk's random draw) now use a counter-based RNG keyed on
  ``(random_state, cycle)`` and the candidate's pool position
  (``learnm8/utils/rng.py``). Draws are identical regardless of chunking and
  host memory, but **differ from v0.11** for the same seed. The sampling
  distributions are unchanged.
- **Pruning** on the streaming path uses ``StreamingTopK`` to find the exact
  ``n_to_remove`` worst predictions (matching ``ScoreBasedPruner``'s count;
  boundary ties broken by smaller pool index) and marks them in the master
  DataFrame via a join (``mark_pruned_by_ids``) rather than an ``is_in`` over a
  multi-million-ID Python list. Per-cycle ``pruned_ids`` is an empty list on the
  streaming path (already dropped from ``cycle_metrics.csv``); ``pruned_count``
  remains exact.
- **Benchmark mode** still runs streaming for pass 1/2, but its ground-truth
  ranking metrics (and the CSV oracle / ``original_pool``) remain O(N) by
  design; the memory win targets **run mode** at scale.
- **Bug fix**: ``predict_with_batching``'s OOM-retry recursion now forwards
  ``compute_uncertainty`` (previously dropped), so OOM sub-chunks no longer
  compute uncertainty the cycle opted out of.
- Internal: ``PredictionStats`` value object carries the per-cycle metric
  reductions (computed once from the parquet on the streaming path, from arrays
  on the legacy path — value-identical); ``_select_and_measure`` split to share
  the oracle/label tail (``_measure_and_label``) with the streaming path.

### Removed — Dead-code cleanup

Unused public surface and empty stub packages removed. None of these
symbols were referenced by ``run_active_learning()`` or any built-in
component; removal does not affect standard API or CLI usage.

- ``AdvancedRandomForestLearner`` class and the ``advanced_rf`` learner key
  (the class was never registered in ``LEARNER_REGISTRY``).
- Unused sklearn learner introspection methods: ``get_feature_importance``,
  ``get_tree_stats``, ``get_coefficients``, ``get_intercept``,
  ``get_booster_stats``, ``get_learned_hyperparameters``. ``get_oob_score``
  is retained.
- Unused ensemble methods: ``get_ensemble_statistics``,
  ``get_individual_predictions``, ``add_learner``, ``remove_learner``.
- Pruning ``get_pruning_stats`` — both the abstract
  ``DesignSpacePruner.get_pruning_stats`` and the ``ScoreBasedPruner``
  implementation.
- TorchLearner ``save_model``, ``load_model``, ``get_training_history``
  and the backing ``training_history`` attribute.
- Module ``learnm8/cli/formatting.py`` (``format_cycle_metrics_table``).
- Module ``learnm8/utils/data_loaders.py``.
- Empty stub packages ``learnm8/features/skfp_2d/`` and
  ``learnm8/features/skfp_3d/``. The 39 featurizers are defined in the
  ``_FEATURIZER_CONFIG`` factory in ``learnm8/features/__init__.py``; new
  featurizers are added there, not as files in those packages.

## [0.11.0] — 2026-05-13

### Added — Acquisition-Aware Uncertainty Computation (feature 023)

- `Learner.predict()` gains a keyword-only ``compute_uncertainty: bool = True``
  argument. The active-learning cycle resolves
  ``compute_uncertainty = force_uncertainty OR acq_func.requires_uncertainty()``
  once at cycle start and threads it through
  ``_predict_pool → predict_with_batching → learner.predict``. Skip-eligible
  learners genuinely elide the uncertainty-specific compute path.
- `run_active_learning(force_uncertainty=False)` API parameter and
  ``learnm8 run --force-uncertainty`` CLI flag for diagnostic workflows
  (asymmetric override: cannot disable uncertainty when an acquisition like
  UCB/EI/PI semantically requires it).
- Per-cycle parquet drops the ``uncertainty`` column entirely when skipped
  (column-presence semantics, not fabricated nulls); ``has_uncertainty``
  metric still reports False with ``uncertainty_*`` keys present at None.
- 8 skip-eligible learners: CPU ``gp``, ``rf``, ``dt``, ``lr`` and
  GPU/gpytorch siblings ``rf_fil``, ``ridge_cuml``, ``gpu_gp``, ``svgp``.
  RF unifies its mean source across both branches for bit-identity under
  the force_uncertainty toggle (D9). Ensembles cascade the flag to members
  AND short-circuit their outer std-reduction (D10 — saves ~4 GB at
  100M × N-member scale).
- ``mc_dropout`` stays uniform-contract: continues to run N stochastic
  forward passes when ``compute_uncertainty=False`` (skipping them would
  change mean semantics per Gal & Ghahramani 2016), only the std return is
  suppressed.

### Breaking changes

`Learner.predict()` adds a **required keyword-only** parameter
``compute_uncertainty: bool = True``. Third-party ``Learner`` subclasses
with the old signature will raise on first cycle predict:

    TypeError: predict() got an unexpected keyword argument 'compute_uncertainty'

If you do NOT subclass ``Learner`` (you use built-in learners via
``learner='rf'`` etc.), **this change does not affect you**. No code or
config update needed.

If you DO subclass ``Learner``, apply this 5-line diff:

```python
 def predict(
     self,
     features: np.ndarray,
-    smiles: list[str] | None = None,
+    smiles: list[str] | None = None,
+    *,
+    compute_uncertainty: bool = True,
 ) -> tuple[np.ndarray, np.ndarray | None]:
     ...
+    # Optional: honour compute_uncertainty=False to skip uncertainty work.
+    if not compute_uncertainty:
+        return predictions, None
     return predictions, uncertainties
```

To locate subclasses in your codebase:

    git grep -nE '\bclass\s+\w+\s*\(.*Learner.*\)' -- '*.py'

A test-time ``inspect.signature`` audit of ``LEARNER_REGISTRY`` plus a
parametrized behavioural check (every registered learner returns ``None``
when called with ``compute_uncertainty=False``) is now part of the test
suite (``tests/core/test_learner_registry_signatures.py``,
``tests/learners/test_compute_uncertainty_behavior.py``,
``tests/core/test_third_party_subclass_breakage.py``).

### Notes

- No runtime shim and no DeprecationWarning — old signatures literally
  reject the new kwarg with a natural ``TypeError``. The CHANGELOG entry
  IS the migration guide; ``Learner.predict.__doc__`` links here.
- External validation: the same optimisation was merged in
  [scikit-learn #31374](https://github.com/scikit-learn/scikit-learn/issues/31374)
  (~96% wall-clock saving on the GP variance solve when both
  ``return_std=False, return_cov=False``).
- Forward-compatibility note for downstream parquet readers: every existing
  consumer column-gates on ``'uncertainty' in df.columns`` and reads single
  files (no glob/concat). A future glob-based multi-cycle reader would need
  ``missing_columns="insert"`` (polars ≥ 1.30) or per-file iteration.

## [0.10.0] — 2026-05-12

### Added

#### MCDropout Chunked Predict + OOM Retry
- `predict_batch_size` parameter on `MCDropoutLearner` (default auto-computed via `estimate_batch_size()`)
- Chunked prediction splits input by `predict_batch_size`, processes per-chunk with GPU memory cleanup
- OOM retry with halving: catches `torch.cuda.OutOfMemoryError` / `RuntimeError('out of memory')`, halves chunk size, retries up to 3 times before raising `LearnerError`
- Effective minimum chunk size of 256 (below which chunk is too small — requires more GPU memory or reduced T)

#### Atomic Writes Everywhere
- `_atomic_write()` helper in `learnm8/core/persistence.py`: writes to `<path>.tmp`, fsyncs file + parent dir, then `os.replace()` for crash-safe atomic commits
- EXDEV fallback: if `os.replace` fails cross-device, falls back to `shutil.copyfile` + `os.unlink` with `LearnM8Warning`
- All persistence outputs routed through atomic writes: compounds_final, cycle_metrics, selection_history, validation_report, config.json, prediction_cycle_N parquet files
- Orphan `.tmp` files detected at startup → silently overwritten with INFO log

#### Parquet Output Threshold with Auto-Detection
- `_resolve_output_format(fmt, n_rows, file_label)` → auto-selects parquet (>1M rows) or CSV (≤1M rows)
- `sink_parquet` streaming for >1M rows (constant RAM), `write_parquet` for smaller
- `_apply_parquet_schema()` enforces stable dtypes (ID→Utf8, prediction/uncertainty→Float32, etc.)
- Parquet `key_value_metadata` for self-documenting output; sidecar `.metadata.json` for sink_parquet path
- `cycle_metrics` always CSV regardless of threshold
- Result loader in `animation.py` auto-detects `.parquet` vs `.csv`

#### Descriptor-Class Hard Guard
- `check_large_feature_guard()` in `learnm8/features/guard.py`: detects descriptor-class featurizers (feature_type='continuous', per-molecule bytes > 1024) on pools > 10M
- Raises `ConfigurationError` before any RDKit calls with predicted cache size and remediation message
- Acknowledged via `large_features_ack=True` (API) or `--allow-large-features` (CLI) → proceeds with `LearnM8Warning`
- Dynamic detection via `feature_type`, `dtype.itemsize`, single-molecule probe fallback

#### Selection History Optimization
- Pre-built ID→(SMILES, measured) lookup dict from single `compounds_df` scan
- Replaces N × cycle row scans with one scan + O(total_selected) iterations

#### New API Parameters
- `large_features_ack: bool = False` — opt-in for descriptor-class featurizers on large pools
- `output_format: Literal['auto', 'csv', 'parquet'] = 'auto'` — control output file format

#### New CLI Flags
- `--allow-large-features` (run subcommand, Advanced group)
- `--output-format` (run subcommand, Output group, choices: auto/csv/parquet)

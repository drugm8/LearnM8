# Changelog

## [1.0.0] — 2026-10-19

### Added — `learnm8 suggest` / `suggest_next_batch()` for slow lab assays

One call reads the `batch_cycle_N.csv` files in a folder, trains on the measured
values, writes the next batch file and exits. Nothing runs between rounds; the
folder is the only state. Cycle 0 is a random batch; later cycles reuse
`execute_cycle`. Empty cells are pending, finite numbers are measured, anything
else is failed (never reselected). Batch files start with a `sep=,` line so
Excel splits columns correctly; reading detects separator, decimal mark and
encoding automatically, recovers IDs whose leading zeros a spreadsheet dropped
when the match is unique, and rejects ambiguous numbers with an error naming the
file, ID and cell. New files only: existing batch files are never edited.
Re-running before any result is entered raises `ConfigurationError` (CLI exit
code 2). No pruning, multi-stage schedules or censored values. See
`docs/tutorials/lab-campaigns.md`.

### Removed — the `simulated_annealing` acquisition strategy (BREAKING)

`SimulatedAnnealingAcquisition` and its registry entry `'simulated_annealing'`
are gone. `learnm8.acquisition.simulated_annealing` no longer exists, and
`get_acquisition_function('simulated_annealing')` raises `KeyError` listing the
eight remaining strategies. Migration: use `ucb` (balanced exploration via
uncertainty), `topk` (a random draw from the top-scoring slice), or `random`,
depending on which property of the annealing schedule you relied on.

Two consequences inside the library:

- **Every built-in acquisition strategy is now streamable.** The non-streamable
  branch of `is_streaming_eligible` is retained — a third-party
  `AcquisitionFunction` subclass that implements only `select()` still falls
  back to the legacy path — but no shipped strategy reaches it. Streaming
  eligibility now turns on `output_dir` and the pruning strategy in practice.
- The strategy colour map in `visualization/animation.py` lost its
  `simulated_annealing` entry; unknown strategies already fall back to `tab10`.

Removed alongside it: the `docs/components/acquisition/diversity.md` page (it
documented only this strategy) and the simulated-annealing runs from the
publication benchmark table and figures.

### Fixed — silently dropped user parameters (`acquisition_params` and the config-file path)

Every `--acquisition-params` value passed alongside `--cycles` (or a config
file's `cycles:` list) was discarded. `parse_cycle_schedule` declared the
parameter and then used only the per-block value in its `cycles is not None`
branch — the branch the CLI always takes — so runs completed normally, reported
success, and used the acquisition defaults. A UCB `beta` sweep produced
byte-identical trajectories at every cycle. Fixed by mirroring the top-level
pruning fallback: a block without its own
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

### Fixed — `enable_chemprop_fine_tuning` ignored for `learner='chemprop'`

`run_active_learning(learner='chemprop', enable_chemprop_fine_tuning=True)`
built a plain `ChempropLearner` and silently trained from scratch every cycle;
only `'chemprop_ensemble'` honoured the flag. The `'chemprop'` shortcut now
passes `enable_fine_tuning=True` and the run's checkpoint directory, keeping
the run's `random_state`.

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

### Removed — the bare `ensemble` learner name (BREAKING)

`'ensemble'` is no longer a key of `LEARNER_REGISTRY`, so `learner='ensemble'`
and `--learner ensemble` now raise `ValueError: Unknown learner`. It mapped to
the generic `EnsembleLearner`, which needs an explicit `learners=[...]` list,
so it could never be built from a name. There is no alias.

Migration: use `dt_ensemble` or `xgb_ensemble` (uncertainty from member
spread) or `mixed_ensemble` (RF + LR + XGBoost), or pass an
`EnsembleLearner([...])` instance for custom members.

### Removed — the `rf_ensemble` learner (BREAKING)

`RFEnsemble` and its registry entry `'rf_ensemble'` are gone (deprecated since
0.10.0). `learner='rf_ensemble'` and `--learner rf_ensemble` now raise
`ValueError: Unknown learner`; `from learnm8 import RFEnsemble` fails. There is
no alias.

Migration: use `rf`, which reports uncertainty from the spread across its
trees. For a member-spread ensemble use `dt_ensemble`, `xgb_ensemble` or
`mixed_ensemble`.

### Added — `failed` status for compounds the oracle cannot measure

A Python (run-mode) oracle can now fail on individual compounds. If it omits a
selected compound, or returns null, NaN, infinity or a non-numeric value for it,
that compound gets the new status `failed`. It keeps its row, its
`selected_cycle` and `labeled_cycle` are set to the selecting cycle, and its
target stays null. It is excluded from training and never selected again, and
the run no longer aborts with `OracleError: Oracle did not return measurements
...`. Cycle metrics gain `n_failed` (0 when none), the aggregates gain
`total_failed`, and a warning is logged for every cycle with failures. If the
whole initial batch fails, `run_active_learning` raises `OracleError` before
any model is trained. `VALID_STATUSES` now has a fourth value, `failed`, which
also appears in `compounds_final`; the master DataFrame still has 7 columns.
See `docs/components/oracles/custom-oracles.md`.

### Changed — `CSVOracle` raises for IDs missing from its ground truth

`CSVOracle.measure` used to log a warning and return a partial result when a
requested ID was absent from the oracle CSV. It now raises `OracleError`
(listing up to five IDs), so benchmark mode stays strict: every compound in the
pool must have a ground-truth value. Null target values still raise
`ValueError`.

### Changed (BREAKING) — feature cache file names and safety

- Cache files are now named `features_{featurizer}_{dim}_{storage_dtype}.h5`.
  The same featurizer at two sizes (for example Morgan 2048 and 1024 bits) no
  longer shares, and overwrites, one file. Caches under the old name are not
  found and rebuild once. To keep a large cache, rename it by hand to the new
  name.
- The cache never deletes, renames or overwrites an existing file. A corrupt
  file, a wrong schema version or a wrong feature dimension raises
  `PersistenceError` and leaves the file untouched; move or delete it, or use
  another `cache_dir`, to rebuild. Before, such a file was silently deleted and
  rebuilt.
- The default cache lock timeout is 3600 s instead of 5 s, so concurrent runs
  no longer fail while a first writer builds a large cache. Override it with
  `LEARNM8_LOCK_TIMEOUT_S`.

### Changed — `Learner.train` overrides accept `smiles`

Every built-in `train` override now accepts the optional `smiles` argument of
`Learner.train` (and ignores it unless the learner uses SMILES), so all learners
share one signature.

### Changed — packaging and dependencies

- Version 1.0.0, read from the package metadata (`learnm8.__version__`).
- The runtime dependencies PyTorch, Lightning, XGBoost, scikit-fingerprints,
  Chemprop, fastprop and mordredcommunity are now declared. A plain
  `pip install learnm8` previously failed at import because they were missing.
- The wheel contains only the `learnm8` package. `tests`, `examples` and
  `publication` were being installed as top-level packages. The example oracles
  are used from a repository checkout (`from examples.oracles import ...`).
- PyPI metadata added: readme, MIT license expression, authors, project URLs,
  classifiers. Inside the conda environment install the package with
  `pip install --no-deps -e .`.
- `mypy learnm8/` passes with no ignored modules, and the code is formatted
  with `ruff format`.

### Fixed

- **Ensemble learners now forward `feature_type` to their members.** Members
  stayed on the binary NaN path for continuous featurizers (descriptors, count
  fingerprints) instead of using median imputation.
- **`count=True` for `pubchem`, `klekota_roth`, `laggner`, `avalon` and
  `rdkit`.** Counts were truncated to bits by `np.packbits`. They are now stored
  as CSR uint16 and treated as continuous. Existing `count=True` caches for these
  featurizers are not reused.
- **Tanimoto similarity of two all-zero fingerprints is defined as 1.0.** It was
  implicit RDKit behaviour, and RDKit 2026.03 changed it to 0.0, so the
  diversity metric would have differed between RDKit versions for compounds with
  empty fingerprints.
- **`PythonOracle` given a path that is not a loadable `.py` module** raises
  `OracleError` instead of an `AttributeError`.
- **`FastpropLearner.predict`** raises `LearnerError` when the trainer returns no
  predictions, instead of failing inside `torch.cat`.
- **`evaluate_cycle`** raises `ConfigurationError` when benchmark ranking
  metrics are requested without ground truth, instead of an `AttributeError`.
- **Documentation** corrected against the code: 8 acquisition strategies, the
  real diversity metric keys (`bit_position_uniformity_entropy_*`), the
  `skip_validation` and `streaming` parameters, the missing CLI flags, and the
  cache design.
- **MC Dropout predictions no longer depend on chunking or free memory.**
  Dropout was reseeded at the start of every prediction chunk, so a compound's
  dropout masks depended on where it fell in a chunk, and chunk sizes depend on
  free memory (the outer `predict_with_batching` split, and, once a
  `torch.device`-for-`str` bug that silently disabled it was fixed, the
  learner's own memory-aware chunking). The same seed could therefore give
  different results on different machines. Masks are now a pure function of
  `(random_state, feature-row content, layer, pass, unit)`, generated with a
  counter-based integer hash on the device, with the same Bernoulli(1 − p)
  keep probability and 1 / (1 − p) scaling as `nn.Dropout`. Predictions are
  identical, up to float32 rounding, whatever the chunk size, row order or
  pool, on CPU or GPU, and prediction no longer touches the global torch RNG.
  Outputs differ from 0.11.0 by Monte Carlo noise (synthetic 20K × 2048
  benchmark, same trained weights: Spearman 0.986 for the means, 84% overlap
  of the top 1%). GPU prediction is about 2× slower for small MLPs (mask
  hashing outweighs the matrix multiplies); CPU prediction is slightly faster.
- **`learnm8.log` is written by the CLI and by every run in one process.** The
  CLI configured the console first, and logging setup returned early once any
  handler existed, so `learnm8 run` / `learnm8 suggest` wrote no log file, and a
  second run in one Python session logged into the first run's directory.
- **`fingerprint_used` names Morgan(2, 2048) for every run.** For binary
  featurizers it named the run featurizer (e.g. MACCS), although the diversity
  metrics are always computed on Morgan(2, 2048). Metric values are unchanged.
- **Failed compounds count in the cumulative diversity metrics** (they were
  selected; they used to be left out).
- **Cache integrity violations raise `PersistenceError`**, like every other
  corrupt or incompatible cache file (was `FeatureExtractionError`).
- **`learnm8 suggest` prints the batch file it wrote**, not the
  highest-numbered file in the folder (they differed when a newer header-only
  file existed). Batch rows with an empty SMILES cell now log a warning that
  they skipped the pool cross-check. Only Polars parse errors are reported as
  "cannot parse CSV". `PythonOracle` no longer logs a duplicate
  missing-results warning.

## [0.11.0] — 2026-05-13

### Added — Acquisition-Aware Uncertainty Computation

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
  the force_uncertainty toggle. Ensembles cascade the flag to members
  AND short-circuit their outer std-reduction (saves ~4 GB at
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

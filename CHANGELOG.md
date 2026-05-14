# Changelog

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
- 9 skip-eligible learners: CPU ``gp``, ``rf``, ``advanced_rf``, ``dt``,
  ``lr`` and GPU/gpytorch siblings ``rf_fil``, ``ridge_cuml``, ``gpu_gp``,
  ``svgp``. RF/AdvancedRF unify their mean source across both branches for
  bit-identity under the force_uncertainty toggle (D9). Ensembles cascade
  the flag to members AND short-circuit their outer std-reduction (D10 —
  saves ~4 GB at 100M × N-member scale).
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

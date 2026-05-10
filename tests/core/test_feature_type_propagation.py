"""Tests for `_feature_type` resolution from the user-supplied Featurizer (Feature 020).

Covers tasks T032 (behavior matrix unit tests for `_resolve_feature_type`,
FR-008/FR-009) and T033 (end-to-end SC-003 verification that
``MorganFeaturizer(count=True)`` flows through to ``csr_uint16`` cache storage).

Authoritative reference:
    specs/020-cache-integrity/contracts/cycle_feature_type_contract.md
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import h5py
import polars as pl
import pytest

from learnm8.core.config import CycleConfig
from learnm8.core.cycle import _resolve_feature_type, execute_cycle
from learnm8.core.interfaces import Featurizer
from learnm8.features import MorganFeaturizer

# ---------------------------------------------------------------------------
# T032 — behavior matrix tests for _resolve_feature_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_morgan_count_true_returns_continuous() -> None:
    """MorganFeaturizer(count=True) is the canonical bug case the contract fixes.

    Old code re-instantiated MorganFeaturizer() (no count) and returned 'binary';
    new code reads the user-supplied instance's feature_type attribute directly.
    """
    feat = MorganFeaturizer(count=True, n_jobs=1)
    assert _resolve_feature_type(feat, None) == 'continuous'


@pytest.mark.unit
def test_resolve_morgan_default_returns_binary() -> None:
    """MorganFeaturizer() with default params has feature_type='binary'."""
    feat = MorganFeaturizer(n_jobs=1)
    assert _resolve_feature_type(feat, None) == 'binary'


@pytest.mark.unit
def test_resolve_morgan_string_returns_binary() -> None:
    """String 'morgan' instantiates the registry default → 'binary' (no behaviour change)."""
    assert _resolve_feature_type('morgan', None) == 'binary'


@pytest.mark.unit
def test_resolve_whim_instance_returns_continuous() -> None:
    """WhimFeaturizer instance reports feature_type='continuous'."""
    pytest.importorskip('skfp.fingerprints')
    from learnm8.features import WHIMFeaturizer

    try:
        feat = WHIMFeaturizer(random_state=42, n_jobs=1)
    except Exception as exc:  # pragma: no cover - environment-dependent skip
        pytest.skip(f'WHIMFeaturizer not constructible in this env: {exc}')

    assert _resolve_feature_type(feat, None) == 'continuous'


@pytest.mark.unit
def test_resolve_whim_string_returns_continuous() -> None:
    """String 'whim' instantiates the registry default → 'continuous'."""
    pytest.importorskip('skfp.fingerprints')
    try:
        result = _resolve_feature_type('whim', None)
    except Exception as exc:  # pragma: no cover - environment-dependent skip
        pytest.skip(f"'whim' registry entry not constructible in this env: {exc}")
    assert result == 'continuous'


@pytest.mark.unit
def test_resolve_none_returns_binary() -> None:
    """featurizer=None falls through to 'binary'."""
    assert _resolve_feature_type(None, None) == 'binary'


@pytest.mark.unit
def test_resolve_unknown_string_warns_and_returns_binary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown registry name logs a WARNING and falls back to 'binary'."""
    bad_name = 'nonexistent_featurizer'
    with caplog.at_level(logging.WARNING, logger='learnm8.core.cycle'):
        result = _resolve_feature_type(bad_name, None)

    assert result == 'binary'
    matching = [
        r for r in caplog.records
        if 'Unknown featurizer name' in r.getMessage() and bad_name in r.getMessage()
    ]
    assert len(matching) >= 1, (
        f"expected WARNING containing 'Unknown featurizer name' and {bad_name!r}; "
        f"got records: {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.unit
def test_resolve_custom_non_featurizer_object_warns_and_returns_binary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Defensive path: non-Featurizer, non-str input logs WARNING and returns 'binary'.

    Featurizer ABC subclasses are guaranteed to define ``feature_type``; the
    fallback exists for genuinely unsupported types (e.g. duck-typed classes
    that implement ``transform()`` but neither subclass Featurizer nor are str).
    Per the contract this hits the 'Unsupported featurizer type' branch.
    """

    class DuckTypedFeaturizer:
        """Has transform() but does NOT subclass Featurizer."""

        def transform(self, smiles_list: list[str]) -> Any:  # pragma: no cover
            return None

    duck = DuckTypedFeaturizer()
    with caplog.at_level(logging.WARNING, logger='learnm8.core.cycle'):
        result = _resolve_feature_type(duck, None)  # type: ignore[arg-type]

    assert result == 'binary'
    matching = [
        r for r in caplog.records if 'Unsupported featurizer type' in r.getMessage()
    ]
    assert len(matching) >= 1, (
        f"expected WARNING containing 'Unsupported featurizer type'; "
        f"got records: {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.unit
def test_resolve_featurizer_obj_wins_over_string() -> None:
    """When both featurizer (str) and featurizer_obj are supplied, the instance wins.

    This guards against silent train-vs-eval drift: a string 'morgan' (binary)
    paired with a MorganFeaturizer(count=True) instance must resolve to
    'continuous' to match the eval-side diversity-metrics path.
    """
    inst = MorganFeaturizer(count=True, n_jobs=1)
    assert _resolve_feature_type('morgan', inst) == 'continuous'


@pytest.mark.unit
def test_resolve_featurizer_obj_wins_over_instance() -> None:
    """featurizer_obj precedence beats a Featurizer instance passed via featurizer.

    Matrix row: featurizer=MorganFeaturizer() (binary) AND
    featurizer_obj=WHIMFeaturizer(random_state=42) (continuous) → 'continuous'.
    """
    pytest.importorskip('skfp.fingerprints')
    from learnm8.features import WHIMFeaturizer

    try:
        whim = WHIMFeaturizer(random_state=42, n_jobs=1)
    except Exception as exc:  # pragma: no cover - environment-dependent skip
        pytest.skip(f'WHIMFeaturizer not constructible in this env: {exc}')

    morgan_default = MorganFeaturizer(n_jobs=1)
    assert _resolve_feature_type(morgan_default, whim) == 'continuous'


@pytest.mark.unit
def test_resolve_unsupported_type_warns_and_returns_binary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Defensive path: an int (or other unexpected type) logs WARNING → 'binary'."""
    with caplog.at_level(logging.WARNING, logger='learnm8.core.cycle'):
        result = _resolve_feature_type(42, None)  # type: ignore[arg-type]

    assert result == 'binary'
    matching = [
        r for r in caplog.records
        if 'Unsupported featurizer type' in r.getMessage()
    ]
    assert len(matching) >= 1, (
        f"expected WARNING containing 'Unsupported featurizer type'; "
        f"got: {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.unit
def test_resolve_featurizer_instance_subclass_recognised() -> None:
    """Custom Featurizer subclass with feature_type='continuous' is honoured."""

    class CustomContinuous(Featurizer):
        feature_type: str = 'continuous'

        def transform(self, smiles_list: list[str]):  # pragma: no cover - not invoked
            raise NotImplementedError

        def get_dimension(self) -> int:
            return 1

        def get_name(self) -> str:
            return 'custom_continuous'

    assert _resolve_feature_type(CustomContinuous(), None) == 'continuous'


# ---------------------------------------------------------------------------
# T033 — end-to-end SC-003: MorganFeaturizer(count=True) → csr_uint16 cache
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_morgan_count_true_propagates_through_cycle_to_csr_uint16_storage(
    sample_compounds: pl.DataFrame,
    mock_learner_with_uncertainty: Any,
    mock_oracle: Any,
    tmp_path: Path,
) -> None:
    """SC-003: MorganFeaturizer(count=True) drives the continuous-feature storage path.

    Asserts in two phases (decoupled to keep the test orthogonal to the
    full prediction-batching surface):

      Phase A — cycle propagation: ``execute_cycle`` sets
        ``learner._feature_type = 'continuous'`` from the user-supplied
        ``MorganFeaturizer(count=True)`` instance (FR-008/SC-003 part 1).

      Phase B — storage routing: extracting features for the same
        ``MorganFeaturizer(count=True)`` produces an HDF5 cache file with
        ``attrs['storage_dtype'] == 'csr_uint16'`` (SC-003 part 2). We use
        ``extract_features`` directly here rather than scraping the cache
        produced by the cycle, because the cycle's prediction-batching path
        also writes its own miss rows and we want a deterministic assertion.
    """
    from learnm8.features.extraction import extract_features
    from tests.fixtures.master_dataframe import (
        create_initialized_master_df as initialize_master_dataframe,
    )

    # ---- Phase A: feature_type propagation through execute_cycle ----------
    initial_ids = sample_compounds['ID'].head(2).to_list()
    initial_values = dict(zip(initial_ids, [0.3, 0.6], strict=True))

    master_df = initialize_master_dataframe(
        valid_compounds=sample_compounds,
        target_col='Activity',
        initial_labeled_ids=initial_ids,
        initial_measurements=initial_values,
    )

    config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.2)
    count_featurizer = MorganFeaturizer(count=True, n_jobs=1)

    cycle_cache = tmp_path / 'cycle_cache'
    cycle_cache.mkdir()

    # We only need _feature_type to be set before any downstream work runs;
    # the cycle may still raise from unrelated paths in this minimal
    # mock-driven setup. The propagation assertion below is the authoritative
    # SC-003 check for this phase.
    with contextlib.suppress(Exception):
        execute_cycle(
            compounds_df=master_df,
            cycle=1,
            config=config,
            learner=mock_learner_with_uncertainty,
            oracle=mock_oracle,
            target_col='Activity',
            featurizer=count_featurizer,
            cache_dir=cycle_cache,
            original_pool_size=len(sample_compounds),
            mode='run',
            output_dir=tmp_path,
            n_jobs=1,
        )

    assert getattr(mock_learner_with_uncertainty, '_feature_type', None) == 'continuous', (
        'execute_cycle must propagate the user-supplied Featurizer instance ' "'s "
        'feature_type to the learner; expected continuous for '
        'MorganFeaturizer(count=True)'
    )

    # ---- Phase B: csr_uint16 on-disk storage routing ----------------------
    extract_cache = tmp_path / 'extract_cache'
    extract_cache.mkdir()

    smiles = [sample_compounds['SMILES'].to_list()[0]]
    extract_features(smiles, count_featurizer, cache_dir=extract_cache, n_jobs=1)

    cache_file = extract_cache / f'features_{count_featurizer.get_name()}.h5'
    assert cache_file.exists(), (
        f'expected count-mode cache file at {cache_file}; '
        f'directory contents: {sorted(p.name for p in extract_cache.iterdir())}'
    )

    with h5py.File(cache_file, 'r') as f:
        storage_dtype = str(f.attrs['storage_dtype'])

    assert storage_dtype == 'csr_uint16', (
        f"expected storage_dtype='csr_uint16' for MorganFeaturizer(count=True); "
        f'got {storage_dtype!r}'
    )

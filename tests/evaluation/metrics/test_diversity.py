"""Tests for the redesigned diversity-metrics module (feature 013).

Covers the public ``compute_diversity_metrics`` API plus the private
primitives. Replaces ``test_similarity.py``.
"""

from __future__ import annotations

import math
import time

import numpy as np
import polars as pl
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from learnm8.evaluation.metrics.similarity import (
    DIVERSITY_KEYS,
    FALLBACK_FP_SIZE,
    N_PAIRS_MAX,
    N_PAIRS_MIN,
    N_PILOT_PAIRS,
    TARGET_EPSILON,
    Z_95,
    RunCache,
    _derive_rng,
    _get_or_build_fps,
    _mean_tanimoto_adaptive_sampled,
    _scaffold_diversity_index,
    _bit_marginal_entropy_from_bit_sum,
    compute_diversity_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SMILES_DRUGLIKE = [
    "CCO",                                  # ethanol (acyclic)
    "c1ccccc1",                             # benzene
    "CC(=O)Oc1ccccc1C(=O)O",                # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",         # caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",        # ibuprofen
    "C1=CC=C(C=C1)C(=O)O",                  # benzoic acid
    "CCN(CC)CC",                            # triethylamine (acyclic)
    "C1CCCCC1",                             # cyclohexane
    "Nc1ncnc2[nH]cnc12",                    # adenine
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",           # theobromine
]


def _morgan_fps(smiles: list[str], n_bits: int = FALLBACK_FP_SIZE):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def _df_with_smiles(smiles: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "ID": [f"mol_{i}" for i in range(len(smiles))],
        "SMILES": smiles,
    })


@pytest.fixture
def run_cache() -> RunCache:
    return RunCache()


# ---------------------------------------------------------------------------
# Module-level constants (sanity)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModuleConstants:
    def test_diversity_keys_shape(self):
        assert isinstance(DIVERSITY_KEYS, tuple)
        assert len(DIVERSITY_KEYS) == 6
        assert all(k.endswith("_batch") or k.endswith("_cumulative") for k in DIVERSITY_KEYS)

    def test_constants_present(self):
        assert N_PILOT_PAIRS == 500
        assert TARGET_EPSILON == 0.005
        assert N_PAIRS_MIN == 1_000
        assert N_PAIRS_MAX == 100_000
        assert pytest.approx(1.96) == Z_95


# ---------------------------------------------------------------------------
# Primitives — known-answer + edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMeanTanimotoAdaptive:
    def test_returns_none_below_two_compounds(self, run_cache):
        rng = _derive_rng(42, 0)
        fps = _morgan_fps(["CCO"])
        assert _mean_tanimoto_adaptive_sampled(fps, rng) is None
        assert _mean_tanimoto_adaptive_sampled([], rng) is None

    def test_identical_compounds_yield_one(self):
        rng = _derive_rng(42, 0)
        fps = _morgan_fps(["CCO"] * 5)
        result = _mean_tanimoto_adaptive_sampled(fps, rng)
        assert result == pytest.approx(1.0)

    def test_returns_value_in_unit_interval(self):
        rng = _derive_rng(42, 0)
        fps = _morgan_fps(SMILES_DRUGLIKE)
        result = _mean_tanimoto_adaptive_sampled(fps, rng)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_seeded_deterministic(self):
        # Build a larger set to actually engage adaptive sampling.
        smis = SMILES_DRUGLIKE * 80
        fps = _morgan_fps(smis)
        a = _mean_tanimoto_adaptive_sampled(fps, _derive_rng(123, 7))
        b = _mean_tanimoto_adaptive_sampled(fps, _derive_rng(123, 7))
        assert a == b

    def test_both_empty_fp_tanimoto_one(self):
        # RDKit edge case: BulkTanimotoSimilarity on two all-zero fps is 1.0.
        rng = _derive_rng(42, 0)
        # Helium has no atoms in RDKit's standard parsing so we fabricate
        # an all-zero ExplicitBitVect by computing an atom-less mol's fp;
        # if RDKit produces non-empty bits, fall back to a minimal acyclic
        # fragment whose fp can still be all-zero in some RDKit builds.
        from rdkit.DataStructs.cDataStructs import ExplicitBitVect
        empty_a = ExplicitBitVect(FALLBACK_FP_SIZE)
        empty_b = ExplicitBitVect(FALLBACK_FP_SIZE)
        result = _mean_tanimoto_adaptive_sampled([empty_a, empty_b], rng)
        assert result == pytest.approx(1.0)


@pytest.mark.unit
class TestScaffoldDiversityIndex:
    def test_all_unique_scaffolds(self):
        # Each compound has a different ring system (or all are acyclic).
        smis = ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCNCC1"]
        cache: dict[str, str] = {}
        result = _scaffold_diversity_index(smis, cache)
        assert result == pytest.approx(1.0)

    def test_acyclic_compounds_collapse_to_one_scaffold(self):
        smis = ["CCO", "CCC", "CCN", "CCCC"]  # all acyclic → "" scaffold
        cache: dict[str, str] = {}
        result = _scaffold_diversity_index(smis, cache)
        assert result == pytest.approx(1.0 / 4.0)

    def test_returns_none_on_empty_input(self):
        assert _scaffold_diversity_index([], {}) is None

    def test_invalid_smiles_excluded(self):
        cache: dict[str, str] = {}
        smis = ["c1ccccc1", "NOT_A_SMILES", "C1CCCCC1"]
        result = _scaffold_diversity_index(smis, cache)
        # 2 valid compounds, 2 unique scaffolds
        assert result == pytest.approx(1.0)

    def test_cache_population(self):
        cache: dict[str, str] = {}
        _scaffold_diversity_index(["c1ccccc1", "c1ccccc1"], cache)
        assert "c1ccccc1" in cache


@pytest.mark.unit
class TestShannonEntropy:
    def test_zero_compounds_returns_none(self):
        assert _bit_marginal_entropy_from_bit_sum(np.zeros(8), 0) is None

    def test_all_zero_bit_sum_returns_none(self):
        assert _bit_marginal_entropy_from_bit_sum(np.zeros(8), 5) is None

    def test_uniform_distribution_hits_log2_n_bits(self):
        # All bits active in all compounds → frequencies all equal → entropy = log2(n_bits).
        n_bits = 16
        bit_sum = np.full(n_bits, 5)  # 5 compounds, all bits on
        result = _bit_marginal_entropy_from_bit_sum(bit_sum, 5)
        assert result == pytest.approx(math.log2(n_bits))

    def test_upper_bound_property(self):
        # Random binary fingerprints should have entropy ≤ log2(n_bits).
        rng = np.random.default_rng(seed=0)
        n_bits = 64
        n_compounds = 50
        matrix = rng.integers(0, 2, size=(n_compounds, n_bits), dtype=np.uint8)
        bit_sum = matrix.sum(axis=0)
        result = _bit_marginal_entropy_from_bit_sum(bit_sum, n_compounds)
        assert result is not None
        assert result <= math.log2(n_bits) + 1e-9


# ---------------------------------------------------------------------------
# RunCache
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunCacheLifecycle:
    def test_clear_evicts_all_state(self):
        cache = RunCache()
        cache.fp_cache["CCO"] = object()  # type: ignore[assignment]
        cache.scaffold_cache["CCO"] = ""
        cache.cumulative_fp_buffer.append(object())  # type: ignore[arg-type]
        cache.cumulative_smiles_seen.add("CCO")
        cache.bit_sum_buffer = np.zeros(8, dtype=np.int64)

        cache.clear()
        assert not cache.fp_cache
        assert not cache.scaffold_cache
        assert cache.cumulative_fp_buffer == []
        assert not cache.cumulative_smiles_seen
        assert cache.bit_sum_buffer is None

    def test_instances_are_isolated(self):
        a = RunCache()
        b = RunCache()
        a.fp_cache["x"] = object()  # type: ignore[assignment]
        assert "x" not in b.fp_cache
        a.clear()
        b.fp_cache["y"] = object()  # type: ignore[assignment]
        assert "y" not in a.fp_cache


# ---------------------------------------------------------------------------
# Fingerprint sourcing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFingerprintSourcing:
    def test_morgan_fallback_label_when_featurizer_none(self):
        cache: dict[str, object] = {}
        fps, packed, label, valid = _get_or_build_fps(SMILES_DRUGLIKE, None, None, cache)  # type: ignore[arg-type]
        assert label == "morgan_2_2048_float32"
        assert len(fps) == len(SMILES_DRUGLIKE)
        assert packed.shape == (len(SMILES_DRUGLIKE), FALLBACK_FP_SIZE)
        assert valid == list(range(len(SMILES_DRUGLIKE)))

    def test_invalid_smiles_skipped(self):
        cache: dict[str, object] = {}
        smis = ["c1ccccc1", "NOT_A_SMILES", "CCO"]
        fps, packed, _label, valid = _get_or_build_fps(smis, None, None, cache)  # type: ignore[arg-type]
        assert len(fps) == 2
        assert packed.shape == (2, FALLBACK_FP_SIZE)
        assert valid == [0, 2]

    def test_cache_populates(self):
        cache: dict[str, object] = {}
        _get_or_build_fps(["c1ccccc1"], None, None, cache)  # type: ignore[arg-type]
        assert "c1ccccc1" in cache


# ---------------------------------------------------------------------------
# Public API: cycle-0 invariant + disable handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeDiversityMetricsPublic:
    def test_returns_six_keys_plus_provenance(self, run_cache):
        df = _df_with_smiles(SMILES_DRUGLIKE[:4])
        result = compute_diversity_metrics(
            newly_selected_df=df,
            cumulative_selected_df=df,
            cycle=0,
            run_cache=run_cache,
            random_state=42,
        )
        assert set(result.keys()) == set(DIVERSITY_KEYS) | {"fingerprint_used"}
        for k in DIVERSITY_KEYS:
            assert result[k] is None or isinstance(result[k], float)
        assert result["fingerprint_used"] == "morgan_2_2048_float32"

    def test_cycle_zero_batch_equals_cumulative(self, run_cache):
        # FR-004a: at cycle 0 the cumulative set IS the batch, so the two
        # halves of every metric pair must coincide exactly.
        df = _df_with_smiles(SMILES_DRUGLIKE[:6])
        result = compute_diversity_metrics(
            newly_selected_df=df,
            cumulative_selected_df=df,
            cycle=0,
            run_cache=run_cache,
            random_state=42,
        )
        assert result["scaffold_diversity_index_batch"] == result["scaffold_diversity_index_cumulative"]
        assert result["bit_marginal_entropy_batch"] == pytest.approx(
            result["bit_marginal_entropy_cumulative"]
        )

    def test_disable_true_emits_none_for_all(self, run_cache):
        df = _df_with_smiles(SMILES_DRUGLIKE[:4])
        result = compute_diversity_metrics(
            newly_selected_df=df,
            cumulative_selected_df=df,
            cycle=0,
            run_cache=run_cache,
            disable=True,
        )
        for k in DIVERSITY_KEYS:
            assert result[k] is None
        assert result["fingerprint_used"] is None

    def test_disable_iterable_emits_none_for_named_only(self, run_cache):
        df = _df_with_smiles(SMILES_DRUGLIKE[:4])
        result = compute_diversity_metrics(
            newly_selected_df=df,
            cumulative_selected_df=df,
            cycle=0,
            run_cache=run_cache,
            random_state=42,
            disable=["scaffold_diversity_index_batch"],
        )
        assert result["scaffold_diversity_index_batch"] is None
        # Other keys should be computed.
        assert result["scaffold_diversity_index_cumulative"] is not None
        assert result["bit_marginal_entropy_batch"] is not None
        assert result["fingerprint_used"] == "morgan_2_2048_float32"

    def test_disable_unknown_key_raises(self, run_cache):
        df = _df_with_smiles(SMILES_DRUGLIKE[:2])
        with pytest.raises(ValueError):
            compute_diversity_metrics(
                newly_selected_df=df,
                cumulative_selected_df=df,
                cycle=0,
                run_cache=run_cache,
                disable=["not_a_real_key"],
            )

    def test_missing_smiles_column_raises(self, run_cache):
        bad = pl.DataFrame({"ID": ["a", "b"]})
        good = _df_with_smiles(SMILES_DRUGLIKE[:2])
        with pytest.raises(ValueError):
            compute_diversity_metrics(
                newly_selected_df=bad,
                cumulative_selected_df=good,
                cycle=0,
                run_cache=run_cache,
            )

    def test_seeded_runs_produce_identical_metrics(self):
        df = _df_with_smiles(SMILES_DRUGLIKE)
        ra = compute_diversity_metrics(
            newly_selected_df=df,
            cumulative_selected_df=df,
            cycle=3,
            run_cache=RunCache(),
            random_state=12345,
        )
        rb = compute_diversity_metrics(
            newly_selected_df=df,
            cumulative_selected_df=df,
            cycle=3,
            run_cache=RunCache(),
            random_state=12345,
        )
        for k in DIVERSITY_KEYS:
            assert ra[k] == rb[k], f"key {k} differs: {ra[k]} != {rb[k]}"

    def test_cumulative_buffer_grows_across_cycles(self, run_cache):
        cycle1 = _df_with_smiles(SMILES_DRUGLIKE[:3])
        cycle2 = _df_with_smiles(SMILES_DRUGLIKE[3:6])
        cumulative = _df_with_smiles(SMILES_DRUGLIKE[:6])
        compute_diversity_metrics(
            newly_selected_df=cycle1,
            cumulative_selected_df=cycle1,
            cycle=0,
            run_cache=run_cache,
            random_state=42,
        )
        compute_diversity_metrics(
            newly_selected_df=cycle2,
            cumulative_selected_df=cumulative,
            cycle=1,
            run_cache=run_cache,
            random_state=42,
        )
        assert len(run_cache.cumulative_fp_buffer) == 6
        assert run_cache.bit_sum_buffer is not None


# ---------------------------------------------------------------------------
# Performance smoke (regression tripwire — fast)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformanceSmoke:
    def test_smoke_under_half_second(self, run_cache):
        # 200-compound batch + 500-compound cumulative set should comfortably
        # land under 0.5 s. Catches gross regressions without slow-marker.
        rng = np.random.default_rng(0)
        # Generate diverse SMILES cheaply by repeating the seed list.
        batch = (SMILES_DRUGLIKE * 25)[:200]
        cumulative = (SMILES_DRUGLIKE * 60)[:500]
        # Inject a small amount of random ordering variance.
        batch = list(rng.permutation(batch))
        cumulative = list(rng.permutation(cumulative))
        start = time.perf_counter()
        compute_diversity_metrics(
            newly_selected_df=_df_with_smiles(batch),
            cumulative_selected_df=_df_with_smiles(cumulative),
            cycle=4,
            run_cache=run_cache,
            random_state=42,
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"diversity smoke regressed: {elapsed:.3f}s > 0.5s"

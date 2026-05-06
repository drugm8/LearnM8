"""Integration tests for the diversity-metrics overhaul (feature 013).

End-to-end exercise of ``run_active_learning`` to verify:
- All six DIVERSITY_KEYS plus ``fingerprint_used`` land in cycle_metrics.csv
- Schema is stable (Float64) regardless of ``disable_molecular_similarity``
- Disable flag (bool and Iterable[str]) takes effect at the persistence boundary
- Determinism: two runs with the same random_state yield identical metric values
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from learnm8 import run_active_learning
from learnm8.evaluation.metrics.similarity import DIVERSITY_KEYS


SMILES_POOL = [
    "CCO", "CCC", "CCCO", "c1ccccc1", "C1CCCCC1", "Nc1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "C1=CC=C(C=C1)C(=O)O",
    "CCN(CC)CC", "CCCC", "CCCCC", "CCCCCC", "Nc1ncnc2[nH]cnc12",
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C", "Oc1ccccc1", "CCOc1ccccc1",
    "ClCC(Cl)Cl", "BrCCBr", "CC(=O)O", "CCO[H]", "CCN", "CCCCCCN",
    "CCSCC", "Cc1ccccc1", "Cc1ccc(C)cc1", "CC(C)C", "CC(C)(C)C",
    "C1CCNCC1",
]


def _make_dataset(n: int = 30) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    smis = (SMILES_POOL * ((n // len(SMILES_POOL)) + 1))[:n]
    return pl.DataFrame({
        "ID": [f"mol_{i}" for i in range(n)],
        "SMILES": smis,
        "Activity": rng.uniform(-1.0, 1.0, size=n).tolist(),
    })


def _read_metrics_csv(path: Path) -> pl.DataFrame:
    schema_overrides = {key: pl.Float64 for key in DIVERSITY_KEYS}
    schema_overrides["fingerprint_used"] = pl.Utf8
    return pl.read_csv(path, comment_prefix="#", schema_overrides=schema_overrides)


@pytest.fixture
def workdir():
    tmp = Path(tempfile.mkdtemp(prefix="learnm8_diversity_it_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.integration
class TestDiversityMetricsIntegration:

    def test_all_six_keys_persist_with_correct_dtypes(self, workdir):
        df = _make_dataset(n=30)
        out = workdir / "run_a"
        run_active_learning(
            compound_pool=df,
            oracle=df.select(["ID", "Activity"]),
            learner="rf",
            target_col="Activity",
            featurizer="morgan",
            n_cycles=3,
            batch_fraction=0.1,
            random_state=42,
            output_dir=out,
        )
        metrics = _read_metrics_csv(out / "cycle_metrics.csv")
        for key in DIVERSITY_KEYS:
            assert key in metrics.columns, f"missing column {key}"
            assert metrics.schema[key] == pl.Float64, (
                f"{key} dtype should be Float64, got {metrics.schema[key]}"
            )
        assert "fingerprint_used" in metrics.columns
        assert metrics.schema["fingerprint_used"] == pl.Utf8

        # Cycle 0 + 2 active learning cycles = 3 rows.
        assert metrics.height >= 3
        # At least some cycles should have computed values.
        non_null_count = metrics.select(
            pl.col("scaffold_diversity_index_batch").is_not_null().sum()
        ).item()
        assert non_null_count >= 1

        used = metrics.get_column("fingerprint_used").drop_nulls().to_list()
        assert all(label.startswith("morgan") for label in used)

    def test_disable_true_emits_null_columns_with_stable_schema(self, workdir):
        df = _make_dataset(n=20)
        out = workdir / "run_disabled"
        run_active_learning(
            compound_pool=df,
            oracle=df.select(["ID", "Activity"]),
            learner="rf",
            target_col="Activity",
            featurizer="morgan",
            n_cycles=2,
            batch_fraction=0.15,
            random_state=42,
            output_dir=out,
            disable_molecular_similarity=True,
        )
        metrics = _read_metrics_csv(out / "cycle_metrics.csv")
        for key in DIVERSITY_KEYS:
            assert metrics.schema[key] == pl.Float64
            # All values null when fully disabled.
            null_count = metrics.select(pl.col(key).is_null().sum()).item()
            assert null_count == metrics.height, f"{key} should be all-null"

    def test_disable_iterable_emits_subset_only(self, workdir):
        df = _make_dataset(n=24)
        out = workdir / "run_partial"
        run_active_learning(
            compound_pool=df,
            oracle=df.select(["ID", "Activity"]),
            learner="rf",
            target_col="Activity",
            featurizer="morgan",
            n_cycles=2,
            batch_fraction=0.15,
            random_state=42,
            output_dir=out,
            disable_molecular_similarity=["scaffold_diversity_index_batch"],
        )
        metrics = _read_metrics_csv(out / "cycle_metrics.csv")
        # Disabled key all-null, others computed.
        assert metrics.select(pl.col("scaffold_diversity_index_batch").is_null().sum()).item() == metrics.height
        scaffold_cum_non_null = metrics.select(
            pl.col("scaffold_diversity_index_cumulative").is_not_null().sum()
        ).item()
        assert scaffold_cum_non_null >= 1

    def test_seeded_runs_produce_identical_metric_columns(self, workdir):
        df = _make_dataset(n=24)
        out_a = workdir / "run_seed_a"
        out_b = workdir / "run_seed_b"
        for out in (out_a, out_b):
            run_active_learning(
                compound_pool=df,
                oracle=df.select(["ID", "Activity"]),
                learner="rf",
                target_col="Activity",
                featurizer="morgan",
                n_cycles=2,
                batch_fraction=0.15,
                random_state=12345,
                output_dir=out,
            )
        ma = _read_metrics_csv(out_a / "cycle_metrics.csv")
        mb = _read_metrics_csv(out_b / "cycle_metrics.csv")
        for key in DIVERSITY_KEYS:
            va = ma.get_column(key).to_list()
            vb = mb.get_column(key).to_list()
            for i, (a, b) in enumerate(zip(va, vb)):
                if a is None and b is None:
                    continue
                assert a is not None and b is not None, (
                    f"row {i} key {key} mismatch on null: {a} vs {b}"
                )
                assert np.isclose(a, b, equal_nan=True), (
                    f"row {i} key {key} differs: {a} vs {b}"
                )

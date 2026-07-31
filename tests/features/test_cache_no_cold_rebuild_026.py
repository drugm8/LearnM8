"""Feature 026 must not invalidate any existing feature cache.

026 changes two things that sit next to the cache key without being part of it:
where RDKit ``Mol`` objects are built (``SkfpFeaturizer.transform``) and how the
BLAKE2b digest loop is executed (``_cache_keys_bytes16``). Neither may move a
key. A cache written before the upgrade must be a 100% hit after it, and the
values read back must be identical.

The proof runs in two directions:

  - a cache populated through the pre-026 code path is read by the post-026
    path with ``misses == 0``
  - the digest for a fixed featurizer config is unchanged, whichever dispatch
    path (serial or loky) computed it

Contrast with feature 025 Item 1, which deliberately DID orphan the cache by
widening the config-hash denylist. 026 makes no such change, so no
``_warn_orphaned_cache_once``-style notice is warranted.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from learnm8.features import cache as cache_module
from learnm8.features import create_featurizer, extract_features

CACHE_COUNTER_RE = re.compile(r'cache featurizer=(\S+) hits=(\d+) misses=(\d+)')


@pytest.fixture(scope='module')
def fixture_smiles() -> list[str]:
    """Real ZINC molecules (§3.4 red line 1: no synthetic molecular data)."""
    path = Path(__file__).parent.parent / 'data' / 'diverse_molecules.csv'
    with path.open() as handle:
        return [row['SMILES'] for row in csv.DictReader(handle)]


@pytest.fixture(autouse=True)
def restore_log_propagation():
    """Let caplog observe the cache hit/miss counters regardless of test order.

    ``learnm8.utils.logging.setup_logging`` sets ``propagate = False`` on the
    ``learnm8`` logger, so once any earlier test calls it the cache's DEBUG
    counters never reach pytest's root-level caplog handler. Without this the
    ``misses == 0`` assertions below silently degrade into "no counters found"
    depending on which tests ran first.
    """
    learnm8_logger = logging.getLogger('learnm8')
    previous = learnm8_logger.propagate
    learnm8_logger.propagate = True
    yield
    learnm8_logger.propagate = previous


def _counters(caplog: pytest.LogCaptureFixture) -> list[tuple[str, int, int]]:
    """Parse every 'cache featurizer=... hits=N misses=M' debug line."""
    found = []
    for record in caplog.records:
        match = CACHE_COUNTER_RE.search(record.getMessage())
        if match:
            found.append((match.group(1), int(match.group(2)), int(match.group(3))))
    return found


def _write_cache_the_pre_026_way(
    smiles: list[str], featurizer_name: str, cache_dir: Path
) -> np.ndarray:
    """Populate the cache with features built the way 026 replaced.

    Monkeypatching is not needed: the pre-026 body is exactly
    ``fingerprint.transform(mol_from_smiles.transform(smiles))``, so calling
    that directly and letting the normal cache wrapper store the result
    reproduces a pre-upgrade cache file byte for byte.
    """
    featurizer = create_featurizer(featurizer_name, n_jobs=1)

    @cache_module.cache_features(cache_dir)
    def _legacy_extract(smiles_list, feat, *args, **kwargs):
        mols = feat.mol_from_smiles.transform(smiles_list)
        raw = np.asarray(feat.fingerprint.transform(mols))
        return raw.astype(np.dtype(feat.get_compute_dtype()), copy=False)

    return _legacy_extract(smiles, featurizer, cache_dir=cache_dir)


@pytest.mark.integration
@pytest.mark.parametrize('featurizer_name', ['morgan', 'maccs', 'estate'])
def test_pre_026_cache_is_a_total_hit_after_upgrade(
    featurizer_name: str,
    fixture_smiles: list[str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cache written the old way yields misses == 0 and identical values."""
    legacy_features = _write_cache_the_pre_026_way(
        fixture_smiles, featurizer_name, tmp_path
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=cache_module.__name__):
        new_features = extract_features(
            fixture_smiles, featurizer_name, n_jobs=1, cache_dir=tmp_path
        )

    counters = _counters(caplog)
    assert counters, 'expected the cache to log its hit/miss counters'
    name, hits, misses = counters[-1]
    assert misses == 0, (
        f'{featurizer_name}: {misses} miss(es) against a pre-026 cache — '
        f'feature 026 must not change any cache key'
    )
    assert hits == len(fixture_smiles)
    assert name == featurizer_name

    npt.assert_array_equal(new_features, legacy_features)


@pytest.mark.integration
def test_second_read_is_also_a_total_hit(
    fixture_smiles: list[str], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Sanity: the cache genuinely round-trips under the new path alone."""
    first = extract_features(fixture_smiles, 'morgan', n_jobs=1, cache_dir=tmp_path)

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=cache_module.__name__):
        second = extract_features(
            fixture_smiles, 'morgan', n_jobs=1, cache_dir=tmp_path
        )

    _, hits, misses = _counters(caplog)[-1]
    assert (hits, misses) == (len(fixture_smiles), 0)
    npt.assert_array_equal(first, second)


@pytest.mark.integration
def test_cache_keys_are_dispatch_path_independent(
    fixture_smiles: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache written with serial hashing is a total hit when read in parallel.

    This is the REQ-7/REQ-8 guarantee at the integration level: the loky path
    must not be able to write keys the serial path cannot find.
    """
    featurizer = create_featurizer('morgan', n_jobs=1)

    serial_keys = cache_module._cache_keys_bytes16(fixture_smiles, featurizer)

    monkeypatch.setattr(cache_module, 'PARALLEL_HASH_MIN_ROWS', 8)
    parallel_keys = cache_module._cache_keys_bytes16(
        fixture_smiles, featurizer, n_jobs=4
    )

    npt.assert_array_equal(serial_keys, parallel_keys)


@pytest.mark.integration
def test_smiles_consuming_featurizer_cache_also_unchanged(
    fixture_smiles: list[str], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """REQ-3 featurizers keep their cache too, not just the fast-path ones.

    ``lingo`` is used rather than mhfp/secfp: those two abort on one fixture
    molecule under RDKit 2026.03.2, independently of 026 (see
    ``test_transform_dispatch.RDKIT_STEREO_BUG_SMILES``).
    """
    legacy = _write_cache_the_pre_026_way(fixture_smiles, 'lingo', tmp_path)

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=cache_module.__name__):
        current = extract_features(
            fixture_smiles, 'lingo', n_jobs=1, cache_dir=tmp_path
        )

    _, hits, misses = _counters(caplog)[-1]
    assert misses == 0, 'lingo cache must survive feature 026'
    assert hits == len(fixture_smiles)
    npt.assert_array_equal(current, legacy)

"""Pre-computed feature fixtures for LearnM8 tests.

Provides session-scoped Morgan fingerprints computed once per test session
(or per xdist worker), replacing ~150 individual extract_features() calls.
"""

import pytest
import numpy as np
from learnm8.features.extraction import extract_features


@pytest.fixture(scope="session")
def _small_real_morgan_features_raw(small_real_compounds, tmp_path_factory):
    """Session-cached Morgan fingerprints. Internal — use small_real_morgan_features instead."""
    cache_dir = tmp_path_factory.mktemp("morgan_cache")
    return extract_features(small_real_compounds['SMILES'].to_list(), 'morgan', cache_dir)


@pytest.fixture
def small_real_morgan_features(_small_real_morgan_features_raw):
    """Defensive copy of session-cached Morgan fingerprints. Safe to mutate."""
    return _small_real_morgan_features_raw.copy()


@pytest.fixture(scope="session")
def _small_real_maccs_features_raw(small_real_compounds, tmp_path_factory):
    """Session-cached MACCS fingerprints. Internal — use small_real_maccs_features instead."""
    cache_dir = tmp_path_factory.mktemp("maccs_cache")
    return extract_features(small_real_compounds['SMILES'].to_list(), 'maccs', cache_dir)


@pytest.fixture
def small_real_maccs_features(_small_real_maccs_features_raw):
    """Defensive copy of session-cached MACCS fingerprints. Safe to mutate."""
    return _small_real_maccs_features_raw.copy()


@pytest.fixture(scope="session")
def _small_real_ecfp6_features_raw(small_real_compounds, tmp_path_factory):
    """Session-cached ECFP6 fingerprints. Internal — use small_real_ecfp6_features instead."""
    cache_dir = tmp_path_factory.mktemp("ecfp6_cache")
    return extract_features(small_real_compounds['SMILES'].to_list(), 'ecfp6', cache_dir)


@pytest.fixture
def small_real_ecfp6_features(_small_real_ecfp6_features_raw):
    """Defensive copy of session-cached ECFP6 fingerprints. Safe to mutate."""
    return _small_real_ecfp6_features_raw.copy()

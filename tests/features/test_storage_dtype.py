"""Interface contract tests for ``Featurizer.get_storage_dtype()`` (T004, FR-001/FR-014/FR-015)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learnm8.core.interfaces import Featurizer
from learnm8.exceptions import ConfigurationError
from learnm8.features import (
    FEATURIZER_REGISTRY,
    MordredFeaturizer,
    MorganFeaturizer,
)
from learnm8.features.cache import (
    ALLOWED_STORAGE_DTYPES,
    STORAGE_FLOAT32,
    STORAGE_PACKED,
    _normalize_storage_dtype,
)
from learnm8.features.extraction import extract_features

BINARY_FEATURIZER_NAMES = {
    'morgan', 'ecfp', 'ecfp6', 'morgan_feat',
    'maccs', 'pubchem', 'klekota_roth', 'laggner',
    'avalon', 'atom_pair', 'topological_torsion',
    'rdkit', 'pattern', 'layered',
    'map4', 'mhfp', 'secfp', 'lingo',
    'e3fp',
}

# Continuous-by-design featurizers (skfp default + ones overriding feature_type='continuous').
# All others in the registry default to 'binary' via SkfpFeaturizer.feature_type.
CONTINUOUS_FEATURIZER_NAMES = {
    'mordred', 'descriptors', 'rdkit_2d_descriptors',
    'estate', 'ghose_crippen', 'mqns', 'vsa', 'bcut2d',
    'physiochemical', 'pharmacophore', 'functional_groups',
    'erg',
}

THREE_D_FEATURIZER_NAMES = {
    'whim', 'usr', 'usrcat', 'getaway',
    'morse', 'rdf', 'autocorr', 'electroshape',
}


@pytest.mark.unit
def test_default_is_float32():
    class _Stub(Featurizer):
        def transform(self, smiles_list):
            return np.zeros((len(smiles_list), 4), dtype=np.float32)

        def get_dimension(self) -> int:
            return 4

        def get_name(self) -> str:
            return 'stub'

    assert _Stub().get_storage_dtype() == 'float32'


@pytest.mark.unit
def test_morgan_returns_packed_uint8():
    assert MorganFeaturizer().get_storage_dtype() == 'packed_uint8'


@pytest.mark.unit
def test_mordred_returns_float32():
    assert MordredFeaturizer().get_storage_dtype() == 'float32'


@pytest.mark.unit
@pytest.mark.parametrize('name', sorted(BINARY_FEATURIZER_NAMES))
def test_binary_skfp_subclasses_return_packed_uint8(name):
    cls = FEATURIZER_REGISTRY[name]
    feat = cls(n_jobs=1)
    assert feat.get_storage_dtype() == 'packed_uint8', (
        f"Expected packed_uint8 for {name}, got {feat.get_storage_dtype()!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize('name', sorted(CONTINUOUS_FEATURIZER_NAMES | THREE_D_FEATURIZER_NAMES))
def test_continuous_and_3d_subclasses_return_float32(name):
    cls = FEATURIZER_REGISTRY[name]
    feat = cls(n_jobs=1)
    assert feat.get_storage_dtype() == 'float32', (
        f"Expected float32 for {name}, got {feat.get_storage_dtype()!r}"
    )


@pytest.mark.unit
def test_normalize_rejects_unknown_dtype():
    with pytest.raises(ConfigurationError, match=r"unknown storage_dtype"):
        _normalize_storage_dtype('float16')


@pytest.mark.unit
def test_unknown_dtype_raises_configuration_error_at_extract(tmp_path: Path):
    class _Bad(MorganFeaturizer):
        def get_storage_dtype(self) -> str:
            return 'float16'

    with pytest.raises(ConfigurationError, match=r"unknown storage_dtype"):
        extract_features(['CCO'], _Bad(), cache_dir=tmp_path)


@pytest.mark.unit
def test_full_registry_dispatch_yields_allowed_values():
    for name, cls in FEATURIZER_REGISTRY.items():
        feat = cls(n_jobs=1)
        dtype = feat.get_storage_dtype()
        assert dtype in ALLOWED_STORAGE_DTYPES, (
            f"{name} returned unsupported storage_dtype {dtype!r}"
        )
        if name in BINARY_FEATURIZER_NAMES:
            assert dtype == STORAGE_PACKED
        elif name in CONTINUOUS_FEATURIZER_NAMES | THREE_D_FEATURIZER_NAMES:
            assert dtype == STORAGE_FLOAT32

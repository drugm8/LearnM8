"""
Feature extraction module for molecular fingerprints and descriptors.

This module provides a config-driven factory for all scikit-fingerprints
featurizers. The ~1,700 LOC of thin wrapper classes have been replaced by a
single _FEATURIZER_CONFIG dict and a create_featurizer() factory function.
"""

from difflib import get_close_matches
from types import MappingProxyType
from typing import Any

from skfp.fingerprints import (
    AtomPairFingerprint,
    AutocorrFingerprint,
    AvalonFingerprint,
    BCUT2DFingerprint,
    E3FPFingerprint,
    ECFPFingerprint,
    ElectroShapeFingerprint,
    ERGFingerprint,
    EStateFingerprint,
    FunctionalGroupsFingerprint,
    GETAWAYFingerprint,
    GhoseCrippenFingerprint,
    KlekotaRothFingerprint,
    LaggnerFingerprint,
    LayeredFingerprint,
    LingoFingerprint,
    MACCSFingerprint,
    MAPFingerprint,
    MHFPFingerprint,
    MordredFingerprint,
    MORSEFingerprint,
    MQNsFingerprint,
    PatternFingerprint,
    PharmacophoreFingerprint,
    PhysiochemicalPropertiesFingerprint,
    PubChemFingerprint,
    RDFFingerprint,
    RDKit2DDescriptorsFingerprint,
    RDKitFingerprint,
    SECFPFingerprint,
    TopologicalTorsionFingerprint,
    USRCATFingerprint,
    USRFingerprint,
    VSAFingerprint,
    WHIMFingerprint,
)

from learnm8.exceptions import ConfigurationError
from learnm8.features.base import SkfpFeaturizer
from learnm8.features.extraction import extract_features

__all__: list[str] = []


def _morgan_defaults() -> dict[str, Any]:
    return {
        'radius': 2,
        'fp_size': 2048,
        'include_chirality': False,
        'use_bond_types': True,
        'use_features': False,
        'count': False,
        'n_jobs': -1,
        'verbose': 0,
        'auto_generate_conformers': False,
    }


def _atom_pair_defaults() -> dict[str, Any]:
    return {
        'fp_size': 2048,
        'min_distance': 1,
        'max_distance': 30,
        'include_chirality': False,
        'use_2D': True,
        'count': False,
        'n_jobs': -1,
        'verbose': 0,
        'auto_generate_conformers': False,
    }


def _topological_torsion_defaults() -> dict[str, Any]:
    return {
        'fp_size': 2048,
        'include_chirality': False,
        'count': False,
        'n_jobs': -1,
        'verbose': 0,
        'auto_generate_conformers': False,
    }


def _simple_2d_defaults(**extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {'n_jobs': -1, 'verbose': 0, 'auto_generate_conformers': False}
    d.update(extra)
    return d


def _3d_defaults(**extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        'auto_generate_conformers': True,
        'num_conformers': 1,
        'optimize_force_field': None,
        'n_jobs': -1,
        'verbose': 0,
    }
    d.update(extra)
    return d


_FEATURIZER_CONFIG: MappingProxyType[str, Any] = MappingProxyType(
    {
        # ── Circular fingerprints ──────────────────────────────────────────
        'morgan': {
            'cls': ECFPFingerprint,
            'defaults': _morgan_defaults(),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'count_override': {
                'storage_dtype': 'csr_uint16',
                'feature_type': 'continuous',
            },
            'description': 'ECFP4 circular fingerprints (2048-bit)',
        },
        'ecfp': {
            'cls': ECFPFingerprint,
            'defaults': _morgan_defaults(),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'count_override': {
                'storage_dtype': 'csr_uint16',
                'feature_type': 'continuous',
            },
            'description': 'ECFP4 circular fingerprints (2048-bit)',
        },
        'ecfp6': {
            'cls': ECFPFingerprint,
            'defaults': {**_morgan_defaults(), 'radius': 3},
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'count_override': {
                'storage_dtype': 'csr_uint16',
                'feature_type': 'continuous',
            },
            'description': 'ECFP6 circular fingerprints (2048-bit)',
        },
        'morgan_feat': {
            'cls': ECFPFingerprint,
            'defaults': {**_morgan_defaults(), 'use_features': True},
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'count_override': {
                'storage_dtype': 'csr_uint16',
                'feature_type': 'continuous',
            },
            'description': 'FCFP4 circular fingerprints (2048-bit, feature-based)',
        },
        # ── Structural keys ────────────────────────────────────────────────
        'maccs': {
            'cls': MACCSFingerprint,
            'defaults': _simple_2d_defaults(count=False),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'count_override': {'storage_dtype': 'uint8', 'feature_type': 'continuous'},
            'description': 'MACCS structural keys (167 predefined patterns)',
        },
        'pubchem': {
            'cls': PubChemFingerprint,
            'defaults': _simple_2d_defaults(count=False),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'PubChem CACTVS fingerprints (881 structural keys)',
        },
        'klekota_roth': {
            'cls': KlekotaRothFingerprint,
            'defaults': _simple_2d_defaults(count=False),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'Klekota-Roth substructure fingerprints (4860-bit)',
        },
        'laggner': {
            'cls': LaggnerFingerprint,
            'defaults': _simple_2d_defaults(count=False),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'Laggner substructure fingerprints (307-bit)',
        },
        # ── Topological ────────────────────────────────────────────────────
        'avalon': {
            'cls': AvalonFingerprint,
            'defaults': _simple_2d_defaults(fp_size=512, count=False),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'Avalon fingerprints (512-bit)',
        },
        'atom_pair': {
            'cls': AtomPairFingerprint,
            'defaults': _atom_pair_defaults(),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'count_override': {
                'storage_dtype': 'csr_uint16',
                'feature_type': 'continuous',
            },
            'description': 'Atom pair fingerprints (2048-bit)',
        },
        'topological_torsion': {
            'cls': TopologicalTorsionFingerprint,
            'defaults': _topological_torsion_defaults(),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'count_override': {
                'storage_dtype': 'csr_uint16',
                'feature_type': 'continuous',
            },
            'description': 'Topological torsion fingerprints (2048-bit)',
        },
        'rdkit': {
            'cls': RDKitFingerprint,
            'defaults': _simple_2d_defaults(
                fp_size=2048,
                min_path=1,
                max_path=7,
                linear_paths_only=False,
                use_bond_order=True,
                count=False,
            ),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'RDKit topological fingerprints (2048-bit)',
        },
        'pattern': {
            'cls': PatternFingerprint,
            'defaults': _simple_2d_defaults(fp_size=2048),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'Pattern fingerprints (2048-bit)',
        },
        'layered': {
            'cls': LayeredFingerprint,
            'defaults': _simple_2d_defaults(fp_size=2048),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'Layered fingerprints (2048-bit)',
        },
        # ── Hashed fingerprints ────────────────────────────────────────────
        'map4': {
            'cls': MAPFingerprint,
            'defaults': _simple_2d_defaults(fp_size=2048, radius=2),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'MAP4 MinHashed atom-pair fingerprints (2048-bit)',
        },
        'mhfp': {
            'cls': MHFPFingerprint,
            'defaults': _simple_2d_defaults(fp_size=2048, radius=3),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'MHFP MinHashed fingerprints (2048-bit)',
        },
        'secfp': {
            'cls': SECFPFingerprint,
            'defaults': _simple_2d_defaults(fp_size=2048, radius=3),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'SECFP SMILES extended connectivity fingerprints (2048-bit)',
        },
        'lingo': {
            'cls': LingoFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': False,
            'description': 'Lingo SMILES substring fingerprints',
        },
        'erg': {
            'cls': ERGFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'ERG extended reduced graph fingerprints',
        },
        # ── 2D Descriptors ─────────────────────────────────────────────────
        'mordred': {
            'cls': MordredFingerprint,
            'defaults': _simple_2d_defaults(ignore_3D=True),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'Mordred 2D molecular descriptors (1613-D)',
        },
        'descriptors': {
            'cls': MordredFingerprint,
            'defaults': _simple_2d_defaults(ignore_3D=True),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'Mordred 2D molecular descriptors (1613-D)',
        },
        'rdkit_2d_descriptors': {
            'cls': RDKit2DDescriptorsFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'RDKit 2D molecular descriptors (~200-D)',
        },
        'estate': {
            'cls': EStateFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'EState electrotopological state indices (79-D)',
        },
        'ghose_crippen': {
            'cls': GhoseCrippenFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'Ghose-Crippen LogP atomic contributions',
        },
        'mqns': {
            'cls': MQNsFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'uint8',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'MQNs molecular quantum numbers (42-D)',
        },
        'vsa': {
            'cls': VSAFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'VSA Van der Waals surface area descriptors',
        },
        'bcut2d': {
            'cls': BCUT2DFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'BCUT2D Burden-CAS-UT descriptors',
        },
        'physiochemical': {
            'cls': PhysiochemicalPropertiesFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'csr_uint16',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'Physiochemical molecular properties',
        },
        'pharmacophore': {
            'cls': PharmacophoreFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'csr_uint16',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'Pharmacophore feature patterns',
        },
        'functional_groups': {
            'cls': FunctionalGroupsFingerprint,
            'defaults': _simple_2d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': False,
            'description': 'Functional group presence patterns',
        },
        # ── 3D fingerprints ────────────────────────────────────────────────
        'whim': {
            'cls': WHIMFingerprint,
            'defaults': _3d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': True,
            'description': 'WHIM 3D weighted holistic invariant molecular descriptors (114-D)',
        },
        'usr': {
            'cls': USRFingerprint,
            'defaults': _3d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': True,
            'description': 'USR 3D ultrafast shape recognition descriptors (12-D)',
        },
        'usrcat': {
            'cls': USRCATFingerprint,
            'defaults': _3d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': True,
            'description': 'USRCAT 3D ultrafast shape recognition with CREDO atom types (60-D)',
        },
        'e3fp': {
            'cls': E3FPFingerprint,
            'defaults': _3d_defaults(fp_size=2048, level=5, radius_multiplier=1.718),
            'storage_dtype': 'packed_uint8',
            'feature_type': 'binary',
            'requires_conformers': True,
            'description': 'E3FP extended 3D fingerprints (2048-bit)',
        },
        'getaway': {
            'cls': GETAWAYFingerprint,
            'defaults': _3d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': True,
            'description': 'GETAWAY 3D geometry, topology, and atom-weights assembly descriptors',
        },
        'morse': {
            'cls': MORSEFingerprint,
            'defaults': _3d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': True,
            'description': 'MORSE 3D molecule representation of structures based on electron diffraction',
        },
        'rdf': {
            'cls': RDFFingerprint,
            'defaults': _3d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': True,
            'description': 'RDF 3D radial distribution function descriptors',
        },
        'autocorr': {
            'cls': AutocorrFingerprint,
            'defaults': _3d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': True,
            'description': 'Autocorr 3D autocorrelation descriptors',
        },
        'electroshape': {
            'cls': ElectroShapeFingerprint,
            'defaults': _3d_defaults(),
            'storage_dtype': 'float32',
            'feature_type': 'continuous',
            'requires_conformers': True,
            'description': 'ElectroShape 3D electrostatic shape descriptors (15-D)',
        },
    }
)

FEATURIZER_REGISTRY: frozenset[str] = frozenset(_FEATURIZER_CONFIG.keys())

FEATURIZERS_3D: list[str] = [
    'whim',
    'usr',
    'usrcat',
    'e3fp',
    'getaway',
    'morse',
    'rdf',
    'autocorr',
    'electroshape',
]

FEATURIZERS_2D: list[str] = [
    name for name in _FEATURIZER_CONFIG if name not in FEATURIZERS_3D
]


def _apply_param_mappings(name: str, kwargs: dict[str, Any]) -> None:
    if name in ('morgan', 'ecfp', 'ecfp6', 'morgan_feat'):
        if 'use_features' in kwargs:
            kwargs['use_pharmacophoric_invariants'] = kwargs.pop('use_features')
    elif name == 'atom_pair' and 'use_2D' in kwargs:
        kwargs['use_3D'] = not kwargs.pop('use_2D')
    elif name in ('mordred', 'descriptors') and 'ignore_3D' in kwargs:
        kwargs['use_3D'] = not kwargs.pop('ignore_3D')


def create_featurizer(name: str, **overrides: Any) -> SkfpFeaturizer:
    if name not in _FEATURIZER_CONFIG:
        available = sorted(_FEATURIZER_CONFIG.keys())
        suggestions = get_close_matches(name, available, n=5)
        raise ConfigurationError(
            f"Unknown featurizer: '{name}'. "
            f'Available: {", ".join(available[:10])}... '
            f'{"Did you mean: " + ", ".join(suggestions) + "?" if suggestions else ""}'
            f"Use 'learnm8 list featurizers' to see all options."
        )

    config = _FEATURIZER_CONFIG[name]
    defaults: dict[str, Any] = dict(config['defaults'])

    allowed = set(defaults.keys())
    unknown = set(overrides.keys()) - allowed
    if unknown:
        raise TypeError(
            f'create_featurizer({name!r}) got unexpected keyword arguments: '
            f'{", ".join(sorted(unknown))}. '
            f'Allowed: {", ".join(sorted(allowed))}.'
        )

    defaults.update(overrides)

    conformer_keys = {
        'auto_generate_conformers',
        'num_conformers',
        'optimize_force_field',
    }
    fp_kwargs = {k: v for k, v in defaults.items() if k not in conformer_keys}

    auto_generate_conformers = defaults.get('auto_generate_conformers', False)
    num_conformers = defaults.get('num_conformers', 1)
    optimize_force_field = defaults.get('optimize_force_field')

    conformer_params: dict[str, Any] = {}
    if num_conformers != 1:
        conformer_params['num_conformers'] = num_conformers
    if optimize_force_field:
        conformer_params['optimize_force_field'] = optimize_force_field

    _apply_param_mappings(name, fp_kwargs)

    count = fp_kwargs.get('count', False)
    storage_dtype: str = config['storage_dtype']
    feature_type: str = config['feature_type']
    if count and 'count_override' in config:
        storage_dtype = config['count_override']['storage_dtype']
        feature_type = config['count_override']['feature_type']

    fingerprint_instance = config['cls'](**fp_kwargs)

    return SkfpFeaturizer(
        fingerprint_instance,
        auto_generate_conformers=auto_generate_conformers,
        conformer_params=conformer_params if conformer_params else None,
        n_jobs=fp_kwargs.get('n_jobs', -1),
        verbose=fp_kwargs.get('verbose', 0),
        storage_dtype=storage_dtype,
        feature_type=feature_type,
        fingerprint_name=name,
        fingerprint_params=fp_kwargs.copy(),
        description=config.get('description'),
    )


def list_available_featurizers() -> dict[str, list[str]]:
    return {
        '2d': sorted(FEATURIZERS_2D),
        '3d': sorted(FEATURIZERS_3D),
        'all': sorted(FEATURIZER_REGISTRY),
    }


__all__ = [
    'FEATURIZERS_2D',
    'FEATURIZERS_3D',
    'FEATURIZER_REGISTRY',
    'SkfpFeaturizer',
    'create_featurizer',
    'extract_features',
    'list_available_featurizers',
]

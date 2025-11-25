"""
Feature extraction module for molecular fingerprints and descriptors.

This module provides both functional API (extract_features) and class-based API (Featurizer).
All 43 scikit-fingerprints featurizers available, including 9 3D conformational types.
"""

from learnm8.core.interfaces import Featurizer
from learnm8.features.extraction import extract_features

# Base class
from learnm8.features.base import SkfpFeaturizer

# 2D featurizers - Circular and structural keys
from learnm8.features.skfp_2d.morgan import MorganFeaturizer
from learnm8.features.skfp_2d.maccs import MACCSFeaturizer
from learnm8.features.skfp_2d.avalon import AvalonFeaturizer
from learnm8.features.skfp_2d.pubchem import PubChemFeaturizer
from learnm8.features.skfp_2d.klekota_roth import KlekotaRothFeaturizer
from learnm8.features.skfp_2d.laggner import LaggnerFeaturizer

# 2D featurizers - Topological
from learnm8.features.skfp_2d.atom_pair import AtomPairFeaturizer
from learnm8.features.skfp_2d.topological_torsion import TopologicalTorsionFeaturizer
from learnm8.features.skfp_2d.rdkit import RDKitFeaturizer
from learnm8.features.skfp_2d.pattern import PatternFeaturizer
from learnm8.features.skfp_2d.layered import LayeredFeaturizer

# 2D featurizers - Hashed
from learnm8.features.skfp_2d.map4 import MAP4Featurizer
from learnm8.features.skfp_2d.mhfp import MHFPFeaturizer
from learnm8.features.skfp_2d.secfp import SECFPFeaturizer
from learnm8.features.skfp_2d.lingo import LingoFeaturizer
from learnm8.features.skfp_2d.erg import ERGFeaturizer

# 2D featurizers - Descriptors
from learnm8.features.skfp_2d.mordred import MordredFeaturizer
from learnm8.features.skfp_2d.rdkit_2d_descriptors import RDKit2DDescriptorsFeaturizer
from learnm8.features.skfp_2d.estate import EStateFeaturizer
from learnm8.features.skfp_2d.ghose_crippen import GhoseCrippenFeaturizer
from learnm8.features.skfp_2d.mqns import MQNsFeaturizer
from learnm8.features.skfp_2d.vsa import VSAFeaturizer
from learnm8.features.skfp_2d.bcut2d import BCUT2DFeaturizer
from learnm8.features.skfp_2d.physiochemical import PhysiochemicalPropertiesFeaturizer
from learnm8.features.skfp_2d.pharmacophore import PharmacophoreFeaturizer
from learnm8.features.skfp_2d.functional_groups import FunctionalGroupsFeaturizer

# 3D featurizers
from learnm8.features.skfp_3d.whim import WHIMFeaturizer
from learnm8.features.skfp_3d.usr import USRFeaturizer
from learnm8.features.skfp_3d.usrcat import USRCATFeaturizer
from learnm8.features.skfp_3d.e3fp import E3FPFeaturizer
from learnm8.features.skfp_3d.getaway import GETAWAYFeaturizer
from learnm8.features.skfp_3d.morse import MORSEFeaturizer
from learnm8.features.skfp_3d.rdf import RDFFeaturizer
from learnm8.features.skfp_3d.autocorr import AutocorrFeaturizer
from learnm8.features.skfp_3d.electroshape import ElectroShapeFeaturizer


# Wrapper classes for registry entries with custom defaults
class ECFP6Featurizer(MorganFeaturizer):
    def __init__(self, **kwargs):
        kwargs.setdefault('radius', 3)
        super().__init__(**kwargs)


class MorganFeatFeaturizer(MorganFeaturizer):
    def __init__(self, **kwargs):
        kwargs.setdefault('use_features', True)
        super().__init__(**kwargs)


FEATURIZER_REGISTRY = {
    # Circular fingerprints
    'morgan': MorganFeaturizer,
    'ecfp': MorganFeaturizer,
    'ecfp6': ECFP6Featurizer,
    'morgan_feat': MorganFeatFeaturizer,

    # Structural keys
    'maccs': MACCSFeaturizer,
    'pubchem': PubChemFeaturizer,
    'klekota_roth': KlekotaRothFeaturizer,
    'laggner': LaggnerFeaturizer,

    # Topological
    'avalon': AvalonFeaturizer,
    'atom_pair': AtomPairFeaturizer,
    'topological_torsion': TopologicalTorsionFeaturizer,
    'rdkit': RDKitFeaturizer,
    'pattern': PatternFeaturizer,
    'layered': LayeredFeaturizer,

    # Hashed fingerprints
    'map4': MAP4Featurizer,
    'mhfp': MHFPFeaturizer,
    'secfp': SECFPFeaturizer,
    'lingo': LingoFeaturizer,
    'erg': ERGFeaturizer,

    # 2D Descriptors
    'mordred': MordredFeaturizer,
    'descriptors': MordredFeaturizer,
    'rdkit_2d_descriptors': RDKit2DDescriptorsFeaturizer,
    'estate': EStateFeaturizer,
    'ghose_crippen': GhoseCrippenFeaturizer,
    'mqns': MQNsFeaturizer,
    'vsa': VSAFeaturizer,
    'bcut2d': BCUT2DFeaturizer,
    'physiochemical': PhysiochemicalPropertiesFeaturizer,
    'pharmacophore': PharmacophoreFeaturizer,
    'functional_groups': FunctionalGroupsFeaturizer,

    # 3D fingerprints
    'whim': WHIMFeaturizer,
    'usr': USRFeaturizer,
    'usrcat': USRCATFeaturizer,
    'e3fp': E3FPFeaturizer,
    'getaway': GETAWAYFeaturizer,
    'morse': MORSEFeaturizer,
    'rdf': RDFFeaturizer,
    'autocorr': AutocorrFeaturizer,
    'electroshape': ElectroShapeFeaturizer,
}

FEATURIZERS_3D = [
    'whim', 'usr', 'usrcat', 'e3fp', 'getaway', 'morse',
    'rdf', 'autocorr', 'electroshape'
]

FEATURIZERS_2D = [
    name for name in FEATURIZER_REGISTRY.keys()
    if name not in FEATURIZERS_3D
]


def list_available_featurizers() -> dict:
    """List all registered featurizer names categorized by type.

    Returns:
        Dictionary with '2d' and '3d' keys containing sorted lists
    """
    return {
        '2d': sorted(FEATURIZERS_2D),
        '3d': sorted(FEATURIZERS_3D),
        'all': sorted(FEATURIZER_REGISTRY.keys())
    }


__all__ = [
    'Featurizer',
    'extract_features',
    'FEATURIZER_REGISTRY',
    'FEATURIZERS_2D',
    'FEATURIZERS_3D',
    'list_available_featurizers',
    'SkfpFeaturizer',
    # Circular and structural keys
    'MorganFeaturizer',
    'MACCSFeaturizer',
    'AvalonFeaturizer',
    'PubChemFeaturizer',
    'KlekotaRothFeaturizer',
    'LaggnerFeaturizer',
    # Topological
    'AtomPairFeaturizer',
    'TopologicalTorsionFeaturizer',
    'RDKitFeaturizer',
    'PatternFeaturizer',
    'LayeredFeaturizer',
    # Hashed
    'MAP4Featurizer',
    'MHFPFeaturizer',
    'SECFPFeaturizer',
    'LingoFeaturizer',
    'ERGFeaturizer',
    # Descriptors
    'MordredFeaturizer',
    'RDKit2DDescriptorsFeaturizer',
    'EStateFeaturizer',
    'GhoseCrippenFeaturizer',
    'MQNsFeaturizer',
    'VSAFeaturizer',
    'BCUT2DFeaturizer',
    'PhysiochemicalPropertiesFeaturizer',
    'PharmacophoreFeaturizer',
    'FunctionalGroupsFeaturizer',
    # 3D featurizers
    'WHIMFeaturizer',
    'USRFeaturizer',
    'USRCATFeaturizer',
    'E3FPFeaturizer',
    'GETAWAYFeaturizer',
    'MORSEFeaturizer',
    'RDFFeaturizer',
    'AutocorrFeaturizer',
    'ElectroShapeFeaturizer',
    # Wrapper classes
    'ECFP6Featurizer',
    'MorganFeatFeaturizer',
]

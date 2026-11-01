# Topological Fingerprints

Fingerprints built from paths, atom pairs, torsions or substructure patterns in the molecular graph.

| Name | What it is | Defaults | Storage dtype |
|------|-----------|----------|---------------|
| `avalon` | Avalon | `fp_size=512` | `packed_uint8` |
| `atom_pair` | Atom pairs | `fp_size=2048`, `min_distance=1`, `max_distance=30`, `include_chirality=False`, `use_2D=True` | `packed_uint8` |
| `topological_torsion` | Topological torsions | `fp_size=2048`, `include_chirality=False` | `packed_uint8` |
| `rdkit` | RDKit path-based fingerprint | `fp_size=2048`, `min_path=1`, `max_path=7`, `linear_paths_only=False`, `use_bond_order=True` | `packed_uint8` |
| `pattern` | RDKit pattern fingerprint | `fp_size=2048` | `packed_uint8` |
| `layered` | RDKit layered fingerprint | `fp_size=2048` | `packed_uint8` |

`avalon`, `atom_pair`, `topological_torsion` and `rdkit` default to `count=False`; with `count=True` they become continuous counts stored as `csr_uint16`. `pattern` and `layered` have no count mode. None requires conformers.

## Usage

```bash
learnm8 run compounds.csv --target Activity --learner xgb --featurizer atom_pair
```

```python
from learnm8 import extract_features

features = extract_features(['CCO', 'c1ccccc1'], 'avalon')
```

See the [featurizers overview](overview.md) for the other categories.

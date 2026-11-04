# Featurizers

Featurizers turn SMILES strings into fixed-length numeric vectors that learners can train on. They are scikit-fingerprints fingerprints and descriptors wrapped by LearnM8, selected by name.

## When a featurizer is required

Every learner needs `featurizer=` (`--featurizer`) except `chemprop` and `chemprop_ensemble`, which work directly on SMILES. If you give those two a featurizer, it is used as extra descriptors alongside the graph representation (hybrid mode):

```bash
learnm8 run compounds.csv --target Activity --learner chemprop --featurizer descriptors
```

## Categories

| Category | Featurizers |
|----------|-------------|
| [Circular](circular.md) | `morgan`, `ecfp`, `ecfp6`, `morgan_feat` |
| [Structural keys](keys.md) | `maccs`, `pubchem`, `klekota_roth`, `laggner` |
| [Topological](topological.md) | `avalon`, `atom_pair`, `topological_torsion`, `rdkit`, `pattern`, `layered` |
| [Hashed](hashed.md) | `map4`, `mhfp`, `lingo`, `erg`, `secfp` |
| [2D descriptors](descriptors.md) | `mordred`, `descriptors`, `rdkit_2d_descriptors`, `estate`, `ghose_crippen`, `mqns`, `vsa`, `bcut2d`, `physicochemical`, `pharmacophore`, `functional_groups`, `autocorr` |
| [3D](3d.md) | `whim`, `usr`, `usrcat`, `e3fp`, `getaway`, `morse`, `rdf`, `electroshape` |

That is 39 names: 31 2D and 8 3D (37 distinct featurizers plus the aliases `ecfp` and `descriptors`).

List the names from Python:

```python
from learnm8.features import list_available_featurizers

print(list_available_featurizers()['all'])
```

## Extracting features directly

`run_active_learning()` featurizes for you. To featurize on your own, use `extract_features`:

```python
from pathlib import Path
from learnm8 import extract_features

features = extract_features(['CCO', 'CCC'], 'morgan', cache_dir=Path('.shared_cache'))
# shape: (n_compounds, n_features)
```

`featurizer` is a name or a featurizer instance, `n_jobs` defaults to `-1`, and `show_progress` defaults to `False`. To change a featurizer's parameters, build an instance with `create_featurizer(name, **overrides)` from `learnm8.features` and pass it as `featurizer`. Features are cached on disk; set `cache_dir` to share the cache between runs.

## Custom featurizers

Featurizers are entries in `_FEATURIZER_CONFIG` in `learnm8/features/__init__.py`; see [Extending LearnM8](../../customization/extending.md).

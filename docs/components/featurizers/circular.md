# Circular Fingerprints

Circular (ECFP/FCFP) fingerprints encode the neighbourhood of each atom out to a given radius, hashed into a fixed-length bit vector. All four are built on scikit-fingerprints' `ECFPFingerprint`.

| Name | What it is | Defaults | Storage dtype |
|------|-----------|----------|---------------|
| `morgan` | ECFP4 | `radius=2`, `fp_size=2048` | `packed_uint8` |
| `ecfp` | Alias of `morgan` | `radius=2`, `fp_size=2048` | `packed_uint8` |
| `ecfp6` | ECFP6 | `radius=3`, `fp_size=2048` | `packed_uint8` |
| `morgan_feat` | FCFP4 (feature-based invariants) | `radius=2`, `fp_size=2048`, `use_features=True` | `packed_uint8` |

All four also default to `include_chirality=False`, `use_bond_types=True` and `count=False`. With `count=True` the features become count vectors (continuous) stored as `csr_uint16`. None of them requires conformers.

## Usage

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan
```

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    target_col='Activity',
    learner='rf',
    featurizer='ecfp6',
    n_cycles=10,
)
```

To change parameters, build the featurizer with `create_featurizer` and pass the instance:

```python
from learnm8.features import create_featurizer

featurizer = create_featurizer('morgan', radius=3, fp_size=4096, count=True)
```

See the [featurizers overview](overview.md) for the other categories.

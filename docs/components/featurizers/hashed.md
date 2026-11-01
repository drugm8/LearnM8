# Hashed Fingerprints

Fingerprints computed from hashed string or shingle representations, or from reduced graphs.

| Name | What it is | Defaults | Storage dtype |
|------|-----------|----------|---------------|
| `map4` | MAP4 MinHashed atom-pair fingerprint | `fp_size=2048`, `radius=2` | `packed_uint8` |
| `mhfp` | MHFP MinHashed fingerprint | `fp_size=2048`, `radius=3` | `packed_uint8` |
| `secfp` | SECFP, SMILES extended connectivity fingerprint | `fp_size=2048`, `radius=3` | `packed_uint8` |
| `lingo` | Lingo SMILES substring fingerprint | scikit-fingerprints defaults | `packed_uint8` |
| `erg` | ERG extended reduced graph (continuous) | scikit-fingerprints defaults | `float32` |

None requires conformers. `erg` is the only continuous featurizer in this group.

## Usage

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer mhfp
```

```python
from learnm8 import extract_features

features = extract_features(['CCO', 'c1ccccc1'], 'map4')
```

See the [featurizers overview](overview.md) for the other categories.

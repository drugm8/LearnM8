# Structural Keys

Structural-key fingerprints test each molecule against a fixed dictionary of substructure patterns; each bit means one pattern is present.

| Name | What it is | Dimension / defaults | Storage dtype |
|------|-----------|----------------------|---------------|
| `maccs` | MACCS structural keys | 166 keys | `packed_uint8` |
| `pubchem` | PubChem CACTVS keys | 881 keys | `packed_uint8` |
| `klekota_roth` | Klekota-Roth substructure keys | 4860 bits | `packed_uint8` |
| `laggner` | Laggner substructure keys | 307 bits | `packed_uint8` |

All four default to `count=False`. With `count=True` they become continuous count features: `maccs` is stored as `uint8`, the others as `csr_uint16`. None requires conformers.

## Usage

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer maccs
```

```python
from learnm8 import extract_features

features = extract_features(['CCO', 'c1ccccc1'], 'maccs')
```

See the [featurizers overview](overview.md) for the other categories.

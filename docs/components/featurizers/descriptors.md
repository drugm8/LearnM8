# 2D Descriptors

Continuous descriptors and count vectors computed from the 2D structure (no conformers). Because they are not binary, tree learners and linear models see real-valued features.

| Name | What it is | Dimension | Storage dtype |
|------|-----------|-----------|---------------|
| `mordred` | Mordred 2D descriptors (`ignore_3D=True`) | 1613 | `float32` |
| `descriptors` | Alias of `mordred` | 1613 | `float32` |
| `rdkit_2d_descriptors` | RDKit 2D descriptors | 200 | `float32` |
| `estate` | EState electrotopological indices | 79 | `float32` |
| `ghose_crippen` | Ghose-Crippen LogP atomic contributions | 110 | `float32` |
| `mqns` | Molecular quantum numbers | 42 | `uint8` |
| `vsa` | Van der Waals surface area descriptors | 36 | `float32` |
| `bcut2d` | BCUT2D Burden-CAS-UT descriptors | 8 | `float32` |
| `physicochemical` | Physicochemical property count fingerprint | 2048 | `csr_uint16` |
| `pharmacophore` | Pharmacophore count fingerprint | 39,972 | `csr_uint16` |
| `functional_groups` | Functional group patterns | 85 | `float32` |
| `autocorr` | 2D autocorrelation descriptors | 192 | `float32` |

Dimensions were measured with `extract_features`. None requires conformers. Descriptors can produce NaN or infinite values; learners replace these with 0.0 and drop zero-variance columns during training.

## Usage

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer rdkit_2d_descriptors
```

```python
from learnm8 import extract_features

features = extract_features(['CCO', 'c1ccccc1'], 'estate')
```

Descriptors are also used as the extra features in Chemprop hybrid mode (`--learner chemprop --featurizer descriptors`); see the [featurizers overview](overview.md).

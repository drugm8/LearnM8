# Available Featurizers

LearnM8 provides 39 molecular featurizers (38 unique implementations; `mordred` and `descriptors` are aliases). They are organised into five categories based on their representation type.

## Quick Reference

| Category | Featurizers | Storage | Dimensions |
|----------|-------------|---------|------------|
| 2D Circular | `morgan`, `ecfp`, `ecfp6`, `morgan_feat`, `secfp` | `packed_uint8` | 2048 |
| 2D Keys | `maccs`, `pubchem`, `klekota_roth`, `laggner` | `packed_uint8` | 167 – 4860 |
| 2D Topological | `avalon`, `atom_pair`, `topological_torsion`, `rdkit`, `pattern`, `layered` | `packed_uint8` / `csr_uint16` | 512 – 2048 |
| 2D Hashed | `map4`, `mhfp`, `lingo`, `erg` | `packed_uint8` / `uint8` | 1024 – 2048 |
| 2D Descriptors | `mordred`, `descriptors`, `rdkit_2d_descriptors`, `estate`, `ghose_crippen`, `mqns`, `vsa`, `bcut2d`, `physiochemical`, `pharmacophore`, `functional_groups` | `float32` | 1 – 1826 |
| 3D (conformer) | `whim`, `usr`, `usrcat`, `e3fp`, `getaway`, `morse`, `rdf`, `autocorr`, `electroshape` | `float32` | 114 – 2048 |

**Storage notes:**

- `packed_uint8` — binary fingerprints stored with `np.packbits` (~32× space saving vs float32).
- `csr_uint16` — sparse integer count vectors (e.g., `atom_pair`, `topological_torsion` in count mode).
- `uint8` — small-range integer counts (e.g., ERG, MQNs sub-ranges).
- `float32` — continuous-valued descriptors and 3D features.
- Tree learners (RF, XGB, DT) can request `preferred_dtype='uint8'` to skip float32 inflation (4× working-set reduction on 2048-bit Morgan).

---

## 2D Circular Fingerprints

Circular (ECFP-family) fingerprints encode atom environments up to a given radius. They are the most widely validated molecular representations for property prediction.

| Name | Radius | Dimensions | Notes |
|------|--------|------------|-------|
| `morgan` | 2 | 2048 | Default; standard circular FP |
| `ecfp` | 2 | 2048 | Alias for Morgan-style ECFP |
| `ecfp6` | 3 | 2048 | ECFP6 (diameter=6); captures more distant contexts |
| `morgan_feat` | 2 | 2048 | Feature-based encoding (pharmacophore properties, not atom types) |
| `secfp` | 3 | 2048 | Spherical environment FP; includes SMILES-based substructures |

**Recommendation:** Start with `morgan`. Use `ecfp6` for large molecules (>30 heavy atoms) or long-range effects. Use `morgan_feat` for scaffold-hopping and pharmacophore tasks.

---

## 2D Key-Based Fingerprints

Predefined structural keys where each bit has an explicit chemical meaning. Smaller and faster than circular fingerprints; useful when interpretability matters.

| Name | Dimensions | Notes |
|------|------------|-------|
| `maccs` | 167 | MACCS structural keys; fastest, smallest, most interpretable |
| `pubchem` | 881 | PubChem substructure keys |
| `klekota_roth` | 4860 | Klekota–Roth keys; broad coverage of drug-like substructures |
| `laggner` | 307 | Laggner pharmaceutical keys |

**Recommendation:** Use `maccs` for very large libraries (>100k) where speed and memory are constraints.

---

## 2D Topological Fingerprints

Path- and topology-based fingerprints that encode connectivity patterns without explicit circular neighborhoods.

| Name | Dimensions | Storage | Notes |
|------|------------|---------|-------|
| `avalon` | 512 | `packed_uint8` | Avalon toolkit FP; fast and reliable |
| `atom_pair` | 2048 | `packed_uint8` / `csr_uint16` (count mode) | Encodes all atom-pair distances |
| `topological_torsion` | 2048 | `packed_uint8` / `csr_uint16` (count mode) | Four-atom torsion fragments |
| `rdkit` | 2048 | `packed_uint8` | RDKit path-based fingerprint |
| `pattern` | 2048 | `packed_uint8` | RDKit pattern fingerprint |
| `layered` | 2048 | `packed_uint8` | RDKit layered fingerprint |

---

## 2D Hashed Fingerprints

Alternative hashing schemes and MinHash-based representations, useful for large-scale similarity search and diversity-aware selection.

| Name | Dimensions | Storage | Notes |
|------|------------|---------|-------|
| `map4` | 2048 | `packed_uint8` | MAP4 (MinHashed atom-pair fingerprint, diameter=4) |
| `mhfp` | 2048 | `packed_uint8` | MinHashed fingerprint |
| `lingo` | 1024 | `packed_uint8` | LINGO substring fingerprint |
| `erg` | varies | `uint8` | Extended reduced graph (pharmacophore nodes) |

---

## 2D Descriptor-Based Featurizers

Numeric physicochemical descriptors; continuous-valued. These provide the richest representation but are slower to compute and higher-dimensional. All stored as `float32`.

| Name | Dimensions | Notes |
|------|------------|-------|
| `mordred` / `descriptors` | 1613 | Mordred descriptors (these two names are aliases); recommended for Chemprop hybrid mode |
| `rdkit_2d_descriptors` | 200 | RDKit standard 2D descriptors |
| `estate` | 79 | Estate electronegativity descriptors |
| `ghose_crippen` | 2 | LogP + MR (Ghose–Crippen atom contributions) |
| `mqns` | 42 | Molecular quantum numbers |
| `vsa` | 12 | Van der Waals surface area contributions |
| `bcut2d` | 8 | BCUT2D eigenvalue-based descriptors |
| `physiochemical` | varies | Physicochemical property descriptors |
| `pharmacophore` | varies | Pharmacophore feature counts |
| `functional_groups` | varies | Functional group counts |

**Note:** `mordred` and `descriptors` resolve to the same class (`MordredFingerprint`). Both names are accepted anywhere a featurizer string is used.

---

## 3D Fingerprints (Conformer-Based)

These featurizers require a 3D conformer, which LearnM8 generates automatically using RDKit ETKDG. They capture shape, volume, and 3D electronic distribution. All stored as `float32`.

Conformer generation uses `random_state=0xf00d` by default (RDKit ETKDG convention). The seed is recorded in the cache key, so changing it invalidates the existing 3D cache for those featurizers.

| Name | Dimensions | Notes |
|------|------------|-------|
| `whim` | 114 | WHIM shape descriptors |
| `usr` | 12 | Ultrafast shape recognition |
| `usrcat` | 60 | USR with pharmacophore categories |
| `e3fp` | 2048 | Extended 3D fingerprint (3D ECFP analogue) |
| `getaway` | 273 | GETAWAY surface area descriptors |
| `morse` | 224 | 3D-MoRSE descriptors |
| `rdf` | 210 | Radial distribution function descriptors |
| `autocorr` | 80 | 3D autocorrelation descriptors |
| `electroshape` | 15 | Electroshape (charge-weighted shape) |

**Requirements:** 3D featurizers require conformer generation. Invalid SMILES or molecules where conformers cannot be generated will return NaN vectors. Validate your compound pool first (`learnm8 validate`).

---

## Learner Compatibility

All featurizers are compatible with all learners **except**:

- **Chemprop / ChempropEnsemble**: No featurizer required (SMILES-native). Accepts `descriptors` or `mordred` for hybrid graph + descriptor mode.
- **Fastprop / FastpropEnsemble**: Requires a featurizer (works with feature vectors, not SMILES directly).

---

## Choosing a Featurizer

| Scenario | Recommended featurizer |
|----------|------------------------|
| General purpose | `morgan` |
| Very large library (>100k, speed critical) | `maccs` |
| Large molecules / long-range effects | `ecfp6` |
| Scaffold hopping, pharmacophore tasks | `morgan_feat` |
| QSAR with interpretable features | `mordred` / `descriptors` |
| Chemprop hybrid mode | `descriptors` |
| 3D shape similarity | `usr` or `usrcat` |
| 3D ECFP-like | `e3fp` |
| Tree learners, memory constrained | `morgan` with `preferred_dtype='uint8'` |

---

## Listing All Featurizers

```python
from learnm8.features import FEATURIZER_REGISTRY
print(sorted(FEATURIZER_REGISTRY))
```

```python
from learnm8.features import FEATURIZER_REGISTRY
print(sorted(FEATURIZER_REGISTRY))
```

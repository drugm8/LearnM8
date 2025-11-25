# LearnM8 Example Oracles

This directory contains example oracle implementations that demonstrate how to create custom scoring functions for active learning in molecular screening.

## Available Oracles

### 1. SimilarityOracle (2D Fingerprint-Based)

Calculates molecular similarity to a reference compound using various 2D fingerprints.

**Features:**
- Fast 2D-based calculations (no 3D conformer generation needed)
- Multiple fingerprint types: Morgan (ECFP), MACCS keys, Topological (RDK)
- Multiple similarity metrics: Tanimoto, Dice
- Configurable fingerprint parameters (radius, bit size)

**Example Usage:**

```python
import polars as pl
from learnm8.oracles.examples import SimilarityOracle

# Create oracle with Morgan fingerprints
oracle = SimilarityOracle(
    reference_smiles='c1ccccc1O',  # Reference molecule (phenol)
    fingerprint_type='morgan',      # Options: 'morgan', 'maccs', 'topological'
    metric='tanimoto',              # Options: 'tanimoto', 'dice'
    radius=2,                       # Morgan fingerprint radius
    n_bits=2048                     # Fingerprint size
)

# Measure similarity
compounds = pl.DataFrame({
    'ID': ['comp1', 'comp2', 'comp3'],
    'SMILES': ['c1ccccc1N', 'c1ccccc1', 'CCO']
})

results = oracle.measure(compounds, ['similarity'])
print(results)
```

**Use Cases:**
- Scaffold hopping
- Analog identification
- Library diversity assessment
- Quick screening for structural similarity

---

### 2. Pharmacophore2DOracle (2D Pharmacophore Fingerprints)

Calculates pharmacophore similarity using RDKit's Gobbi 2D pharmacophore fingerprints.

**Features:**
- Topological pharmacophore features (no 3D conformers needed)
- Uses standardized Gobbi 2D pharmacophore definition
- Captures functional group patterns and relationships
- Fast computation suitable for large libraries

**Example Usage:**

```python
from learnm8.oracles.examples import Pharmacophore2DOracle

# Create oracle with 2D pharmacophore fingerprints
oracle = Pharmacophore2DOracle(
    reference_smiles='c1ccc(O)cc1C(=O)O',  # Reference (salicylic acid)
    metric='tanimoto'                       # Options: 'tanimoto', 'dice'
)

# Measure pharmacophore similarity
results = oracle.measure(compounds, ['pharmacophore_similarity'])
print(results)
```

**Use Cases:**
- Functional group pattern matching
- Bioisostere identification
- Pharmacophore-based screening
- Finding molecules with similar chemical features

---

### 3. CDPKitPharmacophoreOracle (3D Pharmacophore Alignment)

Performs 3D pharmacophore generation and alignment using CDPKit.

**Features:**
- Full 3D pharmacophore analysis with spatial features
- 3D conformer generation using RDKit ETKDG
- CDPKit pharmacophore generation (H, AR, HBD, HBA, PI, NI features)
- Pharmacophore alignment scoring
- Multiple scoring schemes

**Dependencies:**
```bash
pip install cdpkit
```

**Example Usage:**

```python
from learnm8.oracles.examples import CDPKitPharmacophoreOracle

# Create oracle with 3D pharmacophore alignment
oracle = CDPKitPharmacophoreOracle(
    reference_smiles='c1ccc(O)cc1C(=O)O',
    score_type='alignment_score',      # Options: 'alignment_score', 'rmsd', 'feature_overlap'
    exhaustive=False,                   # Exhaustive conformer search
    min_feature_match=3,                # Minimum matching features
    max_conformers=10                   # Max conformers per molecule
)

# Measure 3D pharmacophore similarity
results = oracle.measure(compounds, ['ph4_3d_score'])
print(results)
```

**Score Types:**
- `alignment_score`: Feature match count (higher = better alignment)
- `feature_overlap`: Normalized by maximum features (0-1 range)
- `rmsd`: RMSD-based similarity score

**Use Cases:**
- 3D pharmacophore screening
- Structure-based ligand design
- Identifying molecules with similar spatial features
- Virtual screening with spatial constraints

**Performance Note:** 3D pharmacophore calculation is computationally expensive due to conformer generation. Consider using for focused libraries or as a refinement step after 2D filtering.

---

### 4. VinaOracle (Molecular Docking)

Performs molecular docking using AutoDock Vina to score protein-ligand binding affinity.

**Features:**
- AutoDock Vina scoring function
- Automatic 3D conformer generation
- PDBQT conversion via Meeko
- Configurable search box and exhaustiveness
- Returns best binding affinity (kcal/mol)

**Dependencies:**
```bash
conda install -c conda-forge vina meeko
```

**Example Usage:**

```python
from learnm8.oracles.examples import VinaOracle

# Create docking oracle
oracle = VinaOracle(
    receptor_path='receptor.pdbqt',         # Prepared receptor file
    center=(10.0, 15.0, 20.0),              # Search box center (x, y, z)
    box_size=(20.0, 20.0, 20.0),            # Search box size (Å)
    exhaustiveness=8,                        # Search exhaustiveness (higher = more thorough)
    n_poses=1,                              # Number of docking poses to return
    energy_range=3.0                        # Energy range for multiple poses
)

# Perform docking
results = oracle.measure(compounds, ['binding_affinity'])
print(results)

# Lower (more negative) binding affinity = stronger binding
```

**Use Cases:**
- Structure-based virtual screening
- Lead optimization with known protein target
- Binding affinity prediction
- Structure-activity relationship (SAR) studies

**Performance Note:** Docking is computationally intensive (seconds per molecule). Best suited for:
- Final screening of prioritized compounds
- Small focused libraries (<1000 compounds)
- High-value predictions where accuracy justifies cost

**Receptor Preparation:**
```python
# Receptor must be in PDBQT format
# Prepare using AutoDock Tools or similar:
# 1. Remove water molecules
# 2. Add hydrogens
# 3. Assign charges
# 4. Convert to PDBQT format
```

---

## Integration with LearnM8

### Using Oracles in Active Learning

```python
from learnm8 import run_active_learning
from learnm8.oracles.examples import SimilarityOracle
import polars as pl

# Load compound library
compounds = pl.read_csv('compound_library.csv')

# Create oracle
oracle = SimilarityOracle(
    reference_smiles='c1ccccc1O',
    fingerprint_type='morgan'
)

# Run active learning
results = run_active_learning(
    compound_pool=compounds,
    oracle=oracle,
    target_col='similarity',           # Property name from oracle.measure()
    learner='gp',                      # Gaussian Process learner
    featurizer='morgan',          # Feature type for ML model
    cycles=[
        ('random', 0.01),              # 1% random initial sampling
        ('ucb', 0.005),                # 0.5% exploitation (5 cycles)
        ('diverse', 0.01)              # 1% final diversity
    ],
    export_csv=True
)
```

### Combining Multiple Oracles

```python
from learnm8.oracles.examples import SimilarityOracle, Pharmacophore2DOracle
import polars as pl

# Create multiple oracles
similarity_oracle = SimilarityOracle(
    reference_smiles='c1ccccc1O',
    fingerprint_type='morgan'
)

pharmacophore_oracle = Pharmacophore2DOracle(
    reference_smiles='c1ccccc1O'
)

# Measure with first oracle
compounds_sim = similarity_oracle.measure(compounds, ['similarity'])

# Measure with second oracle
compounds_ph4 = pharmacophore_oracle.measure(compounds, ['pharmacophore'])

# Combine scores
combined = compounds_sim.join(
    compounds_ph4.select(['ID', 'pharmacophore']),
    on='ID'
).with_columns(
    ((pl.col('similarity') + pl.col('pharmacophore')) / 2).alias('combined_score')
)
```

### Custom Weighted Scoring

```python
# Create custom oracle with weighted scoring
class WeightedOracle:
    def __init__(self, oracles_and_weights):
        self.oracles_and_weights = oracles_and_weights

    def measure(self, compounds, properties):
        results = compounds.select('ID')

        for prop in properties:
            weighted_scores = []

            for oracle, weight in self.oracles_and_weights:
                scores = oracle.measure(compounds, [prop])
                weighted_scores.append(scores[prop] * weight)

            # Combine weighted scores
            combined = sum(weighted_scores)
            results = results.with_columns(combined.alias(prop))

        return results

# Use weighted oracle
weighted_oracle = WeightedOracle([
    (SimilarityOracle('c1ccccc1O', fingerprint_type='morgan'), 0.6),
    (Pharmacophore2DOracle('c1ccccc1O'), 0.4)
])
```

---

## Performance Considerations

### Computational Cost (per molecule)

| Oracle | Speed | Use Case |
|--------|-------|----------|
| **SimilarityOracle** | ~0.001s | Large libraries (>100K compounds) |
| **Pharmacophore2DOracle** | ~0.01s | Medium libraries (10K-100K) |
| **CDPKitPharmacophoreOracle** | ~1-5s | Focused libraries (<10K) |
| **VinaOracle** | ~10-60s | Final screening (<1K compounds) |

### Optimization Strategies

1. **Hierarchical Screening:**
   ```python
   # Stage 1: Fast 2D similarity filter (100K → 10K)
   sim_oracle = SimilarityOracle('c1ccccc1O', fingerprint_type='morgan')

   # Stage 2: 2D pharmacophore refinement (10K → 1K)
   ph4_2d_oracle = Pharmacophore2DOracle('c1ccccc1O')

   # Stage 3: 3D pharmacophore validation (1K → 100)
   ph4_3d_oracle = CDPKitPharmacophoreOracle('c1ccccc1O')

   # Stage 4: Docking for top hits (100 → 10)
   docking_oracle = VinaOracle('receptor.pdbqt', center=(10, 15, 20))
   ```

2. **Parallel Processing:**
   - Split large libraries into batches
   - Process batches in parallel
   - Merge results

3. **Caching:**
   - Cache computed fingerprints for reuse
   - Save 3D conformers for repeated docking
   - Store intermediate results

---

## Testing

Run tests with the oracles environment:

```bash
# Activate oracles environment
conda activate learnm8-oracles

# Run all oracle tests
pytest tests/oracles/test_example_oracles.py -v

# Run specific oracle test
pytest tests/oracles/test_example_oracles.py::TestSimilarityOracle -v

# Run with coverage
pytest tests/oracles/ --cov=learnm8.oracles.examples --cov-report=html
```

---

## Troubleshooting

### Import Errors

```python
# Check which oracles are available
from learnm8.oracles.examples import (
    SimilarityOracle,        # Always available (RDKit only)
    Pharmacophore2DOracle,   # Always available (RDKit only)
    CDPKitPharmacophoreOracle,  # Requires: pip install cdpkit
    VinaOracle               # Requires: conda install -c conda-forge vina meeko
)
```

### Invalid SMILES Handling

All oracles gracefully handle invalid SMILES by returning `None`:

```python
compounds = pl.DataFrame({
    'ID': ['good', 'bad', 'ugly'],
    'SMILES': ['CCO', 'INVALID', None]
})

results = oracle.measure(compounds, ['score'])
# Returns: [0.95, None, None]
```

### Memory Issues with Large Libraries

```python
# Process in batches
batch_size = 1000
all_results = []

for i in range(0, len(compounds), batch_size):
    batch = compounds[i:i+batch_size]
    batch_results = oracle.measure(batch, ['score'])
    all_results.append(batch_results)

final_results = pl.concat(all_results)
```

---

## Creating Custom Oracles

Extend the `Oracle` interface from `learnm8.core.interfaces`:

```python
from learnm8.core.interfaces import Oracle
import polars as pl
from typing import List

class MyCustomOracle(Oracle):
    def __init__(self, **config):
        self.config = config
        # Initialize your scoring system

    def measure(self, compounds: pl.DataFrame, properties: List[str]) -> pl.DataFrame:
        """
        Measure properties for compounds.

        Args:
            compounds: Polars DataFrame with 'ID' and 'SMILES' columns
            properties: List of property names to measure

        Returns:
            Polars DataFrame with 'ID' column and measured property columns
            MUST preserve input row order
        """
        # Validate input
        if 'SMILES' not in compounds.columns or 'ID' not in compounds.columns:
            raise ValueError("Compounds must have 'ID' and 'SMILES' columns")

        # Compute scores
        scores = []
        for smiles in compounds['SMILES'].to_list():
            score = self._calculate_score(smiles)
            scores.append(score)

        # Build result preserving order
        result = compounds.select('ID').with_row_index('_order')

        for prop in properties:
            result = result.with_columns(
                pl.Series(name=prop, values=scores)
            )

        return result.sort('_order').drop('_order')

    def _calculate_score(self, smiles: str):
        # Your scoring implementation
        pass
```

---

## References

- **RDKit Documentation**: https://www.rdkit.org/docs/
- **CDPKit Documentation**: https://cdpkit.org/
- **AutoDock Vina**: https://vina.scripps.edu/
- **Meeko Documentation**: https://github.com/forlilab/Meeko

---

## Citation

If you use these oracles in your research, please cite:

```bibtex
@software{learnm8_oracles,
  title = {LearnM8 Example Oracles},
  author = {LearnM8 Development Team},
  year = {2025},
  url = {https://github.com/volkamerlab/LearnM8}
}
```

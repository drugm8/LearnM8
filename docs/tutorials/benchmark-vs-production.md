# Benchmark vs Production Modes

LearnM8 operates in two distinct modes optimized for different use cases: **Benchmark Mode** for algorithm testing and **Production Mode** for real screening campaigns.

## Benchmark Mode

### What is Benchmark Mode

Benchmark mode uses pre-computed ground truth data from a CSV file to evaluate active learning performance. The oracle predicts on the **entire dataset** each cycle to calculate correct enrichment metrics.

**Key characteristics:**

- Ground truth available for all compounds
- Full dataset prediction for accurate metrics
- Used for algorithm comparison and validation
- Higher computational cost (predicts all compounds)

### When to Use Benchmark Mode

- Testing active learning strategies
- Comparing acquisition functions
- Validating model performance
- Published datasets (ESSENCE, MoleculeNet)
- Algorithm research and development

### Using CSVOracle

The CSVOracle reads ground truth from a CSV file containing compound IDs and measured properties.

**Python API Example:**

```python
from learnm8 import run_active_learning
from learnm8.oracles import CSVOracle

oracle = CSVOracle('ground_truth.csv')

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=oracle,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01
)

print(f"Final enrichment: {results['aggregate_metrics']['enrichment_factor_1']:.2f}")
```

**CLI Alternative:**

```bash
learnm8 run compounds.csv \
  --target Activity \
  --learner gp \
  --featurizer morgan \
  --n-cycles 10 \
  --batch-fraction 0.01
```

When `compounds.csv` contains the target column, LearnM8 auto-detects benchmark mode.

### ESSENCE Benchmark Example

The ESSENCE benchmark dataset provides validated ground truth for testing active learning algorithms.

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='ESSENCE_benchmark_input/ADA.csv',
    oracle=None,  # Auto-detect from compound pool
    learner='ensemble',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=15,
    batch_fraction=0.01
)
```

**CLI alternative:**

```bash
learnm8 run ESSENCE_benchmark_input/ADA.csv \
  --target Activity \
  --learner ensemble \
  --featurizer morgan \
  --n-cycles 15 \
  --batch-fraction 0.01
```

**CSV Structure:**

```csv
ID,SMILES,Activity
comp1,CCO,0.8
comp2,CCC,0.6
comp3,CCCO,0.9
```

## Production Mode

### What is Production Mode

Production mode uses custom scoring functions for expensive measurements (docking, synthesis, assays). The oracle predicts only **unlabeled compounds** to minimize computational cost.

**Key characteristics:**

- No ground truth available upfront
- Unlabeled pool prediction only
- Custom oracle functions (docking, ML models, experiments)
- Lower computational cost (smaller prediction set)

### When to Use Production Mode

- Real drug discovery campaigns
- Virtual screening with docking
- Expensive experimental measurements
- Custom property prediction
- Production deployment

### Using PythonOracle

PythonOracle executes user-defined scoring functions for compound evaluation.

**Oracle Function Interface:**

```python
# scoring_module.py
from typing import List
import numpy as np

def calculate_binding_affinity(compound_ids: List[str]) -> dict:
    """Score compounds using custom logic.

    Args:
        compound_ids: List of compound IDs to score

    Returns:
        Dictionary with 'ID' and property columns
    """
    import polars as pl

    scores = []
    for cid in compound_ids:
        score = your_scoring_logic(cid)
        scores.append(score)

    return pl.DataFrame({
        'ID': compound_ids,
        'binding_affinity': scores
    })
```

**Python API Example:**

```python
from learnm8 import run_active_learning
from learnm8.oracles import PythonOracle

oracle = PythonOracle(
    module_path='scoring_module.py',
    function_name='calculate_binding_affinity'
)

results = run_active_learning(
    compound_pool='compound_library.csv',
    oracle=oracle,
    learner='mc_dropout',
    target_col='binding_affinity',
    featurizer='morgan',
    n_cycles=20,
    batch_fraction=0.005
)
```

**CLI Alternative:**

```bash
learnm8 run compound_library.csv scoring_module.py:calculate_binding_affinity \
  --target binding_affinity \
  --learner mc_dropout \
  --featurizer morgan \
  --n-cycles 20 \
  --batch-fraction 0.005
```

### Custom Oracle Example: Docking

```python
# docking_oracle.py
import polars as pl
from typing import List
from rdkit import Chem
from vina import Vina

def dock_compounds(compound_ids: List[str]) -> pl.DataFrame:
    """Score compounds using AutoDock Vina.

    Args:
        compound_ids: Compound IDs from master DataFrame

    Returns:
        DataFrame with docking scores
    """
    compound_data = load_compound_data()

    scores = []
    for cid in compound_ids:
        smiles = compound_data[cid]['SMILES']
        mol = Chem.MolFromSmiles(smiles)

        score = run_docking(mol, receptor='receptor.pdbqt')
        scores.append(score)

    return pl.DataFrame({
        'ID': compound_ids,
        'docking_score': scores
    })

def run_docking(mol, receptor):
    v = Vina(sf_name='vina')
    v.set_receptor(receptor)
    v.compute_vina_maps(center=[15, 53, 16], box_size=[20, 20, 20])

    v.set_ligand_from_string(Chem.MolToMolBlock(mol))
    v.optimize()

    energy = v.score()[0]
    return energy
```

**Using the docking oracle:**

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compound_library.csv',
    oracle='docking_oracle.py:dock_compounds',
    learner='ensemble',
    target_col='docking_score',
    featurizer='morgan',
    score_direction='lower',
    n_cycles=25
)
```

**CLI alternative:**

```bash
learnm8 run compound_library.csv docking_oracle.py:dock_compounds \
  --target docking_score \
  --learner ensemble \
  --featurizer morgan \
  --score-direction lower \
  --n-cycles 25
```

## Mode Auto-Detection

LearnM8 automatically detects the appropriate mode based on oracle type:

**Auto-detected as Benchmark:**

- CSVOracle instance
- Compound pool CSV contains target column
- Same file used for pool and oracle

**Auto-detected as Production:**

- PythonOracle instance
- Separate oracle specification
- `module.py:function` syntax

### Overriding Auto-Detection

Explicitly specify mode when auto-detection is ambiguous:

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=oracle,
    mode='production',
    learner='gp',
    target_col='Activity',
    featurizer='morgan'
)
```

## Performance Differences

### Computational Cost Comparison

| Aspect               | Benchmark Mode               | Production Mode      |
| -------------------- | ---------------------------- | -------------------- |
| **Prediction Scope** | Full dataset (all compounds) | Unlabeled pool only  |
| **Oracle Calls**     | Once per cycle               | Once per cycle       |
| **Prediction Cost**  | High (N compounds)           | Low (unlabeled only) |
| **Metric Accuracy**  | Exact enrichment             | Approximate          |
| **Use Case**         | Algorithm testing            | Real screening       |

**Example Dataset: 100,000 compounds, 10 cycles, 1% batch**

| Cycle | Benchmark Predictions | Production Predictions |
| ----- | --------------------- | ---------------------- |
| 0     | 100,000               | 99,000                 |
| 1     | 100,000               | 98,000                 |
| 2     | 100,000               | 97,000                 |
| 5     | 100,000               | 94,000                 |
| 10    | 100,000               | 90,000                 |

**Total predictions:**

- Benchmark: 1,100,000
- Production: 950,000 (14% reduction)

### When Full Prediction Matters

**Benchmark mode required for:**

- Accurate enrichment factor calculation
- Hit rate validation
- Coverage metrics
- Algorithm comparison studies

**Production mode sufficient for:**

- Real screening campaigns
- Acquisition function testing
- Model performance evaluation
- Computational efficiency priority

## Best Practices

### Benchmark Mode

1. **Use consistent datasets**: Standard benchmarks (ESSENCE, MoleculeNet) for reproducibility
2. **Include ground truth**: All compounds must have measured values
3. **Compare strategies**: Test multiple acquisition functions and learners
4. **Report metrics**: Include enrichment factors, hit rates, AUC-PR

### Production Mode

1. **Validate oracle**: Test scoring function on small subset before full run
2. **Handle failures**: Implement error handling for measurement failures
3. **Cache results**: Store expensive oracle evaluations
4. **Monitor progress**: Track acquisition strategy performance without ground truth

### Transitioning Between Modes

Start with benchmark mode for algorithm selection, then deploy in production mode:

```python
from learnm8 import run_active_learning, CycleConfig

# 1. Test strategies on benchmark
benchmark_results = run_active_learning(
    compound_pool='ESSENCE_benchmark_input/ADA.csv',
    oracle=None,
    learner='ensemble',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('ucb', n_cycles=8, batch_fraction=0.01),
        CycleConfig('simulated_annealing', n_cycles=1, batch_fraction=0.01)
    ]
)

# 2. Deploy best strategy in production
production_results = run_active_learning(
    compound_pool='screening_library.csv',
    oracle='docking_oracle.py:score',
    learner='ensemble',
    target_col='affinity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('ucb', n_cycles=8, batch_fraction=0.01),
        CycleConfig('simulated_annealing', n_cycles=1, batch_fraction=0.01)
    ]
)
```

**CLI alternative:**

```bash
# 1. Test strategies on benchmark
learnm8 run ESSENCE_benchmark_input/ADA.csv \
  --target Activity \
  --learner ensemble \
  --featurizer morgan \
  --cycles "random:0.02 ucb:0.01*8 simulated_annealing:0.01"

# 2. Deploy best strategy in production
learnm8 run screening_library.csv docking_oracle.py:score \
  --target affinity \
  --learner ensemble \
  --featurizer morgan \
  --cycles "random:0.02 ucb:0.01*8 simulated_annealing:0.01"
```

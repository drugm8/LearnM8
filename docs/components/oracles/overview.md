# Oracles Overview

Oracles are the measurement and scoring functions in LearnM8 that provide ground truth values for molecular properties. They serve as the interface between the active learning framework and property evaluation methods, whether experimental, computational, or database-driven.

## What are Oracles?

In the context of active learning, an oracle is a function that can measure or score compounds to provide ground truth labels. LearnM8 supports two oracle types designed for different scenarios:

- **CSVOracle**: For benchmark mode with pre-computed ground truth data
- **PythonOracle**: For production mode with custom scoring functions

The oracle's role in the active learning cycle is to measure the properties of compounds selected by the acquisition function, providing feedback that trains the learner for the next cycle.

## Oracle Interface

All oracles implement the `Oracle` protocol defined in `learnm8.core.interfaces`:

```python
from abc import ABC, abstractmethod
from typing import List
import polars as pl

class Oracle(ABC):
    @abstractmethod
    def measure(self, compounds: pl.DataFrame, properties: List[str]) -> pl.DataFrame:
        """
        Measure properties for given compounds.

        Args:
            compounds: Polars DataFrame with 'ID' and 'SMILES' columns
            properties: List of property names to measure

        Returns:
            Polars DataFrame with 'ID' column and measured property columns
        """
        pass
```

**Key Requirements:**
- Accept a Polars DataFrame with `ID` and `SMILES` columns
- Return a DataFrame with `ID` and requested property columns
- Preserve input row order for correct alignment
- Handle measurement failures gracefully

## CSVOracle (Benchmark Mode)

CSVOracle reads ground truth data from a CSV file, making it ideal for benchmarking active learning strategies against known results.

### When to Use CSVOracle

- **Algorithm testing**: Compare different learners and acquisition strategies
- **Method validation**: Verify active learning approach on historical data
- **Performance benchmarking**: Establish baselines before production deployment
- **Research**: Evaluate novel strategies with reproducible ground truth

### How CSVOracle Works

CSVOracle loads a complete dataset with known property values and performs lookups as compounds are selected during active learning cycles. In benchmark mode, LearnM8 predicts on the full dataset each cycle to correctly calculate enrichment factors and other metrics.

### Basic Usage

```python
from learnm8.oracles import CSVOracle

oracle = CSVOracle('ground_truth.csv')
```

The CSV file should contain:
- An `ID` column (or specify custom ID column name)
- A `SMILES` column
- One or more property columns

```csv
ID,SMILES,Activity,Solubility
comp1,CCO,0.85,2.3
comp2,CCC,0.42,1.8
comp3,CCCC,0.91,1.2
```

### CLI Usage

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10
```

When the oracle parameter is omitted and `compound_pool` is a CSV file, LearnM8 automatically creates a CSVOracle from that file (benchmark mode).

Explicit oracle specification:

```bash
learnm8 run compounds.csv oracle.csv --target Activity --learner rf --featurizer morgan
```

### API Usage

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
    n_cycles=10
)
```

### Custom ID Column

If your CSV uses a different ID column name:

```python
oracle = CSVOracle('data.csv', id_column='compound_id')
```

## PythonOracle (Production Mode)

PythonOracle executes custom Python functions to score compounds, enabling integration with computational tools, web APIs, or experimental measurements.

### When to Use PythonOracle

- **Production screening**: Real-world compound selection campaigns
- **Computational scoring**: Integration with docking software, QM calculations
- **External APIs**: Web-based property prediction services
- **Custom workflows**: Specialized measurement pipelines

### How PythonOracle Works

PythonOracle loads a Python module and executes a user-defined function that accepts compound IDs and returns scores. In production mode, LearnM8 predicts only on unlabeled compounds to minimize computational cost, since there is no complete ground truth dataset.

### Basic Usage

Create a Python file with your scoring function:

```python
# scoring.py
def calculate_affinity(compound_ids):
    """Calculate binding affinity for compounds."""
    import pandas as pd

    scores = []
    for compound_id in compound_ids:
        # Your scoring logic here
        score = your_scoring_function(compound_id)
        scores.append(score)

    return pd.DataFrame({
        'ID': compound_ids,
        'binding_affinity': scores
    })
```

Use with LearnM8:

```python
from learnm8.oracles import PythonOracle

oracle = PythonOracle('scoring.py', 'calculate_affinity')
```

### CLI Usage

Specify the oracle using `module.py:function` syntax:

```bash
learnm8 run compounds.csv scoring.py:calculate_affinity --target binding_affinity \
  --learner ensemble --featurizer morgan --n-cycles 20
```

### API Usage

```python
from learnm8 import run_active_learning
from learnm8.oracles import PythonOracle

oracle = PythonOracle('scoring.py', 'calculate_affinity')

results = run_active_learning(
    compound_pool='compound_library.csv',
    oracle=oracle,
    learner='mc_dropout',
    target_col='binding_affinity',
    featurizer='morgan',
    n_cycles=20
)
```

## Mode Auto-Detection

LearnM8 automatically detects the operating mode based on the oracle type:

| Oracle Type | Mode | Prediction Scope | Use Case |
|-------------|------|------------------|----------|
| CSVOracle | benchmark | Full dataset | Algorithm testing, validation |
| PythonOracle | run | Unlabeled pool only | Production screening |
| Custom Oracle | run (default) | Unlabeled pool only | Custom workflows |

### Automatic Detection

```python
# CSV file → benchmark mode
results = run_active_learning(
    compound_pool='data.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan'
)

# Python function → run mode
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='scoring.py:score',
    learner='rf',
    target_col='score',
    featurizer='morgan'
)
```

### Manual Override

Override auto-detection with the `mode` parameter:

```python
results = run_active_learning(
    compound_pool='data.csv',
    oracle=oracle,
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    mode='benchmark'  # Force benchmark mode
)
```

Valid mode values: `'benchmark'`, `'run'`

## Benchmark vs Production Differences

### Benchmark Mode (CSVOracle)

**Characteristics:**
- Complete ground truth dataset available
- Full dataset prediction each cycle
- Accurate enrichment factor calculation
- Reproducible results for testing

**Performance:**
- Higher computational cost (predicting all compounds)
- Suitable for datasets up to ~100k compounds
- Focus on algorithm validation over speed

**Metrics:**
- Accurate enrichment factors across all cycles
- Complete ROC/PR curves
- Full dataset coverage statistics

### Production Mode (PythonOracle)

**Characteristics:**
- No complete ground truth (measurements are expensive)
- Predict only unlabeled compounds
- Optimized for computational efficiency
- Real-world screening scenarios

**Performance:**
- Lower computational cost (predicting subset)
- Suitable for large compound libraries (100k+)
- Focus on practical screening efficiency

**Metrics:**
- Approximate enrichment (based on labeled subset)
- Cumulative performance tracking
- Cost-benefit analysis

## Common Oracle Patterns

### Database Lookup

```python
from learnm8.core.interfaces import Oracle
import polars as pl
import sqlite3

class DatabaseOracle(Oracle):
    def __init__(self, db_path, table_name):
        self.conn = sqlite3.connect(db_path)
        self.table_name = table_name

    def measure(self, compounds, properties):
        compound_ids = compounds['ID'].to_list()
        placeholders = ','.join(['?' for _ in compound_ids])

        query = f"SELECT ID, {','.join(properties)} FROM {self.table_name} WHERE ID IN ({placeholders})"

        result = pl.read_database(query, self.conn, params=compound_ids)
        return result
```

### Computational Docking

```python
class DockingOracle(Oracle):
    def __init__(self, protein_file, docking_config):
        self.protein = protein_file
        self.config = docking_config

    def measure(self, compounds, properties):
        results = {'ID': []}
        for prop in properties:
            results[prop] = []

        for row in compounds.iter_rows(named=True):
            compound_id = row['ID']
            smiles = row['SMILES']

            # Run docking simulation
            score = self._dock_compound(smiles)

            results['ID'].append(compound_id)
            results['score'].append(score)

        return pl.DataFrame(results)

    def _dock_compound(self, smiles):
        # Integration with docking software
        pass
```

### Cached Measurements

```python
class CachedOracle(Oracle):
    def __init__(self, base_oracle, cache_file='oracle_cache.csv'):
        self.base_oracle = base_oracle
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def measure(self, compounds, properties):
        # Split into cached and uncached
        cached_ids = set(self.cache['ID'].to_list())
        compound_ids = set(compounds['ID'].to_list())

        uncached_ids = compound_ids - cached_ids

        if uncached_ids:
            # Measure uncached compounds
            uncached_compounds = compounds.filter(pl.col('ID').is_in(list(uncached_ids)))
            new_results = self.base_oracle.measure(uncached_compounds, properties)

            # Update cache
            self.cache = pl.concat([self.cache, new_results])
            self._save_cache()

        # Return results from cache
        return self.cache.filter(pl.col('ID').is_in(list(compound_ids)))
```

## Best Practices

**Input Validation:**
- Validate SMILES before expensive measurements
- Handle missing or malformed compound data gracefully
- Check for duplicate IDs

**Error Handling:**
- Return NaN for failed measurements rather than raising exceptions
- Log measurement failures for debugging
- Provide meaningful error messages

**Performance:**
- Implement caching for expensive measurements
- Use batch processing when possible
- Consider rate limits for API-based oracles

**Reproducibility:**
- Use fixed random seeds for stochastic scoring
- Document oracle configuration and parameters
- Version control oracle code alongside experiments

## Next Steps

- Learn how to implement custom oracles: [Custom Oracles](custom-oracles.md)
- Understand benchmark vs production workflows: [Benchmark vs Production](../../tutorials/benchmark-vs-production.md)
- Explore acquisition strategies: [Acquisition Functions](../acquisition/overview.md)

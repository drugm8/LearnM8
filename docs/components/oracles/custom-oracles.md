# Custom Oracles

This guide shows how to implement custom oracles for integrating LearnM8 with your own scoring functions, computational tools, experimental measurements, or external services.

## Writing a Python Oracle Function

The simplest way to create a custom oracle is to write a Python function that scores compounds. PythonOracle executes your function during active learning cycles.

### Function Signature

Your oracle function must accept a list of compound IDs and return a pandas or Polars DataFrame:

```python
def my_oracle_function(compound_ids: List[str]) -> pd.DataFrame:
    """
    Score compounds for active learning.

    Args:
        compound_ids: List of compound identifiers

    Returns:
        DataFrame with 'ID' column and score columns
    """
    pass
```

**Requirements:**
- **Input**: List of compound ID strings
- **Output**: DataFrame with `ID` column plus one or more score columns
- **Return type**: pandas DataFrame or Polars DataFrame

### Simple Scoring Function Example

```python
# simple_scorer.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def molecular_weight_scorer(compound_ids):
    """Score compounds based on molecular weight preference."""
    scores = []

    for compound_id in compound_ids:
        # Load compound (assuming ID is SMILES for this example)
        smiles = compound_id
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # Handle invalid SMILES
            scores.append({
                'ID': compound_id,
                'mw_score': 0.0
            })
            continue

        # Calculate molecular weight
        mw = Descriptors.MolWt(mol)

        # Score: prefer MW around 400 (drug-like)
        score = 1.0 / (1.0 + abs(mw - 400) / 100)

        scores.append({
            'ID': compound_id,
            'mw_score': score
        })

    return pd.DataFrame(scores)
```

Use this function with LearnM8:

```bash
learnm8 run compounds.csv simple_scorer.py:molecular_weight_scorer \
  --target mw_score --learner rf --featurizer morgan
```

### Multi-Property Scoring

Return multiple scores in a single function:

```python
# multi_property_scorer.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def drug_likeness_scorer(compound_ids):
    """Score compounds on multiple drug-likeness criteria."""
    results = []

    for compound_id in compound_ids:
        smiles = compound_id
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            results.append({
                'ID': compound_id,
                'mw_score': 0.0,
                'logp_score': 0.0,
                'hbd_score': 0.0,
                'composite_score': 0.0
            })
            continue

        # Calculate properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)

        # Individual scores
        mw_score = 1.0 if 200 <= mw <= 500 else 0.0
        logp_score = 1.0 if -0.5 <= logp <= 5.0 else 0.0
        hbd_score = 1.0 if hbd <= 5 else 0.0

        # Composite score
        composite = (mw_score + logp_score + hbd_score) / 3

        results.append({
            'ID': compound_id,
            'mw_score': mw_score,
            'logp_score': logp_score,
            'hbd_score': hbd_score,
            'composite_score': composite
        })

    return pd.DataFrame(results)
```

## Module Structure

For more complex oracles, organize your code into well-structured modules.

### Single-File Module

```python
# docking_scorer.py
"""Molecular docking scoring oracle for LearnM8."""

import pandas as pd
from pathlib import Path
from typing import List

# Module configuration
PROTEIN_FILE = "protein.pdb"
DOCKING_CONFIG = {
    'exhaustiveness': 8,
    'num_modes': 9
}

def docking_scorer(compound_ids: List[str]) -> pd.DataFrame:
    """Score compounds using molecular docking."""
    results = []

    for compound_id in compound_ids:
        try:
            score = _run_docking(compound_id)
            results.append({
                'ID': compound_id,
                'docking_score': score
            })
        except Exception as e:
            print(f"Warning: Docking failed for {compound_id}: {e}")
            results.append({
                'ID': compound_id,
                'docking_score': float('nan')
            })

    return pd.DataFrame(results)

def _run_docking(compound_id: str) -> float:
    """Execute docking simulation."""
    # Your docking implementation
    # This could call external tools like AutoDock Vina
    pass

def _prepare_ligand(smiles: str) -> Path:
    """Prepare ligand structure for docking."""
    pass

def _parse_docking_results(results_file: Path) -> float:
    """Extract best score from docking results."""
    pass
```

### Multi-File Module

For complex workflows, use a package structure:

```
my_oracle/
├── __init__.py
├── scorer.py
├── preprocessing.py
├── config.py
└── utils.py
```

```python
# my_oracle/__init__.py
from .scorer import comprehensive_scorer

__all__ = ['comprehensive_scorer']
```

```python
# my_oracle/scorer.py
import pandas as pd
from .preprocessing import prepare_compounds
from .config import SCORING_CONFIG
from .utils import calculate_properties

def comprehensive_scorer(compound_ids):
    """Main oracle function."""
    # Preprocess compounds
    prepared = prepare_compounds(compound_ids)

    # Calculate scores
    scores = []
    for compound_id, data in prepared.items():
        properties = calculate_properties(data)
        score = _aggregate_scores(properties)
        scores.append({'ID': compound_id, 'score': score})

    return pd.DataFrame(scores)

def _aggregate_scores(properties):
    """Combine multiple properties into final score."""
    pass
```

## Using Python Oracles

### CLI Usage

Specify oracle using `module.py:function` syntax:

```bash
# Single file
learnm8 run compounds.csv scorer.py:my_function --target score --learner gp

# Package
learnm8 run compounds.csv my_oracle:comprehensive_scorer --target score --learner rf
```

### API Usage

```python
from learnm8 import run_active_learning
from learnm8.oracles import PythonOracle

# Create oracle instance
oracle = PythonOracle('scorer.py', 'my_function')

# Run active learning
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=oracle,
    learner='gp',
    target_col='score',
    featurizer_type='morgan',
    n_cycles=20
)
```

### Auto-Detection

If your module contains only one function, PythonOracle auto-detects it:

```python
# No need to specify function name
oracle = PythonOracle('scorer.py')
```

For modules with multiple functions, use common names (`oracle`, `oracle_function`, `measure`, `evaluate`) or explicitly specify:

```python
oracle = PythonOracle('scorer.py', 'my_oracle_function')
```

## Advanced: Oracle Class

For oracles requiring state management, configuration, or complex initialization, implement the `Oracle` protocol as a class.

### When to Use Oracle Classes

- **Persistent connections**: Database connections, API clients
- **Expensive initialization**: Loading large models, preparing data
- **State management**: Caching, request tracking
- **Complex configuration**: Multiple parameters, validation

### Implementing the Oracle Protocol

```python
# advanced_oracle.py
from learnm8.core.interfaces import Oracle
import polars as pl
from typing import List
import requests

class WebAPIOracle(Oracle):
    """Oracle that queries a web API for compound scores."""

    def __init__(self, api_url: str, api_key: str, timeout: int = 30):
        """
        Initialize web API oracle.

        Args:
            api_url: Base URL for the scoring API
            api_key: Authentication key
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout

        # Create session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })

        # Validate connection
        self._validate_connection()

    def measure(self, compounds: pl.DataFrame, properties: List[str]) -> pl.DataFrame:
        """
        Query API for compound scores.

        Args:
            compounds: DataFrame with 'ID' and 'SMILES' columns
            properties: Property names (not used by this API)

        Returns:
            DataFrame with 'ID' and 'score' columns
        """
        results = []

        for row in compounds.iter_rows(named=True):
            compound_id = row['ID']
            smiles = row['SMILES']

            try:
                score = self._query_api(smiles)
                results.append({
                    'ID': compound_id,
                    'score': score
                })
            except Exception as e:
                print(f"Warning: API query failed for {compound_id}: {e}")
                results.append({
                    'ID': compound_id,
                    'score': float('nan')
                })

        return pl.DataFrame(results)

    def _query_api(self, smiles: str) -> float:
        """Query API for single compound."""
        payload = {'smiles': smiles}
        response = self.session.post(
            f"{self.api_url}/score",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()['score']

    def _validate_connection(self):
        """Verify API is accessible."""
        try:
            response = self.session.get(
                f"{self.api_url}/health",
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to API: {e}")

    def __del__(self):
        """Clean up session on deletion."""
        if hasattr(self, 'session'):
            self.session.close()
```

Use the class-based oracle:

```python
from advanced_oracle import WebAPIOracle

oracle = WebAPIOracle(
    api_url='https://api.example.com',
    api_key='your-api-key',
    timeout=60
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=oracle,
    learner='ensemble',
    target_col='score',
    featurizer_type='morgan'
)
```

### State Management Example

Oracle with caching and request tracking:

```python
from learnm8.core.interfaces import Oracle
import polars as pl
from typing import List, Dict
import sqlite3
import hashlib

class CachedDockingOracle(Oracle):
    """Docking oracle with persistent caching."""

    def __init__(self, protein_file: str, cache_db: str = 'docking_cache.db'):
        self.protein_file = protein_file
        self.cache_db = cache_db

        # Initialize cache database
        self.conn = sqlite3.connect(cache_db)
        self._init_cache_table()

        # Track statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'new_calculations': 0
        }

    def measure(self, compounds: pl.DataFrame, properties: List[str]) -> pl.DataFrame:
        """Measure with caching."""
        results = []

        for row in compounds.iter_rows(named=True):
            compound_id = row['ID']
            smiles = row['SMILES']

            self.stats['total_requests'] += 1

            # Check cache
            cached_score = self._get_cached_score(smiles)

            if cached_score is not None:
                self.stats['cache_hits'] += 1
                score = cached_score
            else:
                # Perform docking
                self.stats['new_calculations'] += 1
                score = self._dock_compound(smiles)
                self._cache_score(smiles, score)

            results.append({
                'ID': compound_id,
                'docking_score': score
            })

        return pl.DataFrame(results)

    def _init_cache_table(self):
        """Create cache table if not exists."""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS docking_cache (
                smiles_hash TEXT PRIMARY KEY,
                smiles TEXT,
                score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def _get_cached_score(self, smiles: str) -> float:
        """Retrieve score from cache."""
        smiles_hash = hashlib.md5(smiles.encode()).hexdigest()
        cursor = self.conn.execute(
            'SELECT score FROM docking_cache WHERE smiles_hash = ?',
            (smiles_hash,)
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def _cache_score(self, smiles: str, score: float):
        """Store score in cache."""
        smiles_hash = hashlib.md5(smiles.encode()).hexdigest()
        self.conn.execute(
            'INSERT OR REPLACE INTO docking_cache (smiles_hash, smiles, score) VALUES (?, ?, ?)',
            (smiles_hash, smiles, score)
        )
        self.conn.commit()

    def _dock_compound(self, smiles: str) -> float:
        """Perform docking calculation."""
        # Your docking implementation
        pass

    def get_statistics(self) -> Dict:
        """Return cache statistics."""
        cache_hit_rate = self.stats['cache_hits'] / max(self.stats['total_requests'], 1)
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate
        }

    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
```

## Integration with External Tools

### Docking Software (AutoDock Vina)

```python
# vina_oracle.py
import pandas as pd
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

def vina_scorer(compound_ids):
    """Score compounds using AutoDock Vina."""
    results = []
    work_dir = Path('vina_work')
    work_dir.mkdir(exist_ok=True)

    for compound_id in compound_ids:
        try:
            # Prepare ligand
            smiles = compound_id  # Assuming ID is SMILES
            ligand_file = _prepare_ligand(smiles, work_dir)

            # Run Vina
            score = _run_vina(ligand_file, work_dir)

            results.append({
                'ID': compound_id,
                'vina_score': score
            })
        except Exception as e:
            print(f"Warning: Vina scoring failed for {compound_id}: {e}")
            results.append({
                'ID': compound_id,
                'vina_score': float('nan')
            })

    return pd.DataFrame(results)

def _prepare_ligand(smiles, work_dir):
    """Convert SMILES to PDBQT format."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    # Save to file and convert to PDBQT
    pdb_file = work_dir / 'ligand.pdb'
    pdbqt_file = work_dir / 'ligand.pdbqt'

    Chem.MolToPDBFile(mol, str(pdb_file))

    # Use Open Babel to convert to PDBQT
    subprocess.run([
        'obabel',
        str(pdb_file),
        '-O', str(pdbqt_file),
        '-h'
    ], check=True)

    return pdbqt_file

def _run_vina(ligand_file, work_dir):
    """Execute Vina docking."""
    output_file = work_dir / 'output.pdbqt'
    log_file = work_dir / 'vina.log'

    cmd = [
        'vina',
        '--receptor', 'protein.pdbqt',
        '--ligand', str(ligand_file),
        '--out', str(output_file),
        '--log', str(log_file),
        '--center_x', '0', '--center_y', '0', '--center_z', '0',
        '--size_x', '20', '--size_y', '20', '--size_z', '20'
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    # Parse best score from log
    with open(log_file) as f:
        for line in f:
            if line.strip().startswith('1'):
                score = float(line.split()[1])
                return score

    raise ValueError("Failed to parse Vina score")
```

### Web API Integration

```python
# chembl_oracle.py
import pandas as pd
import requests
import time

def chembl_similarity_scorer(compound_ids):
    """Score compounds based on ChEMBL similarity to known actives."""
    api_base = 'https://www.ebi.ac.uk/chembl/api/data'
    results = []

    for compound_id in compound_ids:
        smiles = compound_id

        try:
            # Rate limiting
            time.sleep(0.5)

            # Search for similar compounds
            response = requests.get(
                f"{api_base}/similarity/{smiles}/70",
                headers={'Accept': 'application/json'},
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Calculate score based on similarity to active compounds
            score = _calculate_similarity_score(data)

            results.append({
                'ID': compound_id,
                'chembl_similarity': score
            })
        except Exception as e:
            print(f"Warning: ChEMBL query failed for {compound_id}: {e}")
            results.append({
                'ID': compound_id,
                'chembl_similarity': 0.0
            })

    return pd.DataFrame(results)

def _calculate_similarity_score(similarity_data):
    """Calculate aggregate similarity score."""
    if not similarity_data.get('molecules'):
        return 0.0

    # Average similarity to top 10 hits
    similarities = [m['similarity'] for m in similarity_data['molecules'][:10]]
    return sum(similarities) / len(similarities) if similarities else 0.0
```

### Caching for Expensive Operations

```python
# cached_oracle.py
import pandas as pd
import pickle
from pathlib import Path
import hashlib

CACHE_DIR = Path('oracle_cache')
CACHE_DIR.mkdir(exist_ok=True)

def expensive_scorer(compound_ids):
    """Score with persistent caching."""
    results = []

    for compound_id in compound_ids:
        # Check cache
        cache_key = hashlib.md5(compound_id.encode()).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.pkl"

        if cache_file.exists():
            # Load from cache
            with open(cache_file, 'rb') as f:
                score = pickle.load(f)
        else:
            # Perform expensive calculation
            score = _expensive_calculation(compound_id)

            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(score, f)

        results.append({
            'ID': compound_id,
            'score': score
        })

    return pd.DataFrame(results)

def _expensive_calculation(compound_id):
    """Perform expensive computation."""
    # Your expensive scoring logic
    pass
```

## Example Workflow

Complete workflow integrating custom oracle with LearnM8:

```python
# production_screening.py
"""Production screening workflow with custom oracle."""

from learnm8 import run_active_learning
from learnm8.oracles import PythonOracle
import pandas as pd
from pathlib import Path

# Configuration
COMPOUND_LIBRARY = 'compound_library.csv'
ORACLE_MODULE = 'docking_oracle.py'
ORACLE_FUNCTION = 'vina_scorer'
OUTPUT_DIR = 'screening_results'

def main():
    # Load compound library
    compounds = pd.read_csv(COMPOUND_LIBRARY)
    print(f"Loaded {len(compounds)} compounds")

    # Create oracle
    oracle = PythonOracle(ORACLE_MODULE, ORACLE_FUNCTION)
    print(f"Initialized oracle: {ORACLE_FUNCTION}")

    # Run active learning
    results = run_active_learning(
        compound_pool=compounds,
        oracle=oracle,
        learner='ensemble',
        target_col='vina_score',
        featurizer_type='morgan',
        n_cycles=20,
        batch_fraction=0.01,
        strategy='ucb',
        output_dir=OUTPUT_DIR,
        export_csv=True
    )

    # Analyze results
    final_cycle = results['cycle_metrics'][-1]
    print(f"\nScreening complete:")
    print(f"  Compounds evaluated: {final_cycle['n_labeled']}")
    print(f"  Best score found: {final_cycle['best_score']:.3f}")
    print(f"  Top 1% enrichment: {final_cycle['enrichment_1%']:.2f}")

    # Export selected compounds
    selected = results['master_df'].filter(
        pl.col('status') == 'labeled'
    ).sort('vina_score')

    selected.write_csv(Path(OUTPUT_DIR) / 'selected_compounds.csv')
    print(f"\nSelected compounds saved to {OUTPUT_DIR}/selected_compounds.csv")

if __name__ == '__main__':
    main()
```

Run the workflow:

```bash
python production_screening.py
```

## Best Practices

**Error Handling:**
- Return NaN for failed measurements rather than raising exceptions
- Log warnings for debugging without stopping execution
- Validate inputs before expensive operations

**Performance:**
- Implement caching for expensive calculations
- Use batch processing when possible
- Consider parallelization for independent measurements

**Reproducibility:**
- Use fixed random seeds for stochastic operations
- Log oracle version and configuration
- Document dependencies and requirements

**Testing:**
- Test with small compound sets before production
- Validate output format matches LearnM8 requirements
- Handle edge cases (invalid SMILES, missing data)

**Security:**
- Never expose API keys in code (use environment variables)
- Validate and sanitize compound inputs
- Use secure connections for web APIs

## Next Steps

- Understand oracle operating modes: [Oracles Overview](overview.md)
- Learn about benchmark vs production workflows: [Benchmark vs Production](../../tutorials/benchmark-vs-production.md)
- Explore learners for your use case: [Learners Overview](../learners/overview.md)

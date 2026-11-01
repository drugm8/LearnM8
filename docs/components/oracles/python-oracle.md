# Python Oracle

Use a Python oracle when values are not in a table: computed scores, calls to external tools, or any function you write. It puts a run into run mode.

## Oracle function

`PythonOracle` loads a `.py` file and calls one function from it. The function takes exactly one parameter, a list of compound IDs, and returns a pandas or Polars DataFrame with an `ID` column and the target column named by `target_col`:

```python
# scoring.py
import pandas as pd

def calculate_affinity(compound_ids):
    scores = [lookup_or_compute(cid) for cid in compound_ids]   # your logic
    return pd.DataFrame({'ID': compound_ids, 'binding_affinity': scores})
```

The function receives IDs only, not SMILES. If you need the structure, look it up from your own data by ID, or write an [Oracle subclass](#writing-an-oracle-subclass), whose `measure` receives both. Rows are realigned to the requested order by `ID`.

### Choosing the function

The string form `oracle='module.py:function'` names the function explicitly (a path without `:function` raises an error). When `PythonOracle('module.py')` is built directly without `function_name`, it uses the only public function in the module, or else the single one named `oracle`, `oracle_function`, `measure` or `evaluate`; otherwise it raises an `OracleError` listing the candidates.

## Compounds that cannot be measured

If a compound cannot be measured (crash, failed assay, timeout), return `None` or `NaN` for it, or leave its row out. Do not substitute a placeholder such as `0.0`. A missing row, null, `NaN`, infinite or non-numeric value all mark the compound as `failed`: it is never trained on or selected again, a warning is logged, and the `n_failed` cycle metric counts it. If every compound of the initial batch fails, the run stops with an `OracleError`. Errors raised by your function (`ValueError`, `RuntimeError`, `TypeError`, `OSError`, `KeyError`) are re-raised as `OracleError`.

## CLI

```bash
learnm8 run compounds.csv scoring.py:calculate_affinity --target binding_affinity \
  --learner rf --featurizer morgan --n-cycles 20
```

## API

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='scoring.py:calculate_affinity',
    target_col='binding_affinity',
    learner='rf',
    featurizer='morgan',
    n_cycles=20,
)
```

Or build it explicitly:

```python
from learnm8.oracles import PythonOracle

oracle = PythonOracle('scoring.py', function_name='calculate_affinity')
```

## Writing an Oracle subclass

For stateful oracles, or when you need SMILES, subclass `Oracle` and pass an instance as `oracle=`. It runs in run mode.

```python
import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors
from learnm8.core.interfaces import Oracle

class MolWeightOracle(Oracle):
    def measure(self, compounds: pl.DataFrame, properties: list[str]) -> pl.DataFrame:
        weights = []
        for smiles in compounds['SMILES']:
            mol = Chem.MolFromSmiles(smiles)
            weights.append(Descriptors.MolWt(mol) if mol is not None else None)
        result = compounds.select('ID')
        for prop in properties:
            result = result.with_columns(pl.Series(prop, weights))
        return result
```

`measure` must return rows in the input order. Returning null for a compound marks it `failed`, as above. Working implementations are in [Example oracles](example-oracles.md).

## See also

- [Oracles overview](overview.md)
- [Benchmark mode vs run mode](../../tutorials/running-experiments.md#benchmark-mode-vs-run-mode)

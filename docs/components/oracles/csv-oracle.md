# CSV Oracle

`CSVOracle` looks up target values in a CSV file of ground truth. It is used for benchmarking learners and acquisition strategies on data where every value is already known, and it puts a run into benchmark mode.

## File format

The CSV needs an ID column (`ID` by default), and the target column(s) you pass as `target_col`. A `SMILES` column is used when present.

```csv
ID,SMILES,Activity
comp1,CCO,0.85
comp2,CCC,0.42
comp3,CCCC,0.91
```

Behaviour verified in the loader:

- String columns that hold numbers mixed with text (for example `no_score`) are converted to numeric, with the text becoming null.
- Duplicate IDs are reduced to the first occurrence, with a `LearnM8Warning`.
- It is strict: every requested compound must have a row, and a null in a requested target column raises an error (use a [Python oracle](python-oracle.md) if some compounds cannot be measured).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | required | Path to the CSV file |
| `id_column` | str | `'ID'` | Column renamed to `ID` |

## CLI

With a CSV compound pool, the pool itself is the oracle:

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10
```

Pass a separate oracle file as the second positional argument:

```bash
learnm8 run compounds.csv oracle.csv --target Activity --learner rf --featurizer morgan
```

## API

```python
from learnm8 import run_active_learning
from learnm8.oracles import CSVOracle

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=CSVOracle('ground_truth.csv', id_column='compound_id'),
    target_col='Activity',
    learner='gp',
    featurizer='morgan',
    n_cycles=10,
)
```

Passing `oracle=None` with a CSV `compound_pool` path builds the `CSVOracle` from that file; `oracle='ground_truth.csv'` does the same for a separate file, and a Polars DataFrame passed as `oracle` is wrapped the same way. `compound_pool` as a DataFrame requires an explicit oracle.

## See also

- [Oracles overview](overview.md)
- [Benchmark mode vs run mode](../../tutorials/running-experiments.md#benchmark-mode-vs-run-mode)

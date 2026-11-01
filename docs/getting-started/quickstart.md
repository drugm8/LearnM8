# Quickstart

Get up and running with LearnM8 in a few minutes. Install first (see [Installation](installation.md)).

## Your first experiment

LearnM8 needs a CSV file with an `ID` column, a `SMILES` column and a target property column. Passing `oracle=None` uses that CSV as the oracle, which is benchmark mode (the true values are known, so LearnM8 can report discovery and ranking metrics).

```python
import polars as pl
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',  # CSV with ID, SMILES, Activity columns
    oracle=None,                    # use the CSV as the oracle
    learner='rf',                   # Random Forest
    target_col='Activity',          # column to optimize
    featurizer='morgan',            # Morgan circular fingerprints
    n_cycles=10,
)
```

Or with the CLI:

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan --n-cycles 10
```

## What happens

- **Cycle 0**: randomly selects 1% of the pool as the initial labeled set.
- **Cycles 1 to 9**: trains the learner on the labeled compounds, predicts the rest, and greedily selects the 1% with the best predicted score.
- Results are saved to a timestamped directory (`learnm8_results_YYYYMMDD_HHMMSS/`) unless you pass `output_dir`.

## Reading the results

```python
# Best value found
print(results['aggregate_metrics']['best_compound_value'])

# Top-10 discovery (benchmark mode only)
print(results['aggregate_metrics']['final_top_10_discovery'])

# Labeled compounds (Polars)
labeled = results['compounds_df'].filter(pl.col('status') == 'labeled')
print(labeled.sort('Activity', descending=True).head(10))

# Per-cycle progress
for m in results['cycle_metrics']:
    print(m['cycle'], m['best_so_far'], m['cumulative_labeled'])
```

## Column name variations

If your CSV uses other column names:

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=None,
    learner='rf',
    target_col='pIC50',
    featurizer='morgan',
    smiles_column='Smiles',
    id_column='CompoundID',
)
```

## Next steps

- [Core Concepts](concepts.md): cycles, oracles, and benchmark vs run mode
- [Running Experiments](../tutorials/running-experiments.md): full workflow, learner choice, output layout
- [CLI Reference](../user-guide/cli-reference.md) and [API Reference](../user-guide/api-reference.md)

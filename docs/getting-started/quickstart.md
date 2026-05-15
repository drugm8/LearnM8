# Quickstart

Get up and running with LearnM8 in a few minutes.

## Install

```bash
conda env create -f environment.yml
conda activate learnm8
pip install -e .
```

## Your First Experiment

LearnM8 needs a CSV file with at least an `ID` column, a `SMILES` column, and a target property column. When you pass a CSV as `compound_pool`, LearnM8 automatically uses it as the benchmark oracle (the `oracle` parameter can be omitted).

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',  # CSV with ID, SMILES, Activity columns
    learner='rf',                   # Random Forest (good general-purpose default)
    target_col='Activity',          # Column to optimize
    featurizer='morgan',            # Morgan circular fingerprints
    n_cycles=10                     # 10 active learning cycles
)
```

Or with the CLI:

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan --n-cycles 10
```

## What Happens

- **Cycle 0**: Randomly selects 1% of the pool as the initial labeled set
- **Cycles 1–9**: Trains a Random Forest on labeled compounds, predicts the rest, and greedily selects the top 1% by predicted score
- Results are saved to a timestamped directory (`learnm8_results_YYYYMMDD_HHMMSS/`)

## Reading the Results

```python
# Best compound found
best = results['aggregate_metrics']['best_compound_value']
print(f"Best Activity: {best:.3f}")

# Discovery rate for top-10 compounds (benchmark mode)
top10 = results['aggregate_metrics']['final_top_10_discovery']
print(f"Top-10 discovery: {top10:.1%}")

# Full labeled DataFrame (Polars) — filter the compounds_df
labeled = results['compounds_df'].filter(pl.col('status') == 'labeled')
print(labeled.sort('Activity', descending=True).head(10))

# Per-cycle progress
for m in results['cycle_metrics']:
    print(f"Cycle {m['cycle']:2d}: best={m['best_so_far']:.3f}, n_labeled={m['n_labeled']}")
```

## Column Name Variations

If your CSV uses non-standard column names:

```python
results = run_active_learning(
    compound_pool='compounds.csv',
    learner='rf',
    target_col='pIC50',
    featurizer='morgan',
    smiles_column='Smiles',     # if not 'SMILES'
    id_column='CompoundID'      # if not 'ID'
)
```

## Next Steps

- **[Core Concepts](concepts.md)** — understand active learning cycles, benchmark vs production mode, and how to choose a learner
- **[Running Experiments](../tutorials/running-experiments.md)** — multi-stage cycles, pruning, GPU acceleration
- **[API Reference](../user-guide/api-reference.md)** — complete parameter reference
- **[CLI Reference](../user-guide/cli-reference.md)** — all command-line options

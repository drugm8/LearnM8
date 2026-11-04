# Examples

Practical code examples for common LearnM8 workflows. For full parameter documentation, see the [API Reference](api-reference.md).

## Basic Benchmark Experiment

The minimal workflow: run active learning against a CSV dataset to evaluate strategy performance.

```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10,
    batch_fraction=0.01,
    random_state=42,
)

for m in results['cycle_metrics']:
    print(f"Cycle {m['cycle']:2d} | strategy={m['strategy']:<8} | "
          f"labeled={m['cumulative_labeled']:4d} | best={m['best_so_far']}")
```

## Multi-Stage Acquisition with CycleConfig

Control exactly which strategy runs in each phase — random warm-start, uncertainty-guided exploration, then greedy exploitation.

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random',  n_cycles=1, batch_fraction=0.02),
        CycleConfig('ucb',     n_cycles=6, batch_fraction=0.005,
                    acquisition_params={'beta': 2.0}),
        CycleConfig('greedy',  n_cycles=3, batch_fraction=0.005),
    ],
    random_state=42,
)
```

## Run Mode with a Python Oracle

Use a Python function as the oracle when measurements are expensive (docking, synthesis, etc.). The function receives a list of compound IDs and returns a DataFrame with an `ID` column and the target column. See [Python oracle](../components/oracles/python-oracle.md).

```python
# scoring.py
def calculate_binding(compound_ids):
    import pandas as pd
    scores = [run_docking(cid) for cid in compound_ids]
    return pd.DataFrame({'ID': compound_ids, 'binding_score': scores})
```

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='virtual_library.csv',
    oracle='scoring.py:calculate_binding',
    learner='rf',
    target_col='binding_score',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('ei',     n_cycles=9, batch_fraction=0.005),
    ],
    n_jobs=-1,
)
```

## Gaussian Process with Uncertainty

GP provides uncertainty estimates, so it can be paired with UCB, EI, PI, Thompson sampling or entropy.

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random',   n_cycles=1, batch_fraction=0.02),
        CycleConfig('thompson', n_cycles=9, batch_fraction=0.005),
    ],
    random_state=42,
)
```

## Skipping Uncertainty

When the acquisition strategy does not need uncertainty (for example greedy), learners that can skip it (including `gp`) do so automatically. Use `force_uncertainty=True` to always compute it, for example for diagnostics.

```python
from learnm8 import run_active_learning, CycleConfig

# Uncertainty skipped automatically — greedy doesn't need it
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=9, batch_fraction=0.005),
    ],
)

# Force uncertainty computation even for non-uncertainty strategies
results_diag = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=9, batch_fraction=0.005),
    ],
    force_uncertainty=True,
)
```

## Large-Scale Screening (100M Compounds)

Parquet output, diversity metrics disabled, a shared feature cache and the large-feature acknowledgement.

```python
from learnm8 import run_active_learning, CycleConfig
from pathlib import Path

results = run_active_learning(
    compound_pool='virtual_library_100M.csv',
    oracle='scoring.py:score',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.001),
        CycleConfig('greedy', n_cycles=9, batch_fraction=0.0005),
    ],
    cache_dir=Path('.shared_cache'),
    output_format='parquet',
    disable_molecular_similarity=True,
    large_features_ack=True,
    n_jobs=-1,
    random_state=42,
)
```

## Chemprop Graph Neural Network

Chemprop operates directly on SMILES — no featurizer required. Add a featurizer to enable hybrid graph + descriptor mode.

```python
from learnm8 import run_active_learning, CycleConfig

# SMILES-native (no featurizer)
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',
    target_col='Activity',
    n_cycles=10,
    batch_fraction=0.01,
)

# Hybrid: graph + Mordred descriptors
results_hybrid = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='chemprop',
    target_col='Activity',
    featurizer='descriptors',
    n_cycles=10,
    batch_fraction=0.01,
)
```

## Ensemble with Uncertainty

Ensembles estimate uncertainty via member disagreement (std dev across predictions). `xgb_ensemble` adds uncertainty to XGBoost, which has none of its own, so uncertainty strategies such as UCB and Thompson sampling can be used with it.

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='xgb_ensemble',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random',   n_cycles=1, batch_fraction=0.02),
        CycleConfig('ucb',      n_cycles=5, batch_fraction=0.005,
                    acquisition_params={'beta': 1.5}),
        CycleConfig('thompson', n_cycles=4, batch_fraction=0.005),
    ],
    random_state=42,
)

# Inspect uncertainty estimates from final cycle
last = results['cycle_metrics'][-1]
print(f"Has uncertainty: {last['has_uncertainty']}")
print(f"Mean uncertainty: {last['uncertainty_mean']}")
```

## Score-Based Pruning

Remove unpromising compounds from the pool during the run to focus resources on the interesting region.

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer='morgan',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('ucb',    n_cycles=5, batch_fraction=0.005,
                    pruning_strategy='score',
                    pruning_params={'pruning_fraction': 0.3}),
        CycleConfig('greedy', n_cycles=4, batch_fraction=0.005),
    ],
    random_state=42,
)
```

## Minimisation Campaign

Set `score_direction='lower'` to target the lowest values (e.g., minimise toxicity or binding energy).

```python
from learnm8 import run_active_learning, CycleConfig

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='rf',
    target_col='Toxicity',
    featurizer='descriptors',
    cycles=[
        CycleConfig('random', n_cycles=1, batch_fraction=0.02),
        CycleConfig('greedy', n_cycles=9, batch_fraction=0.005),
    ],
    score_direction='lower',
)
```

## Direct Feature Extraction

Extract features independently of the active learning loop.

```python
from learnm8 import extract_features
from pathlib import Path

smiles = ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O']

# Binary fingerprint
morgan_fp = extract_features(smiles, featurizer='morgan', cache_dir=Path('.cache'))

# Mordred descriptors
descriptors = extract_features(smiles, featurizer='descriptors',
                               cache_dir=Path('.cache'), show_progress=True)

# Optimised storage for tree learners (uint8 fast path)
fp_uint8 = extract_features(smiles, featurizer='morgan',
                             preferred_dtype='uint8')

print(morgan_fp.shape)    # (3, 2048)
print(descriptors.shape)  # (3, n_descriptors)
```

## Compound Pool Validation

Validate SMILES before starting an experiment.

```python
from learnm8 import validate_compound_pool
import polars as pl

compounds = pl.read_csv('compounds.csv')
result = validate_compound_pool(
    compounds,
    n_jobs=-1,
    progress=True,
    target_col='Activity',
)

print(f"Valid:   {len(result.valid_compounds)}")
print(f"Invalid: {len(result.invalid_compounds)}")

for cid, error in result.validation_errors.items():
    print(f"  {cid}: {error}")
```

## Accessing Results

```python
results = run_active_learning(...)

# Final compounds DataFrame — 7 columns
df = results['compounds_df']
# Columns: ID, SMILES, status, labeled_cycle, selected_cycle, pruned_cycle, <target_col>

# Per-cycle metrics
for m in results['cycle_metrics']:
    print(m['cycle'], m['strategy'], m['cumulative_labeled'], m['avg_score_selected'])

# Aggregate summary
agg = results['aggregate_metrics']
print(agg['best_compound_value'], agg['total_labeled'])

# Output files written
for name, path in results['saved_files'].items():
    print(name, path)
```

## CLI Quick Reference

```bash
# Benchmark experiment
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan --n-cycles 10

# Run mode (Python oracle)
learnm8 run compounds.csv scoring.py:score --target score --learner rf --featurizer morgan --n-cycles 20

# Custom strategy
learnm8 run compounds.csv --target Activity --learner gp --featurizer ecfp6 \
  --strategy ucb --acquisition-params '{"beta": 2.0}' --n-cycles 15

# Multi-stage cycles
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --cycles "random:0.02 ucb:0.005*7 greedy:0.005*2"

# Large-scale with shared cache
learnm8 run large_lib.csv --target Activity --learner rf --featurizer morgan \
  --cache-dir .shared_cache --output-format parquet \
  --n-cycles 10

# Force uncertainty even with greedy
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan \
  --strategy greedy --force-uncertainty --n-cycles 10

# Minimise (lower is better)
learnm8 run compounds.csv --target Toxicity --learner rf --featurizer descriptors \
  --strategy greedy --score-direction lower --n-cycles 10
```

## See also

- [API Reference](api-reference.md)
- [CLI Reference](cli-reference.md)
- [Building Custom Cycles](../tutorials/building-custom-cycles.md)

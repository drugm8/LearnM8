# pruning

Tests score-based pruning strategies that reduce the active search space during
active learning cycles. Validates that pruning improves efficiency without
significantly harming discovery performance on the AmpC 30K dataset.

## Scripts

| Script | Description |
|---|---|
| `validate_score_based_pruning.py` | Sweeps pruning fractions and compares discovery metrics against an unpruned baseline |

## How to Run

Run the script from the repo root:

```bash
PYTHONPATH=. python validation/pruning/scripts/validate_score_based_pruning.py
```

Run the full category via the orchestrator:

```bash
PYTHONPATH=. python validation/run_all_validations.py --category pruning
```

## Outputs

Results are written to:

```
validation/reports/pruning/<script_name>/
```

The script produces per-cycle discovery metrics for each pruning fraction and a
summary JSON comparing hit rates and efficiency gains against the baseline.

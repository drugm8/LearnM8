# uncertainty

Assesses the quality of uncertainty estimates produced by ensemble and other
uncertainty-capable learners. Measures calibration and correlation between
predicted uncertainty and actual prediction error on the AmpC 30K dataset.

## Scripts

| Script | Description |
|---|---|
| `validate_uncertainty_correlations.py` | Correlates predicted uncertainty with observed error across GP, MC Dropout, and ensemble learners |

## How to Run

Run the script from the repo root:

```bash
PYTHONPATH=. python validation/uncertainty/scripts/validate_uncertainty_correlations.py
```

Run the full category via the orchestrator:

```bash
PYTHONPATH=. python validation/run_all_validations.py --category uncertainty
```

## Outputs

Results are written to:

```
validation/reports/uncertainty/<script_name>/
```

The script produces a CSV of per-cycle uncertainty vs. error statistics and a
summary JSON with Spearman/Pearson correlation coefficients per learner.

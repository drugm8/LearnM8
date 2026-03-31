# large_scale

Validates active learning on the full AmpC 1M dataset. Scripts exercise different
learner families at production scale to verify correctness, memory safety, and
discovery performance under realistic conditions.

## Scripts

| Script | Description |
|---|---|
| `validate_ampc_1M_chemprop_mixed.py` | Chemprop with a mixed acquisition schedule on 1M compounds |
| `validate_ampc_1M_chemprop_molpal_params.py` | Chemprop using MolPAL-style hyperparameters for literature comparison |
| `validate_ampc_1M_fastprop_mixed.py` | Fastprop with a mixed acquisition schedule on 1M compounds |
| `validate_ampc_1M_mcdropout_mixed.py` | MC Dropout MLP with a mixed acquisition schedule on 1M compounds |
| `validate_ampc_1M_rf_ensemble_mixed.py` | RF ensemble with a mixed acquisition schedule on 1M compounds |

These scripts are slow (GPU recommended). They are excluded from the default
orchestrator run and must be opted in explicitly.

## How to Run

Run a single script from the repo root:

```bash
PYTHONPATH=. python validation/large_scale/scripts/validate_ampc_1M_chemprop_mixed.py
PYTHONPATH=. python validation/large_scale/scripts/validate_ampc_1M_fastprop_mixed.py
PYTHONPATH=. python validation/large_scale/scripts/validate_ampc_1M_mcdropout_mixed.py
PYTHONPATH=. python validation/large_scale/scripts/validate_ampc_1M_rf_ensemble_mixed.py
PYTHONPATH=. python validation/large_scale/scripts/validate_ampc_1M_chemprop_molpal_params.py
```

Run the full category via the orchestrator:

```bash
PYTHONPATH=. python validation/run_all_validations.py --category large_scale
```

## Outputs

Results are written to:

```
validation/reports/large_scale/<script_name>/
```

Each run produces per-cycle discovery metrics, timing stats, and a summary JSON.

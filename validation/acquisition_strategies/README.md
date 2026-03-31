# acquisition_strategies

Tests acquisition functions against the AmpC 30K benchmark dataset. Each script
runs a parameter sweep for one strategy and compares discovery metrics against
the greedy and random baselines.

## Scripts

| Script | Description |
|---|---|
| `validate_ucb.py` | Validates Upper Confidence Bound with beta parameter sweep |
| `validate_ei.py` | Validates Expected Improvement with xi parameter sweep |
| `validate_pi.py` | Validates Probability of Improvement with xi parameter sweep |
| `validate_thompson.py` | Validates Thompson Sampling over multiple random seeds |
| `validate_entropy.py` | Validates Entropy-based acquisition with temperature sweep |
| `validate_simulated_annealing.py` | Validates Simulated Annealing with temperature schedule sweep |

All scripts require an uncertainty-capable learner (GP or ensemble).

## How to Run

Run a single script from the repo root:

```bash
PYTHONPATH=. python validation/acquisition_strategies/scripts/validate_ucb.py
PYTHONPATH=. python validation/acquisition_strategies/scripts/validate_ei.py
PYTHONPATH=. python validation/acquisition_strategies/scripts/validate_pi.py
PYTHONPATH=. python validation/acquisition_strategies/scripts/validate_thompson.py
PYTHONPATH=. python validation/acquisition_strategies/scripts/validate_entropy.py
PYTHONPATH=. python validation/acquisition_strategies/scripts/validate_simulated_annealing.py
```

Run the full category via the orchestrator:

```bash
PYTHONPATH=. python validation/run_all_validations.py --category acquisition_strategies
```

## Outputs

Results are written to:

```
validation/reports/acquisition_strategies/<script_name>/
```

Each run produces cycle metrics CSVs and a summary JSON comparing the strategy
against the greedy and random baselines.

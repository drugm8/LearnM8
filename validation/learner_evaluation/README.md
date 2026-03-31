# learner_evaluation

Evaluates learner-featurizer and learner-acquisition combinations, producing
comparison matrices and a comprehensive analysis of Chemprop. Scripts run on the
AmpC 30K benchmark dataset.

## Scripts

| Script | Description |
|---|---|
| `validate_learner_featurizer_matrix.py` | Full grid: all learners x featurizers, records discovery metrics per combination |
| `validate_learner_featurizer_matrix_single.py` | Single learner-featurizer pair evaluation for fast targeted checks |
| `validate_learner_acquisition_matrix.py` | Grid of uncertainty learners x acquisition strategies, records performance |
| `validate_chemprop_comprehensive.py` | Deep evaluation of Chemprop: hyperparameter sensitivity, hybrid mode, ensemble vs single |

## How to Run

Run a single script from the repo root:

```bash
PYTHONPATH=. python validation/learner_evaluation/scripts/validate_learner_featurizer_matrix.py
PYTHONPATH=. python validation/learner_evaluation/scripts/validate_learner_featurizer_matrix_single.py
PYTHONPATH=. python validation/learner_evaluation/scripts/validate_learner_acquisition_matrix.py
PYTHONPATH=. python validation/learner_evaluation/scripts/validate_chemprop_comprehensive.py
```

Run the full category via the orchestrator:

```bash
PYTHONPATH=. python validation/run_all_validations.py --category learner_evaluation
```

## Outputs

Results are written to:

```
validation/reports/learner_evaluation/<script_name>/
```

Matrix scripts produce CSV heat-map data and a summary JSON. The Chemprop script
produces per-configuration cycle metrics and a consolidated comparison report.

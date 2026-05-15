# LearnM8: Active Learning Framework for Molecular Screening

LearnM8 is an active learning framework for molecular property prediction and compound screening. Built with a pure functional architecture, it enables researchers to efficiently explore chemical space through intelligent compound selection, uncertainty-guided decision-making, and state-of-the-art machine learning models.

The framework addresses a fundamental challenge in computational chemistry: how to select the most informative molecules for experimental testing when resources are limited. Through iterative cycles of prediction, selection, and measurement, LearnM8 helps researchers identify promising compounds faster than traditional screening approaches while minimizing experimental costs.

## Key Features

**Comprehensive Model Suite**

- 21 machine learning models including scikit-learn, PyTorch neural networks, Chemprop GNNs, and ensemble methods
- Uncertainty quantification for 17 of 21 learners (rf, gp, dt, lr, mc_dropout, all 8 ensembles, and 4 GPU learners)
- Optional GPU acceleration via GPyTorch and RAPIDS cuML

**Rich Acquisition Strategies**

- 9 selection strategies: basic (greedy, random, topk), uncertainty-based (UCB, EI, PI, Thompson, entropy), and optimization-based (simulated annealing)
- Uncertainty-based strategies automatically skip uncertainty computation when not needed

**Performance Optimizations**

- HDF5-based feature caching (~100× speedup on repeated extraction)
- Automatic parallelization for feature extraction
- Vectorized Polars operations for all DataFrame work
- Streaming parquet output for large pools (>1M rows, constant RAM)

**Two Operating Modes**

- **Benchmark mode** (CSV oracle): full discovery, enrichment, and ranking metrics with ground truth
- **Production mode** (Python oracle): integration with real assays or docking software
- Auto-detected from oracle type; no manual flag required

## Quick Example

```python
from learnm8 import run_active_learning

# Benchmark mode — oracle auto-detected from the same CSV
results = run_active_learning(
    compound_pool='compounds.csv',
    learner='rf',
    target_col='Activity',
    featurizer='morgan',
    n_cycles=10
)

print(f"Best compound: {results['aggregate_metrics']['best_compound_value']:.3f}")
print(f"Top-10 discovery rate: {results['aggregate_metrics']['final_top_10_discovery']:.1%}")
```

Or use the CLI:

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan --n-cycles 10
```

## Component Overview

### Learners (21 total)

| Shortcut | Model | Uncertainty | CPU | GPU | Typical Pool Size |
|----------|-------|-------------|-----|-----|-------------------|
| `rf` | Random Forest | Tree std dev | ✓ | — | 1K – 10M+ |
| `gp` | Gaussian Process | GP posterior | ✓ | — | < 10K |
| `gpu_gp` | GPyTorch Exact GP | GP posterior | ✓ | ✓ | < 10K |
| `svgp` | Scalable Variational GP | Variational | ✓ | ✓ | 10K – 100K+ |
| `xgb` | XGBoost | — | ✓ | ✓ | 1K – 10M+ |
| `lr` | Linear Regression | Leverage-based | ✓ | — | any |
| `dt` | Decision Tree | Leaf impurity | ✓ | — | 1K – 10M+ |
| `mlp` | MLP Neural Network | — | ✓ | ✓ | 10K – 1M+ |
| `mc_dropout` | MC Dropout | MC sampling | ✓ | ✓ | 5K – 500K |
| `fastprop` | FastProp | — | ✓ | ✓ | 10K – 1M+ |
| `chemprop` | Chemprop GNN | — | ✓ | ✓ | 5K – 500K |
| `rf_fil` | RF + FIL inference | Tree std dev | — | ✓ | 1K – 10M+ |
| `ridge_cuml` | Ridge (cuML) | Leverage-based | — | ✓ | any |
| `chemprop_ensemble` | Chemprop × 3 | Ensemble std | ✓ | ✓ | 1K – 200K |
| `rf_ensemble` | RF × N | Ensemble std | ✓ | — | 1K – 10M+ |
| `lr_ensemble` | LR × N | Ensemble std | ✓ | ✓ | any |
| `xgb_ensemble` | XGBoost × N | Ensemble std | ✓ | ✓ | 1K – 10M+ |
| `dt_ensemble` | DT × N | Ensemble std | ✓ | — | 1K – 10M+ |
| `mixed_ensemble` | RF + LR + XGB | Ensemble std | ✓ | ✓ | 1K – 50K |
| `fastprop_ensemble` | FastProp × N | Ensemble std | ✓ | ✓ | 5K – 500K |
| `ensemble` | Generic wrapper (explicit member config required) | Ensemble std | ✓ | — | any |

`gpu_gp` and `svgp` require GPyTorch ≥ 1.11 and GAUCHE ≥ 0.1.6. `rf_fil` and `ridge_cuml` require RAPIDS cuML ≥ 25.04.

### Acquisition Strategies (9 total)

| Shortcut | Requires Uncertainty | Strategy |
|----------|---------------------|----------|
| `greedy` | No | Highest predicted value (pure exploitation) |
| `random` | No | Random selection (baseline) |
| `topk` | No | Top-K ranked selection |
| `ucb` | Yes | Mean + β × uncertainty |
| `ei` | Yes | Expected improvement over current best |
| `pi` | Yes | Probability of improving over current best |
| `thompson` | Yes | Sample from posterior predictive distribution |
| `entropy` | Yes | Maximum uncertainty (maximum information gain) |
| `simulated_annealing` | No | Temperature-based probabilistic selection |

Uncertainty-based strategies (`ucb`, `ei`, `pi`, `thompson`, `entropy`) require a learner with an uncertainty method. Uncertainty computation is automatically skipped when the active strategy does not require it.

### Featurizers (39 registered names, 38 unique)

| Category | Count | Examples |
|----------|-------|---------|
| 2D Circular | 5 | morgan, ecfp, ecfp6, morgan_feat, secfp |
| 2D Structural Keys | 4 | maccs, pubchem, klekota_roth, laggner |
| 2D Topological | 6 | avalon, atom_pair, topological_torsion, rdkit, pattern, layered |
| 2D Hashed | 4 | map4, mhfp, lingo, erg, secfp |
| 2D Descriptors | 11 | mordred, rdkit_2d_descriptors, estate, ghose_crippen, mqns, vsa, bcut2d, physiochemical, pharmacophore, functional_groups |
| 3D (conformer) | 9 | whim, usr, usrcat, e3fp, getaway, morse, rdf, autocorr, electroshape |

## Next Steps

- **[Installation Guide](getting-started/installation.md)**: Set up LearnM8 with conda and optional dependencies
- **[Quickstart Tutorial](getting-started/quickstart.md)**: Run your first experiment
- **[Core Concepts](getting-started/concepts.md)**: Understand active learning and LearnM8's design
- **[CLI Reference](user-guide/cli-reference.md)**: Full command-line documentation
- **[API Reference](user-guide/api-reference.md)**: Complete Python API reference

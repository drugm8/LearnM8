# LearnM8: Active Learning Framework for Molecular Screening

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](docs/)
[![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

LearnM8 is a comprehensive active learning framework for molecular property prediction and compound screening. It combines state-of-the-art machine learning models with sophisticated acquisition strategies to enable efficient exploration of chemical space.

Built on a pure functional architecture, LearnM8 provides explicit cycle control, comprehensive uncertainty quantification, and molecular-specific optimizations. The framework supports both benchmark analysis (with known ground truth) and production screening (with expensive experimental measurements), making it suitable for both research and real-world drug discovery applications.

**Key Capabilities:** 15+ ML models (including Chemprop GNNs), 11+ acquisition strategies, HDF5 caching for 100x speedup, automatic parallelization, and design space pruning for large-scale screening.

## 📚 Documentation

**[Read the full documentation →](docs/index.md)**

- [Installation Guide](docs/getting-started/installation.md)
- [Quickstart Tutorial](docs/getting-started/quickstart.md)
- [Core Concepts](docs/getting-started/concepts.md)
- [CLI Reference](docs/user-guide/cli-reference.md)
- [API Reference](docs/user-guide/api-reference.md)
- [Learners](docs/components/learners/overview.md) | [Acquisition](docs/components/acquisition/overview.md) | [Featurizers](docs/components/featurizers/overview.md)

## 🚀 Quick Start

### Installation

```bash
conda env create -f environment.yml
conda activate base
pip install -e .
```

### Your First Experiment

**Python API:**
```python
from learnm8 import run_active_learning

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner='gp',
    target_col='Activity',
    featurizer_type='morgan',
    n_cycles=10
)
```

**CLI Alternative:**
```bash
learnm8 validate compounds.csv
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --n-cycles 10
```

**Memory Management (Large Libraries):**
```python
# Auto-calculated batch size (recommended)
results = run_active_learning(
    compound_pool='large_library.csv',  # 100k+ compounds
    learner='rf',
    target_col='Activity',
    featurizer_type='morgan',
    n_cycles=10
)

# Manual override for specific hardware
results = run_active_learning(
    compound_pool='large_library.csv',
    learner='rf',
    target_col='Activity',
    featurizer_type='morgan',
    n_cycles=10,
    prediction_batch_size=5000  # Process 5000 compounds per batch
)
```

```bash
# CLI with custom batch size
learnm8 run large_library.csv --target Activity --learner rf --featurizer morgan \
  --prediction-batch-size 5000
```

## 📖 Examples

Learn LearnM8 through hands-on examples:

**Notebooks:** ([examples/notebooks/](examples/notebooks/))
- [Quickstart](examples/notebooks/01_quickstart.ipynb) (10 min): First active learning experiment (benchmark mode)
- [Custom Oracles](examples/notebooks/02_custom_oracles.ipynb) (15 min): Custom oracles in production (run mode)
- [Advanced Configuration](examples/notebooks/03_advanced_configuration.ipynb) (15 min): Custom learners and multi-stage workflows
- [Production Screening](examples/notebooks/04_production_screening.ipynb) (20 min): Real-world deployment

**Example Oracles:** ([examples/oracles/](examples/oracles/))
- `SimilarityOracle`: 2D fingerprint-based similarity
- `Pharmacophore2DOracle`: Functional group pattern matching
- `CDPKitPharmacophoreOracle`: 3D pharmacophore alignment (requires CDPKit)
- `VinaOracle`: Molecular docking (requires Vina)

See [examples/README.md](examples/README.md) for complete guide.

## ✨ Key Features

**Architecture & Performance:**
- Pure functional design with explicit cycle control
- HDF5 caching for 100x speedup on repeated operations
- Automatic parallelization (5-10x faster feature extraction)
- Early validation with datamol (catch SMILES errors before experiments)
- Memory-efficient batch prediction for large compound libraries (auto-calculated or manual override)

**Machine Learning Models:**
- 15+ models: Scikit-learn (RF, GP, XGB), PyTorch (MLP, MC Dropout, FastProp), GNNs (Chemprop)
- Ensemble learning with automatic uncertainty quantification
- Chemprop with fine-tuning for active learning

**Acquisition Strategies:**
- 11+ strategies: Basic (greedy, random), Uncertainty-based (UCB, EI, Thompson), Diversity (BitBIRCH, simulated annealing)
- Multi-stage cycles with per-cycle configuration
- Predefined schedules (quick, standard, intensive, diverse)

**Molecular Screening:**
- Two modes: Benchmark (known ground truth) and Production (expensive measurements)
- Design space pruning for large libraries (>100k compounds)
- Multiple featurizers: Morgan, MACCS, ECFP6, Mordred descriptors
- RDKit integration for molecular handling

## 📊 Components Overview

| Component | Count | Examples |
|-----------|-------|----------|
| **Learners** | 18 | RandomForest, GaussianProcess, XGBoost, MLP, MCDropout, FastProp, Chemprop, Ensembles |
| **Acquisition** | 11 | greedy, random, ucb, ei, pi, thompson, entropy, bitbirch, simulated_annealing |
| **Featurizers** | 5 | morgan (2048-bit), maccs (167-bit), ecfp6 (2048-bit), morgan_feat (2048-bit), descriptors (1613-D) |
| **Pruning** | 1 | score_based |

See the [documentation](docs/) for complete details on all components.

## 🔗 Links

- **Documentation:** [docs/](docs/index.md)
- **Tutorials:** [Running Experiments](docs/tutorials/running-experiments.md) | [Custom Cycles](docs/tutorials/building-custom-cycles.md) | [Advanced Workflows](docs/tutorials/advanced-workflows.md)
- **Customization:** [Custom Learners](docs/customization/custom-learners.md) | [Custom Acquisition](docs/customization/custom-acquisition.md)
- **Examples:** See [docs/tutorials/](docs/tutorials/) for comprehensive examples

## 🤝 Contributing

Contributions are welcome! LearnM8 follows a functional architecture with modular design. When contributing:

1. Maintain pure functional patterns
2. Use Polars for all internal DataFrame operations
3. Follow existing code conventions
4. Add comprehensive tests with real molecular data
5. Update documentation for new features

See the [extending framework guide](docs/customization/extending-framework.md) for details on adding new components.

## 📄 License

LearnM8 is developed for molecular screening and active learning research.

## 📚 Citation

If you use LearnM8 in your research, please cite our work (citation details to be added).

## 🧪 Testing

```bash
pytest tests/
pytest tests/ --cov=learnm8 --cov-report=html
```

---

**LearnM8**: Intelligent molecular screening through active learning 🧬🤖

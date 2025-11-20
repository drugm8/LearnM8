# LearnM8 Examples

Practical examples and tutorials for using LearnM8 in molecular screening workflows.

## Quick Navigation

- **[Notebooks](notebooks/)**: Step-by-step tutorials
- **[Oracles](oracles/)**: Ready-to-use scoring functions
- **[Documentation](../docs/)**: Full API reference

## 📓 Notebooks

Learn LearnM8 through hands-on examples:

| Notebook | Time | Description |
|----------|------|-------------|
| [01_quickstart.ipynb](notebooks/01_quickstart.ipynb) | 10 min | Your first active learning experiment (benchmark mode) |
| [02_custom_oracles.ipynb](notebooks/02_custom_oracles.ipynb) | 15 min | Custom oracles in production (run mode) |
| [03_advanced_configuration.ipynb](notebooks/03_advanced_configuration.ipynb) | 15 min | Custom learners and multi-stage workflows |
| [04_production_screening.ipynb](notebooks/04_production_screening.ipynb) | 20 min | Real-world deployment workflow |

**Learning Path:** Start with 01 → 02 → 03 → 04 for comprehensive introduction.

See [notebooks/README.md](notebooks/README.md) for detailed notebook guide.

## 🔬 Example Oracles

Ready-to-use scoring functions for common screening tasks:

| Oracle | Speed | Use Case |
|--------|-------|----------|
| **SimilarityOracle** | ~0.001s/mol | Scaffold hopping, analog finding |
| **Pharmacophore2DOracle** | ~0.01s/mol | Feature-based screening |
| **CDPKitPharmacophoreOracle** | ~1-5s/mol | 3D pharmacophore alignment |
| **VinaOracle** | ~10-60s/mol | Molecular docking |

See [oracles/README.md](oracles/README.md) for complete oracle documentation.

## 🚀 Quick Start

### Run Your First Experiment

```bash
# Install LearnM8
conda env create -f environment.yml
conda activate learnm8
pip install -e .

# Launch Jupyter
jupyter notebook examples/notebooks/01_quickstart.ipynb
```

### Use Example Oracle

```python
from examples.oracles import SimilarityOracle
from learnm8 import run_active_learning
import polars as pl

# Create oracle
oracle = SimilarityOracle(
    reference_smiles='c1ccccc1O',
    fingerprint_type='morgan'
)

# Run active learning
results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=oracle,
    target_col='similarity',
    learner='gp',
    featurizer_type='morgan',
    n_cycles=10
)
```

## 📚 Additional Resources

- **[Documentation](../docs/)**: Full API reference and guides
- **[Validation](../validation/)**: Internal validation notebooks
- **[Tests](../tests/oracles/)**: Oracle test suite

## 🤝 Contributing Examples

To contribute new examples:

1. Follow existing notebook structure
2. Keep notebooks concise (10-20 minutes)
3. Include clear learning objectives
4. Test on standard datasets
5. Add to navigation tables

## 💡 Tips

- **Notebooks run faster** with feature caching enabled
- **Use small datasets** for learning (validation subsets available)
- **Check notebook outputs** before running to see expected results
- **Refer to docs** for advanced customization

---

**Questions?** See [documentation](../docs/) or open an issue.

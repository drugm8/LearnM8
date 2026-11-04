# LearnM8 Tutorial Notebooks

Step-by-step tutorials for learning LearnM8.

## Prerequisites

### Installation
```bash
conda env create -f ../../environment.yml
conda activate learnm8
pip install --no-deps -e ../..
```

### Jupyter Setup
```bash
pip install jupyter matplotlib
jupyter notebook
```

## Notebooks

### 01_quickstart.ipynb (10 minutes)
**Goal:** Run your first active learning experiment

**What you'll learn:**
- Basic `run_active_learning()` API
- Loading compound libraries
- Inspecting results
- Simple visualization

**Prerequisites:** None

**Outcome:** A working active learning experiment from a single `run_active_learning()` call

---

### 02_custom_oracles.ipynb (15 minutes)
**Goal:** Use and create custom scoring functions

**What you'll learn:**
- Using SimilarityOracle for scaffold hopping
- Using Pharmacophore2DOracle for feature matching
- Using VinaOracle for molecular docking (optional)
- Visualizing best score discovery progress

**Prerequisites:** 01_quickstart.ipynb

**Outcome:** Ability to create custom scoring functions for any property

---

### 03_advanced_configuration.ipynb (15 minutes)
**Goal:** Master advanced LearnM8 configuration

**What you'll learn:**
- Configuring custom learners (Chemprop with custom parameters)
- Designing multi-stage cycles (explore → exploit → diversify)
- Comparing acquisition schedules (greedy, and UCB with different `beta` values)
- Analyzing selection quality (not enrichment)

**Prerequisites:** 01_quickstart.ipynb, 02_custom_oracles.ipynb

**Outcome:** Configure custom learners and design workflows for different screening objectives

---

### 04_production_screening.ipynb (20 minutes)
**Goal:** Structure a two-stage screening campaign in run mode

**What you'll learn:**
- Hierarchical screening (fast → slow oracles)
- Narrowing the library between stages using model predictions
- Reusing the feature cache between runs
- Reading per-cycle predictions from `prediction_cycle_N.parquet`
- Exporting and analysing results

**Prerequisites:** All previous notebooks

**Outcome:** A template for a multi-stage run-mode screening workflow

---

## Learning Paths

### New Users (Total: ~40 minutes)
1. Start with 01_quickstart.ipynb (understand basics - benchmark mode)
2. Read 02_custom_oracles.ipynb (custom oracles - run mode)
3. Skim 03_advanced_configuration.ipynb (advanced configuration)
4. Bookmark 04_production_screening.ipynb (for later reference)

### Intermediate Users (Total: ~30 minutes)
1. Review 02_custom_oracles.ipynb (oracle patterns)
2. Deep dive 03_advanced_configuration.ipynb (custom learners and cycle design)
3. Study 04_production_screening.ipynb (multi-stage workflow)

### Multi-Stage Screening (Focus on 04)
1. Run 04_production_screening.ipynb
2. Adapt workflow to your use case
3. Reference docs for advanced configuration

## Datasets

Notebooks use curated real molecular datasets from `../data/`:

| Dataset | Size | Columns | Used In | Description |
|---------|------|---------|---------|-------------|
| `ampc_1k_with_scores.csv` | 1,000 | ID, SMILES, dockscore | Notebook 1 | AmpC compounds with docking scores (CSVOracle) |
| `ampc_1k_no_scores.csv` | 1,000 | ID, SMILES | Notebooks 2, 3 | AmpC compounds for custom oracle testing |
| `ampc_30k_no_scores.csv` | 29,054 | ID, SMILES | Notebook 4 | Full AmpC library for the multi-stage workflow |

**Source:** All datasets derived from AmpC beta-lactamase docking screens, randomly sampled to maintain chemical diversity.

For custom datasets, use CSV format with columns:
- `ID`: Unique compound identifier (required)
- `SMILES`: Molecular structure (required)
- Target column(s): Properties to predict (optional, depends on oracle)

## Troubleshooting

### Import Errors
```python
# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))
```

### Slow Execution
- Reuse the feature cache: `cache_dir='.cache'`
- Use smaller datasets: `compounds.head(1000)`
- Reduce cycles: `n_cycles=3`

### Missing Dependencies
```bash
# Core dependencies
conda install -c conda-forge rdkit pandas polars scikit-learn

# Optional (for specific examples)
pip install cdpkit  # For 3D pharmacophores
conda install -c conda-forge vina meeko  # For docking
```

## Running All Notebooks

```bash
# Convert and execute all notebooks
jupyter nbconvert --to notebook --execute 01_quickstart.ipynb
jupyter nbconvert --to notebook --execute 02_custom_oracles.ipynb
jupyter nbconvert --to notebook --execute 03_advanced_configuration.ipynb
jupyter nbconvert --to notebook --execute 04_production_screening.ipynb
```

## Expected Outputs

Some cells contain saved outputs from an earlier run, and cells edited since then have none. To clear all outputs:

```bash
jupyter nbconvert --clear-output --inplace *.ipynb
```

---

**Next Steps:** Run [01_quickstart.ipynb](01_quickstart.ipynb) to begin!

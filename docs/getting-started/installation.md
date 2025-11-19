# Installation

This guide walks you through installing LearnM8 and its dependencies.

## Prerequisites

**Python Version:** LearnM8 requires Python 3.11.9 or compatible versions. We recommend using Conda for environment management.

**System Requirements:**
- 64-bit operating system (Linux, macOS, or Windows)
- 4+ GB RAM (8+ GB recommended for large compound libraries)
- Internet connection for initial installation

## Basic Installation

### Step 1: Create Conda Environment

LearnM8 provides an environment configuration file for easy setup:

```bash
conda env create -f environment.yml
conda activate learnm8
```

This creates a new Conda environment named `learnm8` with all core dependencies.

### Step 2: Install LearnM8

Install LearnM8 in editable mode:

```bash
pip install -e .
```

The `-e` flag installs in development/editable mode, allowing you to modify the source code if needed.

### Step 3: Verify Installation

Test that LearnM8 is correctly installed:

```bash
learnm8 --help
```

You should see the main help message with available subcommands (`run`, `validate`, `list`).

## Optional Dependencies

LearnM8's core functionality works with the basic installation. Additional features require optional dependencies.

### PyTorch Models (MLP, MCDropout, Fastprop)

PyTorch-based neural network models require PyTorch:

```bash
pip install torch
```

For GPU acceleration:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Visit [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific installation instructions.

### Chemprop (Graph Neural Networks)

For state-of-the-art graph neural network models:

```bash
pip install chemprop
```

Chemprop requires PyTorch. It works directly with SMILES strings without requiring featurizers.

### XGBoost (Gradient Boosting)

For high-performance gradient boosting:

```bash
pip install xgboost
```

### BitBIRCH (Molecular Clustering)

For advanced diversity-based acquisition using molecular clustering:

```bash
pip install git+https://github.com/mqcomplab/bitbirch.git
```

BitBIRCH cannot be included in `setup.py` due to its GitHub source, so it must be installed separately.

### All Optional Dependencies

Install all optional features at once:

```bash
pip install torch
pip install chemprop
pip install xgboost
pip install git+https://github.com/mqcomplab/bitbirch.git
```

## Testing Your Installation

### Basic Functionality Test

Verify core functionality with pytest:

```bash
pytest tests/ -m unit
```

This runs unit tests that don't require optional dependencies.

### Full Test Suite

If you have all optional dependencies installed:

```bash
pytest tests/
```

### Quick CLI Test

Test the command-line interface:

```bash
# List available learners (confirms model registry)
learnm8 list learners

# List acquisition strategies
learnm8 list acquisition

# List featurizers
learnm8 list featurizers
```

## Development Installation

If you plan to contribute to LearnM8 or modify the source code:

### Install with Test Dependencies

```bash
pip install -e .[test]
```

This installs additional testing tools:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- `hypothesis` - Property-based testing

### Run Tests with Coverage

```bash
pytest tests/ --cov=learnm8 --cov-report=html
```

Coverage reports are generated in `htmlcov/index.html`.

### Install Documentation Dependencies

For building documentation locally:

```bash
pip install mkdocs-material
```

## Troubleshooting

### RDKit Installation Issues

RDKit is included in the Conda environment file. If you encounter issues:

```bash
conda install -c conda-forge rdkit
```

### SMILES Validation Fails

LearnM8 uses datamol for SMILES validation. Ensure it's installed:

```bash
pip install datamol
```

### Cache Directory Errors

Feature caching requires write permissions. If you see cache-related errors:

```bash
# Specify a custom cache directory
learnm8 run compounds.csv --target Activity --learner gp --featurizer morgan --cache-dir /tmp/learnm8_cache
```

### GPU Not Detected (PyTorch Models)

Verify PyTorch sees your GPU:

```python
import torch
print(torch.cuda.is_available())
```

If `False`, reinstall PyTorch with CUDA support (see PyTorch installation section above).

## Platform-Specific Notes

### Linux

No special considerations. All dependencies install cleanly on modern Linux distributions.

### macOS

Conda installation is recommended on macOS. If using Apple Silicon (M1/M2):

```bash
# Use ARM64 native environment
CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml
```

### Windows

Use Conda for Windows. Windows Subsystem for Linux (WSL2) is recommended for best compatibility.

## Updating LearnM8

To update to the latest version:

```bash
cd /path/to/LearnM8
git pull
pip install -e .
```

If dependencies have changed, recreate the Conda environment:

```bash
conda env remove -n learnm8
conda env create -f environment.yml
conda activate learnm8
pip install -e .
```

## Next Steps

With LearnM8 installed, you're ready to:

- **[Run your first experiment](quickstart.md)** - Get started in under 5 minutes
- **[Understand core concepts](concepts.md)** - Learn active learning fundamentals
- **[Explore tutorials](../tutorials/running-experiments.md)** - Detailed workflow guides

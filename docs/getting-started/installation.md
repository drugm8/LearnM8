# Installation

LearnM8 supports Linux. It requires Python 3.11 or newer; the provided Conda environment installs Python 3.12.

## Install from source

```bash
git clone https://github.com/drugm8/LearnM8.git
cd LearnM8
conda env create -f environment.yml
conda activate learnm8
pip install --no-deps -e .
```

The Conda environment provides every dependency, including PyTorch, Chemprop, XGBoost, RDKit and the GPU stack. `--no-deps` stops pip from replacing Conda-managed packages.

All core dependencies are required: LearnM8 fails at import if one is missing.

## Install from PyPI

```bash
pip install learnm8
```

This pulls the core dependencies from PyPI. The GPU learners (`gpu_gp`, `svgp`, `rf_fil`, `ridge_cuml`) need extra packages, available as the `gpu`, `gpu-gp` and `gpu-rapids` extras (for example `pip install "learnm8[gpu]"`); the Conda environment is the supported way to get RAPIDS cuML.

## GPU learners

`gpu_gp` and `svgp` need GPyTorch and GAUCHE; `rf_fil` and `ridge_cuml` need RAPIDS cuML. All 19 learners are always listed. If GPyTorch is missing, creating `gpu_gp` or `svgp` raises `ConfigurationError`; if RAPIDS cuML is missing, `rf_fil` and `ridge_cuml` raise `LearnerError` when they train.

## Verify the installation

```bash
learnm8 --help
```

You should see the subcommands `run`, `validate` and `suggest`. To list the available components:

```bash
python -c "from learnm8.api import list_available_learners; print(list_available_learners())"
```

## Troubleshooting

**Cache directory errors.** The feature cache needs a writable local filesystem; NFS, Lustre and GlusterFS are not supported. Point it at a local path with `--cache-dir` (or `cache_dir=` in Python).

**GPU not detected.** Check that PyTorch sees the device:

```python
import torch
print(torch.cuda.is_available())
```

**Updating.** Pull the latest source and reinstall. If `environment.yml` changed, recreate the environment (`conda env remove -n learnm8`, then repeat the install steps).

Contributors: see `CONTRIBUTING.md` in the repository.

## Next steps

- [Quickstart](quickstart.md)
- [Core Concepts](concepts.md)
- [Running Experiments](../tutorials/running-experiments.md)

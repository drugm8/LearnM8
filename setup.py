"""Setup script for LearnM8 package."""

from setuptools import setup, find_packages

setup(
    name="learnm8",
    version="0.10.0",
    description="Active Learning for Molecular Screening - Modern Architecture with Polars",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "polars>=1.0.0",
        "pyarrow>=15.0.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "rdkit>=2022.03.0",
        "joblib>=1.0.0",
        "rich>=10.0.0",
        "h5py>=3.0.0",
        "hdf5plugin>=3.0.0",
        "pyyaml>=5.0.0",
        "datamol>=0.12.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "hypothesis>=6.0.0",
            "psutil>=5.0.0",
        ],
        "diversity": [
            "umap-learn>=0.5.0",  # For UMAP dimensionality reduction
            "hdbscan>=0.8.0",     # For improved clustering (optional)
        ],
        "bitbirch": [
            # BitBIRCH installation via pip install git+https://github.com/mqcomplab/bitbirch.git
            # Note: This cannot be specified as a regular dependency due to git source
        ],
        "cli": [
            "tqdm>=4.0.0",  # For progress bars (optional, graceful fallback)
        ],
        "full": [
            "umap-learn>=0.5.0",
            "hdbscan>=0.8.0",
            "tqdm>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "learnm8=learnm8.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
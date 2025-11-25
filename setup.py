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
        "matplotlib>=3.5.0",
    ],
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
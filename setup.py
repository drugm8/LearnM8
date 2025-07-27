"""Setup script for LearnM8 package."""

from setuptools import setup, find_packages

setup(
    name="learnm8",
    version="0.2.0",
    description="Active Learning for Molecular Screening",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "rdkit>=2022.03.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "learnm8=learnm8.cli.__main__:main",
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
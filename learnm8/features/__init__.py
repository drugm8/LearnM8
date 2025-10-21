"""
Feature extraction module for molecular fingerprints and descriptors.

This module provides a clean, function-based API for extracting molecular features
with automatic HDF5 caching and parallel processing optimization.

Main API:
    extract_features: Extract features with caching and parallelization
"""

from .extraction import extract_features

__all__ = ['extract_features']

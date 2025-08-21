"""Data generation module for clustering experiments."""

from .synthetic import SyntheticDataGenerator
from .real_datasets import RealDatasetLoader

__all__ = ["SyntheticDataGenerator", "RealDatasetLoader"]
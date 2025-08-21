"""
Clustering Analysis Package

A comprehensive framework for comparative analysis of clustering algorithms.
This package provides implementations of multiple clustering methods with
rigorous evaluation metrics and visualization capabilities.
"""

from .core.experiment import ClusteringExperiment
from .algorithms import (
    KMeansClusterer,
    FuzzyCMeansClusterer, 
    GaussianMixtureClusterer,
    DBSCANClusterer,
    SpectralClusterer
)
from .data_generation import SyntheticDataGenerator
from .evaluation import ClusteringMetrics
from .visualization import ClusteringVisualizer

__version__ = "1.0.0"
__author__ = "José"

__all__ = [
    "ClusteringExperiment",
    "KMeansClusterer",
    "FuzzyCMeansClusterer",
    "GaussianMixtureClusterer", 
    "DBSCANClusterer",
    "SpectralClusterer",
    "SyntheticDataGenerator",
    "ClusteringMetrics",
    "ClusteringVisualizer"
]
"""Density-based clustering algorithms implementation."""

import numpy as np
import time
import warnings
from typing import Optional, Tuple
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from loguru import logger

from .base import BaseClusterer


class DBSCANClusterer(BaseClusterer):
    """DBSCAN clustering algorithm with parameter optimization."""
    
    def __init__(self, n_clusters: Optional[int] = None, **kwargs):
        """
        Initialize DBSCAN clusterer.
        
        Args:
            n_clusters: Target number of clusters (for optimization)
            **kwargs: Additional parameters
        """
        default_params = {
            'eps': 0.5,
            'min_samples': 5,
            'eps_range': [0.05, 12.0],
            'eps_step': 0.05,
            'min_samples_range': [1, 50],
            'min_samples_step': 2
        }
        default_params.update(kwargs)
        super().__init__("DBSCAN", n_clusters=n_clusters, **default_params)
        self.best_params = {}
        
    def fit(self, X: np.ndarray, optimize: bool = True, 
            true_labels: Optional[np.ndarray] = None) -> 'DBSCANClusterer':
        """
        Fit DBSCAN to data with optional parameter optimization.
        
        Args:
            X: Input data
            optimize: Whether to optimize parameters
            true_labels: True labels for optimization (if available)
        """
        start_time = time.time()
        
        if optimize and true_labels is not None:
            self._optimize_parameters(X, true_labels)
        else:
            # Use provided or default parameters
            self.model = DBSCAN(
                eps=self.params['eps'],
                min_samples=self.params['min_samples']
            )
            self.labels_ = self.model.fit_predict(X)
            
        self.fit_time = time.time() - start_time
        logger.debug(f"DBSCAN fitted in {self.fit_time:.4f} seconds")
        return self
    
    def _optimize_parameters(self, X: np.ndarray, true_labels: np.ndarray) -> None:
        """Optimize DBSCAN parameters using grid search."""
        eps_values = np.arange(
            self.params['eps_range'][0],
            self.params['eps_range'][1], 
            self.params['eps_step']
        )
        min_samples_values = list(range(
            self.params['min_samples_range'][0],
            self.params['min_samples_range'][1],
            self.params['min_samples_step']
        ))
        
        best_score = -1
        best_eps = None
        best_min_samples = None
        best_labels = None
        
        logger.info("Optimizing DBSCAN parameters...")
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(X)
                score = adjusted_rand_score(true_labels, dbscan_labels)
                
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = dbscan_labels
                    
                # Early stopping for good scores
                if score > 0.95:
                    break
            if score > 0.9:
                break
                
        self.best_params = {
            'eps': best_eps,
            'min_samples': best_min_samples,
            'score': best_score
        }
        
        # Fit final model with best parameters
        self.model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        self.labels_ = best_labels
        
        logger.info(f"Best DBSCAN params: eps={best_eps:.3f}, "
                   f"min_samples={best_min_samples}, score={best_score:.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.fit_predict(X)


class SpectralClusterer(BaseClusterer):
    """Spectral clustering algorithm with parameter optimization."""
    
    def __init__(self, n_clusters: int = 3, **kwargs):
        """
        Initialize Spectral clusterer.
        
        Args:
            n_clusters: Number of clusters
            **kwargs: Additional parameters
        """
        default_params = {
            'eigen_solver': 'arpack',
            'affinity': 'nearest_neighbors',
            'n_neighbors': 10,
            'n_neighbors_range': [2, 50]
        }
        default_params.update(kwargs)
        super().__init__("Spectral Clustering", n_clusters=n_clusters, **default_params)
        self.best_params = {}
        
    def fit(self, X: np.ndarray, optimize: bool = True,
            true_labels: Optional[np.ndarray] = None) -> 'SpectralClusterer':
        """
        Fit Spectral Clustering to data with optional parameter optimization.
        
        Args:
            X: Input data
            optimize: Whether to optimize parameters
            true_labels: True labels for optimization (if available)
        """
        start_time = time.time()
        
        if optimize and true_labels is not None:
            self._optimize_parameters(X, true_labels)
        else:
            # Use provided or default parameters
            self.model = SpectralClustering(
                n_clusters=self.params['n_clusters'],
                eigen_solver=self.params['eigen_solver'],
                affinity=self.params['affinity'],
                n_neighbors=self.params['n_neighbors']
            )
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.labels_ = self.model.fit_predict(X)
                
        self.fit_time = time.time() - start_time
        logger.debug(f"Spectral Clustering fitted in {self.fit_time:.4f} seconds")
        return self
    
    def _optimize_parameters(self, X: np.ndarray, true_labels: np.ndarray) -> None:
        """Optimize Spectral Clustering parameters."""
        n_neighbors_values = range(
            self.params['n_neighbors_range'][0],
            self.params['n_neighbors_range'][1]
        )
        
        best_score = -1
        best_n_neighbors = None
        best_labels = None
        
        logger.info("Optimizing Spectral Clustering parameters...")
        
        for n_neighbors in n_neighbors_values:
            sc = SpectralClustering(
                n_clusters=self.params['n_clusters'],
                eigen_solver=self.params['eigen_solver'],
                affinity=self.params['affinity'],
                n_neighbors=n_neighbors
            )
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                sc_labels = sc.fit_predict(X)
                
            score = adjusted_rand_score(true_labels, sc_labels)
            
            if score > best_score:
                best_score = score
                best_n_neighbors = n_neighbors
                best_labels = sc_labels
                
            # Early stopping
            if score > 0.9:
                break
                
        self.best_params = {
            'n_neighbors': best_n_neighbors,
            'score': best_score
        }
        
        # Fit final model with best parameters
        self.model = SpectralClustering(
            n_clusters=self.params['n_clusters'],
            eigen_solver=self.params['eigen_solver'],
            affinity=self.params['affinity'],
            n_neighbors=best_n_neighbors
        )
        
        self.labels_ = best_labels
        
        logger.info(f"Best Spectral params: n_neighbors={best_n_neighbors}, "
                   f"score={best_score:.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Note: Spectral clustering doesn't have predict method,
        # so we need to refit for new data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return self.model.fit_predict(X)
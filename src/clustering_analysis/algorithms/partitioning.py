"""Partitioning clustering algorithms implementation."""

import numpy as np
import time
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
from loguru import logger

from .base import BaseClusterer


class KMeansClusterer(BaseClusterer):
    """K-Means clustering algorithm implementation."""
    
    def __init__(self, n_clusters: int = 3, **kwargs):
        """
        Initialize K-Means clusterer.
        
        Args:
            n_clusters: Number of clusters
            **kwargs: Additional parameters for KMeans
        """
        default_params = {
            'n_init': 10,
            'max_iter': 10000,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__("K-Means", n_clusters=n_clusters, **default_params)
        
    def fit(self, X: np.ndarray) -> 'KMeansClusterer':
        """Fit K-Means to data."""
        start_time = time.time()
        
        self.model = KMeans(
            n_clusters=self.params['n_clusters'],
            n_init=self.params['n_init'],
            max_iter=self.params['max_iter'],
            random_state=self.params.get('random_state')
        )
        
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.fit_time = time.time() - start_time
        
        logger.debug(f"K-Means fitted in {self.fit_time:.4f} seconds")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class FuzzyCMeansClusterer(BaseClusterer):
    """Fuzzy C-Means clustering algorithm implementation."""
    
    def __init__(self, n_clusters: int = 3, **kwargs):
        """
        Initialize Fuzzy C-Means clusterer.
        
        Args:
            n_clusters: Number of clusters
            **kwargs: Additional parameters for FCM
        """
        default_params = {
            'max_iter': 150,
            'error': 1e-5,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__("Fuzzy C-Means", n_clusters=n_clusters, **default_params)
        
    def fit(self, X: np.ndarray) -> 'FuzzyCMeansClusterer':
        """Fit Fuzzy C-Means to data."""
        start_time = time.time()
        
        # Use scikit-fuzzy for Fuzzy C-Means clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, self.params['n_clusters'], 2, 
            error=self.params['error'], 
            maxiter=self.params['max_iter']
        )
        
        # Store results
        self.cluster_centers_ = cntr
        self.membership_matrix = u
        
        # Convert fuzzy memberships to hard labels
        self.labels_ = np.argmax(u, axis=0)
        self.fit_time = time.time() - start_time
        
        logger.debug(f"Fuzzy C-Means fitted in {self.fit_time:.4f} seconds")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # For FCM, we return the hard clustering (most likely cluster)
        return self.labels_
    
    def get_membership_matrix(self) -> Optional[np.ndarray]:
        """Get fuzzy membership matrix."""
        return getattr(self, 'membership_matrix', None)


class GaussianMixtureClusterer(BaseClusterer):
    """Gaussian Mixture Model clustering algorithm implementation."""
    
    def __init__(self, n_clusters: int = 3, **kwargs):
        """
        Initialize Gaussian Mixture clusterer.
        
        Args:
            n_clusters: Number of components
            **kwargs: Additional parameters for GaussianMixture
        """
        default_params = {
            'covariance_type': 'full',
            'max_iter': 100,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__("Gaussian Mixture", n_clusters=n_clusters, **default_params)
        
    def fit(self, X: np.ndarray) -> 'GaussianMixtureClusterer':
        """Fit Gaussian Mixture Model to data."""
        start_time = time.time()
        
        self.model = GaussianMixture(
            n_components=self.params['n_clusters'],
            covariance_type=self.params['covariance_type'],
            max_iter=self.params['max_iter'],
            random_state=self.params.get('random_state')
        )
        
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        self.fit_time = time.time() - start_time
        
        logger.debug(f"Gaussian Mixture fitted in {self.fit_time:.4f} seconds")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
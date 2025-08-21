"""Tests for clustering algorithms."""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from src.clustering_analysis.algorithms import (
    KMeansClusterer,
    FuzzyCMeansClusterer,
    GaussianMixtureClusterer,
    DBSCANClusterer,
    SpectralClusterer
)


@pytest.fixture
def sample_data():
    """Create sample clustering data."""
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, 
                      cluster_std=1.5, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


class TestKMeansClusterer:
    """Test KMeans clustering."""
    
    def test_initialization(self):
        """Test proper initialization."""
        clusterer = KMeansClusterer(n_clusters=3)
        assert clusterer.name == "K-Means"
        assert clusterer.params['n_clusters'] == 3
        
    def test_fit_predict(self, sample_data):
        """Test fit and predict functionality."""
        X, y_true = sample_data
        
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(X)
        
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(X)
        assert clusterer.fit_time is not None
        assert clusterer.fit_time > 0
        
    def test_predict(self, sample_data):
        """Test prediction on new data."""
        X, y_true = sample_data
        
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(X)
        predictions = clusterer.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(isinstance(p, (int, np.integer)) for p in predictions)


class TestFuzzyCMeansClusterer:
    """Test Fuzzy C-Means clustering."""
    
    def test_initialization(self):
        """Test proper initialization."""
        clusterer = FuzzyCMeansClusterer(n_clusters=3)
        assert clusterer.name == "Fuzzy C-Means"
        assert clusterer.params['n_clusters'] == 3
        
    def test_fit_predict(self, sample_data):
        """Test fit and predict functionality."""
        X, y_true = sample_data
        
        clusterer = FuzzyCMeansClusterer(n_clusters=3)
        clusterer.fit(X)
        
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(X)
        assert hasattr(clusterer, 'membership_matrix')


class TestGaussianMixtureClusterer:
    """Test Gaussian Mixture Model clustering."""
    
    def test_initialization(self):
        """Test proper initialization."""
        clusterer = GaussianMixtureClusterer(n_clusters=3)
        assert clusterer.name == "Gaussian Mixture"
        assert clusterer.params['n_clusters'] == 3
        
    def test_fit_predict(self, sample_data):
        """Test fit and predict functionality."""
        X, y_true = sample_data
        
        clusterer = GaussianMixtureClusterer(n_clusters=3)
        clusterer.fit(X)
        
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(X)
        
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y_true = sample_data
        
        clusterer = GaussianMixtureClusterer(n_clusters=3)
        clusterer.fit(X)
        probas = clusterer.predict_proba(X[:10])
        
        assert probas.shape == (10, 3)
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1


class TestDBSCANClusterer:
    """Test DBSCAN clustering."""
    
    def test_initialization(self):
        """Test proper initialization."""
        clusterer = DBSCANClusterer(n_clusters=3)
        assert clusterer.name == "DBSCAN"
        
    def test_fit_predict(self, sample_data):
        """Test fit and predict functionality."""
        X, y_true = sample_data
        
        clusterer = DBSCANClusterer(n_clusters=3, eps=0.5, min_samples=5)
        clusterer.fit(X, optimize=False)  # Skip optimization for faster test
        
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(X)


class TestSpectralClusterer:
    """Test Spectral clustering."""
    
    def test_initialization(self):
        """Test proper initialization."""
        clusterer = SpectralClusterer(n_clusters=3)
        assert clusterer.name == "Spectral Clustering"
        assert clusterer.params['n_clusters'] == 3
        
    def test_fit_predict(self, sample_data):
        """Test fit and predict functionality."""
        X, y_true = sample_data
        
        clusterer = SpectralClusterer(n_clusters=3, n_neighbors=10)
        clusterer.fit(X, optimize=False)  # Skip optimization for faster test
        
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(X)


class TestBaseClusterer:
    """Test base clusterer functionality."""
    
    def test_get_set_params(self, sample_data):
        """Test parameter getting and setting."""
        clusterer = KMeansClusterer(n_clusters=3)
        
        # Test get_params
        params = clusterer.get_params()
        assert 'n_clusters' in params
        assert params['n_clusters'] == 3
        
        # Test set_params
        clusterer.set_params(n_clusters=4)
        assert clusterer.params['n_clusters'] == 4
        
    def test_execution_time(self, sample_data):
        """Test execution time recording."""
        X, y_true = sample_data
        
        clusterer = KMeansClusterer(n_clusters=3)
        assert clusterer.get_execution_time() is None
        
        clusterer.fit(X)
        exec_time = clusterer.get_execution_time()
        assert exec_time is not None
        assert exec_time > 0
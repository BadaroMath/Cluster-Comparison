"""Base class for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import time
from loguru import logger


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize base clusterer.
        
        Args:
            name: Name of the clustering algorithm
            **kwargs: Algorithm-specific parameters
        """
        self.name = name
        self.params = kwargs
        self.model = None
        self.fit_time = None
        self.labels_ = None
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterer':
        """
        Fit the clustering algorithm to data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod 
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Cluster labels array of shape (n_samples,)
        """
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the algorithm and predict cluster labels.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Cluster labels array of shape (n_samples,)
        """
        self.fit(X)
        return self.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return self.params.copy()
    
    def set_params(self, **params) -> 'BaseClusterer':
        """Set algorithm parameters."""
        self.params.update(params)
        return self
    
    def get_execution_time(self) -> Optional[float]:
        """Get algorithm execution time in seconds."""
        return self.fit_time
    
    def save_results(self, 
                     labels: np.ndarray, 
                     true_labels: np.ndarray,
                     output_path: str) -> None:
        """
        Save clustering results to CSV.
        
        Args:
            labels: Predicted cluster labels
            true_labels: True cluster labels  
            output_path: Path to save results
        """
        df_results = pd.DataFrame({
            'predicted_labels': labels,
            'true_labels': true_labels
        })
        df_results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    def __repr__(self) -> str:
        """String representation of the clusterer."""
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
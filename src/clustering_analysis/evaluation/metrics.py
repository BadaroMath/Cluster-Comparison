"""Comprehensive clustering evaluation metrics."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    adjusted_rand_score, 
    rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.spatial.distance import pdist, squareform
from loguru import logger


class ClusteringMetrics:
    """Comprehensive clustering evaluation metrics calculator."""
    
    def __init__(self):
        """Initialize clustering metrics calculator."""
        self.metrics = {}
        
    def calculate_all_metrics(self,
                             X: np.ndarray,
                             predicted_labels: np.ndarray,
                             true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all available clustering metrics.
        
        Args:
            X: Original data
            predicted_labels: Predicted cluster labels
            true_labels: True cluster labels (if available)
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Supervised metrics (require true labels)
        if true_labels is not None:
            metrics.update(self._calculate_supervised_metrics(predicted_labels, true_labels))
            
        # Unsupervised metrics (don't require true labels)
        metrics.update(self._calculate_unsupervised_metrics(X, predicted_labels))
        
        self.metrics = metrics
        return metrics
    
    def _calculate_supervised_metrics(self,
                                    predicted_labels: np.ndarray,
                                    true_labels: np.ndarray) -> Dict[str, float]:
        """Calculate metrics that require true labels."""
        metrics = {}
        
        try:
            # Adjusted Rand Index
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)
            
            # Rand Index
            metrics['rand_score'] = rand_score(true_labels, predicted_labels)
            
            # Classification Proportion (accuracy-like measure)
            metrics['classification_proportion'] = self._calculate_classification_proportion(
                true_labels, predicted_labels
            )
            
            # Variation of Information
            metrics['variation_of_information'] = self._calculate_variation_of_information(
                true_labels, predicted_labels
            )
            
        except Exception as e:
            logger.error(f"Error calculating supervised metrics: {e}")
            
        return metrics
    
    def _calculate_unsupervised_metrics(self,
                                      X: np.ndarray,
                                      predicted_labels: np.ndarray) -> Dict[str, float]:
        """Calculate metrics that don't require true labels."""
        metrics = {}
        
        try:
            # Filter out noise points (label -1 in DBSCAN)
            valid_mask = predicted_labels != -1
            if not np.any(valid_mask):
                logger.warning("No valid clusters found (all points labeled as noise)")
                return {}
                
            X_valid = X[valid_mask]
            labels_valid = predicted_labels[valid_mask]
            
            # Check if we have at least 2 clusters
            unique_labels = np.unique(labels_valid)
            if len(unique_labels) < 2:
                logger.warning("Need at least 2 clusters for unsupervised metrics")
                return metrics
                
            # Silhouette Score
            if len(X_valid) > 1:
                metrics['silhouette_score'] = silhouette_score(X_valid, labels_valid)
                
            # Calinski-Harabasz Index
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_valid, labels_valid)
            
            # Davies-Bouldin Index
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_valid, labels_valid)
            
            # Dunn Index
            metrics['dunn_index'] = self._calculate_dunn_index(X_valid, labels_valid)
            
        except Exception as e:
            logger.error(f"Error calculating unsupervised metrics: {e}")
            
        return metrics
    
    def _calculate_classification_proportion(self,
                                           true_labels: np.ndarray,
                                           predicted_labels: np.ndarray) -> float:
        """Calculate classification proportion (similar to accuracy)."""
        # Create contingency matrix
        cm = contingency_matrix(true_labels, predicted_labels)
        
        # Find the best matching between true and predicted clusters
        # This is essentially finding the maximum matching in a bipartite graph
        from scipy.optimize import linear_sum_assignment
        
        # Convert to cost matrix (negative because we want maximum)
        cost_matrix = -cm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Calculate maximum correct classifications
        max_correct = cm[row_indices, col_indices].sum()
        total = len(true_labels)
        
        return max_correct / total if total > 0 else 0.0
    
    def _calculate_variation_of_information(self,
                                          true_labels: np.ndarray,
                                          predicted_labels: np.ndarray) -> float:
        """Calculate Variation of Information."""
        n = len(true_labels)
        
        if n == 0:
            return 0.0
            
        # Calculate contingency matrix
        cm = contingency_matrix(true_labels, predicted_labels)
        
        # Calculate marginals
        a = cm.sum(axis=1)  # True cluster sizes
        b = cm.sum(axis=0)  # Predicted cluster sizes
        
        # Calculate entropies
        def entropy(counts):
            counts = counts[counts > 0]  # Remove zeros
            probs = counts / counts.sum()
            return -(probs * np.log(probs)).sum()
        
        H_true = entropy(a)
        H_pred = entropy(b)
        
        # Calculate mutual information
        cm_nonzero = cm[cm > 0]
        if len(cm_nonzero) == 0:
            return H_true + H_pred
            
        joint_probs = cm_nonzero / n
        marginal_true = np.repeat(a[:, np.newaxis], cm.shape[1], axis=1)[cm > 0] / n
        marginal_pred = np.repeat(b[np.newaxis, :], cm.shape[0], axis=0)[cm > 0] / n
        
        mutual_info = (joint_probs * np.log(joint_probs / (marginal_true * marginal_pred))).sum()
        
        # Variation of Information = H(X) + H(Y) - 2*I(X,Y)
        vi = H_true + H_pred - 2 * mutual_info
        
        return max(0.0, vi)  # Ensure non-negative
    
    def _calculate_dunn_index(self,
                             X: np.ndarray,
                             labels: np.ndarray) -> float:
        """Calculate Dunn Index."""
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return 0.0
            
        # Calculate distances between all points
        distances = pdist(X)
        distance_matrix = squareform(distances)
        
        # Calculate minimum inter-cluster distance
        min_inter_cluster_dist = float('inf')
        
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels[i+1:], i+1):
                # Get points in each cluster
                cluster1_indices = np.where(labels == label1)[0]
                cluster2_indices = np.where(labels == label2)[0]
                
                # Find minimum distance between clusters
                inter_dists = distance_matrix[np.ix_(cluster1_indices, cluster2_indices)]
                min_dist = np.min(inter_dists)
                min_inter_cluster_dist = min(min_inter_cluster_dist, min_dist)
        
        # Calculate maximum intra-cluster distance
        max_intra_cluster_dist = 0.0
        
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 1:
                intra_dists = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                max_dist = np.max(intra_dists)
                max_intra_cluster_dist = max(max_intra_cluster_dist, max_dist)
        
        # Dunn Index = min_inter_cluster_dist / max_intra_cluster_dist
        if max_intra_cluster_dist == 0:
            return float('inf') if min_inter_cluster_dist > 0 else 0.0
            
        return min_inter_cluster_dist / max_intra_cluster_dist
    
    def save_metrics(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        """Save metrics to file."""
        data = {'metrics': self.metrics}
        if metadata:
            data['metadata'] = metadata
            
        df = pd.DataFrame([data])
        df.to_json(filepath, orient='records', indent=2)
        logger.info(f"Metrics saved to {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        if not self.metrics:
            return {}
            
        return {
            'n_metrics': len(self.metrics),
            'metrics': self.metrics,
            'supervised_metrics': {
                k: v for k, v in self.metrics.items() 
                if k in ['adjusted_rand_score', 'rand_score', 'classification_proportion', 
                        'variation_of_information']
            },
            'unsupervised_metrics': {
                k: v for k, v in self.metrics.items()
                if k in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 
                        'dunn_index']
            }
        }
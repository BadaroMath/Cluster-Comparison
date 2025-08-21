"""Real dataset loader for clustering benchmarks."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from loguru import logger
from sklearn.preprocessing import StandardScaler


class RealDatasetLoader:
    """Loader for real clustering benchmark datasets."""
    
    # Dataset configurations
    DATASET_CONFIGS = {
        'compound': {'label_col': 'label', 'expected_clusters': 6},
        'aggregation': {'label_col': 'label', 'expected_clusters': 7},
        'pathbased': {'label_col': 'label', 'expected_clusters': 3},
        's2': {'label_col': 'labels', 'expected_clusters': 15},
        'flame': {'label_col': 'label', 'expected_clusters': 2},
        'face': {'label_col': None, 'expected_clusters': 5}  # No labels
    }
    
    def __init__(self, data_dir: str = "data/real"):
        """
        Initialize real dataset loader.
        
        Args:
            data_dir: Directory containing real datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, 
                     dataset_name: str,
                     standardize: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load a real clustering dataset.
        
        Args:
            dataset_name: Name of dataset to load
            standardize: Whether to standardize features
            
        Returns:
            Tuple of (data, labels) or (data, None) if no labels
        """
        if dataset_name not in self.DATASET_CONFIGS:
            available = list(self.DATASET_CONFIGS.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
            
        config = self.DATASET_CONFIGS[dataset_name]
        filepath = self.data_dir / f"{dataset_name}.csv"
        
        if not filepath.exists():
            logger.warning(f"Dataset file not found: {filepath}")
            return self._create_sample_data(dataset_name, config)
            
        # Load dataset
        df = pd.read_csv(filepath)
        
        # Extract labels if available
        labels = None
        if config['label_col'] and config['label_col'] in df.columns:
            labels = df[config['label_col']].values
            df = df.drop(columns=[config['label_col']])
            
        # Extract features
        X = df.values
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
        logger.info(f"Loaded dataset '{dataset_name}': "
                   f"shape={X.shape}, clusters={len(np.unique(labels)) if labels is not None else 'unknown'}")
        
        return X, labels
    
    def _create_sample_data(self, 
                           dataset_name: str, 
                           config: Dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sample data when dataset file is missing."""
        from sklearn.datasets import make_blobs
        
        logger.warning(f"Creating sample data for {dataset_name}")
        
        # Create sample data based on expected clusters
        n_samples = 500
        n_features = 2
        n_clusters = config['expected_clusters']
        
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,
            n_features=n_features,
            cluster_std=1.5,
            random_state=42
        )
        
        # Save sample data
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
        if config['label_col']:
            df[config['label_col']] = y
            
        filepath = self.data_dir / f"{dataset_name}.csv"
        df.to_csv(filepath, index=False)
        
        return X, y if config['label_col'] else None
    
    def list_available_datasets(self) -> List[str]:
        """List all available real datasets."""
        return list(self.DATASET_CONFIGS.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset."""
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        config = self.DATASET_CONFIGS[dataset_name].copy()
        filepath = self.data_dir / f"{dataset_name}.csv"
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            config.update({
                'n_samples': len(df),
                'n_features': len(df.columns) - (1 if config['label_col'] else 0),
                'file_exists': True
            })
        else:
            config['file_exists'] = False
            
        return config
    
    def create_toy_datasets(self) -> None:
        """Create toy versions of clustering datasets for demonstration."""
        from sklearn.datasets import (make_circles, make_moons, make_blobs,
                                    make_classification)
        
        logger.info("Creating toy clustering datasets...")
        
        # Noisy circles
        X, y = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)
        self._save_toy_dataset(X, y, "noisy_circles")
        
        # Noisy moons  
        X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
        self._save_toy_dataset(X, y, "noisy_moons")
        
        # Varied blobs
        X, y = make_blobs(n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state=42)
        self._save_toy_dataset(X, y, "varied_blobs")
        
        # Anisotropic blobs
        X, y = make_blobs(n_samples=1000, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
        self._save_toy_dataset(X, y, "aniso_blobs")
        
        logger.info("Toy datasets created successfully")
        
    def _save_toy_dataset(self, X: np.ndarray, y: np.ndarray, name: str) -> None:
        """Save toy dataset to file."""
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        df['labels'] = y
        
        filepath = self.data_dir / f"{name}.csv"
        df.to_csv(filepath, index=False)
        
        # Add to configs
        self.DATASET_CONFIGS[name] = {
            'label_col': 'labels',
            'expected_clusters': len(np.unique(y))
        }
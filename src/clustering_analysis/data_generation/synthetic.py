"""Synthetic data generation using MixSim methodology."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from loguru import logger
import subprocess
import os


class SyntheticDataGenerator:
    """Generator for synthetic clustering datasets using MixSim methodology."""
    
    def __init__(self, output_dir: str = "data/synthetic"):
        """
        Initialize synthetic data generator.
        
        Args:
            output_dir: Directory to save generated datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_dataset(self,
                        bar_omega: float,
                        K: int, 
                        p: int,
                        n: int,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic dataset with specified parameters.
        
        Args:
            bar_omega: Average overlap between clusters (0.0 - 1.0)
            K: Number of clusters
            p: Number of dimensions
            n: Number of samples
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (data, labels)
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Check if dataset already exists
        filename = f"simdataset_{bar_omega}_{K}_{p}_{n}.csv"
        filepath = self.output_dir / filename
        
        if filepath.exists():
            logger.info(f"Loading existing dataset: {filename}")
            df = pd.read_csv(filepath)
            X = df.drop(columns=['id']).values
            y = df['id'].values
            return X, y
            
        # Generate using R MixSim if available, otherwise use Python fallback
        try:
            return self._generate_with_mixsim_r(bar_omega, K, p, n)
        except Exception as e:
            logger.warning(f"R MixSim failed: {e}. Using Python fallback.")
            return self._generate_with_sklearn(bar_omega, K, p, n)
    
    def _generate_with_mixsim_r(self,
                               bar_omega: float,
                               K: int,
                               p: int, 
                               n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dataset using R MixSim library."""
        r_script = f'''
        library("MixSim")
        
        # Set working directory
        setwd("{self.output_dir.absolute()}")
        
        # Generate mixture parameters
        Q <- MixSim(BarOmega = {bar_omega}, K = {K}, p = {p}, resN = 100000)
        
        # Generate dataset
        A <- simdataset(n = {n}, Pi = Q$Pi, Mu = Q$Mu, S = Q$S)
        
        # Save to CSV
        write.csv(A, "simdataset_{bar_omega}_{K}_{p}_{n}.csv", row.names=FALSE)
        '''
        
        # Execute R script
        result = subprocess.run(['R', '--vanilla', '-e', r_script], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")
            
        # Load generated data
        filepath = self.output_dir / f"simdataset_{bar_omega}_{K}_{p}_{n}.csv"
        df = pd.read_csv(filepath)
        X = df.drop(columns=['id']).values
        y = df['id'].values
        
        logger.info(f"Generated dataset with MixSim: {bar_omega}, {K}, {p}, {n}")
        return X, y
    
    def _generate_with_sklearn(self,
                              bar_omega: float,
                              K: int,
                              p: int,
                              n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback dataset generation using sklearn."""
        from sklearn.datasets import make_blobs
        
        # Adjust cluster separation based on bar_omega
        # Higher bar_omega means more overlap (lower separation)
        cluster_std = 1.0 + (bar_omega * 2.0)  # Range: 1.0 to 3.0
        
        X, y = make_blobs(
            n_samples=n,
            centers=K,
            n_features=p,
            cluster_std=cluster_std,
            random_state=42
        )
        
        # Save to match MixSim format
        df = pd.DataFrame(X, columns=[f'X.{i+1}' for i in range(p)])
        df['id'] = y
        
        filename = f"simdataset_{bar_omega}_{K}_{p}_{n}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"Generated fallback dataset: {bar_omega}, {K}, {p}, {n}")
        return X, y
    
    def generate_experiment_grid(self, experiment_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate all datasets for an experiment grid.
        
        Args:
            experiment_config: Configuration with parameter grids
            
        Returns:
            List of generated dataset metadata
        """
        datasets = []
        
        for experiment_name, params in experiment_config.items():
            logger.info(f"Generating datasets for {experiment_name}")
            
            bar_omega_values = params.get('bar_omega', [0.0])
            K_values = params.get('K', [3])
            P_values = params.get('P', [5])
            N_values = params.get('N', [5000])
            
            for bar_omega in bar_omega_values:
                for K in K_values:
                    for P in P_values:
                        for N in N_values:
                            try:
                                X, y = self.generate_dataset(bar_omega, K, P, N)
                                
                                datasets.append({
                                    'experiment': experiment_name,
                                    'bar_omega': bar_omega,
                                    'K': K,
                                    'P': P,
                                    'N': N,
                                    'n_samples': len(X),
                                    'n_features': X.shape[1],
                                    'n_clusters': len(np.unique(y))
                                })
                                
                            except Exception as e:
                                logger.error(f"Failed to generate dataset "
                                           f"({bar_omega}, {K}, {P}, {N}): {e}")
                                
        logger.info(f"Generated {len(datasets)} datasets")
        return datasets
    
    def get_dataset_path(self,
                        bar_omega: float,
                        K: int,
                        p: int,
                        n: int) -> Path:
        """Get path to dataset file."""
        filename = f"simdataset_{bar_omega}_{K}_{p}_{n}.csv"
        return self.output_dir / filename
        
    def load_dataset(self,
                    bar_omega: float,
                    K: int,
                    p: int,
                    n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load existing dataset."""
        filepath = self.get_dataset_path(bar_omega, K, p, n)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
            
        df = pd.read_csv(filepath)
        X = df.drop(columns=['id']).values
        y = df['id'].values
        
        return X, y
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        datasets = []
        
        for filepath in self.output_dir.glob("simdataset_*.csv"):
            parts = filepath.stem.split('_')[1:]  # Remove 'simdataset'
            if len(parts) == 4:
                bar_omega, K, p, n = parts
                datasets.append({
                    'bar_omega': float(bar_omega),
                    'K': int(K),
                    'p': int(p),
                    'n': int(n),
                    'filepath': filepath
                })
                
        return sorted(datasets, key=lambda x: (x['bar_omega'], x['K'], x['p'], x['n']))
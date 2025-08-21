"""Configuration management for clustering experiments."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ConfigManager:
    """Manages configuration for clustering experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path or "config/experiment_config.yaml"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "output": {
                "base_dir": "data/results",
                "figures_dir": "data/results/figures", 
                "metrics_dir": "data/results/metrics",
                "clusters_dir": "data/results/clusters"
            },
            "algorithms": {
                "kmeans": {
                    "n_init": 10,
                    "max_iter": 10000,
                    "random_state": 42
                },
                "fuzzy_cmeans": {
                    "max_iter": 150,
                    "error": 1e-5,
                    "random_state": 42
                },
                "gaussian_mixture": {
                    "covariance_type": "full",
                    "max_iter": 100,
                    "random_state": 42
                },
                "dbscan": {
                    "eps_range": [0.05, 12.0],
                    "eps_step": 0.05,
                    "min_samples_range": [1, 50],
                    "min_samples_step": 2
                },
                "spectral": {
                    "eigen_solver": "arpack",
                    "affinity": "nearest_neighbors",
                    "n_neighbors_range": [2, 50]
                }
            },
            "parallel": {
                "n_processes": 8
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def create_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for dir_key in ["base_dir", "figures_dir", "metrics_dir", "clusters_dir"]:
            dir_path = Path(self.get(f"output.{dir_key}"))
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
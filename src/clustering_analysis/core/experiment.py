"""Core clustering experiment orchestration."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from loguru import logger
from sklearn.preprocessing import StandardScaler

from .config import ConfigManager
from ..algorithms import (
    KMeansClusterer, FuzzyCMeansClusterer, GaussianMixtureClusterer,
    DBSCANClusterer, SpectralClusterer
)
from ..data_generation import SyntheticDataGenerator, RealDatasetLoader
from ..evaluation import ClusteringMetrics
from ..visualization import ClusteringVisualizer


class ClusteringExperiment:
    """Main experiment orchestrator for clustering analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize clustering experiment.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.config.create_output_dirs()
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator(
            output_dir=self.config.get('data_generation.synthetic.base_path', 'data/synthetic')
        )
        self.dataset_loader = RealDatasetLoader(
            data_dir=self.config.get('data_generation.real_datasets.base_path', 'data/real')
        )
        self.metrics_calculator = ClusteringMetrics()
        self.visualizer = ClusteringVisualizer(
            output_dir=self.config.get('output.figures_dir', 'data/results/figures')
        )
        
        # Initialize algorithms
        self.algorithms = self._initialize_algorithms()
        
        # Results storage
        self.results = []
        
        logger.info("Clustering experiment initialized")
        
    def _initialize_algorithms(self) -> Dict[str, Any]:
        """Initialize clustering algorithms with config parameters."""
        algorithms = {}
        
        # K-Means
        kmeans_params = self.config.get('algorithms.kmeans', {})
        algorithms['kmeans'] = lambda n_clusters: KMeansClusterer(
            n_clusters=n_clusters, **kmeans_params
        )
        
        # Fuzzy C-Means
        fcm_params = self.config.get('algorithms.fuzzy_cmeans', {})
        algorithms['fuzzy_cmeans'] = lambda n_clusters: FuzzyCMeansClusterer(
            n_clusters=n_clusters, **fcm_params
        )
        
        # Gaussian Mixture
        gmm_params = self.config.get('algorithms.gaussian_mixture', {})
        algorithms['gaussian_mixture'] = lambda n_clusters: GaussianMixtureClusterer(
            n_clusters=n_clusters, **gmm_params
        )
        
        # DBSCAN
        dbscan_params = self.config.get('algorithms.dbscan', {})
        algorithms['dbscan'] = lambda n_clusters: DBSCANClusterer(
            n_clusters=n_clusters, **dbscan_params
        )
        
        # Spectral Clustering
        spectral_params = self.config.get('algorithms.spectral', {})
        algorithms['spectral'] = lambda n_clusters: SpectralClusterer(
            n_clusters=n_clusters, **spectral_params
        )
        
        return algorithms
    
    def run_single_experiment(self,
                             X: np.ndarray,
                             true_labels: np.ndarray,
                             experiment_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run clustering algorithms on a single dataset.
        
        Args:
            X: Input data
            true_labels: True cluster labels
            experiment_params: Experiment parameters
            
        Returns:
            Dictionary with results for all algorithms
        """
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_clusters = experiment_params.get('K', len(np.unique(true_labels)))
        results = {
            'experiment_params': experiment_params,
            'algorithms': {}
        }
        
        # Run each algorithm
        for algo_name, algo_factory in self.algorithms.items():
            logger.info(f"Running {algo_name} on dataset {experiment_params}")
            
            try:
                # Initialize algorithm
                algorithm = algo_factory(n_clusters)
                
                # Fit algorithm
                start_time = time.time()
                algorithm.fit(X_scaled, optimize=True, true_labels=true_labels)
                fit_time = time.time() - start_time
                
                # Get predictions
                predicted_labels = algorithm.labels_
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_all_metrics(
                    X_scaled, predicted_labels, true_labels
                )
                
                # Store results
                results['algorithms'][algo_name] = {
                    'predicted_labels': predicted_labels,
                    'fit_time': fit_time,
                    'metrics': metrics,
                    'algorithm_params': algorithm.get_params()
                }
                
                # Save individual results
                self._save_algorithm_results(
                    algorithm, predicted_labels, true_labels, experiment_params, algo_name
                )
                
            except Exception as e:
                logger.error(f"Error running {algo_name}: {e}")
                results['algorithms'][algo_name] = {
                    'error': str(e),
                    'fit_time': None,
                    'metrics': {},
                    'predicted_labels': None
                }
        
        return results
    
    def run_comparative_analysis(self, 
                                experiment_name: str = 'bar_omega_variation') -> List[Dict]:
        """
        Run comparative analysis across multiple datasets.
        
        Args:
            experiment_name: Name of experiment configuration to run
            
        Returns:
            List of results for all experiments
        """
        logger.info(f"Starting comparative analysis: {experiment_name}")
        
        # Get experiment configuration
        experiment_config = self.config.get(f'experiments.{experiment_name}')
        if not experiment_config:
            raise ValueError(f"Experiment config not found: {experiment_name}")
        
        # Generate datasets
        logger.info("Generating synthetic datasets...")
        datasets_info = self.data_generator.generate_experiment_grid({
            experiment_name: experiment_config
        })
        
        # Run experiments in parallel
        n_processes = self.config.get('parallel.n_processes', 4)
        results = []
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all tasks
            future_to_params = {}
            
            for dataset_info in datasets_info:
                # Load dataset
                X, y = self.data_generator.load_dataset(
                    dataset_info['bar_omega'],
                    dataset_info['K'],
                    dataset_info['P'], 
                    dataset_info['N']
                )
                
                # Submit experiment
                future = executor.submit(
                    self.run_single_experiment,
                    X, y, dataset_info
                )
                future_to_params[future] = dataset_info
            
            # Collect results
            for future in as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed experiment: {future_to_params[future]}")
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
        
        # Save consolidated results
        self._save_consolidated_results(results, experiment_name)
        
        logger.info(f"Comparative analysis completed. {len(results)} experiments run.")
        return results
    
    def run_real_datasets_analysis(self) -> List[Dict]:
        """Run analysis on real benchmark datasets."""
        logger.info("Running analysis on real datasets")
        
        datasets = self.dataset_loader.list_available_datasets()
        results = []
        
        for dataset_name in datasets:
            logger.info(f"Processing dataset: {dataset_name}")
            
            try:
                # Load dataset
                X, true_labels = self.dataset_loader.load_dataset(dataset_name)
                
                # Create experiment parameters
                experiment_params = {
                    'dataset_name': dataset_name,
                    'dataset_type': 'real',
                    'K': len(np.unique(true_labels)) if true_labels is not None else 5,
                    'N': len(X),
                    'P': X.shape[1]
                }
                
                # Run experiment
                result = self.run_single_experiment(X, true_labels, experiment_params)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
        
        # Save results
        self._save_consolidated_results(results, "real_datasets")
        
        return results
    
    def _save_algorithm_results(self,
                               algorithm,
                               predicted_labels: np.ndarray,
                               true_labels: np.ndarray,
                               experiment_params: Dict,
                               algo_name: str) -> None:
        """Save individual algorithm results."""
        # Create results DataFrame
        results_df = pd.DataFrame({
            'predicted_labels': predicted_labels,
            'true_labels': true_labels
        })
        
        # Create filename
        if 'dataset_name' in experiment_params:
            # Real dataset
            filename = f"{algo_name}_{experiment_params['dataset_name']}_results.csv"
        else:
            # Synthetic dataset
            filename = (f"{algo_name}_{experiment_params['bar_omega']}_"
                       f"{experiment_params['K']}_{experiment_params['P']}_"
                       f"{experiment_params['N']}_results.csv")
        
        # Save to clusters directory
        clusters_dir = Path(self.config.get('output.clusters_dir'))
        filepath = clusters_dir / filename
        results_df.to_csv(filepath, index=False)
        
        # Save execution time
        time_results_file = clusters_dir / "execution_times.txt"
        with open(time_results_file, 'a') as f:
            if 'dataset_name' in experiment_params:
                f.write(f"{algo_name},{experiment_params['dataset_name']},{algorithm.get_execution_time():.4f}\n")
            else:
                f.write(f"{algo_name},{experiment_params['bar_omega']},"
                       f"{experiment_params['K']},{experiment_params['P']},"
                       f"{experiment_params['N']},{algorithm.get_execution_time():.4f}\n")
    
    def _save_consolidated_results(self, results: List[Dict], experiment_name: str) -> None:
        """Save consolidated experiment results."""
        # Create metrics summary
        metrics_data = []
        
        for result in results:
            experiment_params = result['experiment_params']
            
            for algo_name, algo_result in result['algorithms'].items():
                if 'metrics' in algo_result and algo_result['metrics']:
                    row = {
                        'experiment': experiment_name,
                        'algorithm': algo_name,
                        'fit_time': algo_result['fit_time'],
                        **experiment_params,
                        **algo_result['metrics']
                    }
                    metrics_data.append(row)
        
        # Save metrics
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_file = Path(self.config.get('output.metrics_dir')) / f"{experiment_name}_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            
            logger.info(f"Saved metrics to {metrics_file}")
    
    def create_visualizations(self, results: Optional[List[Dict]] = None) -> None:
        """Create visualizations for experiment results."""
        if results is None:
            # Load results from files
            metrics_dir = Path(self.config.get('output.metrics_dir'))
            results = []
            
            for metrics_file in metrics_dir.glob("*_metrics.csv"):
                df = pd.read_csv(metrics_file)
                results.extend(df.to_dict('records'))
        
        # Create visualizations using the visualizer
        self.visualizer.create_comprehensive_report(results)
        
        logger.info("Visualizations created successfully")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        metrics_dir = Path(self.config.get('output.metrics_dir'))
        clusters_dir = Path(self.config.get('output.clusters_dir'))
        
        summary = {
            'total_experiments': 0,
            'algorithms_tested': set(),
            'datasets_processed': set(),
            'metrics_files': [],
            'cluster_files': []
        }
        
        # Count metrics files
        for metrics_file in metrics_dir.glob("*.csv"):
            df = pd.read_csv(metrics_file)
            summary['total_experiments'] += len(df)
            summary['algorithms_tested'].update(df.get('algorithm', []))
            summary['datasets_processed'].update(df.get('experiment', []))
            summary['metrics_files'].append(str(metrics_file))
        
        # Count cluster results
        summary['cluster_files'] = list(str(f) for f in clusters_dir.glob("*.csv"))
        
        # Convert sets to lists for JSON serialization
        summary['algorithms_tested'] = list(summary['algorithms_tested'])
        summary['datasets_processed'] = list(summary['datasets_processed'])
        
        return summary
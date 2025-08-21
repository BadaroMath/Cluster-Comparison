#!/usr/bin/env python3
"""Command-line interface for clustering analysis."""

import argparse
import sys
from pathlib import Path

from .core import ClusteringExperiment
from .utils import setup_logging
from loguru import logger


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clustering Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run bar omega variation experiment
  clustering-analysis run --experiment bar_omega_variation
  
  # Run on real datasets with visualization
  clustering-analysis run --real-datasets --visualize
  
  # Generate synthetic data only  
  clustering-analysis generate --bar-omega 0.1 --clusters 3 --samples 1000
  
  # Create visualizations from existing results
  clustering-analysis visualize --results-dir data/results/metrics
        """
    )
    
    # Global options
    parser.add_argument('--config', default='config/experiment_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run experiments
    run_parser = subparsers.add_parser('run', help='Run clustering experiments')
    run_parser.add_argument('--experiment', default='bar_omega_variation',
                           choices=['bar_omega_variation', 'cluster_variation',
                                   'dimension_variation', 'sample_variation'],
                           help='Experiment type to run')
    run_parser.add_argument('--real-datasets', action='store_true',
                           help='Run on real datasets instead of synthetic')
    run_parser.add_argument('--visualize', action='store_true',
                           help='Create visualizations after analysis')
    
    # Generate data
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--bar-omega', type=float, default=0.05,
                           help='Cluster overlap parameter')
    gen_parser.add_argument('--clusters', '-k', type=int, default=3,
                           help='Number of clusters')
    gen_parser.add_argument('--dimensions', '-p', type=int, default=5,
                           help='Number of dimensions')
    gen_parser.add_argument('--samples', '-n', type=int, default=5000,
                           help='Number of samples')
    gen_parser.add_argument('--output-dir', default='data/synthetic',
                           help='Output directory')
    
    # Visualize results
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('--results-dir', default='data/results/metrics',
                           help='Results directory')
    viz_parser.add_argument('--output-dir', default='data/results/figures',
                           help='Output directory for figures')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Handle commands
    if args.command == 'run':
        return run_experiments(args)
    elif args.command == 'generate':
        return generate_data(args)
    elif args.command == 'visualize':
        return create_visualizations(args)
    elif args.command == 'info':
        return show_info()
    else:
        parser.print_help()
        return 1


def run_experiments(args):
    """Run clustering experiments."""
    try:
        experiment = ClusteringExperiment(config_path=args.config)
        
        results = []
        if args.real_datasets:
            logger.info("Running analysis on real datasets...")
            results = experiment.run_real_datasets_analysis()
        else:
            logger.info(f"Running {args.experiment} experiment...")
            results = experiment.run_comparative_analysis(args.experiment)
        
        if args.visualize:
            logger.info("Creating visualizations...")
            experiment.create_visualizations(results)
        
        # Print summary
        summary = experiment.get_experiment_summary()
        logger.info("Experiment completed successfully!")
        logger.info(f"Total experiments: {summary['total_experiments']}")
        logger.info(f"Algorithms tested: {', '.join(summary['algorithms_tested'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return 1


def generate_data(args):
    """Generate synthetic data."""
    try:
        from .data_generation import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(args.output_dir)
        X, y = generator.generate_dataset(
            bar_omega=args.bar_omega,
            K=args.clusters,
            p=args.dimensions,
            n=args.samples,
            random_state=42
        )
        
        logger.info(f"Dataset generated successfully!")
        logger.info(f"Shape: {X.shape}")
        logger.info(f"Clusters: {len(set(y))}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        return 1


def create_visualizations(args):
    """Create visualizations from results."""
    try:
        from .visualization import ClusteringVisualizer
        import pandas as pd
        
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            logger.error(f"Results directory not found: {results_dir}")
            return 1
        
        visualizer = ClusteringVisualizer(args.output_dir)
        
        # Load all results
        results_files = list(results_dir.glob("*_metrics.csv"))
        if not results_files:
            logger.error(f"No results files found in {results_dir}")
            return 1
        
        all_results = []
        for results_file in results_files:
            df = pd.read_csv(results_file)
            all_results.extend(df.to_dict('records'))
        
        logger.info(f"Loaded {len(all_results)} results")
        
        # Create visualizations
        visualizer.create_comprehensive_report(all_results)
        logger.info(f"Visualizations created in {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        return 1


def show_info():
    """Show system information."""
    import platform
    import sklearn
    import numpy as np
    import pandas as pd
    
    logger.info("Clustering Analysis Framework Information")
    logger.info("=" * 50)
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"Pandas: {pd.__version__}")
    logger.info(f"Scikit-learn: {sklearn.__version__}")
    
    # Check algorithm availability
    try:
        from .algorithms import (KMeansClusterer, FuzzyCMeansClusterer, 
                                GaussianMixtureClusterer, DBSCANClusterer, 
                                SpectralClusterer)
        algorithms = ["K-Means", "Fuzzy C-Means", "Gaussian Mixture", "DBSCAN", "Spectral"]
        logger.info(f"Available algorithms: {', '.join(algorithms)}")
    except ImportError as e:
        logger.warning(f"Some algorithms unavailable: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
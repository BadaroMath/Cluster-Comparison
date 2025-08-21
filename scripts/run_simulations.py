#!/usr/bin/env python3
"""Script to run clustering simulations."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from clustering_analysis import ClusteringExperiment
from loguru import logger


def main():
    """Main function for running clustering simulations."""
    parser = argparse.ArgumentParser(description="Run clustering algorithm simulations")
    
    parser.add_argument('--config', default='config/experiment_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment', default='bar_omega_variation',
                       choices=['bar_omega_variation', 'cluster_variation', 
                               'dimension_variation', 'sample_variation'],
                       help='Type of experiment to run')
    parser.add_argument('--real-datasets', action='store_true',
                       help='Run analysis on real datasets')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations after analysis')
    
    args = parser.parse_args()
    
    # Initialize experiment
    logger.info(f"Initializing clustering experiment with config: {args.config}")
    
    try:
        experiment = ClusteringExperiment(config_path=args.config)
    except Exception as e:
        logger.error(f"Failed to initialize experiment: {e}")
        return 1
    
    results = []
    
    # Run experiments
    if args.real_datasets:
        logger.info("Running analysis on real datasets...")
        try:
            real_results = experiment.run_real_datasets_analysis()
            results.extend(real_results)
        except Exception as e:
            logger.error(f"Real dataset analysis failed: {e}")
            return 1
    else:
        logger.info(f"Running {args.experiment} experiment...")
        try:
            synthetic_results = experiment.run_comparative_analysis(args.experiment)
            results.extend(synthetic_results)
        except Exception as e:
            logger.error(f"Synthetic experiment failed: {e}")
            return 1
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("Creating visualizations...")
        try:
            experiment.create_visualizations(results)
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return 1
    
    # Print summary
    summary = experiment.get_experiment_summary()
    logger.info("Experiment Summary:")
    logger.info(f"  Total experiments: {summary['total_experiments']}")
    logger.info(f"  Algorithms tested: {', '.join(summary['algorithms_tested'])}")
    logger.info(f"  Datasets processed: {', '.join(summary['datasets_processed'])}")
    
    logger.info("Clustering simulations completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
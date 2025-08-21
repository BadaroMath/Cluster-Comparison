#!/usr/bin/env python3
"""Script to create visualizations from results."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from clustering_analysis.visualization import ClusteringVisualizer
from loguru import logger


def main():
    """Main function for creating visualizations."""
    parser = argparse.ArgumentParser(description="Create visualizations from clustering results")
    
    parser.add_argument('--results-dir', default='data/results/metrics',
                       help='Directory containing results CSV files')
    parser.add_argument('--output-dir', default='data/results/figures',
                       help='Output directory for figures')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format for figures')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1
    
    # Initialize visualizer
    visualizer = ClusteringVisualizer(args.output_dir)
    
    # Find all results files
    results_files = list(results_dir.glob("*_metrics.csv"))
    
    if not results_files:
        logger.error(f"No metrics files found in {results_dir}")
        return 1
    
    logger.info(f"Found {len(results_files)} results files")
    
    # Load and combine all results
    all_results = []
    
    for results_file in results_files:
        logger.info(f"Loading results from {results_file}")
        
        try:
            df = pd.read_csv(results_file)
            results_dict = df.to_dict('records')
            all_results.extend(results_dict)
        except Exception as e:
            logger.error(f"Failed to load {results_file}: {e}")
            continue
    
    if not all_results:
        logger.error("No valid results loaded")
        return 1
    
    logger.info(f"Loaded {len(all_results)} experiment results")
    
    # Create comprehensive visualizations
    try:
        visualizer.create_comprehensive_report(all_results)
        logger.info(f"Visualizations created successfully in {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
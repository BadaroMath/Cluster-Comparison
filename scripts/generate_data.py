#!/usr/bin/env python3
"""Script to generate synthetic clustering datasets."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from clustering_analysis import SyntheticDataGenerator
from loguru import logger


def main():
    """Main function for data generation script."""
    parser = argparse.ArgumentParser(description="Generate synthetic clustering datasets")
    
    parser.add_argument('--bar-omega', type=float, default=0.05,
                       help='Average overlap between clusters (0.0-1.0)')
    parser.add_argument('--clusters', '-k', type=int, default=3,
                       help='Number of clusters')
    parser.add_argument('--dimensions', '-p', type=int, default=5,
                       help='Number of dimensions')
    parser.add_argument('--samples', '-n', type=int, default=5000,
                       help='Number of samples')
    parser.add_argument('--output-dir', default='data/synthetic',
                       help='Output directory for datasets')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize data generator
    generator = SyntheticDataGenerator(args.output_dir)
    
    logger.info(f"Generating dataset with parameters:")
    logger.info(f"  Bar Omega: {args.bar_omega}")
    logger.info(f"  Clusters: {args.clusters}")
    logger.info(f"  Dimensions: {args.dimensions}")
    logger.info(f"  Samples: {args.samples}")
    
    try:
        # Generate dataset
        X, y = generator.generate_dataset(
            bar_omega=args.bar_omega,
            K=args.clusters,
            p=args.dimensions,
            n=args.samples,
            random_state=args.random_state
        )
        
        logger.info(f"Dataset generated successfully!")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Unique labels: {len(set(y))}")
        logger.info(f"  Saved to: {generator.get_dataset_path(args.bar_omega, args.clusters, args.dimensions, args.samples)}")
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
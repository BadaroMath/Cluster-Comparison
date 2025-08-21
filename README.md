# Clustering Algorithms Comparative Analysis

A comprehensive research project implementing and comparing multiple clustering algorithms using both synthetic and real datasets. This project provides a systematic evaluation framework for clustering methods including K-Means, Fuzzy C-Means, Gaussian Mixture Models, DBSCAN, and Spectral Clustering.

## 📊 Research Overview

This project implements a rigorous comparative analysis of clustering algorithms across various scenarios:

- **Synthetic Data Generation**: Using MixSim library for controlled experimental conditions
- **Multiple Clustering Methods**: Implementation of 5 major clustering algorithms
- **Comprehensive Evaluation**: Multiple metrics including Adjusted Rand Index, Silhouette Coefficient, Dunn Index
- **Scalability Analysis**: Performance evaluation across different data dimensions and sizes
- **Visualization**: Comprehensive plotting and statistical analysis

## 🏗️ Project Structure

```
clustering-analysis/
├── src/                          # Main source code
│   └── clustering_analysis/      # Python package
│       ├── __init__.py
│       ├── algorithms/           # Clustering algorithm implementations
│       ├── data_generation/      # Synthetic data generation
│       ├── evaluation/           # Metrics and evaluation functions
│       ├── visualization/        # Plotting and graphics
│       └── utils/               # Utility functions
├── data/                        # Data storage
│   ├── synthetic/              # Generated synthetic datasets
│   ├── real/                   # Real benchmark datasets
│   └── results/                # Clustering results
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_generation.ipynb
│   ├── 02_clustering_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── scripts/                     # Execution scripts
│   ├── run_simulations.py
│   ├── generate_data.py
│   └── create_visualizations.py
├── tests/                       # Unit tests
├── docs/                        # Documentation
│   ├── TCC_Document.pdf         # Original thesis document
│   └── methodology.md           # Detailed methodology
├── config/                      # Configuration files
│   └── experiment_config.yaml
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── environment.yml              # Conda environment
```

## 🔬 Algorithms Implemented

### Partitioning Methods
- **K-Means**: Classic centroid-based clustering
- **Fuzzy C-Means (FCM)**: Soft clustering with membership degrees
- **Gaussian Mixture Models (GMM)**: Probabilistic clustering approach

### Density-Based Methods
- **DBSCAN**: Density-based spatial clustering with noise detection
- **Spectral Clustering**: Graph-based clustering using eigenvalue decomposition

## 📈 Evaluation Metrics

- **Adjusted Rand Index (ARI)**: Measures clustering agreement corrected for chance
- **Rand Index (RI)**: Basic measure of clustering similarity
- **Silhouette Coefficient**: Measures cluster cohesion and separation
- **Dunn Index**: Ratio of minimum inter-cluster to maximum intra-cluster distance
- **Variation of Information (VI)**: Information-theoretic clustering comparison
- **Classification Proportion**: Accuracy of cluster assignments

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd clustering-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.clustering_analysis import ClusteringExperiment

# Initialize experiment
experiment = ClusteringExperiment(config_path='config/experiment_config.yaml')

# Run clustering analysis
results = experiment.run_comparative_analysis(
    bar_omega=0.05,
    n_clusters=3,
    n_dimensions=5,
    n_samples=5000
)

# Generate visualizations
experiment.create_visualizations(results)
```

### Command Line Usage

```bash
# Generate synthetic data
python scripts/generate_data.py --bar-omega 0.05 --clusters 3 --dimensions 5 --samples 5000

# Run clustering algorithms
python scripts/run_simulations.py --config config/experiment_config.yaml

# Create visualizations
python scripts/create_visualizations.py --results-dir data/results/
```

## 📊 Experimental Design

### Synthetic Data Parameters
- **BarOmega (ω̄)**: Overlap measure between clusters (0.0 - 0.6)
- **K**: Number of clusters (2 - 40)
- **P**: Number of dimensions (2 - 200)
- **N**: Number of observations (100 - 2,000,000)

### Real Datasets
- Compound dataset
- Aggregation dataset
- PathBased dataset
- S2 dataset
- Flame dataset
- Face dataset

## 🔧 Dependencies

- **Python**: >= 3.8
- **R**: >= 4.0 (for MixSim simulations and visualizations)
- **Key Libraries**:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - fcmeans
  - MixSim (R)
  - ggplot2 (R)

## 📝 Research Methodology

1. **Data Generation**: Systematic generation of synthetic datasets using MixSim
2. **Algorithm Application**: Implementation of clustering algorithms with parameter optimization
3. **Performance Evaluation**: Multi-metric evaluation including statistical and computational metrics
4. **Comparative Analysis**: Cross-algorithm performance comparison across different scenarios
5. **Visualization**: Comprehensive plotting of results and algorithm behavior

## 📚 Academic References

This project implements methodologies from:
- Statistical clustering analysis literature
- Machine learning comparative studies
- Computational complexity analysis
- Information-theoretic clustering evaluation

## 🤝 Contributing

This is a research project. For academic use, please cite appropriately and follow scientific reproducibility standards.

## 📄 License

Academic/Research License - Please cite appropriately when using this code.

---

**Note**: This project was developed as part of academic research. The methodology and results should be interpreted within the context of the original research objectives.
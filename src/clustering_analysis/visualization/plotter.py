"""Comprehensive clustering visualization and plotting."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class ClusteringVisualizer:
    """Comprehensive clustering visualization and plotting."""
    
    def __init__(self, output_dir: str = "data/results/figures"):
        """
        Initialize clustering visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_clustering_results(self,
                               X: np.ndarray,
                               predicted_labels: np.ndarray,
                               true_labels: Optional[np.ndarray] = None,
                               algorithm_name: str = "Clustering",
                               save_path: Optional[str] = None) -> None:
        """
        Plot clustering results for 2D data.
        
        Args:
            X: Input data (should be 2D for plotting)
            predicted_labels: Predicted cluster labels
            true_labels: True cluster labels (optional)
            algorithm_name: Name of algorithm for title
            save_path: Path to save figure
        """
        if X.shape[1] != 2:
            logger.warning(f"Data has {X.shape[1]} dimensions, cannot plot directly")
            return
            
        fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, 
                                figsize=(12, 5) if true_labels is not None else (6, 5))
        
        if true_labels is not None:
            # Plot true labels
            axes[0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
            axes[0].set_title('True Labels')
            axes[0].set_xlabel('Feature 1')
            axes[0].set_ylabel('Feature 2')
            
            # Plot predicted labels
            axes[1].scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', alpha=0.7)
            axes[1].set_title(f'{algorithm_name} Results')
            axes[1].set_xlabel('Feature 1')
            axes[1].set_ylabel('Feature 2')
        else:
            axes.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', alpha=0.7)
            axes.set_title(f'{algorithm_name} Results')
            axes.set_xlabel('Feature 1')
            axes.set_ylabel('Feature 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Clustering plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def plot_metrics_comparison(self,
                               results_df: pd.DataFrame,
                               metric: str = 'adjusted_rand_score',
                               groupby: str = 'algorithm',
                               save_path: Optional[str] = None) -> None:
        """
        Plot comparison of metrics across different conditions.
        
        Args:
            results_df: DataFrame with results
            metric: Metric to plot
            groupby: Variable to group by
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot
        sns.boxplot(data=results_df, x=groupby, y=metric, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xlabel(groupby.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def plot_parameter_analysis(self,
                               results_df: pd.DataFrame,
                               parameter: str,
                               metric: str = 'adjusted_rand_score',
                               save_path: Optional[str] = None) -> None:
        """
        Plot performance vs parameter variation.
        
        Args:
            results_df: DataFrame with results
            parameter: Parameter to analyze
            metric: Performance metric
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Line plot for each algorithm
        algorithms = results_df['algorithm'].unique()
        
        for algorithm in algorithms:
            algo_data = results_df[results_df['algorithm'] == algorithm]
            
            if len(algo_data) > 1:
                # Group by parameter and calculate mean
                param_analysis = algo_data.groupby(parameter)[metric].agg(['mean', 'std']).reset_index()
                
                ax.errorbar(param_analysis[parameter], param_analysis['mean'], 
                           yerr=param_analysis['std'], label=algorithm, marker='o')
        
        ax.set_xlabel(parameter.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs {parameter.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter analysis saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def create_performance_heatmap(self,
                                  results_df: pd.DataFrame,
                                  index_col: str = 'algorithm',
                                  columns_col: str = 'experiment',
                                  value_col: str = 'adjusted_rand_score',
                                  save_path: Optional[str] = None) -> None:
        """
        Create heatmap of algorithm performance.
        
        Args:
            results_df: DataFrame with results
            index_col: Column for rows
            columns_col: Column for columns  
            value_col: Column for values
            save_path: Path to save figure
        """
        # Pivot data
        pivot_data = results_df.pivot_table(
            index=index_col, columns=columns_col, values=value_col, aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
        ax.set_title(f'{value_col.replace("_", " ").title()} Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance heatmap saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def create_execution_time_analysis(self,
                                     results_df: pd.DataFrame,
                                     save_path: Optional[str] = None) -> None:
        """
        Create execution time analysis plots.
        
        Args:
            results_df: DataFrame with results including fit_time
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Box plot by algorithm
        sns.boxplot(data=results_df, x='algorithm', y='fit_time', ax=axes[0, 0])
        axes[0, 0].set_title('Execution Time by Algorithm')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Scatter plot: fit_time vs performance metric
        if 'adjusted_rand_score' in results_df.columns:
            sns.scatterplot(data=results_df, x='fit_time', y='adjusted_rand_score', 
                           hue='algorithm', ax=axes[0, 1])
            axes[0, 1].set_title('Performance vs Execution Time')
        
        # Histogram of execution times
        axes[1, 0].hist(results_df['fit_time'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Execution Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Execution Times')
        
        # Time vs dataset size (if N column exists)
        if 'N' in results_df.columns:
            sns.scatterplot(data=results_df, x='N', y='fit_time', hue='algorithm', ax=axes[1, 1])
            axes[1, 1].set_title('Execution Time vs Dataset Size')
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Execution time analysis saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def create_comprehensive_report(self, results: List[Dict[str, Any]]) -> None:
        """
        Create comprehensive visualization report.
        
        Args:
            results: List of experiment results
        """
        logger.info("Creating comprehensive visualization report...")
        
        # Convert results to DataFrame
        if not results:
            logger.warning("No results to visualize")
            return
            
        # Flatten results for analysis
        flat_results = []
        for result in results:
            if isinstance(result, dict) and 'algorithms' in result:
                # Results from experiment
                for algo_name, algo_result in result['algorithms'].items():
                    if 'metrics' in algo_result and algo_result['metrics']:
                        flat_result = {
                            'algorithm': algo_name,
                            'fit_time': algo_result['fit_time'],
                            **result['experiment_params'],
                            **algo_result['metrics']
                        }
                        flat_results.append(flat_result)
            else:
                # Direct results
                flat_results.append(result)
        
        if not flat_results:
            logger.warning("No valid results to visualize")
            return
            
        results_df = pd.DataFrame(flat_results)
        
        # 1. Algorithm comparison
        if 'adjusted_rand_score' in results_df.columns:
            self.plot_metrics_comparison(
                results_df, 'adjusted_rand_score', 'algorithm',
                save_path=self.output_dir / 'algorithm_comparison_ari.png'
            )
        
        if 'silhouette_score' in results_df.columns:
            self.plot_metrics_comparison(
                results_df, 'silhouette_score', 'algorithm',
                save_path=self.output_dir / 'algorithm_comparison_silhouette.png'
            )
        
        # 2. Parameter analysis
        parameter_columns = ['bar_omega', 'K', 'P', 'N']
        for param in parameter_columns:
            if param in results_df.columns and results_df[param].nunique() > 1:
                if 'adjusted_rand_score' in results_df.columns:
                    self.plot_parameter_analysis(
                        results_df, param, 'adjusted_rand_score',
                        save_path=self.output_dir / f'parameter_analysis_{param}_ari.png'
                    )
        
        # 3. Performance heatmap
        if len(results_df['algorithm'].unique()) > 1 and 'experiment' in results_df.columns:
            if 'adjusted_rand_score' in results_df.columns:
                self.create_performance_heatmap(
                    results_df, 'algorithm', 'experiment', 'adjusted_rand_score',
                    save_path=self.output_dir / 'performance_heatmap.png'
                )
        
        # 4. Execution time analysis
        if 'fit_time' in results_df.columns:
            self.create_execution_time_analysis(
                results_df, save_path=self.output_dir / 'execution_time_analysis.png'
            )
        
        # 5. Create summary statistics
        self._create_summary_statistics(results_df)
        
        logger.info(f"Comprehensive report created in {self.output_dir}")
    
    def _create_summary_statistics(self, results_df: pd.DataFrame) -> None:
        """Create and save summary statistics."""
        # Overall statistics
        summary_stats = {}
        
        # Metrics by algorithm
        metric_columns = [col for col in results_df.columns 
                         if col in ['adjusted_rand_score', 'rand_score', 'silhouette_score', 
                                   'dunn_index', 'calinski_harabasz_score']]
        
        if metric_columns:
            algorithm_stats = results_df.groupby('algorithm')[metric_columns].agg([
                'count', 'mean', 'std', 'min', 'max'
            ])
            
            # Save to CSV
            stats_path = self.output_dir / 'summary_statistics.csv'
            algorithm_stats.to_csv(stats_path)
            logger.info(f"Summary statistics saved to {stats_path}")
        
        # Create summary report
        report_lines = [
            "# Clustering Analysis Summary Report\n",
            f"Total experiments: {len(results_df)}\n",
            f"Algorithms tested: {', '.join(results_df['algorithm'].unique())}\n",
            f"Datasets processed: {results_df.get('experiment', ['unknown']).nunique()}\n\n"
        ]
        
        if metric_columns:
            report_lines.append("## Performance Summary by Algorithm\n")
            for algorithm in results_df['algorithm'].unique():
                algo_data = results_df[results_df['algorithm'] == algorithm]
                report_lines.append(f"\n### {algorithm}\n")
                
                for metric in metric_columns:
                    if metric in algo_data.columns:
                        mean_score = algo_data[metric].mean()
                        std_score = algo_data[metric].std()
                        report_lines.append(f"- {metric}: {mean_score:.3f} ± {std_score:.3f}\n")
        
        # Save report
        report_path = self.output_dir / 'analysis_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
            
        logger.info(f"Analysis report saved to {report_path}")
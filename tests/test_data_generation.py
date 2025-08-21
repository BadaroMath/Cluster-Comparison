"""Tests for data generation modules."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.clustering_analysis.data_generation import SyntheticDataGenerator, RealDatasetLoader


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
        
    def test_initialization(self, temp_dir):
        """Test generator initialization."""
        generator = SyntheticDataGenerator(temp_dir)
        assert generator.output_dir == Path(temp_dir)
        assert generator.output_dir.exists()
        
    def test_generate_dataset(self, temp_dir):
        """Test dataset generation."""
        generator = SyntheticDataGenerator(temp_dir)
        
        # Generate dataset
        X, y = generator.generate_dataset(
            bar_omega=0.1, K=3, p=2, n=100, random_state=42
        )
        
        # Check output
        assert X.shape == (100, 2)
        assert len(y) == 100
        assert len(np.unique(y)) <= 3  # May have fewer clusters due to small n
        
        # Check file was created
        filepath = generator.get_dataset_path(0.1, 3, 2, 100)
        assert filepath.exists()
        
    def test_load_dataset(self, temp_dir):
        """Test dataset loading."""
        generator = SyntheticDataGenerator(temp_dir)
        
        # Generate and save dataset
        X_orig, y_orig = generator.generate_dataset(
            bar_omega=0.1, K=3, p=2, n=100, random_state=42
        )
        
        # Load dataset
        X_loaded, y_loaded = generator.load_dataset(0.1, 3, 2, 100)
        
        # Check they match
        np.testing.assert_array_almost_equal(X_orig, X_loaded)
        np.testing.assert_array_equal(y_orig, y_loaded)
        
    def test_generate_experiment_grid(self, temp_dir):
        """Test experiment grid generation."""
        generator = SyntheticDataGenerator(temp_dir)
        
        config = {
            'test_experiment': {
                'bar_omega': [0.0, 0.1],
                'K': [2, 3],
                'P': [2],
                'N': [50]
            }
        }
        
        results = generator.generate_experiment_grid(config)
        
        # Should generate 2*2*1*1 = 4 datasets
        assert len(results) == 4
        
        # Check result structure
        assert all('bar_omega' in r for r in results)
        assert all('K' in r for r in results)
        assert all('n_samples' in r for r in results)
        
    def test_list_datasets(self, temp_dir):
        """Test dataset listing."""
        generator = SyntheticDataGenerator(temp_dir)
        
        # Generate a few datasets
        generator.generate_dataset(0.0, 2, 2, 50, random_state=42)
        generator.generate_dataset(0.1, 3, 2, 100, random_state=42)
        
        # List datasets
        datasets = generator.list_datasets()
        
        assert len(datasets) == 2
        assert all('bar_omega' in d for d in datasets)
        assert all('filepath' in d for d in datasets)


class TestRealDatasetLoader:
    """Test real dataset loading."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
        
    def test_initialization(self, temp_dir):
        """Test loader initialization."""
        loader = RealDatasetLoader(temp_dir)
        assert loader.data_dir == Path(temp_dir)
        assert loader.data_dir.exists()
        
    def test_list_available_datasets(self, temp_dir):
        """Test listing available datasets."""
        loader = RealDatasetLoader(temp_dir)
        datasets = loader.list_available_datasets()
        
        expected = ['compound', 'aggregation', 'pathbased', 's2', 'flame', 'face']
        assert all(d in expected for d in datasets)
        
    def test_get_dataset_info(self, temp_dir):
        """Test dataset info retrieval."""
        loader = RealDatasetLoader(temp_dir)
        
        info = loader.get_dataset_info('compound')
        
        assert 'expected_clusters' in info
        assert 'label_col' in info
        assert 'file_exists' in info
        assert info['expected_clusters'] == 6
        
    def test_load_dataset_creates_sample(self, temp_dir):
        """Test dataset loading creates sample data when file missing."""
        loader = RealDatasetLoader(temp_dir)
        
        # Load non-existent dataset (should create sample)
        X, y = loader.load_dataset('compound', standardize=True)
        
        assert X is not None
        assert y is not None
        assert X.shape[0] == len(y)
        
        # Check file was created
        filepath = loader.data_dir / "compound.csv"
        assert filepath.exists()
        
    def test_create_toy_datasets(self, temp_dir):
        """Test toy dataset creation."""
        loader = RealDatasetLoader(temp_dir)
        loader.create_toy_datasets()
        
        # Check files were created
        expected_files = ['noisy_circles.csv', 'noisy_moons.csv', 
                         'varied_blobs.csv', 'aniso_blobs.csv']
        
        for filename in expected_files:
            filepath = loader.data_dir / filename
            assert filepath.exists()
            
        # Check data can be loaded
        X, y = loader.load_dataset('noisy_circles')
        assert X is not None
        assert y is not None
"""
Unit tests for data processing module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path

from mdt_dashboard.data_processing import (
    DataQualityReport,
    ProcessingConfig,
    DataLoader,
    DataQualityAnalyzer,
    DataPreprocessor,
    FeatureEngineer
)


class TestDataQualityReport:
    """Test DataQualityReport dataclass."""
    
    def test_report_creation(self):
        """Test creating a data quality report."""
        report = DataQualityReport(
            total_rows=1000,
            total_columns=5,
            missing_values={'col1': 10, 'col2': 0},
            missing_percentage={'col1': 1.0, 'col2': 0.0},
            duplicate_rows=5,
            numeric_columns=['col1', 'col2'],
            categorical_columns=['col3'],
            datetime_columns=['col4'],
            outliers={'col1': 5, 'col2': 2},
            data_types={'col1': 'float64', 'col2': 'int64'},
            summary_stats={'col1': {'mean': 5.0, 'std': 1.0}}
        )
        
        assert report.total_rows == 1000
        assert report.total_columns == 5
        assert len(report.numeric_columns) == 2
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = DataQualityReport(
            total_rows=100,
            total_columns=3,
            missing_values={},
            missing_percentage={},
            duplicate_rows=0,
            numeric_columns=[],
            categorical_columns=[],
            datetime_columns=[],
            outliers={},
            data_types={},
            summary_stats={}
        )
        
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert 'total_rows' in report_dict


class TestProcessingConfig:
    """Test ProcessingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        
        assert config.missing_strategy == "mean"
        assert config.scaling_method == "standard"
        assert config.train_size == 0.8
        assert config.random_state == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProcessingConfig(
            missing_strategy="median",
            scaling_method="minmax",
            outlier_threshold=2.5
        )
        
        assert config.missing_strategy == "median"
        assert config.scaling_method == "minmax"
        assert config.outlier_threshold == 2.5


class TestDataLoader:
    """Test DataLoader functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader(cache_enabled=False)
    
    def test_load_csv_file_not_found(self):
        """Test loading non-existent CSV file."""
        with pytest.raises(Exception):
            self.loader.load_csv("non_existent_file.csv")
    
    @patch('pandas.read_csv')
    def test_load_csv_success(self, mock_read_csv):
        """Test successful CSV loading."""
        # Mock data
        mock_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_read_csv.return_value = mock_data
        
        result = self.loader.load_csv("test.csv")
        
        assert isinstance(result, pd.DataFrame)
        mock_read_csv.assert_called_once()
    
    def test_load_json_with_list(self):
        """Test loading JSON data as list."""
        import tempfile
        import json
        
        test_data = [
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = self.loader.load_json(temp_path)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert 'name' in result.columns
        finally:
            Path(temp_path).unlink()


class TestDataQualityAnalyzer:
    """Test DataQualityAnalyzer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DataQualityAnalyzer()
        
        # Create test dataset
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, 1000),
            'numeric2': np.random.exponential(2, 1000),
            'categorical': np.random.choice(['A', 'B', 'C'], 1000),
            'datetime': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'with_missing': [1, 2, None, 4, 5] * 200
        })
        
        # Add some duplicates
        self.test_data = pd.concat([self.test_data, self.test_data.iloc[:10]])
    
    def test_analyze_basic_info(self):
        """Test basic data analysis functionality."""
        report = self.analyzer.analyze(self.test_data)
        
        assert isinstance(report, DataQualityReport)
        assert report.total_rows == 1010  # 1000 + 10 duplicates
        assert report.total_columns == 5
        assert report.duplicate_rows == 10
    
    def test_analyze_column_types(self):
        """Test column type detection."""
        report = self.analyzer.analyze(self.test_data)
        
        assert 'numeric1' in report.numeric_columns
        assert 'numeric2' in report.numeric_columns
        assert 'categorical' in report.categorical_columns
        assert 'datetime' in report.datetime_columns
    
    def test_analyze_missing_values(self):
        """Test missing value analysis."""
        report = self.analyzer.analyze(self.test_data)
        
        assert 'with_missing' in report.missing_values
        assert report.missing_values['with_missing'] > 0
        assert report.missing_percentage['with_missing'] > 0
    
    def test_analyze_summary_statistics(self):
        """Test summary statistics generation."""
        report = self.analyzer.analyze(self.test_data)
        
        assert 'numeric1' in report.summary_stats
        stats = report.summary_stats['numeric1']
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats


class TestDataPreprocessor:
    """Test DataPreprocessor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig(
            missing_strategy="mean",
            scaling_method="standard"
        )
        self.preprocessor = DataPreprocessor(self.config)
        
        # Create test dataset
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.exponential(2, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Add missing values
        self.test_data.loc[5:10, 'numeric1'] = np.nan
        self.test_data.loc[15:20, 'categorical'] = np.nan
    
    def test_fit_transform(self):
        """Test fit and transform functionality."""
        transformed = self.preprocessor.fit_transform(self.test_data, target_column='target')
        
        assert isinstance(transformed, pd.DataFrame)
        assert self.preprocessor.is_fitted
        assert transformed.isnull().sum().sum() == 0  # No missing values after processing
    
    def test_separate_fit_transform(self):
        """Test separate fit and transform calls."""
        self.preprocessor.fit(self.test_data, target_column='target')
        transformed = self.preprocessor.transform(self.test_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert self.preprocessor.is_fitted
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        with pytest.raises(ValueError, match="must be fitted"):
            self.preprocessor.transform(self.test_data)
    
    def test_missing_value_handling(self):
        """Test missing value imputation."""
        self.preprocessor.fit(self.test_data, target_column='target')
        transformed = self.preprocessor.transform(self.test_data)
        
        # Should have no missing values in numeric columns
        assert transformed['numeric1'].isnull().sum() == 0
        assert transformed['categorical'].isnull().sum() == 0
    
    @patch('joblib.dump')
    def test_save_pipeline(self, mock_dump):
        """Test saving preprocessing pipeline."""
        self.preprocessor.fit(self.test_data, target_column='target')
        self.preprocessor.save_pipeline('test_pipeline.pkl')
        
        mock_dump.assert_called_once()
    
    @patch('joblib.load')
    def test_load_pipeline(self, mock_load):
        """Test loading preprocessing pipeline."""
        # Mock pipeline data
        mock_pipeline_data = {
            'config': self.config.__dict__,
            'scalers': {},
            'imputers': {},
            'encoders': {},
            'outlier_detectors': {},
            'feature_selectors': {},
            'is_fitted': True,
            'feature_names': ['numeric1', 'numeric2', 'categorical'],
            'target_column': 'target'
        }
        mock_load.return_value = mock_pipeline_data
        
        self.preprocessor.load_pipeline('test_pipeline.pkl')
        
        assert self.preprocessor.is_fitted
        mock_load.assert_called_once()


class TestFeatureEngineer:
    """Test FeatureEngineer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()
        
        # Create test dataset with datetime
        self.test_data = pd.DataFrame({
            'datetime_col': pd.date_range('2023-01-01', periods=100, freq='H'),
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.exponential(2, 100),
            'group_col': np.random.choice(['Group1', 'Group2'], 100)
        })
    
    def test_create_temporal_features(self):
        """Test temporal feature creation."""
        result = self.engineer.create_temporal_features(
            self.test_data, 
            ['datetime_col']
        )
        
        assert 'datetime_col_year' in result.columns
        assert 'datetime_col_month' in result.columns
        assert 'datetime_col_day' in result.columns
        assert 'datetime_col_hour' in result.columns
        assert 'datetime_col_is_weekend' in result.columns
    
    def test_create_aggregated_features(self):
        """Test aggregated feature creation."""
        result = self.engineer.create_aggregated_features(
            self.test_data,
            group_by_columns=['group_col'],
            agg_columns=['numeric1'],
            agg_functions=['mean', 'std']
        )
        
        assert 'numeric1_mean_by_group_col' in result.columns
        assert 'numeric1_std_by_group_col' in result.columns
    
    def test_create_ratio_features(self):
        """Test ratio feature creation."""
        result = self.engineer.create_ratio_features(
            self.test_data,
            numerator_cols=['numeric1'],
            denominator_cols=['numeric2']
        )
        
        assert 'numeric1_ratio_numeric2' in result.columns
        
        # Check for division by zero handling
        assert not result['numeric1_ratio_numeric2'].isnull().any()


# Integration tests for the complete preprocessing pipeline
class TestPreprocessingPipeline:
    """Test the complete preprocessing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a complex test dataset
        np.random.seed(42)
        n_samples = 1000
        
        self.test_data = pd.DataFrame({
            'numeric_normal': np.random.normal(0, 1, n_samples),
            'numeric_skewed': np.random.exponential(2, n_samples),
            'categorical_ordered': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'categorical_nominal': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'datetime_feature': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'target_regression': np.random.normal(10, 3, n_samples),
            'target_classification': np.random.randint(0, 3, n_samples)
        })
        
        # Add missing values
        missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        self.test_data.loc[missing_indices, 'numeric_normal'] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        self.test_data.loc[outlier_indices, 'numeric_skewed'] = np.random.normal(50, 10, len(outlier_indices))
    
    def test_end_to_end_preprocessing_regression(self):
        """Test complete preprocessing pipeline for regression."""
        # Data quality analysis
        analyzer = DataQualityAnalyzer()
        quality_report = analyzer.analyze(self.test_data)
        
        assert isinstance(quality_report, DataQualityReport)
        assert quality_report.total_rows == 1000
        
        # Feature engineering
        engineer = FeatureEngineer()
        engineered_data = engineer.create_temporal_features(
            self.test_data, 
            ['datetime_feature']
        )
        
        # Preprocessing
        config = ProcessingConfig(
            missing_strategy="mean",
            scaling_method="standard",
            outlier_method="iqr",
            outlier_action="clip"
        )
        
        preprocessor = DataPreprocessor(config)
        processed_data = preprocessor.fit_transform(
            engineered_data, 
            target_column='target_regression'
        )
        
        assert isinstance(processed_data, pd.DataFrame)
        assert processed_data.isnull().sum().sum() == 0  # No missing values
        assert len(processed_data) <= len(engineered_data)  # May be smaller due to outlier removal
    
    def test_end_to_end_preprocessing_classification(self):
        """Test complete preprocessing pipeline for classification."""
        config = ProcessingConfig(
            missing_strategy="median",
            scaling_method="minmax"
        )
        
        preprocessor = DataPreprocessor(config)
        processed_data = preprocessor.fit_transform(
            self.test_data, 
            target_column='target_classification'
        )
        
        assert isinstance(processed_data, pd.DataFrame)
        assert preprocessor.is_fitted


if __name__ == "__main__":
    pytest.main([__file__])

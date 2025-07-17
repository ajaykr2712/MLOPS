"""
Unit tests for model validation functionality.
Tests model performance metrics, validation checks, and error handling.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.mdt_dashboard.ml_pipeline.model_validation import (
    ModelValidator,
    ValidationResult,
    PerformanceMetrics,
    ModelValidationError
)


class TestModelValidator:
    """Test suite for ModelValidator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data for testing."""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=10,
            n_clusters_per_class=1,
            random_state=42
        )
        return pd.DataFrame(X), pd.Series(y)
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def validator(self):
        """Create ModelValidator instance."""
        config = {
            'min_accuracy': 0.8,
            'min_precision': 0.75,
            'min_recall': 0.75,
            'max_prediction_time': 0.1,
            'required_features': None
        }
        return ModelValidator(config)
    
    def test_model_validator_initialization(self, validator):
        """Test ModelValidator initialization."""
        assert validator.config['min_accuracy'] == 0.8
        assert validator.config['min_precision'] == 0.75
        assert validator.config['min_recall'] == 0.75
        assert validator.config['max_prediction_time'] == 0.1
    
    def test_performance_metrics_calculation(self, validator, trained_model, sample_data):
        """Test performance metrics calculation."""
        X, y = sample_data
        y_pred = trained_model.predict(X)
        
        metrics = validator._calculate_performance_metrics(y, y_pred)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
    
    def test_prediction_time_validation(self, validator, trained_model, sample_data):
        """Test prediction time validation."""
        X, _ = sample_data
        
        # Test with small dataset (should pass)
        small_X = X.head(10)
        is_fast_enough, avg_time = validator._validate_prediction_time(trained_model, small_X)
        
        assert isinstance(is_fast_enough, bool)
        assert isinstance(avg_time, float)
        assert avg_time > 0
    
    def test_feature_validation_success(self, validator, sample_data):
        """Test successful feature validation."""
        X, _ = sample_data
        required_features = list(X.columns[:10])
        
        validator.config['required_features'] = required_features
        is_valid, missing_features = validator._validate_features(X)
        
        assert is_valid is True
        assert len(missing_features) == 0
    
    def test_feature_validation_failure(self, validator, sample_data):
        """Test feature validation with missing features."""
        X, _ = sample_data
        required_features = list(X.columns) + ['missing_feature_1', 'missing_feature_2']
        
        validator.config['required_features'] = required_features
        is_valid, missing_features = validator._validate_features(X)
        
        assert is_valid is False
        assert 'missing_feature_1' in missing_features
        assert 'missing_feature_2' in missing_features
    
    def test_full_model_validation_success(self, validator, trained_model, sample_data):
        """Test complete model validation that should pass."""
        X, y = sample_data
        
        result = validator.validate_model(trained_model, X, y)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.metrics is not None
        assert len(result.errors) == 0
    
    def test_model_validation_performance_failure(self, validator, sample_data):
        """Test model validation that fails due to poor performance."""
        X, y = sample_data
        
        # Create a dummy model that always predicts the same class
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))
        
        dummy_model = DummyModel()
        
        # Set high performance thresholds
        validator.config['min_accuracy'] = 0.9
        validator.config['min_precision'] = 0.9
        validator.config['min_recall'] = 0.9
        
        result = validator.validate_model(dummy_model, X, y)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('accuracy' in error.lower() for error in result.errors)
    
    def test_validation_with_invalid_input(self, validator, trained_model):
        """Test validation with invalid input data."""
        invalid_X = "not_a_dataframe"
        invalid_y = "not_a_series"
        
        with pytest.raises(ModelValidationError):
            validator.validate_model(trained_model, invalid_X, invalid_y)
    
    @patch('time.time')
    def test_prediction_time_measurement(self, mock_time, validator, trained_model, sample_data):
        """Test prediction time measurement accuracy."""
        X, _ = sample_data
        
        # Mock time.time() to return predictable values
        mock_time.side_effect = [0.0, 0.05, 0.05, 0.1, 0.1, 0.15]  # Three predictions taking 0.05s each
        
        is_fast_enough, avg_time = validator._validate_prediction_time(trained_model, X.head(3))
        
        assert abs(avg_time - 0.05) < 0.001  # Should average to 0.05 seconds
    
    def test_validation_result_serialization(self, validator, trained_model, sample_data):
        """Test ValidationResult can be serialized to dict."""
        X, y = sample_data
        
        result = validator.validate_model(trained_model, X, y)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'is_valid' in result_dict
        assert 'metrics' in result_dict
        assert 'errors' in result_dict
        assert 'validation_timestamp' in result_dict
    
    def test_custom_validation_rules(self, sample_data):
        """Test validator with custom validation rules."""
        custom_config = {
            'min_accuracy': 0.95,  # Very high threshold
            'min_precision': 0.9,
            'min_recall': 0.9,
            'max_prediction_time': 0.001,  # Very low threshold
            'custom_checks': {
                'min_feature_importance': 0.01,
                'max_model_size_mb': 100
            }
        }
        
        validator = ModelValidator(custom_config)
        X, y = sample_data
        
        # Create a simple model that won't meet the thresholds
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        result = validator.validate_model(model, X, y)
        
        # Should fail due to strict thresholds
        assert result.is_valid is False


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics class."""
    
    def test_performance_metrics_initialization(self):
        """Test PerformanceMetrics initialization."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            prediction_time=0.05
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.82
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.85
        assert metrics.prediction_time == 0.05
    
    def test_performance_metrics_to_dict(self):
        """Test PerformanceMetrics to_dict method."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            prediction_time=0.05
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['accuracy'] == 0.85
        assert metrics_dict['precision'] == 0.82
        assert metrics_dict['recall'] == 0.88
        assert metrics_dict['f1_score'] == 0.85
        assert metrics_dict['prediction_time'] == 0.05


if __name__ == '__main__':
    pytest.main([__file__])

"""
Unit tests for drift detection algorithms.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from mdt_dashboard.drift_detection.algorithms import (
    DriftResult,
    BaseDriftDetector,
    KolmogorovSmirnovDetector,
    PSIDetector,
    DriftDetectionSuite
)


class TestDriftResult:
    """Test DriftResult dataclass."""
    
    def test_drift_result_creation(self):
        """Test creating a DriftResult."""
        result = DriftResult(
            test_name="Test",
            p_value=0.05,
            statistic=0.1,
            threshold=0.05,
            is_drift=True,
            severity="medium",
            feature_name="feature1"
        )
        
        assert result.test_name == "Test"
        assert result.p_value == 0.05
        assert result.is_drift is True
        assert result.severity == "medium"
    
    def test_drift_result_to_dict(self):
        """Test converting DriftResult to dictionary."""
        result = DriftResult(
            test_name="Test",
            p_value=0.05,
            statistic=0.1,
            threshold=0.05,
            is_drift=True,
            severity="medium"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["test_name"] == "Test"
        assert result_dict["p_value"] == 0.05
        assert result_dict["is_drift"] is True


class TestKolmogorovSmirnovDetector:
    """Test Kolmogorov-Smirnov drift detector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = KolmogorovSmirnovDetector(threshold=0.05)
        
        # Create test data
        np.random.seed(42)
        self.reference_data = np.random.normal(0, 1, 1000)
        self.comparison_data_no_drift = np.random.normal(0, 1, 1000)
        self.comparison_data_with_drift = np.random.normal(2, 1, 1000)
    
    def test_no_drift_detection(self):
        """Test when there is no drift."""
        result = self.detector.detect(
            self.reference_data, 
            self.comparison_data_no_drift,
            feature_name="test_feature"
        )
        
        assert isinstance(result, DriftResult)
        assert result.test_name == "Kolmogorov-Smirnov"
        assert result.feature_name == "test_feature"
        assert result.p_value > 0.05  # No drift expected
        assert result.is_drift is False
    
    def test_drift_detection(self):
        """Test when there is drift."""
        result = self.detector.detect(
            self.reference_data,
            self.comparison_data_with_drift,
            feature_name="test_feature"
        )
        
        assert isinstance(result, DriftResult)
        assert result.p_value < 0.05  # Drift expected
        assert result.is_drift is True
        assert result.severity in ["low", "medium", "high"]
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_data = np.array([])
        
        result = self.detector.detect(
            empty_data,
            self.comparison_data_no_drift,
            feature_name="test_feature"
        )
        
        assert result.is_drift is False
        assert "error" in result.metadata
    
    def test_nan_data_handling(self):
        """Test handling of NaN values."""
        data_with_nan = np.array([1, 2, np.nan, 4, 5])
        
        result = self.detector.detect(
            data_with_nan,
            self.comparison_data_no_drift[:5],
            feature_name="test_feature"
        )
        
        assert isinstance(result, DriftResult)
        assert result.reference_size == 4  # NaN should be removed


class TestPSIDetector:
    """Test Population Stability Index detector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = PSIDetector(threshold=0.2, bins=10)
        
        # Create test data
        np.random.seed(42)
        self.reference_data = np.random.normal(0, 1, 1000)
        self.comparison_data_no_drift = np.random.normal(0, 1, 1000)
        self.comparison_data_with_drift = np.random.normal(3, 2, 1000)
        
        # Categorical data
        self.categorical_ref = np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
        self.categorical_comp_no_drift = np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
        self.categorical_comp_drift = np.random.choice(['A', 'B', 'C'], 1000, p=[0.2, 0.3, 0.5])
    
    def test_continuous_no_drift(self):
        """Test PSI with continuous data and no drift."""
        result = self.detector.detect(
            self.reference_data,
            self.comparison_data_no_drift,
            feature_name="continuous_feature"
        )
        
        assert isinstance(result, DriftResult)
        assert result.test_name == "Population Stability Index"
        assert result.statistic < self.detector.threshold
        assert result.is_drift is False
    
    def test_continuous_with_drift(self):
        """Test PSI with continuous data and drift."""
        result = self.detector.detect(
            self.reference_data,
            self.comparison_data_with_drift,
            feature_name="continuous_feature"
        )
        
        assert result.statistic > self.detector.threshold
        assert result.is_drift is True
    
    def test_categorical_no_drift(self):
        """Test PSI with categorical data and no drift."""
        result = self.detector.detect(
            self.categorical_ref,
            self.categorical_comp_no_drift,
            feature_name="categorical_feature"
        )
        
        assert result.statistic < self.detector.threshold
        assert result.is_drift is False
    
    def test_categorical_with_drift(self):
        """Test PSI with categorical data and drift."""
        result = self.detector.detect(
            self.categorical_ref,
            self.categorical_comp_drift,
            feature_name="categorical_feature"
        )
        
        assert result.statistic > self.detector.threshold
        assert result.is_drift is True


class TestDriftDetectionSuite:
    """Test the comprehensive drift detection suite."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.suite = DriftDetectionSuite()
        
        # Create multi-feature test data
        np.random.seed(42)
        self.reference_data = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'feature3': np.random.choice(['A', 'B', 'C'], 1000)
        }
        
        self.comparison_data_no_drift = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'feature3': np.random.choice(['A', 'B', 'C'], 1000)
        }
        
        self.comparison_data_with_drift = {
            'feature1': np.random.normal(2, 1, 1000),  # Drift in feature1
            'feature2': np.random.normal(10, 2, 1000),  # No drift
            'feature3': np.random.choice(['A', 'B', 'C'], 1000, p=[0.7, 0.2, 0.1])  # Drift
        }
    
    def test_multi_feature_no_drift(self):
        """Test drift detection across multiple features with no drift."""
        results = self.suite.detect_drift(
            self.reference_data,
            self.comparison_data_no_drift
        )
        
        assert isinstance(results, dict)
        assert len(results) == 3  # Three features
        
        for feature_name, feature_results in results.items():
            assert isinstance(feature_results, list)
            assert len(feature_results) > 0
            
            # Most tests should indicate no drift
            drift_count = sum(1 for result in feature_results if result.is_drift)
            assert drift_count <= len(feature_results) // 2  # Allow some false positives
    
    def test_multi_feature_with_drift(self):
        """Test drift detection across multiple features with drift."""
        results = self.suite.detect_drift(
            self.reference_data,
            self.comparison_data_with_drift
        )
        
        # Check that drift is detected in feature1 and feature3
        feature1_results = results['feature1']
        feature3_results = results['feature3']
        
        # At least one test should detect drift for drifted features
        feature1_drift = any(result.is_drift for result in feature1_results)
        feature3_drift = any(result.is_drift for result in feature3_results)
        
        assert feature1_drift or feature3_drift  # At least one should detect drift
    
    def test_missing_features(self):
        """Test handling of missing features in comparison data."""
        incomplete_comparison = {
            'feature1': np.random.normal(0, 1, 1000)
            # feature2 and feature3 missing
        }
        
        results = self.suite.detect_drift(
            self.reference_data,
            incomplete_comparison
        )
        
        # Should only have results for feature1
        assert 'feature1' in results
        assert 'feature2' not in results
        assert 'feature3' not in results


@pytest.fixture
def sample_drift_results():
    """Create sample drift results for testing."""
    return {
        'feature1': [
            DriftResult("KS", 0.01, 0.3, 0.05, True, "medium", "feature1"),
            DriftResult("PSI", 0.8, 0.1, 0.2, False, "none", "feature1")
        ],
        'feature2': [
            DriftResult("KS", 0.001, 0.5, 0.05, True, "high", "feature2"),
            DriftResult("PSI", 0.02, 0.25, 0.2, True, "medium", "feature2")
        ]
    }


def test_drift_detection_suite_report_generation(sample_drift_results):
    """Test generating comprehensive drift reports."""
    suite = DriftDetectionSuite()
    report = suite.generate_drift_report(sample_drift_results)
    
    assert isinstance(report, dict)
    assert 'summary' in report
    assert 'feature_results' in report
    assert 'recommendations' in report
    
    summary = report['summary']
    assert summary['total_features'] == 2
    assert summary['drifted_features'] == 2
    assert summary['overall_severity'] == 'high'


if __name__ == "__main__":
    pytest.main([__file__])

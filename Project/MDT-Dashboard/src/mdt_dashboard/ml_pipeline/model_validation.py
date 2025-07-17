"""
Model validation module for ensuring model quality and performance.

This module provides comprehensive model validation capabilities including:
- Performance metrics calculation
- Prediction time validation
- Feature validation
- Custom validation rules

Classes:
    PerformanceMetrics: Container for model performance metrics
    ValidationResult: Results of model validation
    ModelValidator: Main validation engine
    ModelValidationError: Custom exception for validation errors
"""

import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logger = logging.getLogger(__name__)


class ModelValidationError(Exception):
    """Custom exception for model validation errors."""
    pass


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_time: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class ValidationResult:
    """Results of model validation."""
    
    is_valid: bool
    metrics: Optional[PerformanceMetrics]
    errors: List[str]
    validation_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        result = {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'validation_timestamp': self.validation_timestamp.isoformat()
        }
        
        if self.metrics:
            result['metrics'] = self.metrics.to_dict()
        else:
            result['metrics'] = None
            
        return result


class ModelValidator:
    """
    Comprehensive model validator with configurable validation rules.
    
    This class provides methods to validate models across multiple dimensions:
    - Performance metrics (accuracy, precision, recall, F1)
    - Prediction time requirements
    - Feature requirements
    - Custom validation rules
    
    Args:
        config: Dictionary containing validation configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelValidator with configuration.
        
        Args:
            config: Validation configuration dictionary containing:
                - min_accuracy: Minimum required accuracy (float)
                - min_precision: Minimum required precision (float)
                - min_recall: Minimum required recall (float)
                - max_prediction_time: Maximum allowed prediction time in seconds (float)
                - required_features: List of required feature names (List[str], optional)
                - custom_checks: Dictionary of custom validation rules (Dict, optional)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_keys = ['min_accuracy', 'min_precision', 'min_recall', 'max_prediction_time']
        
        for key in required_keys:
            if key not in self.config:
                raise ModelValidationError(f"Missing required configuration key: {key}")
            
            if not isinstance(self.config[key], (int, float)):
                raise ModelValidationError(f"Configuration key {key} must be numeric")
            
            if self.config[key] < 0:
                raise ModelValidationError(f"Configuration key {key} must be non-negative")
    
    def validate_model(
        self, 
        model: Any, 
        X_test: Union[pd.DataFrame, np.ndarray], 
        y_test: Union[pd.Series, np.ndarray]
    ) -> ValidationResult:
        """
        Perform comprehensive model validation.
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test targets
            
        Returns:
            ValidationResult: Comprehensive validation results
            
        Raises:
            ModelValidationError: If input data is invalid
        """
        self.logger.info("Starting model validation")
        
        # Validate inputs
        if not hasattr(model, 'predict'):
            raise ModelValidationError("Model must have a 'predict' method")
        
        if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
            raise ModelValidationError("X_test must be pandas DataFrame or numpy array")
        
        if not isinstance(y_test, (pd.Series, np.ndarray)):
            raise ModelValidationError("y_test must be pandas Series or numpy array")
        
        errors = []
        
        try:
            # Convert to pandas if numpy arrays
            if isinstance(X_test, np.ndarray):
                X_test = pd.DataFrame(X_test)
            if isinstance(y_test, np.ndarray):
                y_test = pd.Series(y_test)
            
            # Feature validation
            if self.config.get('required_features'):
                is_features_valid, missing_features = self._validate_features(X_test)
                if not is_features_valid:
                    errors.append(f"Missing required features: {missing_features}")
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(y_test, y_pred)
            
            # Validate performance metrics
            performance_errors = self._validate_performance_metrics(metrics)
            errors.extend(performance_errors)
            
            # Validate prediction time
            is_fast_enough, avg_time = self._validate_prediction_time(model, X_test)
            metrics.prediction_time = avg_time
            
            if not is_fast_enough:
                errors.append(
                    f"Prediction time {avg_time:.4f}s exceeds maximum allowed "
                    f"{self.config['max_prediction_time']}s"
                )
            
            # Custom validations
            if 'custom_checks' in self.config:
                custom_errors = self._perform_custom_validations(model, X_test, y_test, metrics)
                errors.extend(custom_errors)
            
            is_valid = len(errors) == 0
            
            self.logger.info(f"Model validation completed. Valid: {is_valid}")
            if errors:
                self.logger.warning(f"Validation errors: {errors}")
            
            return ValidationResult(
                is_valid=is_valid,
                metrics=metrics,
                errors=errors,
                validation_timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error during model validation: {str(e)}")
            raise ModelValidationError(f"Validation failed: {str(e)}")
    
    def _calculate_performance_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            PerformanceMetrics: Calculated metrics
        """
        # Handle binary and multiclass classification
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            prediction_time=0.0  # Will be set by prediction time validation
        )
    
    def _validate_performance_metrics(self, metrics: PerformanceMetrics) -> List[str]:
        """
        Validate performance metrics against thresholds.
        
        Args:
            metrics: Calculated performance metrics
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if metrics.accuracy < self.config['min_accuracy']:
            errors.append(
                f"Accuracy {metrics.accuracy:.4f} below minimum required "
                f"{self.config['min_accuracy']:.4f}"
            )
        
        if metrics.precision < self.config['min_precision']:
            errors.append(
                f"Precision {metrics.precision:.4f} below minimum required "
                f"{self.config['min_precision']:.4f}"
            )
        
        if metrics.recall < self.config['min_recall']:
            errors.append(
                f"Recall {metrics.recall:.4f} below minimum required "
                f"{self.config['min_recall']:.4f}"
            )
        
        return errors
    
    def _validate_prediction_time(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        num_iterations: int = 3
    ) -> Tuple[bool, float]:
        """
        Validate model prediction time.
        
        Args:
            model: Trained model
            X_test: Test features
            num_iterations: Number of timing iterations
            
        Returns:
            Tuple of (is_fast_enough, average_time)
        """
        times = []
        
        # Use a subset of data for timing to get consistent results
        test_subset = X_test.head(min(100, len(X_test)))
        
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model.predict(test_subset)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        is_fast_enough = avg_time <= self.config['max_prediction_time']
        
        return is_fast_enough, avg_time
    
    def _validate_features(self, X_test: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that required features are present.
        
        Args:
            X_test: Test features DataFrame
            
        Returns:
            Tuple of (is_valid, missing_features)
        """
        required_features = self.config['required_features']
        if not required_features:
            return True, []
        
        available_features = set(X_test.columns)
        required_features_set = set(required_features)
        missing_features = list(required_features_set - available_features)
        
        is_valid = len(missing_features) == 0
        
        return is_valid, missing_features
    
    def _perform_custom_validations(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: PerformanceMetrics
    ) -> List[str]:
        """
        Perform custom validation checks.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            metrics: Calculated performance metrics
            
        Returns:
            List of validation error messages
        """
        errors = []
        custom_checks = self.config['custom_checks']
        
        # Example custom validations
        if 'min_feature_importance' in custom_checks:
            if hasattr(model, 'feature_importances_'):
                min_importance = np.min(model.feature_importances_)
                if min_importance < custom_checks['min_feature_importance']:
                    errors.append(
                        f"Minimum feature importance {min_importance:.4f} below "
                        f"required {custom_checks['min_feature_importance']:.4f}"
                    )
        
        if 'max_model_size_mb' in custom_checks:
            # This is a simplified model size check
            # In practice, you might serialize the model to check actual size
            if hasattr(model, 'n_estimators') and hasattr(model, 'n_features_in_'):
                estimated_size = (model.n_estimators * model.n_features_in_) / 1000000  # Rough estimation
                if estimated_size > custom_checks['max_model_size_mb']:
                    errors.append(
                        f"Estimated model size {estimated_size:.2f}MB exceeds "
                        f"maximum {custom_checks['max_model_size_mb']}MB"
                    )
        
        return errors
    
    def validate_data_quality(self, X: pd.DataFrame, y: pd.Series = None) -> ValidationResult:
        """
        Validate data quality before training or prediction.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            ValidationResult: Data quality validation results
        """
        errors = []
        
        # Check for missing values
        missing_counts = X.isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]
        
        if len(features_with_missing) > 0:
            errors.append(f"Features with missing values: {features_with_missing.to_dict()}")
        
        # Check for infinite values
        infinite_counts = np.isinf(X.select_dtypes(include=[np.number])).sum()
        features_with_infinite = infinite_counts[infinite_counts > 0]
        
        if len(features_with_infinite) > 0:
            errors.append(f"Features with infinite values: {features_with_infinite.to_dict()}")
        
        # Check for constant features
        numeric_features = X.select_dtypes(include=[np.number])
        constant_features = []
        
        for col in numeric_features.columns:
            if numeric_features[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            errors.append(f"Constant features detected: {constant_features}")
        
        # Target variable checks (if provided)
        if y is not None:
            if y.isnull().sum() > 0:
                errors.append(f"Target variable has {y.isnull().sum()} missing values")
            
            if len(y.unique()) < 2:
                errors.append("Target variable has less than 2 unique values")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            metrics=None,
            errors=errors,
            validation_timestamp=datetime.now()
        )

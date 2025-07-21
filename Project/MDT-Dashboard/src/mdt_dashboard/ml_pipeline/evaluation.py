"""
Advanced Model Evaluation Pipeline
Comprehensive evaluation metrics, model profiling, and performance analysis
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"
    ])
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, kfold, time_series
    
    # Performance profiling
    profile_memory: bool = True
    profile_compute: bool = True
    profile_inference_time: bool = True
    
    # Fairness evaluation
    evaluate_fairness: bool = True
    protected_attributes: List[str] = field(default_factory=list)
    
    # Robustness testing
    test_adversarial: bool = False
    test_noise_robustness: bool = True
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    
    # Explainability
    generate_explanations: bool = True
    explanation_methods: List[str] = field(default_factory=lambda: ["shap", "lime"])
    
    # Outputs
    save_predictions: bool = True
    save_explanations: bool = True
    generate_report: bool = True


class ModelEvaluator:
    """Comprehensive model evaluation with advanced metrics."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {}
        self.predictions = {}
        self.explanations = {}
        
    def evaluate_model(self, model, X_test, y_test, X_train=None, y_train=None):
        """Comprehensive model evaluation."""
        logger.info("Starting comprehensive model evaluation...")
        
        start_time = time.time()
        
        # Basic predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(X_test)
            self.predictions['predictions'] = predictions
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
                self.predictions['probabilities'] = probabilities
        
        # Standard metrics
        self.results['standard_metrics'] = self._compute_standard_metrics(
            y_test, predictions, probabilities if 'probabilities' in self.predictions else None
        )
        
        # Performance profiling
        if self.config.profile_inference_time:
            self.results['performance'] = self._profile_model_performance(model, X_test)
        
        # Cross-validation
        if X_train is not None and y_train is not None:
            self.results['cross_validation'] = self._cross_validate_model(
                model, X_train, y_train
            )
        
        # Fairness evaluation
        if self.config.evaluate_fairness and self.config.protected_attributes:
            self.results['fairness'] = self._evaluate_fairness(
                model, X_test, y_test, predictions
            )
        
        # Robustness testing
        if self.config.test_noise_robustness:
            self.results['robustness'] = self._test_robustness(model, X_test, y_test)
        
        # Explainability analysis
        if self.config.generate_explanations:
            self.explanations = self._generate_explanations(model, X_test)
        
        # Model complexity analysis
        self.results['complexity'] = self._analyze_model_complexity(model)
        
        # Feature importance
        self.results['feature_importance'] = self._compute_feature_importance(
            model, X_test
        )
        
        total_time = time.time() - start_time
        self.results['evaluation_time'] = total_time
        
        logger.info(f"Model evaluation completed in {total_time:.2f} seconds")
        
        return self.results
    
    def _compute_standard_metrics(self, y_true, y_pred, y_proba=None):
        """Compute standard classification/regression metrics."""
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, average_precision_score,
                mean_squared_error, mean_absolute_error, r2_score,
                classification_report, confusion_matrix
            )
            
            metrics = {}
            
            # Determine if classification or regression
            is_classification = self._is_classification_task(y_true, y_pred)
            
            if is_classification:
                # Classification metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # ROC-AUC and PR-AUC if probabilities available
                if y_proba is not None:
                    try:
                        if y_proba.shape[1] == 2:  # Binary classification
                            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                            metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
                        else:  # Multi-class
                            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                    except Exception as e:
                        logger.warning(f"Could not compute AUC metrics: {e}")
                
                # Confusion matrix
                metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
                
                # Classification report
                metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
                
            else:
                # Regression metrics
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['r2'] = r2_score(y_true, y_pred)
                
                # Additional regression metrics
                metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
                metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing standard metrics: {e}")
            return {}
    
    def _is_classification_task(self, y_true, y_pred):
        """Determine if task is classification or regression."""
        # Check if targets are discrete
        unique_true = len(np.unique(y_true))
        unique_pred = len(np.unique(y_pred))
        
        # If both have few unique values and are integers, likely classification
        return (unique_true <= 20 and unique_pred <= 20 and 
                np.all(np.equal(np.mod(y_true, 1), 0)) and 
                np.all(np.equal(np.mod(y_pred, 1), 0)))
    
    def _profile_model_performance(self, model, X_test):
        """Profile model inference performance."""
        performance_metrics = {}
        
        # Inference time profiling
        sample_sizes = [1, 10, 100, min(1000, len(X_test))]
        
        for size in sample_sizes:
            if size > len(X_test):
                continue
                
            X_sample = X_test[:size]
            
            # Time multiple runs
            times = []
            for _ in range(10):
                start_time = time.time()
                _ = model.predict(X_sample)
                end_time = time.time()
                times.append(end_time - start_time)
            
            performance_metrics[f'inference_time_{size}_samples'] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'per_sample': np.mean(times) / size
            }
        
        # Memory profiling
        if self.config.profile_memory:
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Large prediction to measure memory usage
                if len(X_test) > 100:
                    _ = model.predict(X_test[:100])
                
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                
                performance_metrics['memory_usage'] = {
                    'before_mb': memory_before,
                    'after_mb': memory_after,
                    'increase_mb': memory_after - memory_before
                }
                
            except Exception as e:
                logger.warning(f"Could not profile memory usage: {e}")
        
        return performance_metrics
    
    def _cross_validate_model(self, model, X, y):
        """Perform cross-validation."""
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
            from sklearn.base import clone
            
            # Choose cross-validation strategy
            if self.config.cv_strategy == "stratified":
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            
            cv_results = {}
            
            # Score multiple metrics
            for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
                try:
                    scores = cross_val_score(clone(model), X, y, cv=cv, scoring=metric)
                    cv_results[metric] = {
                        'scores': scores.tolist(),
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'confidence_interval_95': [
                            scores.mean() - 1.96 * scores.std(),
                            scores.mean() + 1.96 * scores.std()
                        ]
                    }
                except Exception as e:
                    logger.warning(f"Could not compute CV score for {metric}: {e}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {}
    
    def _evaluate_fairness(self, model, X_test, y_test, predictions):
        """Evaluate model fairness across protected attributes."""
        fairness_metrics = {}
        
        try:
            # This is a simplified fairness evaluation
            # In practice, you'd use libraries like fairlearn or aif360
            
            for attr in self.config.protected_attributes:
                if attr in X_test.columns if hasattr(X_test, 'columns') else []:
                    attr_values = X_test[attr] if hasattr(X_test, 'columns') else X_test[:, 0]  # Simplified
                    
                    unique_values = np.unique(attr_values)
                    group_metrics = {}
                    
                    for value in unique_values:
                        mask = attr_values == value
                        if np.sum(mask) > 0:
                            group_pred = predictions[mask]
                            group_true = y_test[mask]
                            
                            from sklearn.metrics import accuracy_score
                            group_accuracy = accuracy_score(group_true, group_pred)
                            group_metrics[str(value)] = {
                                'count': int(np.sum(mask)),
                                'accuracy': group_accuracy
                            }
                    
                    fairness_metrics[attr] = group_metrics
            
            return fairness_metrics
            
        except Exception as e:
            logger.error(f"Fairness evaluation failed: {e}")
            return {}
    
    def _test_robustness(self, model, X_test, y_test):
        """Test model robustness to noise and perturbations."""
        robustness_results = {}
        
        try:
            original_predictions = model.predict(X_test)
            
            for noise_level in self.config.noise_levels:
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, X_test.shape)
                X_noisy = X_test + noise
                
                noisy_predictions = model.predict(X_noisy)
                
                # Compute stability metrics
                from sklearn.metrics import accuracy_score
                
                stability = accuracy_score(original_predictions, noisy_predictions)
                
                robustness_results[f'noise_level_{noise_level}'] = {
                    'prediction_stability': stability,
                    'performance_degradation': accuracy_score(y_test, original_predictions) - 
                                            accuracy_score(y_test, noisy_predictions)
                }
            
            return robustness_results
            
        except Exception as e:
            logger.error(f"Robustness testing failed: {e}")
            return {}
    
    def _generate_explanations(self, model, X_test):
        """Generate model explanations using various methods."""
        explanations = {}
        
        # SHAP explanations
        if "shap" in self.config.explanation_methods:
            explanations['shap'] = self._generate_shap_explanations(model, X_test)
        
        # LIME explanations
        if "lime" in self.config.explanation_methods:
            explanations['lime'] = self._generate_lime_explanations(model, X_test)
        
        # Permutation importance
        explanations['permutation_importance'] = self._compute_permutation_importance(model, X_test)
        
        return explanations
    
    def _generate_shap_explanations(self, model, X_test):
        """Generate SHAP explanations."""
        try:
            import shap
            
            # Sample subset for efficiency
            sample_size = min(100, len(X_test))
            X_sample = X_test[:sample_size]
            
            # Choose appropriate explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X_sample)
            else:
                explainer = shap.Explainer(model.predict, X_sample)
            
            shap_values = explainer(X_sample[:10])  # Explain first 10 samples
            
            return {
                'values': shap_values.values.tolist() if hasattr(shap_values, 'values') else [],
                'base_values': shap_values.base_values.tolist() if hasattr(shap_values, 'base_values') else [],
                'feature_names': getattr(shap_values, 'feature_names', [])
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {}
    
    def _generate_lime_explanations(self, model, X_test):
        """Generate LIME explanations."""
        try:
            from lime import lime_tabular
            
            # Sample for background
            sample_size = min(100, len(X_test))
            explainer = lime_tabular.LimeTabularExplainer(
                X_test[:sample_size],
                mode='classification' if hasattr(model, 'predict_proba') else 'regression'
            )
            
            # Explain first few instances
            explanations = []
            for i in range(min(5, len(X_test))):
                exp = explainer.explain_instance(
                    X_test[i], 
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=10
                )
                explanations.append(exp.as_list())
            
            return {'explanations': explanations}
            
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return {}
    
    def _compute_permutation_importance(self, model, X_test):
        """Compute permutation importance."""
        try:
            from sklearn.inspection import permutation_importance
            
            # Use a subset for efficiency
            sample_size = min(500, len(X_test))
            X_sample = X_test[:sample_size]
            
            # Get baseline predictions
            baseline_score = model.score(X_sample, model.predict(X_sample))
            
            importance_scores = []
            feature_names = []
            
            for i in range(X_sample.shape[1]):
                # Permute feature i
                X_permuted = X_sample.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                
                permuted_score = model.score(X_permuted, model.predict(X_permuted))
                importance = baseline_score - permuted_score
                
                importance_scores.append(importance)
                feature_names.append(f'feature_{i}')
            
            return {
                'importance_scores': importance_scores,
                'feature_names': feature_names
            }
            
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            return {}
    
    def _analyze_model_complexity(self, model):
        """Analyze model complexity."""
        complexity_metrics = {}
        
        try:
            # Number of parameters (for tree-based models)
            if hasattr(model, 'tree_'):
                complexity_metrics['n_nodes'] = model.tree_.node_count
                complexity_metrics['max_depth'] = model.tree_.max_depth
            elif hasattr(model, 'n_estimators'):
                complexity_metrics['n_estimators'] = model.n_estimators
            
            # Model size in memory
            import pickle
            import sys
            
            model_size = len(pickle.dumps(model))
            complexity_metrics['model_size_bytes'] = model_size
            complexity_metrics['model_size_mb'] = model_size / (1024 * 1024)
            
            # Training time (if available)
            if hasattr(model, 'fit_time_'):
                complexity_metrics['training_time'] = model.fit_time_
            
            return complexity_metrics
            
        except Exception as e:
            logger.warning(f"Model complexity analysis failed: {e}")
            return {}
    
    def _compute_feature_importance(self, model, X_test):
        """Compute feature importance from the model."""
        try:
            importance_dict = {}
            
            # Built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance_dict['model_importance'] = model.feature_importances_.tolist()
            
            # Coefficients for linear models
            if hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]  # Take first class for multi-class
                importance_dict['coefficients'] = coef.tolist()
            
            # Feature names if available
            if hasattr(X_test, 'columns'):
                importance_dict['feature_names'] = X_test.columns.tolist()
            else:
                importance_dict['feature_names'] = [f'feature_{i}' for i in range(X_test.shape[1])]
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"Feature importance computation failed: {e}")
            return {}
    
    def generate_evaluation_report(self, output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': time.time(),
            'config': self.config.__dict__,
            'results': self.results,
            'predictions_summary': {
                'num_predictions': len(self.predictions.get('predictions', [])),
                'has_probabilities': 'probabilities' in self.predictions
            },
            'explanations_summary': {
                'methods_used': list(self.explanations.keys()),
                'num_explained_instances': len(self.explanations.get('shap', {}).get('values', []))
            }
        }
        
        # Save report
        if self.config.generate_report:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report


# Export main class
__all__ = ["EvaluationConfig", "ModelEvaluator"]

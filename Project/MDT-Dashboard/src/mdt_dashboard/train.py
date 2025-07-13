"""
Advanced ML training pipeline with automated model selection and hyperparameter tuning.
Includes experiment tracking, cross-validation, and production-ready model artifacts.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import joblib
from pathlib import Path
import json
import warnings

# ML algorithms
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Model evaluation
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    KFold, StratifiedKFold
)

# Hyperparameter optimization
from scipy.stats import uniform, randint

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    # Problem type
    problem_type: str = "regression"  # regression, classification, timeseries
    
    # Model selection
    algorithms: List[str] = field(default_factory=lambda: ["random_forest", "xgboost", "lightgbm"])
    
    # Hyperparameter tuning
    tuning_method: str = "random_search"  # grid_search, random_search, bayesian
    n_trials: int = 50
    cv_folds: int = 5
    
    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Model evaluation
    scoring: str = "neg_mean_squared_error"  # For regression
    early_stopping: bool = True
    
    # MLflow tracking
    experiment_name: str = "mdt-model-training"
    run_name: Optional[str] = None
    track_params: bool = True
    track_metrics: bool = True
    track_artifacts: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "problem_type": self.problem_type,
            "algorithms": self.algorithms,
            "tuning_method": self.tuning_method,
            "n_trials": self.n_trials,
            "cv_folds": self.cv_folds,
            "test_size": self.test_size,
            "validation_size": self.validation_size,
            "random_state": self.random_state,
            "scoring": self.scoring,
            "early_stopping": self.early_stopping,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "track_params": self.track_params,
            "track_metrics": self.track_metrics,
            "track_artifacts": self.track_artifacts
        }


@dataclass
class ModelResult:
    """Results from model training and evaluation."""
    
    model_name: str
    algorithm: str
    best_params: Dict[str, Any]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    test_score: float
    train_score: float
    training_time: float
    prediction_time: float
    model_size: int
    feature_importance: Optional[Dict[str, float]] = None
    class_report: Optional[Dict[str, Any]] = None
    conf_matrix: Optional[List[List[int]]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "best_params": self.best_params,
            "cv_scores": self.cv_scores,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "test_score": self.test_score,
            "train_score": self.train_score,
            "training_time": self.training_time,
            "prediction_time": self.prediction_time,
            "model_size": self.model_size,
            "feature_importance": self.feature_importance,
            "class_report": self.class_report,
            "conf_matrix": self.conf_matrix,
            "timestamp": self.timestamp.isoformat()
        }


class ModelRegistry:
    """Registry for managing trained models."""
    
    def __init__(self, registry_path: Union[str, Path] = "models"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.metadata = {}
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        metadata: Dict[str, Any],
        overwrite: bool = False
    ) -> str:
        """Register a trained model."""
        
        model_path = self.registry_path / f"{model_name}.joblib"
        metadata_path = self.registry_path / f"{model_name}_metadata.json"
        
        if model_path.exists() and not overwrite:
            raise ValueError(f"Model {model_name} already exists. Use overwrite=True to replace.")
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata["registered_at"] = datetime.now().isoformat()
        metadata["model_path"] = str(model_path)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update in-memory registry
        self.models[model_name] = model
        self.metadata[model_name] = metadata
        
        logger.info(f"Model {model_name} registered successfully")
        return str(model_path)
    
    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a registered model."""
        
        model_path = self.registry_path / f"{model_name}.joblib"
        metadata_path = self.registry_path / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise ValueError(f"Model {model_name} not found")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Update in-memory registry
        self.models[model_name] = model
        self.metadata[model_name] = metadata
        
        return model, metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        
        models_info = []
        
        for model_file in self.registry_path.glob("*.joblib"):
            model_name = model_file.stem
            metadata_file = self.registry_path / f"{model_name}_metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                models_info.append({
                    "model_name": model_name,
                    "algorithm": metadata.get("algorithm", "unknown"),
                    "registered_at": metadata.get("registered_at", "unknown"),
                    "performance": metadata.get("test_score", "unknown"),
                    "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2)
                })
        
        return sorted(models_info, key=lambda x: x["registered_at"], reverse=True)


class HyperparameterTuner:
    """Advanced hyperparameter tuning with multiple strategies."""
    
    @staticmethod
    def get_param_grid(algorithm: str, tuning_method: str = "random_search") -> Dict[str, Any]:
        """Get parameter grid for algorithm."""
        
        if algorithm == "random_forest":
            if tuning_method == "grid_search":
                return {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            else:
                return {
                    "n_estimators": randint(50, 500),
                    "max_depth": [10, 20, 30, None],
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf": randint(1, 10),
                    "max_features": ["sqrt", "log2", None]
                }
        
        elif algorithm == "xgboost":
            if tuning_method == "grid_search":
                return {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0]
                }
            else:
                return {
                    "n_estimators": randint(50, 500),
                    "max_depth": randint(3, 10),
                    "learning_rate": uniform(0.01, 0.3),
                    "subsample": uniform(0.6, 0.4),
                    "colsample_bytree": uniform(0.6, 0.4)
                }
        
        elif algorithm == "lightgbm":
            if tuning_method == "grid_search":
                return {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "num_leaves": [31, 62, 127]
                }
            else:
                return {
                    "n_estimators": randint(50, 500),
                    "max_depth": randint(3, 10),
                    "learning_rate": uniform(0.01, 0.3),
                    "num_leaves": randint(20, 200),
                    "feature_fraction": uniform(0.6, 0.4)
                }
        
        elif algorithm == "svm":
            if tuning_method == "grid_search":
                return {
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto", 0.001, 0.01]
                }
            else:
                return {
                    "C": uniform(0.1, 100),
                    "gamma": ["scale", "auto"] + list(uniform(0.001, 0.1).rvs(10))
                }
        
        else:
            return {}


class ModelTrainer:
    """Comprehensive model training pipeline."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.registry = ModelRegistry()
        self.results = []
        self.best_model = None
        self.best_score = -np.inf
        
        # Initialize MLflow
        mlflow.set_experiment(self.config.experiment_name)
    
    def get_algorithm_instance(self, algorithm: str, params: Dict[str, Any] = None) -> Any:
        """Get algorithm instance with parameters."""
        
        params = params or {}
        
        if self.config.problem_type == "regression":
            algorithms = {
                "linear_regression": LinearRegression,
                "ridge": Ridge,
                "lasso": Lasso,
                "elastic_net": ElasticNet,
                "random_forest": RandomForestRegressor,
                "extra_trees": ExtraTreesRegressor,
                "gradient_boosting": GradientBoostingRegressor,
                "xgboost": XGBRegressor,
                "lightgbm": LGBMRegressor,
                "svm": SVR,
                "knn": KNeighborsRegressor
            }
        else:  # classification
            algorithms = {
                "logistic_regression": LogisticRegression,
                "random_forest": RandomForestClassifier,
                "extra_trees": ExtraTreesClassifier,
                "gradient_boosting": GradientBoostingClassifier,
                "xgboost": XGBClassifier,
                "lightgbm": LGBMClassifier,
                "svm": SVC,
                "knn": KNeighborsClassifier
            }
        
        if algorithm not in algorithms:
            raise ValueError(f"Algorithm {algorithm} not supported for {self.config.problem_type}")
        
        return algorithms[algorithm](**params)
    
    def tune_hyperparameters(
        self,
        algorithm: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5
    ) -> Tuple[Any, Dict[str, Any], List[float]]:
        """Tune hyperparameters for algorithm."""
        
        # Get parameter grid
        param_grid = HyperparameterTuner.get_param_grid(algorithm, self.config.tuning_method)
        
        if not param_grid:
            # No tuning needed, use default parameters
            model = self.get_algorithm_instance(algorithm)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds, scoring=self.config.scoring, n_jobs=-1
            )
            model.fit(X_train, y_train)
            return model, {}, cv_scores.tolist()
        
        # Setup cross-validation
        if self.config.problem_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
        
        # Base model
        base_model = self.get_algorithm_instance(algorithm)
        
        # Hyperparameter search
        if self.config.tuning_method == "grid_search":
            search = GridSearchCV(
                base_model, param_grid,
                cv=cv, scoring=self.config.scoring,
                n_jobs=-1, verbose=1
            )
        else:  # random_search
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=self.config.n_trials,
                cv=cv, scoring=self.config.scoring,
                n_jobs=-1, verbose=1,
                random_state=self.config.random_state
            )
        
        # Fit search
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_, search.cv_results_["mean_test_score"]
    
    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        if self.config.problem_type == "regression":
            metrics = {
                "train_mse": mean_squared_error(y_train, y_train_pred),
                "test_mse": mean_squared_error(y_test, y_test_pred),
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "train_r2": r2_score(y_train, y_train_pred),
                "test_r2": r2_score(y_test, y_test_pred)
            }
        else:  # classification
            metrics = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "train_precision": precision_score(y_train, y_train_pred, average="weighted"),
                "test_precision": precision_score(y_test, y_test_pred, average="weighted"),
                "train_recall": recall_score(y_train, y_train_pred, average="weighted"),
                "test_recall": recall_score(y_test, y_test_pred, average="weighted"),
                "train_f1": f1_score(y_train, y_train_pred, average="weighted"),
                "test_f1": f1_score(y_test, y_test_pred, average="weighted")
            }
            
            # Add AUC for binary classification
            if len(np.unique(y_test)) == 2:
                y_test_proba = model.predict_proba(X_test)[:, 1]
                metrics["test_auc"] = roc_auc_score(y_test, y_test_proba)
        
        return metrics
    
    def train_single_model(
        self,
        algorithm: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> ModelResult:
        """Train and evaluate a single model."""
        
        with mlflow.start_run(run_name=f"{self.config.run_name}_{algorithm}") as run:
            logger.info(f"Training {algorithm}...")
            
            # Track parameters
            if self.config.track_params:
                mlflow.log_params(self.config.to_dict())
                mlflow.log_param("algorithm", algorithm)
            
            # Start timing
            start_time = datetime.now()
            
            # Tune hyperparameters
            model, best_params, cv_scores = self.tune_hyperparameters(
                algorithm, X_train, y_train, self.config.cv_folds
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Time prediction
            pred_start = datetime.now()
            _ = model.predict(X_test[:min(100, len(X_test))])
            prediction_time = (datetime.now() - pred_start).total_seconds()
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
            
            # Feature importance
            feature_importance = None
            if hasattr(model, "feature_importances_"):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, "coef_"):
                feature_importance = dict(zip(X_train.columns, model.coef_.flatten()))
            
            # Model size
            model_size = len(joblib.dumps(model))
            
            # Track metrics
            if self.config.track_metrics:
                mlflow.log_metrics(metrics)
                mlflow.log_metric("cv_mean", np.mean(cv_scores))
                mlflow.log_metric("cv_std", np.std(cv_scores))
                mlflow.log_metric("training_time", training_time)
                mlflow.log_metric("prediction_time", prediction_time)
            
            # Track artifacts
            if self.config.track_artifacts:
                mlflow.sklearn.log_model(model, "model")
                
                if feature_importance:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
                    features, importances = zip(*sorted_features)
                    plt.barh(features, importances)
                    plt.title(f"Feature Importance - {algorithm}")
                    plt.tight_layout()
                    mlflow.log_figure(plt.gcf(), "feature_importance.png")
                    plt.close()
            
            # Get primary score for model comparison
            if self.config.problem_type == "regression":
                test_score = metrics.get("test_r2", metrics.get("test_mse", 0))
                train_score = metrics.get("train_r2", metrics.get("train_mse", 0))
            else:
                test_score = metrics.get("test_accuracy", 0)
                train_score = metrics.get("train_accuracy", 0)
            
            # Create result
            result = ModelResult(
                model_name=f"{algorithm}_{run.info.run_id[:8]}",
                algorithm=algorithm,
                best_params=best_params,
                cv_scores=cv_scores,
                cv_mean=np.mean(cv_scores),
                cv_std=np.std(cv_scores),
                test_score=test_score,
                train_score=train_score,
                training_time=training_time,
                prediction_time=prediction_time,
                model_size=model_size,
                feature_importance=feature_importance
            )
            
            # Track best model
            if test_score > self.best_score:
                self.best_score = test_score
                self.best_model = model
                
                # Register best model
                self.registry.register_model(
                    model=model,
                    model_name="best_model",
                    metadata=result.to_dict(),
                    overwrite=True
                )
            
            logger.info(f"Completed training {algorithm}: Test Score = {test_score:.4f}")
            return result
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> List[ModelResult]:
        """Train all configured algorithms."""
        
        logger.info(f"Starting training for {len(self.config.algorithms)} algorithms")
        
        for algorithm in self.config.algorithms:
            try:
                result = self.train_single_model(algorithm, X_train, X_test, y_train, y_test)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Failed to train {algorithm}: {str(e)}")
                continue
        
        # Sort results by performance
        self.results.sort(key=lambda x: x.test_score, reverse=True)
        
        logger.info(f"Training completed. Best model: {self.results[0].algorithm} "
                   f"(Score: {self.results[0].test_score:.4f})")
        
        return self.results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all trained models."""
        
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for result in self.results:
            comparison_data.append({
                "Algorithm": result.algorithm,
                "Test Score": result.test_score,
                "CV Mean": result.cv_mean,
                "CV Std": result.cv_std,
                "Training Time (s)": result.training_time,
                "Model Size (bytes)": result.model_size
            })
        
        return pd.DataFrame(comparison_data).sort_values("Test Score", ascending=False)
    
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save training results."""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            "config": self.config.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "best_model_name": self.results[0].model_name if self.results else None,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Training results saved to {filepath}")


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: Optional[ModelConfig] = None
) -> ModelTrainer:
    """Train models with the given configuration."""
    
    config = config or ModelConfig()
    trainer = ModelTrainer(config)
    trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    return trainer
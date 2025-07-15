"""
Advanced ML training pipeline with automated model selection and hyperparameter tuning.
Includes experiment tracking, cross-validation, and production-ready model artifacts.

Refactored for improved code quality, type safety, and maintainability.
"""

import asyncio
import json
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import joblib
except ImportError:
    joblib = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    
    def ConfigDict(**kwargs):
        return None
    
    def Field(default=None, **kwargs):
        return default
    
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None

# ML algorithms - with optional imports
try:
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
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix
    )
    from sklearn.model_selection import (
        cross_val_score, GridSearchCV, RandomizedSearchCV,
        KFold, StratifiedKFold, train_test_split
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMRegressor = LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False

try:
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    uniform = randint = None
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ProblemType(str, Enum):
    """Types of ML problems."""
    
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    TIME_SERIES = "time_series"


class AlgorithmType(str, Enum):
    """Supported ML algorithms."""
    
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    KNN = "knn"
    GRADIENT_BOOSTING = "gradient_boosting"
    EXTRA_TREES = "extra_trees"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"


class TuningMethod(str, Enum):
    """Hyperparameter tuning methods."""
    
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"


class ModelStatus(str, Enum):
    """Model training status."""
    
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelConfig(BaseModel):
    """Configuration for model training."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Problem type
    problem_type: ProblemType = ProblemType.REGRESSION
    
    # Model selection
    algorithms: List[AlgorithmType] = Field(
        default=[AlgorithmType.RANDOM_FOREST, AlgorithmType.XGBOOST, AlgorithmType.LIGHTGBM]
    )
    
    # Hyperparameter tuning
    tuning_method: TuningMethod = TuningMethod.RANDOM_SEARCH
    n_trials: int = Field(default=50, ge=1, le=1000)
    cv_folds: int = Field(default=5, ge=2, le=20)
    
    # Training parameters
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    validation_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(default=42, ge=0)
    
    # Model evaluation
    scoring: str = "neg_mean_squared_error"  # For regression
    early_stopping: bool = True
    
    # MLflow tracking
    experiment_name: str = "mdt-model-training"
    run_name: Optional[str] = None
    track_params: bool = True
    track_metrics: bool = True
    track_artifacts: bool = True
    
    # Performance constraints
    max_training_time_minutes: int = Field(default=60, ge=1)
    max_model_size_mb: int = Field(default=100, ge=1)
    
    @field_validator("algorithms")
    @classmethod
    def validate_algorithms(cls, v: List[AlgorithmType]) -> List[AlgorithmType]:
        """Validate that algorithms are available."""
        if not v:
            raise ValueError("At least one algorithm must be specified")
        
        unavailable_algorithms = []
        for algo in v:
            if algo in [AlgorithmType.XGBOOST] and not XGBOOST_AVAILABLE:
                unavailable_algorithms.append(algo)
            elif algo in [AlgorithmType.LIGHTGBM] and not LIGHTGBM_AVAILABLE:
                unavailable_algorithms.append(algo)
            elif not SKLEARN_AVAILABLE:
                unavailable_algorithms.append(algo)
        
        if unavailable_algorithms:
            logger.warning(f"Removing unavailable algorithms: {unavailable_algorithms}")
            v = [algo for algo in v if algo not in unavailable_algorithms]
        
        if not v:
            raise ValueError("No available algorithms specified")
        
        return v


class ModelResult(BaseModel):
    """Results from model training and evaluation."""
    
    model_config = ConfigDict(extra="forbid")
    
    model_name: str
    algorithm: AlgorithmType
    status: ModelStatus
    best_params: Dict[str, Any] = Field(default_factory=dict)
    cv_scores: List[float] = Field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    test_score: float = 0.0
    train_score: float = 0.0
    training_time_seconds: float = Field(ge=0.0)
    prediction_time_ms: float = Field(ge=0.0)
    model_size_mb: float = Field(ge=0.0)
    feature_importance: Optional[Dict[str, float]] = None
    classification_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class TrainingProgress(BaseModel):
    """Training progress tracking."""
    
    model_config = ConfigDict(extra="forbid")
    
    current_algorithm: Optional[AlgorithmType] = None
    algorithms_completed: int = Field(default=0, ge=0)
    total_algorithms: int = Field(default=1, ge=1)
    current_trial: int = Field(default=0, ge=0)
    total_trials: int = Field(default=1, ge=1)
    elapsed_time_seconds: float = Field(default=0.0, ge=0.0)
    estimated_remaining_seconds: Optional[float] = None
    best_score: Optional[float] = None
    status: ModelStatus = ModelStatus.PENDING
    
    @property
    def progress_percent(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_algorithms == 0:
            return 0.0
        
        algorithm_progress = self.algorithms_completed / self.total_algorithms
        if self.current_algorithm and self.total_trials > 0:
            current_algorithm_progress = self.current_trial / self.total_trials / self.total_algorithms
            return min(100.0, (algorithm_progress + current_algorithm_progress) * 100)
        
        return min(100.0, algorithm_progress * 100)


class AlgorithmFactory:
    """Factory for creating ML algorithm instances."""
    
    @staticmethod
    def create_algorithm(
        algorithm: AlgorithmType,
        problem_type: ProblemType,
        random_state: int = 42,
        **kwargs
    ) -> Any:
        """Create an algorithm instance."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
        
        is_classification = problem_type in [
            ProblemType.CLASSIFICATION,
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION
        ]
        
        algorithm_map = {
            AlgorithmType.RANDOM_FOREST: (
                RandomForestClassifier if is_classification else RandomForestRegressor
            ),
            AlgorithmType.XGBOOST: (
                XGBClassifier if is_classification and XGBOOST_AVAILABLE else
                XGBRegressor if not is_classification and XGBOOST_AVAILABLE else None
            ),
            AlgorithmType.LIGHTGBM: (
                LGBMClassifier if is_classification and LIGHTGBM_AVAILABLE else
                LGBMRegressor if not is_classification and LIGHTGBM_AVAILABLE else None
            ),
            AlgorithmType.LINEAR_REGRESSION: (
                LogisticRegression if is_classification else LinearRegression
            ),
            AlgorithmType.SVM: (
                SVC if is_classification else SVR
            ),
            AlgorithmType.KNN: (
                KNeighborsClassifier if is_classification else KNeighborsRegressor
            ),
            AlgorithmType.GRADIENT_BOOSTING: (
                GradientBoostingClassifier if is_classification else GradientBoostingRegressor
            ),
            AlgorithmType.EXTRA_TREES: (
                ExtraTreesClassifier if is_classification else ExtraTreesRegressor
            ),
            AlgorithmType.RIDGE: Ridge,
            AlgorithmType.LASSO: Lasso,
            AlgorithmType.ELASTIC_NET: ElasticNet,
        }
        
        algorithm_class = algorithm_map.get(algorithm)
        if algorithm_class is None:
            raise ValueError(f"Algorithm {algorithm} not available or not supported for {problem_type}")
        
        # Set random state for reproducibility
        init_kwargs = {"random_state": random_state}
        init_kwargs.update(kwargs)
        
        # Handle algorithm-specific parameters
        if algorithm in [AlgorithmType.RIDGE, AlgorithmType.LASSO, AlgorithmType.ELASTIC_NET]:
            init_kwargs.pop("random_state", None)  # These don't have random_state
        
        return algorithm_class(**init_kwargs)


class HyperparameterTuner:
    """Advanced hyperparameter tuning with multiple strategies."""
    
    @staticmethod
    def get_param_grid(algorithm: AlgorithmType, tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get parameter grid for algorithm."""
        
        if not SCIPY_AVAILABLE and tuning_method == TuningMethod.RANDOM_SEARCH:
            logger.warning("scipy not available, falling back to grid search")
            tuning_method = TuningMethod.GRID_SEARCH
        
        param_grids = {
            AlgorithmType.RANDOM_FOREST: HyperparameterTuner._get_rf_params(tuning_method),
            AlgorithmType.XGBOOST: HyperparameterTuner._get_xgb_params(tuning_method),
            AlgorithmType.LIGHTGBM: HyperparameterTuner._get_lgb_params(tuning_method),
            AlgorithmType.SVM: HyperparameterTuner._get_svm_params(tuning_method),
            AlgorithmType.KNN: HyperparameterTuner._get_knn_params(tuning_method),
            AlgorithmType.GRADIENT_BOOSTING: HyperparameterTuner._get_gb_params(tuning_method),
            AlgorithmType.EXTRA_TREES: HyperparameterTuner._get_et_params(tuning_method),
            AlgorithmType.RIDGE: HyperparameterTuner._get_ridge_params(tuning_method),
            AlgorithmType.LASSO: HyperparameterTuner._get_lasso_params(tuning_method),
            AlgorithmType.ELASTIC_NET: HyperparameterTuner._get_elastic_params(tuning_method),
        }
        
        return param_grids.get(algorithm, {})
    
    @staticmethod
    def _get_rf_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get Random Forest parameters."""
        if tuning_method == TuningMethod.GRID_SEARCH:
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        else:
            return {
                "n_estimators": randint(50, 500) if SCIPY_AVAILABLE else [50, 100, 200, 300, 500],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": randint(2, 20) if SCIPY_AVAILABLE else [2, 5, 10, 15, 20],
                "min_samples_leaf": randint(1, 10) if SCIPY_AVAILABLE else [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", None]
            }
    
    @staticmethod
    def _get_xgb_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get XGBoost parameters."""
        if tuning_method == TuningMethod.GRID_SEARCH:
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            }
        else:
            return {
                "n_estimators": randint(50, 500) if SCIPY_AVAILABLE else [50, 100, 200, 300, 500],
                "max_depth": randint(3, 10) if SCIPY_AVAILABLE else [3, 4, 5, 6, 7, 8, 9, 10],
                "learning_rate": uniform(0.01, 0.3) if SCIPY_AVAILABLE else [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                "subsample": uniform(0.6, 0.4) if SCIPY_AVAILABLE else [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": uniform(0.6, 0.4) if SCIPY_AVAILABLE else [0.6, 0.7, 0.8, 0.9, 1.0]
            }
    
    @staticmethod
    def _get_lgb_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get LightGBM parameters."""
        if tuning_method == TuningMethod.GRID_SEARCH:
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "num_leaves": [31, 62, 127]
            }
        else:
            return {
                "n_estimators": randint(50, 500) if SCIPY_AVAILABLE else [50, 100, 200, 300, 500],
                "max_depth": randint(3, 10) if SCIPY_AVAILABLE else [3, 4, 5, 6, 7, 8, 9, 10],
                "learning_rate": uniform(0.01, 0.3) if SCIPY_AVAILABLE else [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                "num_leaves": randint(20, 200) if SCIPY_AVAILABLE else [20, 31, 50, 100, 150, 200],
                "feature_fraction": uniform(0.6, 0.4) if SCIPY_AVAILABLE else [0.6, 0.7, 0.8, 0.9, 1.0]
            }
    
    @staticmethod
    def _get_svm_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get SVM parameters."""
        if tuning_method == TuningMethod.GRID_SEARCH:
            return {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01]
            }
        else:
            return {
                "C": uniform(0.1, 100) if SCIPY_AVAILABLE else [0.1, 1, 10, 50, 100],
                "gamma": ["scale", "auto"] + ([0.001, 0.01, 0.1] if not SCIPY_AVAILABLE else 
                        list(uniform(0.001, 0.1).rvs(10)) if uniform else [0.001, 0.01, 0.1])
            }
    
    @staticmethod
    def _get_knn_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get KNN parameters."""
        return {
            "n_neighbors": randint(3, 20) if SCIPY_AVAILABLE and tuning_method == TuningMethod.RANDOM_SEARCH 
                          else [3, 5, 7, 9, 11, 15, 20],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }
    
    @staticmethod
    def _get_gb_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get Gradient Boosting parameters."""
        if tuning_method == TuningMethod.GRID_SEARCH:
            return {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        else:
            return {
                "n_estimators": randint(50, 300) if SCIPY_AVAILABLE else [50, 100, 150, 200, 300],
                "max_depth": randint(3, 8) if SCIPY_AVAILABLE else [3, 4, 5, 6, 7, 8],
                "learning_rate": uniform(0.01, 0.3) if SCIPY_AVAILABLE else [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
            }
    
    @staticmethod
    def _get_et_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get Extra Trees parameters."""
        return HyperparameterTuner._get_rf_params(tuning_method)  # Similar to Random Forest
    
    @staticmethod
    def _get_ridge_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get Ridge parameters."""
        return {
            "alpha": uniform(0.1, 10) if SCIPY_AVAILABLE and tuning_method == TuningMethod.RANDOM_SEARCH
                    else [0.1, 1.0, 10.0, 100.0]
        }
    
    @staticmethod
    def _get_lasso_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get Lasso parameters."""
        return HyperparameterTuner._get_ridge_params(tuning_method)
    
    @staticmethod
    def _get_elastic_params(tuning_method: TuningMethod) -> Dict[str, Any]:
        """Get Elastic Net parameters."""
        base_params = HyperparameterTuner._get_ridge_params(tuning_method)
        base_params["l1_ratio"] = uniform(0.1, 0.8) if SCIPY_AVAILABLE and tuning_method == TuningMethod.RANDOM_SEARCH else [0.1, 0.3, 0.5, 0.7, 0.9]
        return base_params


class ModelRegistry:
    """Registry for managing trained models."""
    
    def __init__(self, registry_path: Union[str, Path] = "models"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_model(
        self,
        model: Any,
        result: ModelResult,
        overwrite: bool = False
    ) -> str:
        """Register a trained model."""
        model_name = result.model_name
        model_path = self.registry_path / f"{model_name}.joblib"
        metadata_path = self.registry_path / f"{model_name}_metadata.json"
        
        if model_path.exists() and not overwrite:
            raise ValueError(f"Model {model_name} already exists. Use overwrite=True to replace.")
        
        # Save model
        joblib.dump(model, model_path)
        
        # Prepare metadata
        metadata = result.model_dump()
        metadata["registered_at"] = datetime.now().isoformat()
        metadata["model_path"] = str(model_path)
        metadata["file_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)
        
        # Save metadata
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
                    "status": metadata.get("status", "unknown"),
                    "registered_at": metadata.get("registered_at", "unknown"),
                    "test_score": metadata.get("test_score", "unknown"),
                    "size_mb": metadata.get("file_size_mb", 0)
                })
        
        return sorted(models_info, key=lambda x: x["registered_at"], reverse=True)
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from registry."""
        model_path = self.registry_path / f"{model_name}.joblib"
        metadata_path = self.registry_path / f"{model_name}_metadata.json"
        
        deleted = False
        
        if model_path.exists():
            model_path.unlink()
            deleted = True
        
        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True
        
        # Remove from in-memory registry
        self.models.pop(model_name, None)
        self.metadata.pop(model_name, None)
        
        if deleted:
            logger.info(f"Model {model_name} deleted successfully")
        
        return deleted


class ModelTrainer:
    """Advanced model trainer with progress tracking and async support."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.progress_callbacks: List[callable] = []
        self.current_progress = TrainingProgress()
        
    def add_progress_callback(self, callback: callable) -> None:
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self) -> None:
        """Notify all progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(self.current_progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def train_models_async(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        config: ModelConfig
    ) -> List[ModelResult]:
        """Train models asynchronously with progress tracking."""
        start_time = datetime.now()
        
        # Initialize progress
        self.current_progress = TrainingProgress(
            total_algorithms=len(config.algorithms),
            total_trials=config.n_trials,
            status=ModelStatus.TRAINING
        )
        self._notify_progress()
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=config.random_state
            )
            
            # Setup MLflow if available
            if MLFLOW_AVAILABLE and mlflow:
                mlflow.set_experiment(config.experiment_name)
            
            # Train models using thread pool for CPU-bound tasks
            with ThreadPoolExecutor(max_workers=4) as executor:
                tasks = []
                for algorithm in config.algorithms:
                    task = asyncio.get_event_loop().run_in_executor(
                        executor,
                        self._train_single_model,
                        algorithm,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        config
                    )
                    tasks.append(task)
                
                results = []
                for i, task in enumerate(asyncio.as_completed(tasks)):
                    try:
                        result = await task
                        results.append(result)
                        
                        # Update progress
                        self.current_progress.algorithms_completed = i + 1
                        self.current_progress.elapsed_time_seconds = (
                            datetime.now() - start_time
                        ).total_seconds()
                        self._notify_progress()
                        
                    except Exception as e:
                        logger.error(f"Training failed for algorithm: {e}")
                        error_result = ModelResult(
                            model_name=f"failed_{i}",
                            algorithm=config.algorithms[i] if i < len(config.algorithms) else AlgorithmType.RANDOM_FOREST,
                            status=ModelStatus.FAILED,
                            training_time_seconds=0.0,
                            prediction_time_ms=0.0,
                            model_size_mb=0.0,
                            error_message=str(e)
                        )
                        results.append(error_result)
            
            # Update final progress
            self.current_progress.status = ModelStatus.COMPLETED
            self.current_progress.elapsed_time_seconds = (
                datetime.now() - start_time
            ).total_seconds()
            self._notify_progress()
            
            # Sort by test score (descending)
            successful_results = [r for r in results if r.status == ModelStatus.COMPLETED]
            successful_results.sort(key=lambda x: x.test_score, reverse=True)
            
            return results
            
        except Exception as e:
            self.current_progress.status = ModelStatus.FAILED
            self._notify_progress()
            raise e
    
    def _train_single_model(
        self,
        algorithm: AlgorithmType,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        y_test: Union[pd.Series, pd.DataFrame],
        config: ModelConfig
    ) -> ModelResult:
        """Train a single model."""
        model_name = f"{algorithm.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Update current algorithm in progress
            self.current_progress.current_algorithm = algorithm
            
            # Create algorithm instance
            base_model = AlgorithmFactory.create_algorithm(
                algorithm, config.problem_type, config.random_state
            )
            
            # Get parameter grid
            param_grid = HyperparameterTuner.get_param_grid(algorithm, config.tuning_method)
            
            # Setup cross-validation
            cv = (
                StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
                if config.problem_type in [ProblemType.CLASSIFICATION, ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]
                else KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            )
            
            # Hyperparameter search
            start_time = datetime.now()
            
            if config.tuning_method == TuningMethod.GRID_SEARCH:
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv,
                    scoring=config.scoring,
                    n_jobs=-1,
                    verbose=0
                )
            else:
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=config.n_trials,
                    cv=cv,
                    scoring=config.scoring,
                    n_jobs=-1,
                    random_state=config.random_state,
                    verbose=0
                )
            
            # Fit the model
            search.fit(X_train, y_train)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Get best model
            best_model = search.best_estimator_
            
            # Evaluate on test set
            start_pred = datetime.now()
            y_pred = best_model.predict(X_test)
            prediction_time_ms = (datetime.now() - start_pred).total_seconds() * 1000
            
            # Calculate metrics
            test_score = self._calculate_test_score(y_test, y_pred, config.problem_type)
            train_score = self._calculate_test_score(y_train, best_model.predict(X_train), config.problem_type)
            
            # Get feature importance if available
            feature_importance = self._get_feature_importance(best_model, X_train.columns.tolist())
            
            # Get classification metrics if applicable
            classification_metrics = None
            if config.problem_type in [ProblemType.CLASSIFICATION, ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
                classification_metrics = self._get_classification_metrics(y_test, y_pred)
            
            # Calculate model size
            model_size_mb = self._estimate_model_size(best_model)
            
            # Create result
            result = ModelResult(
                model_name=model_name,
                algorithm=algorithm,
                status=ModelStatus.COMPLETED,
                best_params=search.best_params_,
                cv_scores=search.cv_results_['mean_test_score'].tolist(),
                cv_mean=search.best_score_,
                cv_std=search.cv_results_['std_test_score'][search.best_index_],
                test_score=test_score,
                train_score=train_score,
                training_time_seconds=training_time,
                prediction_time_ms=prediction_time_ms,
                model_size_mb=model_size_mb,
                feature_importance=feature_importance,
                classification_metrics=classification_metrics
            )
            
            # Register model if successful
            if result.status == ModelStatus.COMPLETED:
                self.registry.register_model(best_model, result)
            
            # Track with MLflow if available
            if MLFLOW_AVAILABLE and mlflow and config.track_metrics:
                self._log_to_mlflow(result, config)
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed for {algorithm}: {e}")
            return ModelResult(
                model_name=model_name,
                algorithm=algorithm,
                status=ModelStatus.FAILED,
                training_time_seconds=0.0,
                prediction_time_ms=0.0,
                model_size_mb=0.0,
                error_message=str(e)
            )
    
    def _calculate_test_score(self, y_true, y_pred, problem_type: ProblemType) -> float:
        """Calculate appropriate test score based on problem type."""
        if not SKLEARN_AVAILABLE:
            return 0.0
        
        try:
            if problem_type == ProblemType.REGRESSION:
                return r2_score(y_true, y_pred)
            else:
                return accuracy_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Failed to calculate test score: {e}")
            return 0.0
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances.tolist()))
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # For binary classification
                return dict(zip(feature_names, abs(coef).tolist()))
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
        
        return None
    
    def _get_classification_metrics(self, y_true, y_pred) -> Optional[Dict[str, Any]]:
        """Get classification metrics."""
        if not SKLEARN_AVAILABLE:
            return None
        
        try:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Add ROC AUC for binary classification
            if len(set(y_true)) == 2:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
                except Exception:
                    pass
            
            # Add confusion matrix
            try:
                cm = confusion_matrix(y_true, y_pred)
                metrics["confusion_matrix"] = cm.tolist()
            except Exception:
                pass
            
            # Add classification report
            try:
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics["classification_report"] = report
            except Exception:
                pass
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate classification metrics: {e}")
            return None
    
    def _estimate_model_size(self, model) -> float:
        """Estimate model size in MB."""
        try:
            import pickle
            model_bytes = pickle.dumps(model)
            return len(model_bytes) / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _log_to_mlflow(self, result: ModelResult, config: ModelConfig) -> None:
        """Log results to MLflow."""
        try:
            with mlflow.start_run(run_name=result.model_name):
                # Log parameters
                if config.track_params:
                    mlflow.log_params(result.best_params)
                    mlflow.log_param("algorithm", result.algorithm.value)
                    mlflow.log_param("problem_type", config.problem_type.value)
                
                # Log metrics
                if config.track_metrics:
                    mlflow.log_metric("test_score", result.test_score)
                    mlflow.log_metric("train_score", result.train_score)
                    mlflow.log_metric("cv_mean", result.cv_mean)
                    mlflow.log_metric("cv_std", result.cv_std)
                    mlflow.log_metric("training_time", result.training_time_seconds)
                    mlflow.log_metric("prediction_time_ms", result.prediction_time_ms)
                    mlflow.log_metric("model_size_mb", result.model_size_mb)
                
                # Log model artifact if enabled
                if config.track_artifacts:
                    # Model should be available in registry
                    model_path = self.registry.registry_path / f"{result.model_name}.joblib"
                    if model_path.exists():
                        mlflow.log_artifact(str(model_path))
                        
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")


# Convenience function for quick training
async def train_models(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    config: Optional[ModelConfig] = None,
    registry_path: Optional[Union[str, Path]] = None
) -> List[ModelResult]:
    """
    Convenience function for training models.
    
    Args:
        X: Feature matrix
        y: Target variable
        config: Training configuration
        registry_path: Path for model registry
    
    Returns:
        List of training results
    """
    if config is None:
        config = ModelConfig()
    
    registry = ModelRegistry(registry_path) if registry_path else ModelRegistry()
    trainer = ModelTrainer(registry)
    
    return await trainer.train_models_async(X, y, config)

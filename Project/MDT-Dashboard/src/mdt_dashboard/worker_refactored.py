"""
Enhanced Celery worker configuration and async task definitions for MDT Dashboard.
Handles background processing for model training, drift detection, and performance monitoring.

Refactored for improved code quality, type safety, and maintainability.
"""

import asyncio
import json
import logging
import traceback
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from celery import Celery
    from celery.schedules import crontab
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None
    crontab = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    
    def ConfigDict(**kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    def Field(default=None, **kwargs):
        return default
    
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 15


class TaskType(Enum):
    """Types of background tasks."""
    MODEL_TRAINING = "model_training"
    DRIFT_DETECTION = "drift_detection"
    PERFORMANCE_METRICS = "performance_metrics"
    DATA_QUALITY = "data_quality"
    ALERT_NOTIFICATION = "alert_notification"
    CLEANUP = "cleanup"
    HEALTH_CHECK = "health_check"


# Pydantic models for task validation
if PYDANTIC_AVAILABLE:
    class TaskConfig(BaseModel):
        """Base task configuration."""
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
        
        task_id: Optional[str] = None
        max_retries: int = Field(default=3, ge=0, le=10)
        retry_delay: int = Field(default=60, ge=10, le=3600)
        timeout: int = Field(default=3600, ge=60, le=7200)
        priority: TaskPriority = TaskPriority.NORMAL
        
    class TrainingConfig(TaskConfig):
        """Model training task configuration."""
        data_path: str
        target_column: str
        model_name: str
        model_type: str = "auto"
        hyperparameters: Dict[str, Any] = Field(default_factory=dict)
        validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
        cross_validation: bool = True
        save_artifacts: bool = True
        
        @field_validator('data_path')
        @classmethod
        def validate_data_path(cls, v):
            if not v or not isinstance(v, str):
                raise ValueError("data_path must be a non-empty string")
            return v
    
    class DriftDetectionConfig(TaskConfig):
        """Drift detection task configuration."""
        model_id: str
        time_window_hours: int = Field(default=24, ge=1, le=168)
        detection_methods: List[str] = Field(default=["ks", "psi", "mmd"])
        threshold: float = Field(default=0.05, ge=0.01, le=0.5)
        minimum_samples: int = Field(default=100, ge=10)
        
    class PerformanceMetricsConfig(TaskConfig):
        """Performance metrics calculation configuration."""
        model_id: Optional[str] = None
        time_period_hours: int = Field(default=24, ge=1, le=168)
        include_confidence_metrics: bool = True
        include_prediction_distribution: bool = True
        
    class DataQualityConfig(TaskConfig):
        """Data quality check configuration."""
        model_id: Optional[str] = None
        time_window_hours: int = Field(default=24, ge=1, le=168)
        quality_checks: List[str] = Field(default=["completeness", "accuracy", "consistency", "validity"])
        min_samples_required: int = Field(default=10, ge=1)
        
    class CleanupConfig(TaskConfig):
        """Data cleanup configuration."""
        days_to_keep: int = Field(default=30, ge=1, le=365)
        tables_to_clean: List[str] = Field(default=["predictions", "alerts", "metrics"])
        dry_run: bool = False

else:
    # Fallback dataclasses when Pydantic is not available
    from dataclasses import dataclass, field
    
    @dataclass
    class TaskConfig:
        task_id: Optional[str] = None
        max_retries: int = 3
        retry_delay: int = 60
        timeout: int = 3600
        priority: TaskPriority = TaskPriority.NORMAL
    
    @dataclass
    class TrainingConfig(TaskConfig):
        data_path: str = ""
        target_column: str = ""
        model_name: str = ""
        model_type: str = "auto"
        hyperparameters: Dict[str, Any] = field(default_factory=dict)
        validation_split: float = 0.2
        cross_validation: bool = True
        save_artifacts: bool = True
    
    @dataclass
    class DriftDetectionConfig(TaskConfig):
        model_id: str = ""
        time_window_hours: int = 24
        detection_methods: List[str] = field(default_factory=lambda: ["ks", "psi", "mmd"])
        threshold: float = 0.05
        minimum_samples: int = 100
    
    @dataclass
    class PerformanceMetricsConfig(TaskConfig):
        model_id: Optional[str] = None
        time_period_hours: int = 24
        include_confidence_metrics: bool = True
        include_prediction_distribution: bool = True
    
    @dataclass
    class DataQualityConfig(TaskConfig):
        model_id: Optional[str] = None
        time_window_hours: int = 24
        quality_checks: List[str] = field(default_factory=lambda: ["completeness", "accuracy", "consistency", "validity"])
        min_samples_required: int = 10
    
    @dataclass
    class CleanupConfig(TaskConfig):
        days_to_keep: int = 30
        tables_to_clean: List[str] = field(default_factory=lambda: ["predictions", "alerts", "metrics"])
        dry_run: bool = False


class TaskResult:
    """Task execution result wrapper."""
    
    def __init__(
        self,
        status: TaskStatus,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.status = status
        self.data = data or {}
        self.error = error
        self.execution_time = execution_time
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def is_success(self) -> bool:
        """Check if task was successful."""
        return self.status == TaskStatus.SUCCESS
    
    def is_failure(self) -> bool:
        """Check if task failed."""
        return self.status == TaskStatus.FAILURE


class BaseTask(ABC):
    """Base class for background tasks."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    @abstractmethod
    async def execute(self) -> TaskResult:
        """Execute the task."""
        pass
    
    def validate_config(self) -> bool:
        """Validate task configuration."""
        if PYDANTIC_AVAILABLE and hasattr(self.config, 'model_validate'):
            try:
                self.config.model_validate(self.config.model_dump())
                return True
            except Exception as e:
                logger.error(f"Task configuration validation failed: {e}")
                return False
        return True
    
    async def run_with_monitoring(self) -> TaskResult:
        """Execute task with monitoring and error handling."""
        if not self.validate_config():
            return TaskResult(
                status=TaskStatus.FAILURE,
                error="Invalid task configuration"
            )
        
        self.start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting task {self.__class__.__name__} with config: {self.config}")
            
            result = await self.execute()
            
            self.end_time = datetime.utcnow()
            execution_time = (self.end_time - self.start_time).total_seconds()
            result.execution_time = execution_time
            
            logger.info(f"Task {self.__class__.__name__} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.end_time = datetime.utcnow()
            execution_time = (self.end_time - self.start_time).total_seconds()
            
            logger.error(f"Task {self.__class__.__name__} failed after {execution_time:.2f}s: {e}")
            logger.error(traceback.format_exc())
            
            return TaskResult(
                status=TaskStatus.FAILURE,
                error=str(e),
                execution_time=execution_time
            )


class ModelTrainingTask(BaseTask):
    """Model training background task."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.config: TrainingConfig = config
    
    async def execute(self) -> TaskResult:
        """Execute model training."""
        try:
            # Import here to avoid circular imports
            from .train_refactored import ModelTrainer
            
            trainer = ModelTrainer()
            
            # Convert to async if the trainer supports it
            if hasattr(trainer, 'train_model_async'):
                result = await trainer.train_model_async(
                    data_path=self.config.data_path,
                    target_column=self.config.target_column,
                    model_name=self.config.model_name,
                    model_type=self.config.model_type,
                    hyperparameters=self.config.hyperparameters,
                    validation_split=self.config.validation_split
                )
            else:
                # Run in thread pool for sync trainer
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        trainer.train_model,
                        data_path=self.config.data_path,
                        target_column=self.config.target_column,
                        model_name=self.config.model_name,
                        model_type=self.config.model_type,
                        hyperparameters=self.config.hyperparameters,
                        validation_split=self.config.validation_split
                    )
                    result = future.result()
            
            return TaskResult(
                status=TaskStatus.SUCCESS,
                data={
                    "model_name": self.config.model_name,
                    "performance": result.get("performance", {}),
                    "artifacts": result.get("artifacts", []),
                    "training_metrics": result.get("metrics", {})
                }
            )
            
        except Exception as e:
            return TaskResult(
                status=TaskStatus.FAILURE,
                error=f"Model training failed: {str(e)}"
            )


class DriftDetectionTask(BaseTask):
    """Drift detection background task."""
    
    def __init__(self, config: DriftDetectionConfig):
        super().__init__(config)
        self.config: DriftDetectionConfig = config
    
    async def execute(self) -> TaskResult:
        """Execute drift detection."""
        try:
            # Import here to avoid circular imports
            from .core.database import db_manager
            from .core.models import Model, ReferenceData, Prediction, DriftReport
            from .drift_detection.algorithms import MultivariateDriftDetector
            
            with db_manager.session_scope() as session:
                # Get model
                model = session.query(Model).filter(Model.id == self.config.model_id).first()
                if not model:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        error=f"Model {self.config.model_id} not found"
                    )
                
                # Get reference data
                reference_data = session.query(ReferenceData).filter(
                    ReferenceData.model_id == self.config.model_id,
                    ReferenceData.is_active
                ).first()
                
                if not reference_data:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        error=f"No reference data found for model {self.config.model_id}"
                    )
                
                # Get recent predictions
                cutoff_time = datetime.utcnow() - timedelta(hours=self.config.time_window_hours)
                predictions = session.query(Prediction).filter(
                    Prediction.model_id == self.config.model_id,
                    Prediction.prediction_time >= cutoff_time
                ).all()
                
                if len(predictions) < self.config.minimum_samples:
                    return TaskResult(
                        status=TaskStatus.SUCCESS,
                        data={"status": "skipped", "reason": "insufficient_samples"}
                    )
                
                # Prepare data for drift detection
                if PANDAS_AVAILABLE:
                    current_data = pd.DataFrame([pred.input_data for pred in predictions])
                    
                    if hasattr(pd, 'read_parquet'):
                        reference_df = pd.read_parquet(reference_data.data_path)
                    else:
                        # Fallback to JSON loading
                        with open(reference_data.data_path, 'r') as f:
                            reference_df = pd.DataFrame(json.load(f))
                else:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        error="pandas not available for data processing"
                    )
                
                # Run drift detection
                detector = MultivariateDriftDetector(
                    threshold=self.config.threshold,
                    methods=self.config.detection_methods
                )
                
                drift_result = detector.detect_drift(reference_df, current_data)
                
                # Create drift report
                from .core.models import DriftType
                drift_report = DriftReport(
                    model_id=self.config.model_id,
                    reference_data_id=reference_data.id,
                    report_name=f"Automated_Drift_Check_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    drift_type=DriftType.DATA_DRIFT.value,
                    detection_method="multivariate",
                    start_time=cutoff_time,
                    end_time=datetime.utcnow(),
                    overall_drift_score=drift_result.drift_score,
                    drift_detected=drift_result.drift_detected,
                    p_value=drift_result.p_value,
                    threshold=self.config.threshold,
                    feature_drift_scores=drift_result.feature_drift_scores,
                    drift_statistics=drift_result.statistics,
                    summary=drift_result.summary,
                    num_samples_analyzed=len(predictions)
                )
                
                session.add(drift_report)
                session.commit()
                
                # Create alert if drift detected
                if drift_result.drift_detected:
                    await self._create_drift_alert(session, drift_report)
                
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    data={
                        "drift_detected": drift_result.drift_detected,
                        "drift_score": drift_result.drift_score,
                        "p_value": drift_result.p_value,
                        "report_id": drift_report.id,
                        "samples_analyzed": len(predictions)
                    }
                )
                
        except Exception as e:
            return TaskResult(
                status=TaskStatus.FAILURE,
                error=f"Drift detection failed: {str(e)}"
            )
    
    async def _create_drift_alert(self, session, drift_report):
        """Create alert for detected drift."""
        try:
            from .core.models import Alert, AlertSeverity
            
            severity = AlertSeverity.HIGH if drift_report.overall_drift_score > 0.1 else AlertSeverity.MEDIUM
            
            alert = Alert(
                drift_report_id=drift_report.id,
                model_id=drift_report.model_id,
                alert_type="drift_detection",
                severity=severity.value,
                message=f"Drift detected in model {drift_report.model_id}. Score: {drift_report.overall_drift_score:.4f}",
                metadata={"drift_score": drift_report.overall_drift_score},
                created_at=datetime.utcnow()
            )
            
            session.add(alert)
            
        except Exception as e:
            logger.error(f"Failed to create drift alert: {e}")


class PerformanceMetricsTask(BaseTask):
    """Performance metrics calculation task."""
    
    def __init__(self, config: PerformanceMetricsConfig):
        super().__init__(config)
        self.config: PerformanceMetricsConfig = config
    
    async def execute(self) -> TaskResult:
        """Execute performance metrics calculation."""
        try:
            from .core.database import db_manager
            from .core.models import Model, Prediction, ModelPerformanceMetric
            
            with db_manager.session_scope() as session:
                if self.config.model_id:
                    models = [session.query(Model).filter(Model.id == self.config.model_id).first()]
                    models = [m for m in models if m is not None]
                else:
                    models = session.query(Model).filter(Model.status == "active").all()
                
                results = []
                
                for model in models:
                    result = await self._calculate_model_metrics(session, model)
                    results.append(result)
                
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    data={
                        "models_processed": len(results),
                        "results": results
                    }
                )
                
        except Exception as e:
            return TaskResult(
                status=TaskStatus.FAILURE,
                error=f"Performance metrics calculation failed: {str(e)}"
            )
    
    async def _calculate_model_metrics(self, session, model) -> Dict[str, Any]:
        """Calculate metrics for a single model."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.config.time_period_hours)
            
            # Get predictions with actual values
            from .core.models import Prediction
            predictions = session.query(Prediction).filter(
                Prediction.model_id == model.id,
                Prediction.prediction_time >= cutoff_time,
                Prediction.actual_value.isnot(None)
            ).all()
            
            if len(predictions) < 10:
                return {
                    "model_id": model.id,
                    "status": "skipped",
                    "reason": "insufficient_ground_truth"
                }
            
            # Calculate metrics
            metrics = self._compute_performance_metrics(predictions, model.algorithm)
            
            # Create or update performance metric record
            from .core.models import ModelPerformanceMetric
            
            existing_metric = session.query(ModelPerformanceMetric).filter(
                ModelPerformanceMetric.model_id == model.id,
                ModelPerformanceMetric.metric_date == datetime.utcnow().date()
            ).first()
            
            if existing_metric:
                # Update existing record
                for key, value in metrics.items():
                    if hasattr(existing_metric, key):
                        setattr(existing_metric, key, value)
                existing_metric.num_predictions = len(predictions)
            else:
                # Create new record
                performance_metric = ModelPerformanceMetric(
                    model_id=model.id,
                    metric_date=datetime.utcnow().date(),
                    period_start=cutoff_time,
                    period_end=datetime.utcnow(),
                    num_predictions=len(predictions),
                    num_actual_values=len(predictions),
                    **metrics
                )
                session.add(performance_metric)
            
            session.commit()
            
            return {
                "model_id": model.id,
                "status": "completed",
                "metrics": metrics,
                "num_predictions": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics for model {model.id}: {e}")
            return {
                "model_id": model.id,
                "status": "failed",
                "error": str(e)
            }
    
    def _compute_performance_metrics(self, predictions: List, algorithm: str) -> Dict[str, float]:
        """Compute performance metrics based on algorithm type."""
        try:
            if not NUMPY_AVAILABLE:
                return {}
            
            y_true = [pred.actual_value for pred in predictions]
            y_pred = [pred.prediction for pred in predictions]
            
            metrics = {}
            
            # Determine if classification or regression
            is_classification = algorithm in ["classification", "binary_classification"] or \
                              all(isinstance(val, (int, bool)) or val in [0, 1] for val in y_true)
            
            if is_classification:
                # Classification metrics
                try:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    metrics.update({
                        "accuracy": float(accuracy_score(y_true, y_pred)),
                        "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                        "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                        "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                    })
                    
                    # ROC AUC for binary classification
                    if algorithm == "binary_classification" and len(set(y_true)) == 2:
                        try:
                            from sklearn.metrics import roc_auc_score
                            y_prob = [pred.prediction_probability.get(1, 0.5) if pred.prediction_probability else 0.5 
                                     for pred in predictions]
                            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
                        except Exception:
                            pass
                            
                except ImportError:
                    # Fallback to basic accuracy calculation
                    if len(y_true) > 0:
                        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
                        metrics["accuracy"] = float(accuracy)
            else:
                # Regression metrics
                try:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    
                    metrics.update({
                        "mae": float(mean_absolute_error(y_true, y_pred)),
                        "mse": float(mean_squared_error(y_true, y_pred)),
                        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                        "r2_score": float(r2_score(y_true, y_pred))
                    })
                except ImportError:
                    # Fallback to basic MAE calculation
                    if len(y_true) > 0 and NUMPY_AVAILABLE:
                        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
                        metrics["mae"] = float(mae)
            
            # System metrics
            response_times = [pred.response_time_ms for pred in predictions if pred.response_time_ms]
            if response_times and NUMPY_AVAILABLE:
                metrics["avg_response_time_ms"] = float(np.mean(response_times))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compute performance metrics: {e}")
            return {}


class DataQualityTask(BaseTask):
    """Data quality assessment task."""
    
    def __init__(self, config: DataQualityConfig):
        super().__init__(config)
        self.config: DataQualityConfig = config
    
    async def execute(self) -> TaskResult:
        """Execute data quality assessment."""
        try:
            from .core.database import db_manager
            from .core.models import Model, Prediction, DataQualityReport
            from .data_processing import ComprehensiveDataProcessor
            
            with db_manager.session_scope() as session:
                if self.config.model_id:
                    models = [session.query(Model).filter(Model.id == self.config.model_id).first()]
                    models = [m for m in models if m is not None]
                else:
                    models = session.query(Model).filter(Model.status == "active").all()
                
                results = []
                
                for model in models:
                    result = await self._assess_model_data_quality(session, model)
                    results.append(result)
                
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    data={
                        "models_processed": len(results),
                        "results": results
                    }
                )
                
        except Exception as e:
            return TaskResult(
                status=TaskStatus.FAILURE,
                error=f"Data quality assessment failed: {str(e)}"
            )
    
    async def _assess_model_data_quality(self, session, model) -> Dict[str, Any]:
        """Assess data quality for a single model."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.config.time_window_hours)
            
            from .core.models import Prediction
            predictions = session.query(Prediction).filter(
                Prediction.model_id == model.id,
                Prediction.prediction_time >= cutoff_time
            ).all()
            
            if len(predictions) < self.config.min_samples_required:
                return {
                    "model_id": model.id,
                    "status": "skipped",
                    "reason": "insufficient_samples"
                }
            
            # Analyze data quality
            if PANDAS_AVAILABLE:
                input_data = pd.DataFrame([pred.input_data for pred in predictions])
            else:
                return {
                    "model_id": model.id,
                    "status": "failed",
                    "error": "pandas not available"
                }
            
            processor = ComprehensiveDataProcessor()
            quality_report = processor.analyze_data_quality(input_data)
            
            # Create data quality report
            dq_report = DataQualityReport(
                model_id=model.id,
                report_name=f"Daily_DQ_Check_{datetime.utcnow().strftime('%Y%m%d')}",
                start_time=cutoff_time,
                end_time=datetime.utcnow(),
                overall_quality_score=quality_report.get("overall_score", 0.0),
                completeness_score=quality_report.get("completeness_score", 0.0),
                accuracy_score=quality_report.get("accuracy_score", 0.0),
                consistency_score=quality_report.get("consistency_score", 0.0),
                validity_score=quality_report.get("validity_score", 0.0),
                feature_quality_scores=quality_report.get("feature_scores", {}),
                missing_value_stats=quality_report.get("missing_stats", {}),
                outlier_stats=quality_report.get("outlier_stats", {}),
                schema_violations=quality_report.get("schema_violations", []),
                num_samples_analyzed=len(predictions)
            )
            
            session.add(dq_report)
            session.commit()
            
            return {
                "model_id": model.id,
                "status": "completed",
                "quality_score": quality_report.get("overall_score", 0.0),
                "samples_analyzed": len(predictions),
                "report_id": dq_report.id
            }
            
        except Exception as e:
            logger.error(f"Failed to assess data quality for model {model.id}: {e}")
            return {
                "model_id": model.id,
                "status": "failed",
                "error": str(e)
            }


class CleanupTask(BaseTask):
    """Data cleanup task."""
    
    def __init__(self, config: CleanupConfig):
        super().__init__(config)
        self.config: CleanupConfig = config
    
    async def execute(self) -> TaskResult:
        """Execute data cleanup."""
        try:
            from .core.database import db_manager
            from .core.models import Prediction, Alert, ModelPerformanceMetric
            
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.days_to_keep)
            cleanup_results = {}
            
            with db_manager.session_scope() as session:
                for table_name in self.config.tables_to_clean:
                    if table_name == "predictions":
                        count = await self._cleanup_predictions(session, cutoff_date)
                        cleanup_results["predictions_cleaned"] = count
                    elif table_name == "alerts":
                        count = await self._cleanup_alerts(session, cutoff_date)
                        cleanup_results["alerts_cleaned"] = count
                    elif table_name == "metrics":
                        count = await self._cleanup_metrics(session, cutoff_date)
                        cleanup_results["metrics_cleaned"] = count
                
                if not self.config.dry_run:
                    session.commit()
                else:
                    session.rollback()
                    cleanup_results["note"] = "dry_run_mode_no_changes_made"
            
            return TaskResult(
                status=TaskStatus.SUCCESS,
                data={
                    "cutoff_date": cutoff_date.isoformat(),
                    "dry_run": self.config.dry_run,
                    **cleanup_results
                }
            )
            
        except Exception as e:
            return TaskResult(
                status=TaskStatus.FAILURE,
                error=f"Data cleanup failed: {str(e)}"
            )
    
    async def _cleanup_predictions(self, session, cutoff_date: datetime) -> int:
        """Cleanup old predictions."""
        from .core.models import Prediction
        
        if self.config.dry_run:
            count = session.query(Prediction).filter(
                Prediction.prediction_time < cutoff_date
            ).count()
        else:
            count = session.query(Prediction).filter(
                Prediction.prediction_time < cutoff_date
            ).delete()
        
        return count
    
    async def _cleanup_alerts(self, session, cutoff_date: datetime) -> int:
        """Cleanup old alerts."""
        from .core.models import Alert
        
        if self.config.dry_run:
            count = session.query(Alert).filter(
                Alert.created_at < cutoff_date
            ).count()
        else:
            count = session.query(Alert).filter(
                Alert.created_at < cutoff_date
            ).delete()
        
        return count
    
    async def _cleanup_metrics(self, session, cutoff_date: datetime) -> int:
        """Cleanup old metrics."""
        from .core.models import ModelPerformanceMetric
        
        metric_cutoff = cutoff_date.date()
        
        if self.config.dry_run:
            count = session.query(ModelPerformanceMetric).filter(
                ModelPerformanceMetric.metric_date < metric_cutoff
            ).count()
        else:
            count = session.query(ModelPerformanceMetric).filter(
                ModelPerformanceMetric.metric_date < metric_cutoff
            ).delete()
        
        return count


class CeleryWorkerManager:
    """Enhanced Celery worker manager."""
    
    def __init__(self):
        self.celery_app = None
        self.is_initialized = False
        self._initialize_celery()
    
    def _initialize_celery(self):
        """Initialize Celery application."""
        if not CELERY_AVAILABLE:
            logger.warning("Celery not available, worker functionality disabled")
            return
        
        try:
            from .core.config import get_settings
            settings = get_settings()
            
            self.celery_app = Celery(
                "mdt_dashboard",
                broker=settings.redis.url,
                backend=settings.redis.url,
                include=["mdt_dashboard.worker_refactored"]
            )
            
            # Enhanced Celery configuration
            self.celery_app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
                task_track_started=True,
                task_time_limit=getattr(settings, 'task_time_limit', 3600),
                task_soft_time_limit=getattr(settings, 'task_soft_time_limit', 3300),
                worker_prefetch_multiplier=1,
                task_acks_late=True,
                worker_disable_rate_limits=False,
                task_ignore_result=False,
                result_expires=3600,
                worker_concurrency=getattr(settings, 'worker_concurrency', 4),
                task_routes={
                    'mdt_dashboard.worker_refactored.train_model_async': {'queue': 'training'},
                    'mdt_dashboard.worker_refactored.check_drift_*': {'queue': 'monitoring'},
                    'mdt_dashboard.worker_refactored.calculate_*': {'queue': 'analytics'},
                    'mdt_dashboard.worker_refactored.cleanup_*': {'queue': 'maintenance'},
                },
                task_annotations={
                    'mdt_dashboard.worker_refactored.train_model_async': {
                        'priority': TaskPriority.HIGH.value,
                        'time_limit': 7200
                    },
                    'mdt_dashboard.worker_refactored.check_drift_all_models': {
                        'priority': TaskPriority.NORMAL.value
                    },
                    'mdt_dashboard.worker_refactored.cleanup_old_data': {
                        'priority': TaskPriority.LOW.value
                    }
                }
            )
            
            # Enhanced periodic tasks
            self.celery_app.conf.beat_schedule = {
                'check_drift_all_models': {
                    'task': 'mdt_dashboard.worker_refactored.check_drift_all_models',
                    'schedule': crontab(minute=0),  # Every hour
                    'options': {'priority': TaskPriority.NORMAL.value}
                },
                'calculate_performance_metrics': {
                    'task': 'mdt_dashboard.worker_refactored.calculate_performance_metrics_all_models',
                    'schedule': crontab(minute=30),  # Every hour at 30 minutes
                    'options': {'priority': TaskPriority.NORMAL.value}
                },
                'data_quality_check': {
                    'task': 'mdt_dashboard.worker_refactored.run_data_quality_checks',
                    'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
                    'options': {'priority': TaskPriority.LOW.value}
                },
                'cleanup_old_data': {
                    'task': 'mdt_dashboard.worker_refactored.cleanup_old_data',
                    'schedule': crontab(hour=3, minute=0),  # Daily at 3 AM
                    'options': {'priority': TaskPriority.LOW.value}
                },
                'health_check': {
                    'task': 'mdt_dashboard.worker_refactored.health_check',
                    'schedule': crontab(minute='*/5'),  # Every 5 minutes
                    'options': {'priority': TaskPriority.LOW.value}
                }
            }
            
            self.is_initialized = True
            logger.info("Celery worker manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Celery: {e}")
    
    def get_celery_app(self):
        """Get Celery application instance."""
        return self.celery_app if self.is_initialized else None


# Initialize global worker manager
worker_manager = CeleryWorkerManager()
celery_app = worker_manager.get_celery_app()


# Task wrapper functions for Celery
if CELERY_AVAILABLE and celery_app:
    
    @celery_app.task(bind=True, max_retries=3)
    def train_model_async(self, training_config: Dict[str, Any]):
        """Async model training task wrapper."""
        try:
            config = TrainingConfig(**training_config)
            task = ModelTrainingTask(config)
            
            # Run async task in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(task.run_with_monitoring())
            finally:
                loop.close()
            
            if result.is_success():
                return result.to_dict()
            else:
                if self.request.retries < self.max_retries:
                    raise self.retry(countdown=60 * (2 ** self.request.retries))
                raise Exception(result.error)
                
        except Exception as exc:
            logger.error(f"Model training task failed: {exc}")
            if self.request.retries < self.max_retries:
                raise self.retry(countdown=60 * (2 ** self.request.retries))
            raise exc
    
    @celery_app.task(bind=True, max_retries=2)
    def check_drift_for_model(self, model_id: str, time_window_hours: int = 24):
        """Check drift for a specific model."""
        try:
            config = DriftDetectionConfig(
                model_id=model_id,
                time_window_hours=time_window_hours
            )
            task = DriftDetectionTask(config)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(task.run_with_monitoring())
            finally:
                loop.close()
            
            if result.is_success():
                return result.to_dict()
            else:
                if self.request.retries < self.max_retries:
                    raise self.retry(countdown=60 * (2 ** self.request.retries))
                raise Exception(result.error)
                
        except Exception as exc:
            logger.error(f"Drift detection task failed for model {model_id}: {exc}")
            if self.request.retries < self.max_retries:
                raise self.retry(countdown=60 * (2 ** self.request.retries))
            raise exc
    
    @celery_app.task
    def check_drift_all_models():
        """Check drift for all active models."""
        try:
            from .core.database import db_manager
            from .core.models import Model
            
            with db_manager.session_scope() as session:
                active_models = session.query(Model).filter(Model.status == "active").all()
                
                results = []
                for model in active_models:
                    # Schedule individual drift check tasks
                    task_result = check_drift_for_model.delay(model.id, 24)
                    results.append({
                        "model_id": model.id,
                        "task_id": task_result.id
                    })
                
                logger.info(f"Scheduled drift checks for {len(active_models)} models")
                return {"status": "completed", "models_checked": len(active_models), "tasks": results}
                
        except Exception as exc:
            logger.error(f"Check drift all models failed: {exc}")
            raise exc
    
    @celery_app.task(bind=True, max_retries=2)
    def calculate_model_performance_metrics(self, model_id: Optional[str] = None, time_period_hours: int = 24):
        """Calculate performance metrics for models."""
        try:
            config = PerformanceMetricsConfig(
                model_id=model_id,
                time_period_hours=time_period_hours
            )
            task = PerformanceMetricsTask(config)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(task.run_with_monitoring())
            finally:
                loop.close()
            
            if result.is_success():
                return result.to_dict()
            else:
                if self.request.retries < self.max_retries:
                    raise self.retry(countdown=60 * (2 ** self.request.retries))
                raise Exception(result.error)
                
        except Exception as exc:
            logger.error(f"Performance metrics task failed: {exc}")
            if self.request.retries < self.max_retries:
                raise self.retry(countdown=60 * (2 ** self.request.retries))
            raise exc
    
    @celery_app.task
    def calculate_performance_metrics_all_models():
        """Calculate performance metrics for all active models."""
        try:
            # Schedule task for all models
            task_result = calculate_model_performance_metrics.delay(None, 24)
            return {"status": "scheduled", "task_id": task_result.id}
            
        except Exception as exc:
            logger.error(f"Calculate performance metrics all models failed: {exc}")
            raise exc
    
    @celery_app.task(bind=True, max_retries=2)
    def run_data_quality_checks(self, model_id: Optional[str] = None):
        """Run data quality checks."""
        try:
            config = DataQualityConfig(model_id=model_id)
            task = DataQualityTask(config)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(task.run_with_monitoring())
            finally:
                loop.close()
            
            if result.is_success():
                return result.to_dict()
            else:
                if self.request.retries < self.max_retries:
                    raise self.retry(countdown=60 * (2 ** self.request.retries))
                raise Exception(result.error)
                
        except Exception as exc:
            logger.error(f"Data quality check task failed: {exc}")
            if self.request.retries < self.max_retries:
                raise self.retry(countdown=60 * (2 ** self.request.retries))
            raise exc
    
    @celery_app.task(bind=True, max_retries=1)
    def cleanup_old_data(self, days_to_keep: int = 30, dry_run: bool = False):
        """Clean up old data."""
        try:
            config = CleanupConfig(
                days_to_keep=days_to_keep,
                dry_run=dry_run
            )
            task = CleanupTask(config)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(task.run_with_monitoring())
            finally:
                loop.close()
            
            if result.is_success():
                return result.to_dict()
            else:
                if self.request.retries < self.max_retries:
                    raise self.retry(countdown=300)  # Wait 5 minutes before retry
                raise Exception(result.error)
                
        except Exception as exc:
            logger.error(f"Cleanup task failed: {exc}")
            if self.request.retries < self.max_retries:
                raise self.retry(countdown=300)
            raise exc
    
    @celery_app.task(bind=True, max_retries=3)
    def send_alert_notification(self, alert_id: str):
        """Send alert notification via configured channels."""
        try:
            from .core.database import db_manager
            from .core.models import Alert
            from .monitoring.alerts_refactored import AlertManager
            
            with db_manager.session_scope() as session:
                alert = session.query(Alert).filter(Alert.id == alert_id).first()
                
                if not alert:
                    return {"status": "failed", "error": f"Alert {alert_id} not found"}
                
                alert_manager = AlertManager()
                success = alert_manager.send_notification(alert)
                
                if success:
                    alert.notification_sent = True
                    alert.notification_sent_at = datetime.utcnow()
                    session.commit()
                    
                    return {"status": "completed", "alert_id": alert_id}
                else:
                    if self.request.retries < self.max_retries:
                        raise self.retry(countdown=60 * (2 ** self.request.retries))
                    return {"status": "failed", "error": "Failed to send notification"}
                    
        except Exception as exc:
            logger.error(f"Alert notification task failed: {exc}")
            if self.request.retries < self.max_retries:
                raise self.retry(countdown=60 * (2 ** self.request.retries))
            raise exc
    
    @celery_app.task
    def health_check():
        """Health check task for monitoring worker status."""
        try:
            from .core.database import db_manager
            
            # Test database connection
            with db_manager.session_scope() as session:
                session.execute("SELECT 1")
            
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "worker_id": health_check.request.id,
                "checks": {
                    "database": "ok",
                    "celery": "ok"
                }
            }
            
        except Exception as exc:
            logger.error(f"Health check failed: {exc}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(exc)
            }


def start_worker():
    """Start the Celery worker."""
    if not CELERY_AVAILABLE or not celery_app:
        logger.error("Celery not available, cannot start worker")
        return
    
    try:
        logger.info("Starting Celery worker...")
        celery_app.start()
    except Exception as e:
        logger.error(f"Failed to start Celery worker: {e}")


if __name__ == "__main__":
    start_worker()

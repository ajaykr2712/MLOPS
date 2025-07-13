"""
Celery configuration and task definitions.
"""
from celery import Celery
from celery.schedules import crontab
import logging
from typing import Any, Dict
import pandas as pd
from datetime import datetime, timedelta

from .core.config import get_settings
from .core.database import db_manager
from .core.models import Model, DriftReport, ReferenceData, Alert, ModelPerformanceMetric
from .drift_detection.algorithms import DriftDetector, DriftConfig
from .monitoring.metrics import MetricsCollector
from .utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Celery
settings = get_settings()
celery_app = Celery(
    "mdt_dashboard",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["mdt_dashboard.worker"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_ignore_result=False,
    result_expires=3600,  # 1 hour
)

# Periodic tasks
celery_app.conf.beat_schedule = {
    'check_drift_every_hour': {
        'task': 'mdt_dashboard.worker.check_drift_for_all_models',
        'schedule': crontab(minute=0),  # Every hour
    },
    'calculate_performance_metrics_daily': {
        'task': 'mdt_dashboard.worker.calculate_daily_performance_metrics',
        'schedule': crontab(hour=1, minute=0),  # Daily at 1 AM
    },
    'cleanup_old_data_weekly': {
        'task': 'mdt_dashboard.worker.cleanup_old_data',
        'schedule': crontab(hour=2, minute=0, day_of_week=0),  # Weekly on Sunday at 2 AM
    },
    'health_check_every_5_minutes': {
        'task': 'mdt_dashboard.worker.system_health_check',
        'schedule': crontab(minute='*/5'),  # Every 5 minutes
    },
}


@celery_app.task(bind=True, max_retries=3)
def check_drift_for_model(self, model_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
    """
    Check for drift in a specific model.
    
    Args:
        model_id: Model UUID
        time_window_hours: Time window for drift analysis
        
    Returns:
        Dict with drift analysis results
    """
    try:
        logger.info(f"Starting drift check for model {model_id}")
        
        with db_manager.session_scope() as session:
            # Get model and reference data
            model = session.query(Model).filter(Model.id == model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            reference_data = session.query(ReferenceData).filter(
                ReferenceData.model_id == model_id,
                ReferenceData.is_active
            ).first()
            
            if not reference_data:
                logger.warning(f"No active reference data found for model {model_id}")
                return {"status": "skipped", "reason": "no_reference_data"}
            
            # Get recent predictions
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            from .core.models import Prediction
            predictions = session.query(Prediction).filter(
                Prediction.model_id == model_id,
                Prediction.prediction_time >= start_time,
                Prediction.prediction_time <= end_time
            ).all()
            
            if len(predictions) < 10:  # Minimum sample size
                logger.warning(f"Insufficient predictions ({len(predictions)}) for drift analysis")
                return {"status": "skipped", "reason": "insufficient_data"}
            
            # Prepare data for drift detection
            current_data = pd.DataFrame([pred.input_data for pred in predictions])
            
            # Load reference data
            reference_df = pd.read_parquet(reference_data.data_path)
            
            # Run drift detection
            drift_config = DriftConfig(
                drift_threshold=0.05,
                statistical_tests=["ks", "psi"],
                multivariate_test="mmd"
            )
            
            detector = DriftDetector(drift_config)
            drift_results = detector.detect_drift(
                reference_data=reference_df,
                current_data=current_data,
                feature_names=model.feature_names
            )
            
            # Create drift report
            drift_report = DriftReport(
                model_id=model_id,
                reference_data_id=reference_data.id,
                report_name=f"Automated Drift Check - {datetime.utcnow().isoformat()}",
                drift_type="data_drift",
                detection_method="automated",
                start_time=start_time,
                end_time=end_time,
                overall_drift_score=drift_results["overall_drift_score"],
                drift_detected=drift_results["drift_detected"],
                p_value=drift_results.get("p_value"),
                threshold=drift_config.drift_threshold,
                feature_drift_scores=drift_results["feature_drift_scores"],
                drift_statistics=drift_results["drift_statistics"],
                summary=drift_results["summary"],
                num_samples_analyzed=len(predictions)
            )
            
            session.add(drift_report)
            session.commit()
            
            # Create alert if drift detected
            if drift_results["drift_detected"]:
                alert = Alert(
                    drift_report_id=drift_report.id,
                    model_id=model_id,
                    alert_type="drift_detection",
                    severity="high" if drift_results["overall_drift_score"] > 0.5 else "medium",
                    title=f"Data Drift Detected - {model.name}",
                    message=f"Significant data drift detected with score {drift_results['overall_drift_score']:.3f}",
                    metadata={"drift_score": drift_results["overall_drift_score"]}
                )
                session.add(alert)
                session.commit()
                
                logger.warning(f"Drift detected for model {model_id}: {drift_results['overall_drift_score']:.3f}")
            
            return {
                "status": "completed",
                "drift_detected": drift_results["drift_detected"],
                "drift_score": drift_results["overall_drift_score"],
                "report_id": str(drift_report.id)
            }
            
    except Exception as exc:
        logger.error(f"Drift check failed for model {model_id}: {str(exc)}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True)
def check_drift_for_all_models(self) -> Dict[str, Any]:
    """Check drift for all active models."""
    try:
        logger.info("Starting drift check for all active models")
        
        with db_manager.session_scope() as session:
            active_models = session.query(Model).filter(
                Model.status == "active"
            ).all()
            
            results = []
            for model in active_models:
                # Schedule individual drift checks
                task = check_drift_for_model.delay(str(model.id))
                results.append({
                    "model_id": str(model.id),
                    "model_name": model.name,
                    "task_id": task.id
                })
            
            logger.info(f"Scheduled drift checks for {len(results)} models")
            return {"status": "completed", "models_checked": len(results), "tasks": results}
            
    except Exception as exc:
        logger.error(f"Failed to schedule drift checks: {str(exc)}")
        raise


@celery_app.task(bind=True, max_retries=2)
def calculate_model_performance_metrics(self, model_id: str, date: str) -> Dict[str, Any]:
    """
    Calculate performance metrics for a model for a specific date.
    
    Args:
        model_id: Model UUID
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Dict with calculated metrics
    """
    try:
        logger.info(f"Calculating performance metrics for model {model_id} on {date}")
        
        target_date = datetime.strptime(date, "%Y-%m-%d")
        start_time = target_date
        end_time = target_date + timedelta(days=1)
        
        with db_manager.session_scope() as session:
            # Get model
            model = session.query(Model).filter(Model.id == model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Get predictions with actual values
            from .core.models import Prediction
            predictions = session.query(Prediction).filter(
                Prediction.model_id == model_id,
                Prediction.prediction_time >= start_time,
                Prediction.prediction_time < end_time,
                Prediction.actual_value.isnot(None)
            ).all()
            
            if not predictions:
                logger.info(f"No predictions with actual values found for model {model_id} on {date}")
                return {"status": "skipped", "reason": "no_data_with_ground_truth"}
            
            # Calculate metrics based on model type
            metrics = {}
            
            if model.algorithm in ["classification", "binary_classification"]:
                # Classification metrics
                y_true = [pred.actual_value for pred in predictions]
                y_pred = [pred.prediction for pred in predictions]
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                metrics.update({
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average="weighted"),
                    "recall": recall_score(y_true, y_pred, average="weighted"),
                    "f1_score": f1_score(y_true, y_pred, average="weighted"),
                })
                
                # ROC AUC for binary classification
                if model.algorithm == "binary_classification":
                    y_prob = [pred.prediction_probability[1] if pred.prediction_probability else 0.5 
                             for pred in predictions]
                    metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
                
            else:
                # Regression metrics
                y_true = [pred.actual_value for pred in predictions]
                y_pred = [pred.prediction for pred in predictions]
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                import numpy as np
                
                metrics.update({
                    "mae": mean_absolute_error(y_true, y_pred),
                    "mse": mean_squared_error(y_true, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "r2_score": r2_score(y_true, y_pred)
                })
            
            # System metrics
            response_times = [pred.response_time_ms for pred in predictions if pred.response_time_ms]
            avg_response_time = sum(response_times) / len(response_times) if response_times else None
            
            # Check if metrics already exist
            existing_metric = session.query(ModelPerformanceMetric).filter(
                ModelPerformanceMetric.model_id == model_id,
                ModelPerformanceMetric.metric_date == target_date
            ).first()
            
            if existing_metric:
                # Update existing
                for key, value in metrics.items():
                    setattr(existing_metric, key, value)
                existing_metric.num_predictions = len(predictions)
                existing_metric.num_actual_values = len(predictions)
                existing_metric.avg_response_time_ms = avg_response_time
            else:
                # Create new
                performance_metric = ModelPerformanceMetric(
                    model_id=model_id,
                    metric_date=target_date,
                    period_start=start_time,
                    period_end=end_time,
                    num_predictions=len(predictions),
                    num_actual_values=len(predictions),
                    avg_response_time_ms=avg_response_time,
                    **metrics
                )
                session.add(performance_metric)
            
            session.commit()
            
            logger.info(f"Performance metrics calculated for model {model_id} on {date}")
            return {"status": "completed", "metrics": metrics, "num_predictions": len(predictions)}
            
    except Exception as exc:
        logger.error(f"Failed to calculate performance metrics for model {model_id}: {str(exc)}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True)
def calculate_daily_performance_metrics(self) -> Dict[str, Any]:
    """Calculate performance metrics for all models for the previous day."""
    try:
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"Calculating daily performance metrics for {yesterday}")
        
        with db_manager.session_scope() as session:
            active_models = session.query(Model).filter(
                Model.status == "active"
            ).all()
            
            results = []
            for model in active_models:
                task = calculate_model_performance_metrics.delay(str(model.id), yesterday)
                results.append({
                    "model_id": str(model.id),
                    "model_name": model.name,
                    "task_id": task.id
                })
            
            logger.info(f"Scheduled performance metric calculations for {len(results)} models")
            return {"status": "completed", "date": yesterday, "models": len(results), "tasks": results}
            
    except Exception as exc:
        logger.error(f"Failed to schedule performance metric calculations: {str(exc)}")
        raise


@celery_app.task(bind=True)
def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, Any]:
    """Clean up old data to maintain performance."""
    try:
        logger.info(f"Starting cleanup of data older than {days_to_keep} days")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with db_manager.session_scope() as session:
            # Clean up old predictions (keep recent ones)
            from .core.models import Prediction, SystemMetric
            
            old_predictions = session.query(Prediction).filter(
                Prediction.prediction_time < cutoff_date
            ).count()
            
            session.query(Prediction).filter(
                Prediction.prediction_time < cutoff_date
            ).delete()
            
            # Clean up old system metrics
            old_metrics = session.query(SystemMetric).filter(
                SystemMetric.timestamp < cutoff_date
            ).count()
            
            session.query(SystemMetric).filter(
                SystemMetric.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            
            logger.info(f"Cleaned up {old_predictions} predictions and {old_metrics} system metrics")
            return {
                "status": "completed",
                "cutoff_date": cutoff_date.isoformat(),
                "predictions_cleaned": old_predictions,
                "metrics_cleaned": old_metrics
            }
            
    except Exception as exc:
        logger.error(f"Cleanup failed: {str(exc)}")
        raise


@celery_app.task(bind=True)
def system_health_check(self) -> Dict[str, Any]:
    """Perform system health checks."""
    try:
        logger.info("Performing system health check")
        
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": "unknown",
            "redis": "unknown",
            "celery": "healthy",
            "disk_usage": 0,
            "memory_usage": 0
        }
        
        # Database health check
        try:
            with db_manager.session_scope() as session:
                session.execute("SELECT 1")
                health_status["database"] = "healthy"
        except Exception as e:
            health_status["database"] = "unhealthy"
            logger.error(f"Database health check failed: {e}")
        
        # Redis health check (via Celery broker)
        try:
            celery_app.control.ping(timeout=5)
            health_status["redis"] = "healthy"
        except Exception as e:
            health_status["redis"] = "unhealthy"
            logger.error(f"Redis health check failed: {e}")
        
        # System metrics
        try:
            import psutil
            health_status["disk_usage"] = psutil.disk_usage('/').percent
            health_status["memory_usage"] = psutil.virtual_memory().percent
        except ImportError:
            logger.warning("psutil not installed, skipping system metrics")
        
        # Store health metrics
        metrics_collector = MetricsCollector()
        metrics_collector.record_system_health(health_status)
        
        return health_status
        
    except Exception as exc:
        logger.error(f"Health check failed: {str(exc)}")
        raise


@celery_app.task(bind=True, max_retries=3)
def train_model_async(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronous model training task.
    
    Args:
        training_config: Training configuration dictionary
        
    Returns:
        Dict with training results
    """
    try:
        logger.info(f"Starting async model training: {training_config.get('model_name', 'Unknown')}")
        
        from .train import ModelTrainer
        
        trainer = ModelTrainer()
        
        # Start training
        training_result = trainer.train_model(
            data_path=training_config["data_path"],
            target_column=training_config["target_column"],
            model_name=training_config["model_name"],
            model_type=training_config["model_type"],
            hyperparameters=training_config.get("hyperparameters", {}),
            validation_split=training_config.get("validation_split", 0.2)
        )
        
        logger.info(f"Model training completed: {training_result['model_id']}")
        return {
            "status": "completed",
            "model_id": training_result["model_id"],
            "metrics": training_result["metrics"],
            "model_path": training_result["model_path"]
        }
        
    except Exception as exc:
        logger.error(f"Model training failed: {str(exc)}")
        raise self.retry(exc=exc, countdown=120)


if __name__ == "__main__":
    # For running the worker
    celery_app.start()

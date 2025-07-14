"""
Celery worker configuration and async task definitions for MDT Dashboard.
Handles background processing for model training, drift detection, and performance monitoring.
"""

from celery import Celery
from celery.schedules import crontab
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import traceback

from .core.config import get_settings
from .core.database import db_manager
from .core.models import (
    Model, DriftReport, ReferenceData, Alert, ModelPerformanceMetric,
    DataQualityReport, Prediction, AlertSeverity, DriftType
)
from .drift_detection.algorithms import MultivariateDriftDetector, DriftResult
from .monitoring.metrics import MetricsCollector
from .data_processing import ComprehensiveDataProcessor

logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Initialize Celery
celery_app = Celery(
    "mdt_dashboard",
    broker=settings.redis.url,
    backend=settings.redis.url,
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
    task_time_limit=settings.task_time_limit,
    task_soft_time_limit=settings.task_soft_time_limit,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_ignore_result=False,
    result_expires=3600,
    worker_concurrency=settings.worker_concurrency,
)

# Periodic tasks for automated monitoring
celery_app.conf.beat_schedule = {
    'check_drift_all_models': {
        'task': 'mdt_dashboard.worker.check_drift_all_models',
        'schedule': crontab(minute=0),  # Every hour
    },
    'calculate_performance_metrics': {
        'task': 'mdt_dashboard.worker.calculate_performance_metrics_all_models',
        'schedule': crontab(minute=30),  # Every hour at 30 minutes
    },
    'data_quality_check': {
        'task': 'mdt_dashboard.worker.run_data_quality_checks',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'cleanup_old_predictions': {
        'task': 'mdt_dashboard.worker.cleanup_old_predictions',
        'schedule': crontab(hour=3, minute=0),  # Daily at 3 AM
    },
}


@celery_app.task(bind=True, max_retries=3)
def train_model_async(self, training_config: Dict[str, Any]):
    """
    Async model training task.
    
    Args:
        training_config: Configuration for model training
    """
    try:
        logger.info(f"Starting async model training with config: {training_config}")
        
        # Import here to avoid circular imports
        from .train import ModelTrainer
        
        trainer = ModelTrainer()
        result = trainer.train_model(
            data_path=training_config["data_path"],
            target_column=training_config["target_column"],
            model_name=training_config["model_name"],
            model_type=training_config["model_type"],
            hyperparameters=training_config.get("hyperparameters", {}),
            validation_split=training_config.get("validation_split", 0.2)
        )
        
        logger.info(f"Model training completed successfully: {result['model_id']}")
        return result
        
    except Exception as exc:
        logger.error(f"Model training failed: {exc}")
        logger.error(traceback.format_exc())
        
        # Update model status to failed
        try:
            with db_manager.session_scope() as session:
                model = session.query(Model).filter(
                    Model.name == training_config["model_name"]
                ).first()
                if model:
                    model.status = "failed"
                    session.commit()
        except Exception as db_exc:
            logger.error(f"Failed to update model status: {db_exc}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        raise exc


@celery_app.task(bind=True, max_retries=2)
def check_drift_for_model(self, model_id: str, time_window_hours: int = 24):
    """
    Check drift for a specific model.
    
    Args:
        model_id: Model identifier
        time_window_hours: Time window for drift detection
    """
    try:
        logger.info(f"Starting drift check for model {model_id}")
        
        with db_manager.session_scope() as session:
            # Get model
            model = session.query(Model).filter(Model.id == model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Get reference data
            reference_data = session.query(ReferenceData).filter(
                ReferenceData.model_id == model_id,
                ReferenceData.is_active == True
            ).first()
            
            if not reference_data:
                logger.warning(f"No reference data found for model {model_id}")
                return {"status": "skipped", "reason": "no_reference_data"}
            
            # Get recent predictions for current data
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            predictions = session.query(Prediction).filter(
                Prediction.model_id == model_id,
                Prediction.prediction_time >= cutoff_time
            ).all()
            
            if len(predictions) < 10:
                logger.warning(f"Insufficient predictions ({len(predictions)}) for drift detection")
                return {"status": "skipped", "reason": "insufficient_data"}
            
            # Prepare data for drift detection
            current_data = pd.DataFrame([pred.input_data for pred in predictions])
            
            # Load reference data
            processor = ComprehensiveDataProcessor()
            reference_df = processor.load_data(reference_data.data_path)
            
            # Initialize drift detector
            drift_detector = MultivariateDriftDetector(
                reference_data=reference_df,
                significance_level=settings.drift_detection.ks_test_threshold
            )
            
            # Detect drift
            drift_result = drift_detector.detect_drift(current_data)
            
            # Create drift report
            drift_report = DriftReport(
                model_id=model_id,
                reference_data_id=reference_data.id,
                report_name=f"Automated_Drift_Check_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                drift_type=DriftType.DATA_DRIFT.value,
                detection_method="multivariate_ks",
                start_time=cutoff_time,
                end_time=datetime.utcnow(),
                overall_drift_score=drift_result.drift_score,
                drift_detected=drift_result.drift_detected,
                p_value=drift_result.p_value,
                threshold=settings.drift_detection.ks_test_threshold,
                feature_drift_scores=drift_result.feature_drift_scores,
                drift_statistics=drift_result.statistics,
                summary=drift_result.summary,
                num_samples_analyzed=len(predictions)
            )
            
            session.add(drift_report)
            session.commit()
            
            # Create alert if drift detected
            if drift_result.drift_detected:
                severity = AlertSeverity.HIGH if drift_result.drift_score > 0.1 else AlertSeverity.MEDIUM
                
                alert = Alert(
                    drift_report_id=drift_report.id,
                    model_id=model_id,
                    alert_type="drift_detected",
                    severity=severity.value,
                    title=f"Data Drift Detected for {model.name}",
                    message=f"Drift detected with score {drift_result.drift_score:.4f}. "
                           f"Features affected: {', '.join(drift_result.affected_features[:5])}",
                    metadata={
                        "drift_score": drift_result.drift_score,
                        "affected_features": drift_result.affected_features,
                        "num_samples": len(predictions)
                    }
                )
                
                session.add(alert)
                session.commit()
                
                logger.warning(f"Drift detected for model {model_id}: {drift_result.drift_score:.4f}")
            
            return {
                "status": "completed",
                "drift_detected": drift_result.drift_detected,
                "drift_score": drift_result.drift_score,
                "report_id": str(drift_report.id)
            }
            
    except Exception as exc:
        logger.error(f"Drift check failed for model {model_id}: {exc}")
        logger.error(traceback.format_exc())
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        raise exc


@celery_app.task
def check_drift_all_models():
    """Check drift for all active models."""
    try:
        logger.info("Starting drift check for all active models")
        
        with db_manager.session_scope() as session:
            active_models = session.query(Model).filter(Model.status == "active").all()
            
            for model in active_models:
                # Trigger async drift check for each model
                check_drift_for_model.delay(str(model.id))
                
        logger.info(f"Triggered drift checks for {len(active_models)} models")
        return {"status": "completed", "models_checked": len(active_models)}
        
    except Exception as exc:
        logger.error(f"Failed to check drift for all models: {exc}")
        raise exc


@celery_app.task(bind=True, max_retries=2)
def calculate_model_performance_metrics(self, model_id: str, time_period_hours: int = 24):
    """
    Calculate performance metrics for a model over a time period.
    
    Args:
        model_id: Model identifier
        time_period_hours: Time period for metrics calculation
    """
    try:
        logger.info(f"Calculating performance metrics for model {model_id}")
        
        with db_manager.session_scope() as session:
            # Get model
            model = session.query(Model).filter(Model.id == model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Get predictions with actual values
            cutoff_time = datetime.utcnow() - timedelta(hours=time_period_hours)
            predictions = session.query(Prediction).filter(
                Prediction.model_id == model_id,
                Prediction.prediction_time >= cutoff_time,
                Prediction.actual_value.isnot(None)
            ).all()
            
            if len(predictions) < 10:
                logger.warning(f"Insufficient predictions with ground truth for model {model_id}")
                return {"status": "skipped", "reason": "insufficient_ground_truth"}
            
            # Extract predictions and actual values
            y_pred = [pred.prediction for pred in predictions]
            y_true = [pred.actual_value for pred in predictions]
            
            # Calculate metrics based on model type
            metrics = {}
            
            # Determine if classification or regression
            is_classification = all(isinstance(val, (int, bool)) or val in [0, 1] for val in y_true)
            
            if is_classification:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                metrics.update({
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, average='weighted')),
                    "recall": float(recall_score(y_true, y_pred, average='weighted')),
                    "f1_score": float(f1_score(y_true, y_pred, average='weighted')),
                })
                
                # Try to calculate AUC if probabilities available
                try:
                    y_prob = [pred.prediction_probability.get("1", 0.5) if pred.prediction_probability else 0.5 
                             for pred in predictions]
                    metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
                except Exception:
                    pass
            else:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                
                metrics.update({
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(np.sqrt(mse)),
                    "r2_score": float(r2_score(y_true, y_pred))
                })
            
            # Calculate system metrics
            response_times = [pred.response_time_ms for pred in predictions if pred.response_time_ms]
            avg_response_time = np.mean(response_times) if response_times else None
            
            # Create performance metric record
            performance_metric = ModelPerformanceMetric(
                model_id=model_id,
                metric_date=datetime.utcnow().date(),
                period_start=cutoff_time,
                period_end=datetime.utcnow(),
                num_predictions=len(predictions),
                num_actual_values=len(predictions),
                avg_response_time_ms=avg_response_time,
                error_rate=0.0,  # Calculate based on failed predictions
                **{k: v for k, v in metrics.items() if k in [
                    'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
                    'mae', 'mse', 'rmse', 'r2_score'
                ]},
                custom_metrics={k: v for k, v in metrics.items() if k not in [
                    'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
                    'mae', 'mse', 'rmse', 'r2_score'
                ]}
            )
            
            # Check if record already exists for this date
            existing_metric = session.query(ModelPerformanceMetric).filter(
                ModelPerformanceMetric.model_id == model_id,
                ModelPerformanceMetric.metric_date == datetime.utcnow().date()
            ).first()
            
            if existing_metric:
                # Update existing record
                for key, value in metrics.items():
                    if hasattr(existing_metric, key):
                        setattr(existing_metric, key, value)
                existing_metric.num_predictions = len(predictions)
                existing_metric.avg_response_time_ms = avg_response_time
            else:
                session.add(performance_metric)
            
            session.commit()
            
            logger.info(f"Performance metrics calculated for model {model_id}: {metrics}")
            return {"status": "completed", "metrics": metrics}
            
    except Exception as exc:
        logger.error(f"Performance calculation failed for model {model_id}: {exc}")
        logger.error(traceback.format_exc())
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        raise exc


@celery_app.task
def calculate_performance_metrics_all_models():
    """Calculate performance metrics for all active models."""
    try:
        logger.info("Calculating performance metrics for all active models")
        
        with db_manager.session_scope() as session:
            active_models = session.query(Model).filter(Model.status == "active").all()
            
            for model in active_models:
                # Trigger async performance calculation for each model
                calculate_model_performance_metrics.delay(str(model.id))
                
        logger.info(f"Triggered performance calculations for {len(active_models)} models")
        return {"status": "completed", "models_processed": len(active_models)}
        
    except Exception as exc:
        logger.error(f"Failed to calculate performance for all models: {exc}")
        raise exc


@celery_app.task
def run_data_quality_checks():
    """Run data quality checks for all active models."""
    try:
        logger.info("Running data quality checks for all active models")
        
        with db_manager.session_scope() as session:
            active_models = session.query(Model).filter(Model.status == "active").all()
            
            for model in active_models:
                # Get recent predictions
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                predictions = session.query(Prediction).filter(
                    Prediction.model_id == model.id,
                    Prediction.prediction_time >= cutoff_time
                ).all()
                
                if len(predictions) < 10:
                    continue
                
                # Analyze data quality
                processor = ComprehensiveDataProcessor()
                input_data = pd.DataFrame([pred.input_data for pred in predictions])
                
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
            
        logger.info(f"Data quality checks completed for {len(active_models)} models")
        return {"status": "completed", "models_checked": len(active_models)}
        
    except Exception as exc:
        logger.error(f"Data quality checks failed: {exc}")
        raise exc


@celery_app.task
def cleanup_old_predictions(days_to_keep: int = 30):
    """Clean up old prediction records to manage database size."""
    try:
        logger.info(f"Cleaning up predictions older than {days_to_keep} days")
        
        cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with db_manager.session_scope() as session:
            # Delete old predictions
            deleted_count = session.query(Prediction).filter(
                Prediction.prediction_time < cutoff_time
            ).delete()
            
            session.commit()
            
        logger.info(f"Cleaned up {deleted_count} old prediction records")
        return {"status": "completed", "deleted_count": deleted_count}
        
    except Exception as exc:
        logger.error(f"Cleanup failed: {exc}")
        raise exc


@celery_app.task(bind=True, max_retries=3)
def send_alert_notification(self, alert_id: str):
    """Send alert notification via configured channels."""
    try:
        logger.info(f"Sending notification for alert {alert_id}")
        
        with db_manager.session_scope() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                raise ValueError(f"Alert {alert_id} not found")
            
            # Import notification modules
            from .monitoring.alerts import AlertManager
            
            alert_manager = AlertManager()
            
            # Send notifications
            if settings.alerts.enable_email:
                alert_manager.send_email_alert(alert)
            
            if settings.alerts.enable_slack:
                alert_manager.send_slack_alert(alert)
            
            if settings.alerts.enable_webhook:
                alert_manager.send_webhook_alert(alert)
            
        logger.info(f"Alert notification sent successfully for {alert_id}")
        return {"status": "completed", "alert_id": alert_id}
        
    except Exception as exc:
        logger.error(f"Failed to send alert notification: {exc}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        raise exc


# Health check task
@celery_app.task
def health_check():
    """Health check task for monitoring worker status."""
    try:
        with db_manager.session_scope() as session:
            # Simple database query
            result = session.execute("SELECT 1").scalar()
            
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "worker_id": celery_app.control.inspect().active(),
            "database_connection": "ok" if result == 1 else "error"
        }
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(exc)
        }


if __name__ == "__main__":
    # Run worker
    celery_app.start()

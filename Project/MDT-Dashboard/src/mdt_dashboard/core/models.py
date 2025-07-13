"""
Database models for MDT Dashboard.
"""
from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from enum import Enum

from .database import Base


class ModelStatus(str, Enum):
    """Model status enumeration."""
    TRAINING = "training"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class DriftType(str, Enum):
    """Drift type enumeration."""
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"


class AlertSeverity(str, Enum):
    """Alert severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Model(Base):
    """ML Model registry."""
    __tablename__ = "models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    description = Column(Text)
    algorithm = Column(String(100), nullable=False)
    framework = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default=ModelStatus.TRAINING.value)
    
    # Model metadata
    model_path = Column(String(500), nullable=False)
    metrics = Column(JSON)
    hyperparameters = Column(JSON)
    feature_names = Column(JSON)
    target_names = Column(JSON)
    model_size_mb = Column(Float)
    
    # Tracking
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    
    # MLflow integration
    mlflow_run_id = Column(String(100))
    mlflow_experiment_id = Column(String(100))
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model")
    drift_reports = relationship("DriftReport", back_populates="model")
    performance_metrics = relationship("ModelPerformanceMetric", back_populates="model")
    
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_model_name_version'),
        Index('ix_models_status', 'status'),
        Index('ix_models_created_at', 'created_at'),
        CheckConstraint(status.in_([s.value for s in ModelStatus]), name='ck_model_status')
    )


class ReferenceData(Base):
    """Reference data for drift detection."""
    __tablename__ = "reference_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Data storage
    data_path = Column(String(500), nullable=False)
    data_hash = Column(String(64), nullable=False)  # SHA256
    schema_version = Column(String(20), nullable=False)
    
    # Statistics
    num_samples = Column(Integer, nullable=False)
    feature_statistics = Column(JSON)  # Mean, std, percentiles, etc.
    data_quality_metrics = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    model = relationship("Model")
    drift_reports = relationship("DriftReport", back_populates="reference_data")
    
    __table_args__ = (
        Index('ix_reference_data_model_id', 'model_id'),
        Index('ix_reference_data_active', 'is_active'),
    )


class Prediction(Base):
    """Individual predictions with metadata."""
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    
    # Input/Output
    input_data = Column(JSON, nullable=False)
    prediction = Column(JSON, nullable=False)
    prediction_probability = Column(JSON)  # For classification
    
    # Metadata
    prediction_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    response_time_ms = Column(Float)
    request_id = Column(String(100))
    
    # Data quality
    data_quality_score = Column(Float)
    missing_features = Column(JSON)
    out_of_range_features = Column(JSON)
    
    # Drift scores
    drift_score = Column(Float)
    feature_drift_scores = Column(JSON)
    
    # Performance (if ground truth available)
    actual_value = Column(JSON)
    prediction_error = Column(Float)
    
    # Relationships
    model = relationship("Model", back_populates="predictions")
    
    __table_args__ = (
        Index('ix_predictions_model_id', 'model_id'),
        Index('ix_predictions_time', 'prediction_time'),
        Index('ix_predictions_drift_score', 'drift_score'),
    )


class DriftReport(Base):
    """Drift detection reports."""
    __tablename__ = "drift_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    reference_data_id = Column(UUID(as_uuid=True), ForeignKey("reference_data.id"), nullable=False)
    
    # Report metadata
    report_name = Column(String(255), nullable=False)
    drift_type = Column(String(20), nullable=False)
    detection_method = Column(String(50), nullable=False)
    
    # Time range
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Drift results
    overall_drift_score = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    p_value = Column(Float)
    threshold = Column(Float, nullable=False)
    
    # Detailed results
    feature_drift_scores = Column(JSON)  # Per-feature drift scores
    drift_statistics = Column(JSON)  # Algorithm-specific statistics
    summary = Column(Text)
    
    # Data quality
    num_samples_analyzed = Column(Integer, nullable=False)
    data_quality_issues = Column(JSON)
    
    # Relationships
    model = relationship("Model", back_populates="drift_reports")
    reference_data = relationship("ReferenceData", back_populates="drift_reports")
    alerts = relationship("Alert", back_populates="drift_report")
    
    __table_args__ = (
        Index('ix_drift_reports_model_id', 'model_id'),
        Index('ix_drift_reports_created_at', 'created_at'),
        Index('ix_drift_reports_drift_detected', 'drift_detected'),
        CheckConstraint(drift_type.in_([d.value for d in DriftType]), name='ck_drift_type')
    )


class ModelPerformanceMetric(Base):
    """Model performance metrics over time."""
    __tablename__ = "model_performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    
    # Time period
    metric_date = Column(DateTime, nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    
    # Custom metrics
    custom_metrics = Column(JSON)
    
    # Data volume
    num_predictions = Column(Integer, nullable=False)
    num_actual_values = Column(Integer)  # Ground truth available
    
    # System metrics
    avg_response_time_ms = Column(Float)
    error_rate = Column(Float)
    
    # Relationships
    model = relationship("Model", back_populates="performance_metrics")
    
    __table_args__ = (
        Index('ix_performance_metrics_model_id', 'model_id'),
        Index('ix_performance_metrics_date', 'metric_date'),
        UniqueConstraint('model_id', 'metric_date', name='uq_performance_model_date')
    )


class Alert(Base):
    """Alerts and notifications."""
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drift_report_id = Column(UUID(as_uuid=True), ForeignKey("drift_reports.id"))
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"))
    
    # Alert details
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    
    # Status
    is_resolved = Column(Boolean, default=False, nullable=False)
    is_acknowledged = Column(Boolean, default=False, nullable=False)
    
    # Additional data
    metadata = Column(JSON)
    
    # Relationships
    drift_report = relationship("DriftReport", back_populates="alerts")
    model = relationship("Model")
    
    __table_args__ = (
        Index('ix_alerts_created_at', 'created_at'),
        Index('ix_alerts_severity', 'severity'),
        Index('ix_alerts_resolved', 'is_resolved'),
        CheckConstraint(severity.in_([s.value for s in AlertSeverity]), name='ck_alert_severity')
    )


class DataQualityReport(Base):
    """Data quality assessment reports."""
    __tablename__ = "data_quality_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"))
    
    # Report metadata
    report_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Time range
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    
    # Quality metrics
    overall_quality_score = Column(Float, nullable=False)
    completeness_score = Column(Float)
    accuracy_score = Column(Float)
    consistency_score = Column(Float)
    validity_score = Column(Float)
    
    # Detailed results
    feature_quality_scores = Column(JSON)
    missing_value_stats = Column(JSON)
    outlier_stats = Column(JSON)
    schema_violations = Column(JSON)
    
    # Data volume
    num_samples_analyzed = Column(Integer, nullable=False)
    
    # Relationships
    model = relationship("Model")
    
    __table_args__ = (
        Index('ix_data_quality_reports_model_id', 'model_id'),
        Index('ix_data_quality_reports_created_at', 'created_at'),
    )


class SystemMetric(Base):
    """System performance and health metrics."""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric metadata
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Metric value
    value = Column(Float, nullable=False)
    labels = Column(JSON)  # Additional labels/tags
    
    # Service/component
    service_name = Column(String(100), nullable=False)
    component = Column(String(100))
    
    __table_args__ = (
        Index('ix_system_metrics_timestamp', 'timestamp'),
        Index('ix_system_metrics_service', 'service_name'),
        Index('ix_system_metrics_name', 'metric_name'),
    )

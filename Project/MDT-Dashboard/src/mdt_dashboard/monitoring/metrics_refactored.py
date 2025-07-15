"""
Advanced metrics collection and monitoring system.
Tracks model performance, system health, and business metrics.

Refactored for improved code quality, type safety, and maintainability.
"""

import asyncio
import contextlib
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Protocol
import json
import logging

try:
    import numpy as np
except ImportError:
    np = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback if pydantic not available
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    
    def ConfigDict(**kwargs):
        """Fallback config dict."""
        return None
    
    def Field(default=None, **kwargs):
        """Fallback field function."""
        return default
    
    def field_validator(*args, **kwargs):
        """Fallback field validator."""
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


def safe_mean(values: List[float]) -> float:
    """Calculate mean safely, with fallback if numpy not available."""
    if not values:
        return 0.0
    if np is not None:
        return float(np.mean(values))
    return sum(values) / len(values)


def safe_max(values: List[float]) -> float:
    """Calculate max safely, with fallback if numpy not available."""
    if not values:
        return 0.0
    if np is not None:
        return float(np.max(values))
    return max(values)


def safe_min(values: List[float]) -> float:
    """Calculate min safely, with fallback if numpy not available."""
    if not values:
        return 0.0
    if np is not None:
        return float(np.min(values))
    return min(values)


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    
    PREDICTION = "prediction"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    BUSINESS = "business"


class HealthStatus(str, Enum):
    """Health status levels."""
    
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    NO_DATA = "no_data"


class BaseMetrics(BaseModel):
    """Base class for all metrics with common fields."""
    
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    timestamp: datetime = Field(default_factory=datetime.now)
    metric_type: MetricType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO timestamp."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        return data


class PredictionMetrics(BaseMetrics):
    """Metrics for prediction requests."""
    
    metric_type: MetricType = Field(default=MetricType.PREDICTION, init=False)
    model_name: str = Field(..., min_length=1)
    request_count: int = Field(..., ge=0)
    prediction_count: int = Field(..., ge=0)
    processing_time_ms: float = Field(..., ge=0.0)
    drift_detected: bool = False
    average_confidence: float = Field(..., ge=0.0, le=1.0)
    error_count: int = Field(default=0, ge=0)
    
    @field_validator("prediction_count")
    @classmethod
    def validate_prediction_count(cls, v: int, info) -> int:
        """Ensure prediction count doesn't exceed request count."""
        if "request_count" in info.data and v > info.data["request_count"]:
            raise ValueError("Prediction count cannot exceed request count")
        return v


class SystemMetrics(BaseMetrics):
    """System resource metrics."""
    
    metric_type: MetricType = Field(default=MetricType.SYSTEM, init=False)
    cpu_usage_percent: float = Field(..., ge=0.0, le=100.0)
    memory_usage_percent: float = Field(..., ge=0.0, le=100.0)
    memory_usage_mb: float = Field(..., ge=0.0)
    disk_usage_percent: float = Field(..., ge=0.0, le=100.0)
    network_bytes_sent: int = Field(..., ge=0)
    network_bytes_received: int = Field(..., ge=0)
    active_threads: int = Field(..., ge=0)


class ModelPerformanceMetrics(BaseMetrics):
    """Model performance tracking metrics."""
    
    metric_type: MetricType = Field(default=MetricType.PERFORMANCE, init=False)
    model_name: str = Field(..., min_length=1)
    
    # Classification metrics
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    auc_roc: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Regression metrics
    mse: Optional[float] = Field(None, ge=0.0)
    mae: Optional[float] = Field(None, ge=0.0)
    r2_score: Optional[float] = None
    
    # Additional metrics
    drift_score: Optional[float] = Field(None, ge=0.0)
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    prediction_latency_ms: Optional[float] = Field(None, ge=0.0)
    throughput_per_second: Optional[float] = Field(None, ge=0.0)


class MetricsSummary(BaseModel):
    """Summary statistics for metrics."""
    
    model_config = ConfigDict(extra="forbid")
    
    total_requests: int = Field(ge=0)
    total_predictions: int = Field(ge=0)
    avg_processing_time_ms: float = Field(ge=0.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    drift_detection_rate: float = Field(ge=0.0, le=1.0)
    avg_confidence: float = Field(ge=0.0, le=1.0)
    time_window_minutes: int = Field(ge=0)
    requests_per_minute: float = Field(ge=0.0)


class SystemSummary(BaseModel):
    """Summary of system metrics."""
    
    model_config = ConfigDict(extra="forbid")
    
    avg_cpu_usage: float = Field(ge=0.0, le=100.0)
    max_cpu_usage: float = Field(ge=0.0, le=100.0)
    avg_memory_usage: float = Field(ge=0.0, le=100.0)
    max_memory_usage: float = Field(ge=0.0, le=100.0)
    max_memory_usage_mb: float = Field(ge=0.0)
    avg_disk_usage: float = Field(ge=0.0, le=100.0)
    total_network_sent_mb: float = Field(ge=0.0)
    total_network_received_mb: float = Field(ge=0.0)
    avg_active_threads: float = Field(ge=0.0)
    max_active_threads: int = Field(ge=0)
    time_window_minutes: int = Field(ge=0)


class MetricsStorage(Protocol):
    """Protocol for metrics storage implementations."""
    
    def add_metric(self, metric: BaseMetrics) -> None:
        """Add a metric to storage."""
        ...
    
    def get_metrics(
        self,
        metric_type: MetricType,
        time_window: Optional[timedelta] = None,
        model_name: Optional[str] = None
    ) -> List[BaseMetrics]:
        """Retrieve metrics from storage."""
        ...
    
    def clear_old_metrics(self, older_than: timedelta) -> int:
        """Clear metrics older than specified time."""
        ...


class InMemoryMetricsStorage:
    """In-memory metrics storage with thread safety."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._storage: Dict[MetricType, Deque[BaseMetrics]] = {
            metric_type: deque(maxlen=max_history)
            for metric_type in MetricType
        }
        self._lock = threading.RLock()
    
    def add_metric(self, metric: BaseMetrics) -> None:
        """Add a metric to storage."""
        with self._lock:
            self._storage[metric.metric_type].append(metric)
    
    def get_metrics(
        self,
        metric_type: MetricType,
        time_window: Optional[timedelta] = None,
        model_name: Optional[str] = None
    ) -> List[BaseMetrics]:
        """Retrieve metrics from storage."""
        with self._lock:
            metrics = list(self._storage[metric_type])
            
            # Apply time filter
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            # Apply model name filter
            if model_name and hasattr(metrics[0] if metrics else None, 'model_name'):
                metrics = [m for m in metrics if getattr(m, 'model_name', None) == model_name]
            
            return metrics
    
    def clear_old_metrics(self, older_than: timedelta) -> int:
        """Clear metrics older than specified time."""
        cutoff_time = datetime.now() - older_than
        cleared_count = 0
        
        with self._lock:
            for metric_type, storage in self._storage.items():
                original_length = len(storage)
                # Clear from left while metrics are old
                while storage and storage[0].timestamp < cutoff_time:
                    storage.popleft()
                cleared_count += original_length - len(storage)
        
        return cleared_count


class SystemMonitor:
    """Monitors system resources."""
    
    def __init__(self):
        self._network_baseline = self._get_network_stats()
    
    def _get_network_stats(self) -> Dict[str, int]:
        """Get current network statistics."""
        if psutil is None:
            logger.warning("psutil not available, returning default network stats")
            return {"bytes_sent": 0, "bytes_recv": 0}
            
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv
            }
        except Exception as e:
            logger.warning(f"Failed to get network stats: {e}")
            return {"bytes_sent": 0, "bytes_recv": 0}
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        if psutil is None:
            logger.warning("psutil not available, returning default system metrics")
            return SystemMetrics(
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_percent=0.0,
                network_bytes_sent=0,
                network_bytes_received=0,
                active_threads=threading.active_count()
            )
            
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            current_network = self._get_network_stats()
            network_sent = max(0, current_network["bytes_sent"] - self._network_baseline["bytes_sent"])
            network_received = max(0, current_network["bytes_recv"] - self._network_baseline["bytes_recv"])
            
            # Thread count
            active_threads = threading.active_count()
            
            return SystemMetrics(
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory.percent,
                memory_usage_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_bytes_sent=network_sent,
                network_bytes_received=network_received,
                active_threads=active_threads
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return zero metrics on error
            return SystemMetrics(
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_percent=0.0,
                network_bytes_sent=0,
                network_bytes_received=0,
                active_threads=threading.active_count()
            )


class MetricsAggregator:
    """Aggregates metrics into summaries."""
    
    @staticmethod
    def aggregate_prediction_metrics(
        metrics: List[PredictionMetrics],
        time_window_minutes: int
    ) -> MetricsSummary:
        """Aggregate prediction metrics into summary."""
        if not metrics:
            return MetricsSummary(
                total_requests=0,
                total_predictions=0,
                avg_processing_time_ms=0.0,
                error_rate=0.0,
                drift_detection_rate=0.0,
                avg_confidence=0.0,
                time_window_minutes=time_window_minutes,
                requests_per_minute=0.0
            )
        
        total_requests = sum(m.request_count for m in metrics)
        total_predictions = sum(m.prediction_count for m in metrics)
        total_errors = sum(m.error_count for m in metrics)
        
        # Calculate averages
        processing_times = [m.processing_time_ms for m in metrics if m.processing_time_ms > 0]
        avg_processing_time = safe_mean(processing_times)
        
        drift_detections = sum(1 for m in metrics if m.drift_detected)
        drift_rate = drift_detections / len(metrics) if metrics else 0.0
        
        confidences = [m.average_confidence for m in metrics if m.average_confidence > 0]
        avg_confidence = safe_mean(confidences)
        
        error_rate = total_errors / total_requests if total_requests > 0 else 0.0
        requests_per_minute = total_requests / time_window_minutes if time_window_minutes > 0 else 0.0
        
        return MetricsSummary(
            total_requests=total_requests,
            total_predictions=total_predictions,
            avg_processing_time_ms=avg_processing_time,
            error_rate=error_rate,
            drift_detection_rate=drift_rate,
            avg_confidence=avg_confidence,
            time_window_minutes=time_window_minutes,
            requests_per_minute=requests_per_minute
        )
    
    @staticmethod
    def aggregate_system_metrics(
        metrics: List[SystemMetrics],
        time_window_minutes: int
    ) -> SystemSummary:
        """Aggregate system metrics into summary."""
        if not metrics:
            return SystemSummary(
                avg_cpu_usage=0.0,
                max_cpu_usage=0.0,
                avg_memory_usage=0.0,
                max_memory_usage=0.0,
                max_memory_usage_mb=0.0,
                avg_disk_usage=0.0,
                total_network_sent_mb=0.0,
                total_network_received_mb=0.0,
                avg_active_threads=0.0,
                max_active_threads=0,
                time_window_minutes=time_window_minutes
            )
        
        cpu_usages = [m.cpu_usage_percent for m in metrics]
        memory_usages = [m.memory_usage_percent for m in metrics]
        memory_usage_mbs = [m.memory_usage_mb for m in metrics]
        disk_usages = [m.disk_usage_percent for m in metrics]
        thread_counts = [m.active_threads for m in metrics]
        
        total_network_sent = sum(m.network_bytes_sent for m in metrics)
        total_network_received = sum(m.network_bytes_received for m in metrics)
        
        return SystemSummary(
            avg_cpu_usage=safe_mean(cpu_usages),
            max_cpu_usage=safe_max(cpu_usages),
            avg_memory_usage=safe_mean(memory_usages),
            max_memory_usage=safe_max(memory_usages),
            max_memory_usage_mb=safe_max(memory_usage_mbs),
            avg_disk_usage=safe_mean(disk_usages),
            total_network_sent_mb=total_network_sent / (1024 * 1024),
            total_network_received_mb=total_network_received / (1024 * 1024),
            avg_active_threads=safe_mean([float(x) for x in thread_counts]),
            max_active_threads=int(safe_max([float(x) for x in thread_counts])),
            time_window_minutes=time_window_minutes
        )


class MetricsCollector:
    """Thread-safe metrics collection system with improved architecture."""
    
    def __init__(
        self,
        storage: Optional[MetricsStorage] = None,
        collection_interval: int = 60,
        auto_cleanup_hours: int = 24
    ):
        self.storage = storage or InMemoryMetricsStorage()
        self.collection_interval = collection_interval
        self.auto_cleanup_hours = auto_cleanup_hours
        
        self.system_monitor = SystemMonitor()
        self.aggregator = MetricsAggregator()
        
        # Model statistics tracking
        self.model_stats = defaultdict(lambda: {
            "total_requests": 0,
            "total_predictions": 0,
            "total_errors": 0,
            "avg_processing_time": 0.0,
            "drift_detection_count": 0,
            "last_seen": None
        })
        
        # Background collection control
        self._collection_task: Optional[asyncio.Task] = None
        self._stop_collection = asyncio.Event()
        self._lock = threading.RLock()
        
        logger.info("Metrics collector initialized")
    
    async def start_collection(self) -> None:
        """Start background metrics collection."""
        if self._collection_task is None or self._collection_task.done():
            self._stop_collection.clear()
            self._collection_task = asyncio.create_task(self._collect_system_metrics_loop())
            logger.info("Started background metrics collection")
    
    async def stop_collection(self) -> None:
        """Stop background metrics collection."""
        if self._collection_task and not self._collection_task.done():
            self._stop_collection.set()
            await self._collection_task
            logger.info("Stopped background metrics collection")
    
    async def _collect_system_metrics_loop(self) -> None:
        """Background system metrics collection loop."""
        while not self._stop_collection.is_set():
            try:
                metrics = self.system_monitor.collect_system_metrics()
                self.storage.add_metric(metrics)
                
                # Periodic cleanup
                if datetime.now().hour == 0 and datetime.now().minute < self.collection_interval:
                    self._cleanup_old_metrics()
                    
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            try:
                await asyncio.wait_for(
                    self._stop_collection.wait(),
                    timeout=self.collection_interval
                )
                break
            except asyncio.TimeoutError:
                continue
    
    def record_prediction_metrics(self, metrics: PredictionMetrics) -> None:
        """Record prediction metrics."""
        self.storage.add_metric(metrics)
        
        with self._lock:
            # Update aggregated stats
            stats = self.model_stats[metrics.model_name]
            stats["total_requests"] += metrics.request_count
            stats["total_predictions"] += metrics.prediction_count
            stats["total_errors"] += metrics.error_count
            
            # Update average processing time
            total_requests = stats["total_requests"]
            current_avg = stats["avg_processing_time"]
            new_avg = (
                (current_avg * (total_requests - metrics.request_count)) +
                (metrics.processing_time_ms * metrics.request_count)
            ) / total_requests
            stats["avg_processing_time"] = new_avg
            
            if metrics.drift_detected:
                stats["drift_detection_count"] += 1
            
            stats["last_seen"] = metrics.timestamp.isoformat()
    
    def record_performance_metrics(self, metrics: ModelPerformanceMetrics) -> None:
        """Record model performance metrics."""
        self.storage.add_metric(metrics)
    
    def get_prediction_metrics_summary(
        self,
        model_name: Optional[str] = None,
        time_window_minutes: int = 60
    ) -> MetricsSummary:
        """Get summary of prediction metrics."""
        time_window = timedelta(minutes=time_window_minutes)
        metrics = self.storage.get_metrics(
            MetricType.PREDICTION,
            time_window=time_window,
            model_name=model_name
        )
        
        # Filter to PredictionMetrics
        prediction_metrics = [m for m in metrics if isinstance(m, PredictionMetrics)]
        
        return self.aggregator.aggregate_prediction_metrics(
            prediction_metrics,
            time_window_minutes
        )
    
    def get_system_metrics_summary(self, time_window_minutes: int = 60) -> SystemSummary:
        """Get summary of system metrics."""
        time_window = timedelta(minutes=time_window_minutes)
        metrics = self.storage.get_metrics(MetricType.SYSTEM, time_window=time_window)
        
        # Filter to SystemMetrics
        system_metrics = [m for m in metrics if isinstance(m, SystemMetrics)]
        
        return self.aggregator.aggregate_system_metrics(system_metrics, time_window_minutes)
    
    def get_model_performance_summary(self, model_name: str) -> Dict[str, Any]:
        """Get performance summary for a specific model."""
        metrics = self.storage.get_metrics(
            MetricType.PERFORMANCE,
            model_name=model_name
        )
        
        # Filter to ModelPerformanceMetrics
        performance_metrics = [m for m in metrics if isinstance(m, ModelPerformanceMetrics)]
        
        if not performance_metrics:
            return {"model_name": model_name, "metrics_available": False}
        
        # Get latest metrics
        latest_metrics = performance_metrics[-1]
        
        # Calculate trends (if enough data points)
        trend_data = {}
        if len(performance_metrics) >= 2:
            prev_metrics = performance_metrics[-2]
            
            for attr in ["accuracy", "precision", "recall", "f1_score", "mse", "mae", "r2_score"]:
                current_val = getattr(latest_metrics, attr)
                prev_val = getattr(prev_metrics, attr)
                
                if current_val is not None and prev_val is not None:
                    trend_data[f"{attr}_trend"] = current_val - prev_val
        
        return {
            "model_name": model_name,
            "metrics_available": True,
            "latest_metrics": latest_metrics.to_dict(),
            "trends": trend_data,
            "total_data_points": len(performance_metrics)
        }
    
    def get_all_model_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for all models."""
        with self._lock:
            return dict(self.model_stats)
    
    def export_metrics(
        self,
        filepath: Path,
        time_window_hours: int = 24,
        include_raw_data: bool = False
    ) -> None:
        """Export metrics to file."""
        time_window = timedelta(hours=time_window_hours)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "time_window_hours": time_window_hours,
            "summary": {
                "prediction_metrics": self.get_prediction_metrics_summary(
                    time_window_minutes=time_window_hours * 60
                ).model_dump(),
                "system_metrics": self.get_system_metrics_summary(
                    time_window_minutes=time_window_hours * 60
                ).model_dump(),
                "model_stats": self.get_all_model_stats()
            }
        }
        
        if include_raw_data:
            export_data["raw_data"] = {
                "prediction_metrics": [
                    m.to_dict()
                    for m in self.storage.get_metrics(MetricType.PREDICTION, time_window)
                ],
                "system_metrics": [
                    m.to_dict()
                    for m in self.storage.get_metrics(MetricType.SYSTEM, time_window)
                ],
                "performance_metrics": [
                    m.to_dict()
                    for m in self.storage.get_metrics(MetricType.PERFORMANCE, time_window)
                ]
            }
        
        # Write to file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of metrics collection."""
        system_metrics = self.storage.get_metrics(
            MetricType.SYSTEM,
            time_window=timedelta(minutes=5)
        )
        
        latest_system_metrics = system_metrics[-1] if system_metrics else None
        
        status = HealthStatus.HEALTHY
        if not latest_system_metrics:
            status = HealthStatus.NO_DATA
        elif latest_system_metrics.cpu_usage_percent > 90 or latest_system_metrics.memory_usage_percent > 90:
            status = HealthStatus.WARNING
        
        return {
            "status": status.value,
            "metrics_count": {
                "prediction_metrics": len(self.storage.get_metrics(MetricType.PREDICTION)),
                "system_metrics": len(self.storage.get_metrics(MetricType.SYSTEM)),
                "performance_metrics": len(self.storage.get_metrics(MetricType.PERFORMANCE))
            },
            "collection_running": self._collection_task is not None and not self._collection_task.done(),
            "latest_system_metrics": latest_system_metrics.to_dict() if latest_system_metrics else None,
            "tracked_models": list(self.model_stats.keys())
        }
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics data."""
        older_than = timedelta(hours=self.auto_cleanup_hours)
        cleared_count = self.storage.clear_old_metrics(older_than)
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old metrics (older than {self.auto_cleanup_hours} hours)")


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    
    if _global_collector is None:
        with _collector_lock:
            if _global_collector is None:
                _global_collector = MetricsCollector()
                # Start collection in the background
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(_global_collector.start_collection())
                except RuntimeError:
                    # No event loop running, will be started when one is available
                    pass
    
    return _global_collector


@contextlib.asynccontextmanager
async def metrics_collector_context():
    """Context manager for metrics collector lifecycle."""
    collector = get_metrics_collector()
    await collector.start_collection()
    try:
        yield collector
    finally:
        await collector.stop_collection()

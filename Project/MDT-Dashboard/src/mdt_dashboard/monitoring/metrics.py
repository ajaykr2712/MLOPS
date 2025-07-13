"""
Advanced metrics collection and monitoring system.
Tracks model performance, system health, and business metrics.
"""

import psutil
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Deque
import logging
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for prediction requests."""
    
    model_name: str
    request_count: int
    prediction_count: int
    processing_time_ms: float
    drift_detected: bool
    average_confidence: float
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "request_count": self.request_count,
            "prediction_count": self.prediction_count,
            "processing_time_ms": self.processing_time_ms,
            "drift_detected": self.drift_detected,
            "average_confidence": self.average_confidence,
            "error_count": self.error_count,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_received: int
    active_threads: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_received": self.network_bytes_received,
            "active_threads": self.active_threads,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking metrics."""
    
    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    drift_score: Optional[float] = None
    data_quality_score: Optional[float] = None
    prediction_latency_ms: Optional[float] = None
    throughput_per_second: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "mse": self.mse,
            "mae": self.mae,
            "r2_score": self.r2_score,
            "drift_score": self.drift_score,
            "data_quality_score": self.data_quality_score,
            "prediction_latency_ms": self.prediction_latency_ms,
            "throughput_per_second": self.throughput_per_second,
            "timestamp": self.timestamp.isoformat()
        }


class MetricsCollector:
    """Thread-safe metrics collection system."""
    
    def __init__(self, max_history: int = 10000, collection_interval: int = 60):
        self.max_history = max_history
        self.collection_interval = collection_interval
        
        # Metrics storage
        self.prediction_metrics: Deque[PredictionMetrics] = deque(maxlen=max_history)
        self.system_metrics: Deque[SystemMetrics] = deque(maxlen=max_history)
        self.performance_metrics: Dict[str, Deque[ModelPerformanceMetrics]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        
        # Aggregated metrics
        self.model_stats = defaultdict(lambda: {
            "total_requests": 0,
            "total_predictions": 0,
            "total_errors": 0,
            "avg_processing_time": 0.0,
            "drift_detection_count": 0,
            "last_seen": None
        })
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background collection
        self.collection_thread = None
        self.stop_collection = threading.Event()
        
        # Network baseline
        self.network_baseline = self._get_network_stats()
        
        logger.info("Metrics collector initialized")
    
    def start_collection(self) -> None:
        """Start background metrics collection."""
        if self.collection_thread is None or not self.collection_thread.is_alive():
            self.stop_collection.clear()
            self.collection_thread = threading.Thread(target=self._collect_system_metrics_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            logger.info("Started background metrics collection")
    
    def stop_collection(self) -> None:
        """Stop background metrics collection."""
        if self.collection_thread and self.collection_thread.is_alive():
            self.stop_collection.set()
            self.collection_thread.join()
            logger.info("Stopped background metrics collection")
    
    def _get_network_stats(self) -> Dict[str, int]:
        """Get current network statistics."""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv
            }
        except Exception:
            return {"bytes_sent": 0, "bytes_recv": 0}
    
    def _collect_system_metrics_loop(self) -> None:
        """Background system metrics collection loop."""
        while not self.stop_collection.wait(self.collection_interval):
            try:
                self.collect_system_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            memory_usage_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network stats
            current_network = self._get_network_stats()
            network_bytes_sent = current_network["bytes_sent"] - self.network_baseline["bytes_sent"]
            network_bytes_received = current_network["bytes_recv"] - self.network_baseline["bytes_recv"]
            
            # Thread count
            active_threads = threading.active_count()
            
            metrics = SystemMetrics(
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_usage_percent,
                memory_usage_mb=memory_usage_mb,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=max(0, network_bytes_sent),
                network_bytes_received=max(0, network_bytes_received),
                active_threads=active_threads
            )
            
            with self.lock:
                self.system_metrics.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def record_prediction_metrics(self, metrics: PredictionMetrics) -> None:
        """Record prediction metrics."""
        with self.lock:
            self.prediction_metrics.append(metrics)
            
            # Update aggregated stats
            stats = self.model_stats[metrics.model_name]
            stats["total_requests"] += metrics.request_count
            stats["total_predictions"] += metrics.prediction_count
            stats["total_errors"] += metrics.error_count
            
            # Update average processing time
            total_requests = stats["total_requests"]
            current_avg = stats["avg_processing_time"]
            new_avg = ((current_avg * (total_requests - metrics.request_count)) + 
                      (metrics.processing_time_ms * metrics.request_count)) / total_requests
            stats["avg_processing_time"] = new_avg
            
            if metrics.drift_detected:
                stats["drift_detection_count"] += 1
            
            stats["last_seen"] = metrics.timestamp.isoformat()
    
    def record_performance_metrics(self, metrics: ModelPerformanceMetrics) -> None:
        """Record model performance metrics."""
        with self.lock:
            self.performance_metrics[metrics.model_name].append(metrics)
    
    def get_prediction_metrics_summary(
        self,
        model_name: Optional[str] = None,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get summary of prediction metrics."""
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self.lock:
            # Filter metrics by time and model
            filtered_metrics = []
            for metric in self.prediction_metrics:
                if metric.timestamp >= cutoff_time:
                    if model_name is None or metric.model_name == model_name:
                        filtered_metrics.append(metric)
            
            if not filtered_metrics:
                return {
                    "total_requests": 0,
                    "total_predictions": 0,
                    "avg_processing_time_ms": 0.0,
                    "error_rate": 0.0,
                    "drift_detection_rate": 0.0,
                    "avg_confidence": 0.0,
                    "time_window_minutes": time_window_minutes
                }
            
            # Calculate summary statistics
            total_requests = sum(m.request_count for m in filtered_metrics)
            total_predictions = sum(m.prediction_count for m in filtered_metrics)
            total_errors = sum(m.error_count for m in filtered_metrics)
            
            processing_times = [m.processing_time_ms for m in filtered_metrics if m.processing_time_ms > 0]
            avg_processing_time = np.mean(processing_times) if processing_times else 0.0
            
            drift_detections = sum(1 for m in filtered_metrics if m.drift_detected)
            drift_rate = drift_detections / len(filtered_metrics) if filtered_metrics else 0.0
            
            confidences = [m.average_confidence for m in filtered_metrics if m.average_confidence > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            error_rate = total_errors / total_requests if total_requests > 0 else 0.0
            
            return {
                "total_requests": total_requests,
                "total_predictions": total_predictions,
                "avg_processing_time_ms": avg_processing_time,
                "error_rate": error_rate,
                "drift_detection_rate": drift_rate,
                "avg_confidence": avg_confidence,
                "time_window_minutes": time_window_minutes,
                "requests_per_minute": total_requests / time_window_minutes if time_window_minutes > 0 else 0
            }
    
    def get_system_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of system metrics."""
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self.lock:
            # Filter metrics by time
            filtered_metrics = [
                metric for metric in self.system_metrics 
                if metric.timestamp >= cutoff_time
            ]
            
            if not filtered_metrics:
                return {
                    "avg_cpu_usage": 0.0,
                    "avg_memory_usage": 0.0,
                    "max_memory_usage_mb": 0.0,
                    "avg_disk_usage": 0.0,
                    "total_network_sent_mb": 0.0,
                    "total_network_received_mb": 0.0,
                    "avg_active_threads": 0.0,
                    "time_window_minutes": time_window_minutes
                }
            
            # Calculate summary statistics
            cpu_usages = [m.cpu_usage_percent for m in filtered_metrics]
            memory_usages = [m.memory_usage_percent for m in filtered_metrics]
            memory_usage_mbs = [m.memory_usage_mb for m in filtered_metrics]
            disk_usages = [m.disk_usage_percent for m in filtered_metrics]
            thread_counts = [m.active_threads for m in filtered_metrics]
            
            total_network_sent = sum(m.network_bytes_sent for m in filtered_metrics)
            total_network_received = sum(m.network_bytes_received for m in filtered_metrics)
            
            return {
                "avg_cpu_usage": np.mean(cpu_usages),
                "max_cpu_usage": np.max(cpu_usages),
                "avg_memory_usage": np.mean(memory_usages),
                "max_memory_usage": np.max(memory_usages),
                "max_memory_usage_mb": np.max(memory_usage_mbs),
                "avg_disk_usage": np.mean(disk_usages),
                "total_network_sent_mb": total_network_sent / (1024 * 1024),
                "total_network_received_mb": total_network_received / (1024 * 1024),
                "avg_active_threads": np.mean(thread_counts),
                "max_active_threads": np.max(thread_counts),
                "time_window_minutes": time_window_minutes
            }
    
    def get_model_performance_summary(self, model_name: str) -> Dict[str, Any]:
        """Get performance summary for a specific model."""
        
        with self.lock:
            if model_name not in self.performance_metrics:
                return {"model_name": model_name, "metrics_available": False}
            
            metrics_list = list(self.performance_metrics[model_name])
            if not metrics_list:
                return {"model_name": model_name, "metrics_available": False}
            
            # Get latest metrics
            latest_metrics = metrics_list[-1]
            
            # Calculate trends (if enough data points)
            trend_data = {}
            if len(metrics_list) >= 2:
                prev_metrics = metrics_list[-2]
                
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
                "total_data_points": len(metrics_list)
            }
    
    def get_all_model_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for all models."""
        with self.lock:
            return dict(self.model_stats)
    
    def export_metrics(
        self,
        filepath: Path,
        time_window_hours: int = 24,
        include_raw_data: bool = False
    ) -> None:
        """Export metrics to file."""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        with self.lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "time_window_hours": time_window_hours,
                "summary": {
                    "prediction_metrics": self.get_prediction_metrics_summary(time_window_minutes=time_window_hours*60),
                    "system_metrics": self.get_system_metrics_summary(time_window_minutes=time_window_hours*60),
                    "model_stats": self.get_all_model_stats()
                }
            }
            
            if include_raw_data:
                # Include raw metrics data
                filtered_prediction_metrics = [
                    m.to_dict() for m in self.prediction_metrics 
                    if m.timestamp >= cutoff_time
                ]
                
                filtered_system_metrics = [
                    m.to_dict() for m in self.system_metrics 
                    if m.timestamp >= cutoff_time
                ]
                
                filtered_performance_metrics = {}
                for model_name, metrics_list in self.performance_metrics.items():
                    filtered_performance_metrics[model_name] = [
                        m.to_dict() for m in metrics_list 
                        if m.timestamp >= cutoff_time
                    ]
                
                export_data["raw_data"] = {
                    "prediction_metrics": filtered_prediction_metrics,
                    "system_metrics": filtered_system_metrics,
                    "performance_metrics": filtered_performance_metrics
                }
            
            # Write to file
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of metrics collection."""
        
        with self.lock:
            latest_system_metrics = self.system_metrics[-1] if self.system_metrics else None
            
            return {
                "status": "healthy" if latest_system_metrics else "no_data",
                "metrics_count": {
                    "prediction_metrics": len(self.prediction_metrics),
                    "system_metrics": len(self.system_metrics),
                    "performance_metrics": sum(len(v) for v in self.performance_metrics.values())
                },
                "collection_running": self.collection_thread.is_alive() if self.collection_thread else False,
                "latest_system_metrics": latest_system_metrics.to_dict() if latest_system_metrics else None,
                "tracked_models": list(self.model_stats.keys())
            }
    
    def clear_metrics(self, older_than_hours: int = 24) -> None:
        """Clear old metrics data."""
        
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self.lock:
            # Clear prediction metrics
            while self.prediction_metrics and self.prediction_metrics[0].timestamp < cutoff_time:
                self.prediction_metrics.popleft()
            
            # Clear system metrics
            while self.system_metrics and self.system_metrics[0].timestamp < cutoff_time:
                self.system_metrics.popleft()
            
            # Clear performance metrics
            for model_name, metrics_list in self.performance_metrics.items():
                while metrics_list and metrics_list[0].timestamp < cutoff_time:
                    metrics_list.popleft()
            
            logger.info(f"Cleared metrics older than {older_than_hours} hours")


# Global metrics collector instance
_global_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
        _global_collector.start_collection()
    return _global_collector

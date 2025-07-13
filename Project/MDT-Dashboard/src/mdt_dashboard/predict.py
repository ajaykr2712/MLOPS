"""
Advanced ML prediction service with drift detection and monitoring.
Provides real-time predictions with performance tracking and anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import joblib
from pathlib import Path
import json
import warnings
import time
from threading import Lock
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Custom imports
from .data_processing import ComprehensiveDataProcessor
from .drift_detection.algorithms import MultivariateDriftDetector, DriftResult
from .monitoring.metrics import MetricsCollector, PredictionMetrics

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Structure for prediction requests."""
    
    data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
    model_name: Optional[str] = None
    return_probabilities: bool = False
    return_feature_importance: bool = False
    track_metrics: bool = True
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data if isinstance(self.data, (dict, list)) else self.data.to_dict(),
            "model_name": self.model_name,
            "return_probabilities": self.return_probabilities,
            "return_feature_importance": self.return_feature_importance,
            "track_metrics": self.track_metrics,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PredictionResponse:
    """Structure for prediction responses."""
    
    predictions: Union[List[float], List[int], np.ndarray]
    probabilities: Optional[Union[List[List[float]], np.ndarray]] = None
    feature_importance: Optional[Dict[str, float]] = None
    drift_detected: bool = False
    drift_details: Optional[List[DriftResult]] = None
    confidence_scores: Optional[List[float]] = None
    anomaly_scores: Optional[List[float]] = None
    processing_time_ms: float = 0.0
    model_version: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predictions": self.predictions.tolist() if isinstance(self.predictions, np.ndarray) else self.predictions,
            "probabilities": self.probabilities.tolist() if isinstance(self.probabilities, np.ndarray) else self.probabilities,
            "feature_importance": self.feature_importance,
            "drift_detected": self.drift_detected,
            "drift_details": [dr.to_dict() for dr in self.drift_details] if self.drift_details else None,
            "confidence_scores": self.confidence_scores,
            "anomaly_scores": self.anomaly_scores,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


class ModelCache:
    """Thread-safe model cache with LRU eviction."""
    
    def __init__(self, max_models: int = 10):
        self.max_models = max_models
        self.cache = {}
        self.access_times = {}
        self.lock = Lock()
    
    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache."""
        with self.lock:
            if model_name in self.cache:
                self.access_times[model_name] = time.time()
                return self.cache[model_name]
            return None
    
    def put(self, model_name: str, model: Any) -> None:
        """Put model in cache."""
        with self.lock:
            # Evict oldest if cache is full
            if len(self.cache) >= self.max_models and model_name not in self.cache:
                oldest_model = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_model]
                del self.access_times[oldest_model]
            
            self.cache[model_name] = model
            self.access_times[model_name] = time.time()
    
    def evict(self, model_name: str) -> None:
        """Evict model from cache."""
        with self.lock:
            if model_name in self.cache:
                del self.cache[model_name]
                del self.access_times[model_name]
    
    def clear(self) -> None:
        """Clear all models from cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


class ReferenceDataStore:
    """Store and manage reference data for drift detection."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.reference_data = {}
        self.lock = Lock()
    
    def set_reference_data(self, model_name: str, data: pd.DataFrame) -> None:
        """Set reference data for a model."""
        with self.lock:
            # Sample data if too large
            if len(data) > self.max_samples:
                data = data.sample(n=self.max_samples, random_state=42)
            
            self.reference_data[model_name] = data.copy()
            logger.info(f"Reference data set for {model_name}: {len(data)} samples")
    
    def get_reference_data(self, model_name: str) -> Optional[pd.DataFrame]:
        """Get reference data for a model."""
        with self.lock:
            return self.reference_data.get(model_name)
    
    def update_reference_data(self, model_name: str, new_data: pd.DataFrame, blend_ratio: float = 0.1) -> None:
        """Update reference data with new samples."""
        with self.lock:
            if model_name not in self.reference_data:
                self.set_reference_data(model_name, new_data)
                return
            
            current_data = self.reference_data[model_name]
            
            # Blend new data with existing reference data
            n_new_samples = int(len(current_data) * blend_ratio)
            if len(new_data) >= n_new_samples:
                new_samples = new_data.sample(n=n_new_samples, random_state=42)
                
                # Remove oldest samples and add new ones
                updated_data = pd.concat([
                    current_data.iloc[n_new_samples:],
                    new_samples
                ], ignore_index=True)
                
                self.reference_data[model_name] = updated_data
                logger.info(f"Reference data updated for {model_name}: {n_new_samples} new samples")


class PredictionService:
    """Advanced prediction service with monitoring and drift detection."""
    
    def __init__(
        self,
        model_registry_path: Union[str, Path] = "models",
        enable_drift_detection: bool = True,
        enable_monitoring: bool = True,
        max_cached_models: int = 10
    ):
        self.model_registry_path = Path(model_registry_path)
        self.enable_drift_detection = enable_drift_detection
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.model_cache = ModelCache(max_cached_models)
        self.reference_store = ReferenceDataStore()
        self.drift_detector = MultivariateDriftDetector() if enable_drift_detection else None
        self.metrics_collector = MetricsCollector() if enable_monitoring else None
        
        # Data processors cache
        self.processors = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Prediction service initialized")
    
    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model and metadata."""
        
        # Check cache first
        cached_model = self.model_cache.get(model_name)
        if cached_model is not None:
            return cached_model
        
        # Load from disk
        model_path = self.model_registry_path / f"{model_name}.joblib"
        metadata_path = self.model_registry_path / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        # Cache model
        model_info = (model, metadata)
        self.model_cache.put(model_name, model_info)
        
        return model_info
    
    def load_data_processor(self, model_name: str) -> Optional[ComprehensiveDataProcessor]:
        """Load data processor for a model."""
        
        if model_name in self.processors:
            return self.processors[model_name]
        
        processor_path = self.model_registry_path / f"{model_name}_processor.joblib"
        
        if processor_path.exists():
            try:
                processor = ComprehensiveDataProcessor.load(processor_path)
                self.processors[model_name] = processor
                return processor
            except Exception as e:
                logger.warning(f"Failed to load processor for {model_name}: {e}")
        
        return None
    
    def preprocess_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        model_name: str
    ) -> pd.DataFrame:
        """Preprocess input data."""
        
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Load and apply data processor
        processor = self.load_data_processor(model_name)
        if processor is not None:
            try:
                df = processor.transform(df)
            except Exception as e:
                logger.warning(f"Failed to apply preprocessing for {model_name}: {e}")
        
        return df
    
    def detect_drift(self, model_name: str, input_data: pd.DataFrame) -> Tuple[bool, List[DriftResult]]:
        """Detect drift in input data."""
        
        if not self.enable_drift_detection or self.drift_detector is None:
            return False, []
        
        reference_data = self.reference_store.get_reference_data(model_name)
        if reference_data is None:
            logger.warning(f"No reference data available for drift detection: {model_name}")
            return False, []
        
        try:
            # Align columns
            common_columns = list(set(reference_data.columns) & set(input_data.columns))
            if not common_columns:
                logger.warning("No common columns for drift detection")
                return False, []
            
            ref_aligned = reference_data[common_columns]
            input_aligned = input_data[common_columns]
            
            # Detect drift
            drift_results = self.drift_detector.detect_multivariate(ref_aligned, input_aligned)
            
            # Aggregate results
            all_results = []
            drift_detected = False
            
            for feature, feature_results in drift_results.items():
                all_results.extend(feature_results)
                if any(result.is_drift for result in feature_results):
                    drift_detected = True
            
            return drift_detected, all_results
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return False, []
    
    def calculate_confidence_scores(
        self,
        model: Any,
        input_data: pd.DataFrame,
        predictions: np.ndarray
    ) -> List[float]:
        """Calculate confidence scores for predictions."""
        
        try:
            if hasattr(model, "predict_proba"):
                # For classification models
                probabilities = model.predict_proba(input_data)
                confidence_scores = np.max(probabilities, axis=1).tolist()
            elif hasattr(model, "decision_function"):
                # For SVM models
                decision_scores = model.decision_function(input_data)
                if decision_scores.ndim > 1:
                    confidence_scores = np.max(np.abs(decision_scores), axis=1).tolist()
                else:
                    confidence_scores = np.abs(decision_scores).tolist()
            else:
                # For regression models, use prediction variance if available
                confidence_scores = [1.0] * len(predictions)
            
            return confidence_scores
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence scores: {e}")
            return [1.0] * len(predictions)
    
    async def predict_async(self, request: PredictionRequest) -> PredictionResponse:
        """Async prediction with monitoring and drift detection."""
        
        start_time = time.time()
        
        try:
            # Default model name
            model_name = request.model_name or "best_model"
            
            # Load model
            model, metadata = self.load_model(model_name)
            
            # Preprocess data
            input_data = self.preprocess_data(request.data, model_name)
            
            if input_data.empty:
                raise ValueError("No valid data after preprocessing")
            
            # Make predictions
            predictions = model.predict(input_data)
            
            # Get probabilities if requested and available
            probabilities = None
            if request.return_probabilities and hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)
            
            # Calculate confidence scores
            confidence_scores = self.calculate_confidence_scores(model, input_data, predictions)
            
            # Feature importance
            feature_importance = None
            if request.return_feature_importance:
                if hasattr(model, "feature_importances_"):
                    feature_importance = dict(zip(input_data.columns, model.feature_importances_))
                elif hasattr(model, "coef_"):
                    feature_importance = dict(zip(input_data.columns, model.coef_.flatten()))
            
            # Drift detection
            drift_detected, drift_details = self.detect_drift(model_name, input_data)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = PredictionResponse(
                predictions=predictions,
                probabilities=probabilities,
                feature_importance=feature_importance,
                drift_detected=drift_detected,
                drift_details=drift_details,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time_ms,
                model_version=metadata.get("model_name", model_name),
                request_id=request.request_id
            )
            
            # Track metrics
            if request.track_metrics and self.enable_monitoring and self.metrics_collector:
                metrics = PredictionMetrics(
                    model_name=model_name,
                    request_count=1,
                    prediction_count=len(predictions),
                    processing_time_ms=processing_time_ms,
                    drift_detected=drift_detected,
                    average_confidence=np.mean(confidence_scores) if confidence_scores else 0.0,
                    timestamp=datetime.now()
                )
                self.metrics_collector.record_prediction_metrics(metrics)
            
            # Update reference data (async)
            if self.enable_drift_detection:
                self.executor.submit(self.reference_store.update_reference_data, model_name, input_data)
            
            return response
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Prediction failed: {e}")
            
            # Return error response
            return PredictionResponse(
                predictions=[],
                processing_time_ms=processing_time_ms,
                request_id=request.request_id
            )
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Synchronous prediction wrapper."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.predict_async(request))
        finally:
            loop.close()
    
    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Batch prediction processing."""
        
        async def process_batch():
            tasks = [self.predict_async(request) for request in requests]
            return await asyncio.gather(*tasks)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(process_batch())
        finally:
            loop.close()
    
    def set_reference_data(self, model_name: str, reference_data: pd.DataFrame) -> None:
        """Set reference data for drift detection."""
        if self.enable_drift_detection:
            self.reference_store.set_reference_data(model_name, reference_data)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        
        try:
            model, metadata = self.load_model(model_name)
            
            return {
                "model_name": model_name,
                "algorithm": metadata.get("algorithm", "unknown"),
                "model_type": type(model).__name__,
                "features": metadata.get("feature_names", []),
                "performance": metadata.get("test_score", "unknown"),
                "training_date": metadata.get("timestamp", "unknown"),
                "has_reference_data": self.reference_store.get_reference_data(model_name) is not None,
                "cached": self.model_cache.get(model_name) is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {"error": str(e)}
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        
        models = []
        
        for model_file in self.model_registry_path.glob("*.joblib"):
            model_name = model_file.stem
            if not model_name.endswith("_processor"):
                try:
                    model_info = self.get_model_info(model_name)
                    models.append(model_info)
                except Exception as e:
                    logger.warning(f"Failed to get info for {model_name}: {e}")
        
        return sorted(models, key=lambda x: x.get("training_date", ""), reverse=True)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        
        return {
            "status": "healthy",
            "models_cached": len(self.model_cache.cache),
            "drift_detection_enabled": self.enable_drift_detection,
            "monitoring_enabled": self.enable_monitoring,
            "available_models": len(list(self.model_registry_path.glob("*.joblib"))),
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        
        logger.info("Shutting down prediction service...")
        
        # Clear caches
        self.model_cache.clear()
        self.processors.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Prediction service shutdown complete")


# Convenience functions
def create_prediction_request(
    data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    model_name: Optional[str] = None,
    **kwargs
) -> PredictionRequest:
    """Create a prediction request."""
    return PredictionRequest(data=data, model_name=model_name, **kwargs)


def quick_predict(
    data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    model_name: str = "best_model",
    service: Optional[PredictionService] = None
) -> PredictionResponse:
    """Quick prediction for simple use cases."""
    
    if service is None:
        service = PredictionService()
    
    request = create_prediction_request(data, model_name)
    return service.predict(request)
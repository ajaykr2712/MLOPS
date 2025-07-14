"""
MDT Dashboard - Model Drift Detection & Telemetry Platform
Enterprise-grade solution for automated model performance monitoring.
"""

__version__ = "1.0.0"
__author__ = "Ajay kumar Pondugala"
__email__ = "aponduga@cisco.com"
__description__ = "Enterprise ML monitoring platform with advanced drift detection"
__license__ = "MIT"

from .core.config import settings
from .predict import PredictionService, PredictionRequest, PredictionResponse
from .train import ModelTrainer
from .data_processing import ComprehensiveDataProcessor
from .drift_detection import DriftDetectionSuite
from .monitoring.metrics import MetricsCollector
from .monitoring.alerts import AlertManager

# Quick API access
def create_prediction_service(config=None):
    """Create a configured prediction service."""
    return PredictionService()

def create_model_trainer(config=None):
    """Create a configured model trainer.""" 
    return ModelTrainer()

__all__ = [
    "settings",
    "PredictionService", 
    "PredictionRequest",
    "PredictionResponse",
    "ModelTrainer",
    "ComprehensiveDataProcessor",
    "DriftDetectionSuite", 
    "MetricsCollector",
    "AlertManager",
    "create_prediction_service",
    "create_model_trainer"
]

__all__ = ["settings"]

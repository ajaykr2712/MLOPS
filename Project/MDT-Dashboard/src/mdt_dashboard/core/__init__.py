"""
Core module for MDT Dashboard.
Provides foundational components like configuration, database, and models.
"""

from .config import get_settings, settings
from .database import db_manager, get_db
from .models import (
    Model,
    Prediction,
    DriftReport,
    Alert,
    ModelPerformanceMetric,
    ReferenceData,
    DataQualityReport
)

__all__ = [
    "get_settings",
    "settings", 
    "db_manager",
    "get_db",
    "Model",
    "Prediction",
    "DriftReport",
    "Alert",
    "ModelPerformanceMetric",
    "ReferenceData",
    "DataQualityReport"
]

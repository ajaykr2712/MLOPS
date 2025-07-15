"""
Refactored FastAPI application for MDT Dashboard.
Provides REST API endpoints for model predictions, monitoring, and management.
"""

from __future__ import annotations
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import time
from enum import Enum

# Custom imports
from ..core.config import settings
from ..utils.logging import setup_logging

# Setup logging
setup_logging(settings.monitoring.log_level.value)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


class ProblemType(str, Enum):
    """Machine learning problem types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    AUTO = "auto"


class ResponseStatus(str, Enum):
    """API response status types."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class BaseAPIModel(BaseModel):
    """Base model for API requests/responses."""
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True


class APIResponse(BaseAPIModel):
    """Standard API response wrapper."""
    
    status: ResponseStatus
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

    @classmethod
    def success(
        cls,
        data: Optional[Dict[str, Any]] = None,
        message: str = "Operation completed successfully",
        request_id: Optional[str] = None
    ) -> APIResponse:
        """Create a success response."""
        return cls(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=data,
            request_id=request_id
        )

    @classmethod
    def error(
        cls,
        message: str,
        errors: Optional[List[str]] = None,
        request_id: Optional[str] = None
    ) -> APIResponse:
        """Create an error response."""
        return cls(
            status=ResponseStatus.ERROR,
            message=message,
            errors=errors or [],
            request_id=request_id
        )


class PredictionRequest(BaseAPIModel):
    """Model for prediction requests."""
    
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ..., 
        description="Input data for prediction"
    )
    model_name: str = Field(
        default=settings.default_model_name,
        description="Name of the model to use",
        min_length=1,
        max_length=100
    )
    return_probabilities: bool = Field(
        default=False, 
        description="Return prediction probabilities"
    )
    return_feature_importance: bool = Field(
        default=False, 
        description="Return feature importance"
    )
    track_metrics: bool = Field(
        default=True, 
        description="Track prediction metrics"
    )
    request_id: Optional[str] = Field(
        default=None, 
        description="Unique request identifier",
        max_length=100
    )

    @validator('data')
    def validate_data(cls, v):
        """Validate prediction data format."""
        if isinstance(v, dict):
            if not v:
                raise ValueError("Prediction data cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Prediction data list cannot be empty")
            if len(v) > settings.max_prediction_batch_size:
                raise ValueError(f"Batch size exceeds maximum of {settings.max_prediction_batch_size}")
        return v


class BatchPredictionRequest(BaseAPIModel):
    """Model for batch prediction requests."""
    
    requests: List[PredictionRequest] = Field(
        ...,
        description="List of prediction requests",
        min_items=1,
        max_items=100
    )
    batch_id: Optional[str] = Field(
        default=None,
        description="Unique batch identifier",
        max_length=100
    )


class TrainingRequest(BaseAPIModel):
    """Model for model training requests."""
    
    training_data_path: str = Field(
        ..., 
        description="Path to training data file"
    )
    target_column: str = Field(
        ..., 
        description="Name of the target column",
        min_length=1,
        max_length=100
    )
    problem_type: ProblemType = Field(
        default=ProblemType.AUTO, 
        description="Problem type"
    )
    algorithms: List[str] = Field(
        default=["random_forest", "xgboost"], 
        description="Algorithms to train",
        min_items=1,
        max_items=10
    )
    test_size: float = Field(
        default=0.2, 
        ge=0.1, 
        le=0.5, 
        description="Test set size ratio"
    )
    cv_folds: int = Field(
        default=5, 
        ge=3, 
        le=10, 
        description="Cross-validation folds"
    )
    experiment_name: str = Field(
        default="mdt-api-training", 
        description="MLflow experiment name",
        min_length=1,
        max_length=100
    )

    @validator('algorithms')
    def validate_algorithms(cls, v):
        """Validate algorithm names."""
        supported_algorithms = {
            "random_forest", "xgboost", "lightgbm", 
            "logistic_regression", "linear_regression",
            "svm", "knn", "decision_tree"
        }
        for algo in v:
            if algo not in supported_algorithms:
                raise ValueError(f"Unsupported algorithm: {algo}")
        return v


class HealthResponse(BaseAPIModel):
    """Health check response model."""
    
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    services: Dict[str, str]
    metrics: Dict[str, Any]


class MetricsResponse(BaseAPIModel):
    """System metrics response model."""
    
    predictions_total: int
    predictions_per_minute: float
    avg_latency_ms: float
    error_rate: float
    models_loaded: int
    memory_usage_mb: float
    cpu_usage_percent: float


# Global variables for tracking
app_start_time = time.time()
prediction_count = 0
error_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MDT Dashboard API")
    
    # Initialize services
    try:
        # Initialize prediction service
        # Initialize metrics collector
        # Initialize database connections
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MDT Dashboard API")


# Create FastAPI application
app = FastAPI(
    title="MDT Dashboard API",
    description="Advanced Model Drift Detection and Telemetry Platform",
    version=settings.version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors."""
    global error_count
    error_count += 1
    
    errors = []
    for error in exc.errors():
        errors.append(f"{error['loc'][-1]}: {error['msg']}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=APIResponse.error(
            message="Validation error",
            errors=errors
        ).dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    global error_count
    error_count += 1
    
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse.error(
            message=exc.detail
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    global error_count
    error_count += 1
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=APIResponse.error(
            message="Internal server error"
        ).dict()
    )


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    if not settings.security.enable_auth:
        return {"user_id": "anonymous", "role": "user"}
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Validate JWT token here
    # For now, just return a mock user
    return {"user_id": "test_user", "role": "user"}


def track_prediction():
    """Track prediction metrics."""
    global prediction_count
    prediction_count += 1


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "docs": "/docs" if settings.debug else "disabled",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time
    
    # Check service health
    services = {
        "database": "healthy",  # Add actual health checks
        "redis": "healthy",
        "mlflow": "healthy",
        "prediction_service": "healthy"
    }
    
    metrics = {
        "predictions_total": prediction_count,
        "errors_total": error_count,
        "uptime_seconds": uptime
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.version,
        uptime_seconds=uptime,
        services=services,
        metrics=metrics
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    uptime = time.time() - app_start_time
    predictions_per_minute = prediction_count / (uptime / 60) if uptime > 0 else 0
    error_rate = error_count / prediction_count if prediction_count > 0 else 0
    
    return MetricsResponse(
        predictions_total=prediction_count,
        predictions_per_minute=predictions_per_minute,
        avg_latency_ms=45.0,  # Add actual latency tracking
        error_rate=error_rate,
        models_loaded=1,  # Add actual model count
        memory_usage_mb=512.0,  # Add actual memory usage
        cpu_usage_percent=25.0  # Add actual CPU usage
    )


@app.post("/predict", response_model=APIResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Make a prediction using the specified model."""
    try:
        track_prediction()
        
        # Add background task for metrics tracking
        if request.track_metrics:
            background_tasks.add_task(log_prediction_metrics, request, current_user)
        
        # Mock prediction logic - replace with actual prediction service
        prediction_result = {
            "prediction": 0.85,
            "model_name": request.model_name,
            "confidence": "high",
            "latency_ms": 42,
            "request_id": request.request_id
        }
        
        if request.return_probabilities:
            prediction_result["probabilities"] = [0.15, 0.85]
        
        if request.return_feature_importance:
            prediction_result["feature_importance"] = {
                "feature_1": 0.3,
                "feature_2": 0.7
            }
        
        return APIResponse.success(
            data=prediction_result,
            message="Prediction completed successfully",
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=APIResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Make batch predictions."""
    try:
        results = []
        for pred_request in request.requests:
            track_prediction()
            
            # Mock prediction for each request
            result = {
                "prediction": 0.75,
                "model_name": pred_request.model_name,
                "request_id": pred_request.request_id
            }
            results.append(result)
        
        return APIResponse.success(
            data={
                "batch_id": request.batch_id,
                "results": results,
                "total_predictions": len(results)
            },
            message="Batch prediction completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


async def log_prediction_metrics(request: PredictionRequest, user: dict):
    """Log prediction metrics asynchronously."""
    try:
        # Log metrics to monitoring system
        logger.info(f"Prediction made by user {user['user_id']} for model {request.model_name}")
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_level=settings.monitoring.log_level.value.lower()
    )

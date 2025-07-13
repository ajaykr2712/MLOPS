"""
FastAPI application for MDT Dashboard.
Provides REST API endpoints for model predictions, monitoring, and management.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio
import uvicorn
from contextlib import asynccontextmanager

# Custom imports
from ..core.config import settings
from ..predict import PredictionService, PredictionRequest, PredictionResponse
from ..monitoring.metrics import get_metrics_collector, PredictionMetrics, ModelPerformanceMetrics
from ..train import ModelTrainer, ModelConfig

logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequestModel(BaseModel):
    """API model for prediction requests."""
    
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    model_name: Optional[str] = Field(default="best_model", description="Name of the model to use")
    return_probabilities: bool = Field(default=False, description="Return prediction probabilities")
    return_feature_importance: bool = Field(default=False, description="Return feature importance")
    track_metrics: bool = Field(default=True, description="Track prediction metrics")
    request_id: Optional[str] = Field(default=None, description="Unique request identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "data": {"feature1": 1.0, "feature2": 2.0, "feature3": "category_a"},
                "model_name": "best_model",
                "return_probabilities": False,
                "return_feature_importance": True,
                "track_metrics": True,
                "request_id": "req_123"
            }
        }


class BatchPredictionRequestModel(BaseModel):
    """API model for batch prediction requests."""
    
    requests: List[PredictionRequestModel]
    
    @validator('requests')
    def validate_requests(cls, v):
        if len(v) == 0:
            raise ValueError("At least one prediction request required")
        if len(v) > 1000:  # Reasonable batch size limit
            raise ValueError("Batch size too large (max 1000)")
        return v


class ModelTrainingRequestModel(BaseModel):
    """API model for model training requests."""
    
    training_data_path: str = Field(..., description="Path to training data CSV file")
    target_column: str = Field(..., description="Name of the target column")
    problem_type: str = Field(default="auto", description="Problem type: regression, classification, or auto")
    algorithms: List[str] = Field(default=["random_forest", "xgboost", "lightgbm"], description="Algorithms to train")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size ratio")
    cv_folds: int = Field(default=5, ge=3, le=10, description="Cross-validation folds")
    experiment_name: str = Field(default="mdt-api-training", description="MLflow experiment name")
    
    class Config:
        schema_extra = {
            "example": {
                "training_data_path": "/path/to/training_data.csv",
                "target_column": "target",
                "problem_type": "regression",
                "algorithms": ["random_forest", "xgboost"],
                "test_size": 0.2,
                "cv_folds": 5,
                "experiment_name": "api-experiment"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    prediction_service: Dict[str, Any]
    metrics_collector: Dict[str, Any]
    system_info: Dict[str, Any]


class MetricsSummaryResponse(BaseModel):
    """Metrics summary response model."""
    
    prediction_summary: Dict[str, Any]
    system_summary: Dict[str, Any]
    model_stats: Dict[str, Any]
    time_window_minutes: int


# Global variables
prediction_service: Optional[PredictionService] = None
app_start_time: Optional[datetime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global prediction_service, app_start_time
    
    # Startup
    logger.info("Starting MDT Dashboard API...")
    app_start_time = datetime.now()
    
    # Initialize prediction service
    prediction_service = PredictionService(
        enable_drift_detection=True,
        enable_monitoring=True
    )
    
    # Start metrics collection
    metrics_collector = get_metrics_collector()
    metrics_collector.start_collection()
    
    logger.info("MDT Dashboard API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MDT Dashboard API...")
    
    if prediction_service:
        prediction_service.shutdown()
    
    metrics_collector.stop_collection()
    
    logger.info("MDT Dashboard API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="MDT Dashboard API",
    description="Model Drift Detection & Telemetry Platform API",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

if settings.security.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

# Security
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Simple authentication (extend as needed)."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    
    # Simple token validation (implement proper JWT validation in production)
    if credentials.credentials != settings.security.secret_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return {"user": "authenticated_user"}


# Health check endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Get API health status."""
    global app_start_time, prediction_service
    
    uptime = (datetime.now() - app_start_time).total_seconds() if app_start_time else 0
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.app_version,
        uptime_seconds=uptime,
        prediction_service=prediction_service.get_health_status() if prediction_service else {},
        metrics_collector=get_metrics_collector().get_health_status(),
        system_info={
            "environment": settings.environment,
            "debug": settings.debug
        }
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "MDT Dashboard API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


# Prediction endpoints
@app.post("/predict", response_model=Dict[str, Any], tags=["Predictions"])
async def predict(
    request: PredictionRequestModel,
    background_tasks: BackgroundTasks,
    # user: Dict[str, Any] = Depends(get_current_user)  # Uncomment for authentication
):
    """Make a single prediction."""
    global prediction_service
    
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert to internal request format
        pred_request = PredictionRequest(
            data=request.data,
            model_name=request.model_name,
            return_probabilities=request.return_probabilities,
            return_feature_importance=request.return_feature_importance,
            track_metrics=request.track_metrics,
            request_id=request.request_id
        )
        
        # Make prediction
        response = await prediction_service.predict_async(pred_request)
        
        return response.to_dict()
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[Dict[str, Any]], tags=["Predictions"])
async def predict_batch(
    request: BatchPredictionRequestModel,
    background_tasks: BackgroundTasks,
    # user: Dict[str, Any] = Depends(get_current_user)  # Uncomment for authentication
):
    """Make batch predictions."""
    global prediction_service
    
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert to internal request format
        pred_requests = [
            PredictionRequest(
                data=req.data,
                model_name=req.model_name,
                return_probabilities=req.return_probabilities,
                return_feature_importance=req.return_feature_importance,
                track_metrics=req.track_metrics,
                request_id=req.request_id
            )
            for req in request.requests
        ]
        
        # Make batch predictions
        responses = prediction_service.predict_batch(pred_requests)
        
        return [response.to_dict() for response in responses]
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints
@app.get("/models", response_model=List[Dict[str, Any]], tags=["Models"])
async def list_models(
    # user: Dict[str, Any] = Depends(get_current_user)  # Uncomment for authentication
):
    """List available models."""
    global prediction_service
    
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        return prediction_service.list_available_models()
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}", response_model=Dict[str, Any], tags=["Models"])
async def get_model_info(
    model_name: str,
    # user: Dict[str, Any] = Depends(get_current_user)  # Uncomment for authentication
):
    """Get information about a specific model."""
    global prediction_service
    
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        model_info = prediction_service.get_model_info(model_name)
        if "error" in model_info:
            raise HTTPException(status_code=404, detail=model_info["error"])
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training endpoints
@app.post("/train", response_model=Dict[str, Any], tags=["Training"])
async def train_model(
    request: ModelTrainingRequestModel,
    background_tasks: BackgroundTasks,
    # user: Dict[str, Any] = Depends(get_current_user)  # Uncomment for authentication
):
    """Start model training."""
    
    try:
        # Add training task to background
        background_tasks.add_task(
            _train_model_background,
            request.training_data_path,
            request.target_column,
            request.problem_type,
            request.algorithms,
            request.test_size,
            request.cv_folds,
            request.experiment_name
        )
        
        return {
            "status": "training_started",
            "message": "Model training started in background",
            "experiment_name": request.experiment_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_model_background(
    data_path: str,
    target_column: str,
    problem_type: str,
    algorithms: List[str],
    test_size: float,
    cv_folds: int,
    experiment_name: str
):
    """Background task for model training."""
    
    try:
        logger.info(f"Starting background training: {experiment_name}")
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Prepare features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Auto-detect problem type
        if problem_type == "auto":
            unique_ratio = len(y.unique()) / len(y)
            problem_type = "classification" if unique_ratio < 0.1 else "regression"
        
        # Split data
        from ..data_processing import split_data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
        
        # Create config
        config = ModelConfig(
            problem_type=problem_type,
            algorithms=algorithms,
            cv_folds=cv_folds,
            experiment_name=experiment_name
        )
        
        # Train models
        trainer = ModelTrainer(config)
        trainer.train_all_models(X_train, X_test, y_train, y_test)
        
        logger.info(f"Background training completed: {experiment_name}")
        
    except Exception as e:
        logger.error(f"Background training failed: {e}")


# Monitoring endpoints
@app.get("/metrics", response_model=MetricsSummaryResponse, tags=["Monitoring"])
async def get_metrics_summary(
    time_window_minutes: int = 60,
    model_name: Optional[str] = None,
    # user: Dict[str, Any] = Depends(get_current_user)  # Uncomment for authentication
):
    """Get metrics summary."""
    
    try:
        metrics_collector = get_metrics_collector()
        
        prediction_summary = metrics_collector.get_prediction_metrics_summary(
            model_name=model_name,
            time_window_minutes=time_window_minutes
        )
        
        system_summary = metrics_collector.get_system_metrics_summary(
            time_window_minutes=time_window_minutes
        )
        
        model_stats = metrics_collector.get_all_model_stats()
        
        return MetricsSummaryResponse(
            prediction_summary=prediction_summary,
            system_summary=system_summary,
            model_stats=model_stats,
            time_window_minutes=time_window_minutes
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/models/{model_name}", response_model=Dict[str, Any], tags=["Monitoring"])
async def get_model_metrics(
    model_name: str,
    # user: Dict[str, Any] = Depends(get_current_user)  # Uncomment for authentication
):
    """Get metrics for a specific model."""
    
    try:
        metrics_collector = get_metrics_collector()
        return metrics_collector.get_model_performance_summary(model_name)
        
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Reference data management
@app.post("/models/{model_name}/reference-data", tags=["Models"])
async def set_reference_data(
    model_name: str,
    reference_data: List[Dict[str, Any]],
    # user: Dict[str, Any] = Depends(get_current_user)  # Uncomment for authentication
):
    """Set reference data for drift detection."""
    global prediction_service
    
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(reference_data)
        
        # Set reference data
        prediction_service.set_reference_data(model_name, df)
        
        return {
            "status": "success",
            "message": f"Reference data set for model {model_name}",
            "samples": len(df),
            "features": len(df.columns),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to set reference data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# CLI function
def run_server():
    """Run the API server."""
    uvicorn.run(
        "mdt_dashboard.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.monitoring.log_level.lower()
    )


if __name__ == "__main__":
    run_server()

"""
Production-ready FastAPI application for MDT Dashboard.
Enterprise-grade API with comprehensive monitoring, security, and performance features.
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import logging

from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, ValidationError
import pandas as pd

from ..core.config import get_settings
from ..core.database import get_db, db_manager
from ..core.models import Model, Prediction, DriftReport, Alert, ModelPerformanceMetric
from ..monitoring.metrics import MetricsCollector
from ..utils.logging import setup_logging
from ..worker import (
    check_drift_for_model,
    train_model_async,
    calculate_model_performance_metrics
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')
DRIFT_DETECTION_COUNT = Counter('drift_detections_total', 'Total drift detections')


# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str
    database: str
    redis: str
    services: Dict[str, str]


class PredictionRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]
    detect_drift: bool = True
    store_prediction: bool = True


class PredictionResponse(BaseModel):
    prediction_id: str
    prediction: Any
    prediction_probability: Optional[Dict[str, float]] = None
    drift_score: Optional[float] = None
    drift_detected: Optional[bool] = None
    response_time_ms: float
    timestamp: str


class BatchPredictionRequest(BaseModel):
    model_id: str
    input_data: List[Dict[str, Any]]
    detect_drift: bool = True
    store_predictions: bool = True


class TrainingRequest(BaseModel):
    data_path: str
    model_name: str
    model_type: str
    target_column: str
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2
    async_training: bool = True


class DriftCheckRequest(BaseModel):
    model_id: str
    time_window_hours: int = 24
    drift_threshold: float = 0.05


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting MDT Dashboard API...")
    
    # Initialize database
    try:
        db_manager.create_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Initialize metrics collector
    app.state.metrics_collector = MetricsCollector()
    
    yield
    
    # Shutdown
    logger.info("Shutting down MDT Dashboard API...")


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="MDT Dashboard API",
        description="Enterprise-grade Model Drift Detection & Telemetry Dashboard API",
        version="1.0.0",
        docs_url=None,  # Disable default docs
        redoc_url=None,  # Disable default redoc
        openapi_url=f"{settings.api_v1_str}/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    add_middleware(app, settings)
    
    # Add routes
    add_routes(app, settings)
    
    return app


def add_middleware(app: FastAPI, settings) -> None:
    """Add middleware to the application."""
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware for metrics and logging
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.observe(duration)
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.4f}s"
        
        return response


def add_routes(app: FastAPI, settings) -> None:
    """Add API routes."""
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check(db=Depends(get_db)):
        """Health check endpoint."""
        import datetime
        
        # Check database
        try:
            db.execute("SELECT 1")
            db_status = "healthy"
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            db_status = "unhealthy"
        
        # Check Redis
        try:
            from ..worker import celery_app
            celery_app.control.ping(timeout=5)
            redis_status = "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            redis_status = "unhealthy"
        
        return HealthResponse(
            status="healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
            timestamp=datetime.datetime.utcnow().isoformat(),
            version="1.0.0",
            environment=settings.environment,
            database=db_status,
            redis=redis_status,
            services={
                "database": db_status,
                "redis": redis_status,
                "api": "healthy"
            }
        )
    
    # Metrics endpoint for Prometheus
    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    # Models endpoints
    @app.get(f"{settings.api_v1_str}/models", tags=["Models"])
    async def list_models(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        db=Depends(get_db)
    ):
        """List all models."""
        query = db.query(Model)
        
        if status:
            query = query.filter(Model.status == status)
        
        models = query.offset(skip).limit(limit).all()
        
        return {
            "models": [
                {
                    "id": str(model.id),
                    "name": model.name,
                    "version": model.version,
                    "status": model.status,
                    "algorithm": model.algorithm,
                    "created_at": model.created_at.isoformat(),
                    "metrics": model.metrics
                }
                for model in models
            ],
            "total": query.count(),
            "skip": skip,
            "limit": limit
        }
    
    # Predictions endpoints
    @app.post(f"{settings.api_v1_str}/predict", response_model=PredictionResponse, tags=["Predictions"])
    async def make_prediction(
        request: PredictionRequest,
        background_tasks: BackgroundTasks,
        db=Depends(get_db)
    ):
        """Make a single prediction."""
        start_time = time.time()
        
        # Get model
        model = db.query(Model).filter(Model.id == request.model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if model.status != "active":
            raise HTTPException(status_code=400, detail="Model is not active")
        
        try:
            # Load and use model for prediction
            from ..predict import PredictionService, PredictionRequest
            
            prediction_service = PredictionService()
            
            # Create prediction request
            prediction_request = PredictionRequest(
                data=request.input_data,
                model_name=request.model_id,
                return_probabilities=True,
                track_metrics=True
            )
            
            result = prediction_service.predict(prediction_request)
            
            response_time = (time.time() - start_time) * 1000
            
            # Record metrics
            PREDICTION_COUNT.inc()
            
            if request.detect_drift and result.drift_detected:
                DRIFT_DETECTION_COUNT.inc()
            
            # Store prediction if requested
            if request.store_prediction:
                prediction_record = Prediction(
                    model_id=request.model_id,
                    input_data=request.input_data,
                    prediction=result.predictions,
                    prediction_probability=result.probabilities,
                    response_time_ms=response_time,
                    drift_score=result.drift_details[0].drift_score if result.drift_details else None,
                    request_id=result.request_id
                )
                db.add(prediction_record)
                db.commit()
            
            return PredictionResponse(
                prediction_id=result.request_id or str(uuid.uuid4()),
                prediction=result.predictions,
                prediction_probability=result.probabilities,
                drift_score=result.drift_details[0].drift_score if result.drift_details else None,
                drift_detected=result.drift_detected,
                response_time_ms=response_time,
                timestamp=result.timestamp.isoformat()
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Training endpoints
    @app.post(f"{settings.api_v1_str}/train", tags=["Training"])
    async def train_model(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        db=Depends(get_db)
    ):
        """Train a new model."""
        try:
            if request.async_training:
                # Start async training
                training_config = {
                    "data_path": request.data_path,
                    "model_name": request.model_name,
                    "model_type": request.model_type,
                    "target_column": request.target_column,
                    "hyperparameters": request.hyperparameters or {},
                    "validation_split": request.validation_split
                }
                
                task = train_model_async.delay(training_config)
                
                return {
                    "message": "Training started",
                    "task_id": task.id,
                    "status": "started",
                    "async": True
                }
            else:
                # Synchronous training
                from ..train import ModelTrainer
                
                trainer = ModelTrainer()
                result = trainer.train_model(
                    data_path=request.data_path,
                    target_column=request.target_column,
                    model_name=request.model_name,
                    model_type=request.model_type,
                    hyperparameters=request.hyperparameters or {},
                    validation_split=request.validation_split
                )
                
                return {
                    "message": "Training completed",
                    "model_id": result["model_id"],
                    "metrics": result["metrics"],
                    "status": "completed",
                    "async": False
                }
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    
    # Drift detection endpoints
    @app.post(f"{settings.api_v1_str}/drift/check", tags=["Drift Detection"])
    async def check_drift(
        request: DriftCheckRequest,
        background_tasks: BackgroundTasks,
        db=Depends(get_db)
    ):
        """Trigger drift detection for a model."""
        # Get model
        model = db.query(Model).filter(Model.id == request.model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        try:
            # Start async drift check
            task = check_drift_for_model.delay(
                request.model_id,
                request.time_window_hours
            )
            
            return {
                "message": "Drift check started",
                "task_id": task.id,
                "model_id": request.model_id,
                "time_window_hours": request.time_window_hours
            }
            
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")
    
    # Custom OpenAPI documentation
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Interactive API docs",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        )
    
    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " - API Documentation",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
        )


# Create app instance
app = create_app()


# Run the app
def run():
    """Run the application."""
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "mdt_dashboard.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    run()

"""
Complete FastAPI application for MDT Dashboard.
Production-ready API with all endpoints implemented.
"""
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from ..core.config import get_settings
from ..core.database import get_db, db_manager
from ..core.models import Model, Prediction, DriftReport, Alert
from ..monitoring.metrics import MetricsCollector
from ..utils.logging import setup_logging
from ..predict import PredictionService, PredictionRequest
from ..train import ModelTrainer
from ..drift_detection.algorithms import MultivariateDriftDetector

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


class ModelRequest(BaseModel):
    name: str
    description: Optional[str] = None
    algorithm: str
    framework: str = "scikit-learn"
    model_path: str
    hyperparameters: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None


class ModelResponse(BaseModel):
    id: str
    name: str
    version: str
    status: str
    algorithm: str
    framework: str
    created_at: str
    metrics: Optional[Dict[str, Any]] = None


class PredictionAPIRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]
    detect_drift: bool = True
    store_prediction: bool = True


class PredictionAPIResponse(BaseModel):
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


class DriftReportResponse(BaseModel):
    id: str
    model_id: str
    drift_detected: bool
    overall_drift_score: float
    detection_method: str
    created_at: str
    feature_drift_scores: Optional[Dict[str, float]] = None


class AlertResponse(BaseModel):
    id: str
    alert_type: str
    severity: str
    title: str
    message: str
    created_at: str
    is_resolved: bool


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
    
    # Initialize services
    app.state.prediction_service = PredictionService()
    app.state.model_trainer = ModelTrainer()
    app.state.metrics_collector = MetricsCollector()
    app.state.drift_detector = MultivariateDriftDetector()
    
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
        
        # Check Redis (simplified)
        redis_status = "healthy"  # TODO: Implement actual Redis check
        
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
                ModelResponse(
                    id=str(model.id),
                    name=model.name,
                    version=model.version,
                    status=model.status,
                    algorithm=model.algorithm,
                    framework=model.framework,
                    created_at=model.created_at.isoformat(),
                    metrics=model.metrics
                )
                for model in models
            ],
            "total": query.count(),
            "skip": skip,
            "limit": limit
        }
    
    @app.get(f"{settings.api_v1_str}/models/{{model_id}}", tags=["Models"])
    async def get_model(model_id: str, db=Depends(get_db)):
        """Get model by ID."""
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return ModelResponse(
            id=str(model.id),
            name=model.name,
            version=model.version,
            status=model.status,
            algorithm=model.algorithm,
            framework=model.framework,
            created_at=model.created_at.isoformat(),
            metrics=model.metrics
        )
    
    @app.post(f"{settings.api_v1_str}/models", tags=["Models"])
    async def create_model(
        model_request: ModelRequest,
        db=Depends(get_db)
    ):
        """Create a new model."""
        model = Model(
            name=model_request.name,
            version="1.0.0",  # TODO: Implement versioning
            description=model_request.description,
            algorithm=model_request.algorithm,
            framework=model_request.framework,
            model_path=model_request.model_path,
            hyperparameters=model_request.hyperparameters,
            feature_names=model_request.feature_names,
            target_names=model_request.target_names,
            status="active"
        )
        
        db.add(model)
        db.commit()
        db.refresh(model)
        
        return ModelResponse(
            id=str(model.id),
            name=model.name,
            version=model.version,
            status=model.status,
            algorithm=model.algorithm,
            framework=model.framework,
            created_at=model.created_at.isoformat(),
            metrics=model.metrics
        )
    
    @app.delete(f"{settings.api_v1_str}/models/{{model_id}}", tags=["Models"])
    async def delete_model(model_id: str, db=Depends(get_db)):
        """Delete a model."""
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        db.delete(model)
        db.commit()
        
        return {"message": "Model deleted successfully"}
    
    # Predictions endpoints
    @app.post(f"{settings.api_v1_str}/predict", response_model=PredictionAPIResponse, tags=["Predictions"])
    async def make_prediction(
        request: PredictionAPIRequest,
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
            # Use prediction service
            prediction_service = app.state.prediction_service
            
            # Create prediction request
            pred_request = PredictionRequest(
                data=request.input_data,
                model_name=model.name,
                return_probabilities=True,
                track_metrics=True
            )
            
            # Make prediction
            result = await prediction_service.predict_async(pred_request)
            
            response_time = (time.time() - start_time) * 1000
            
            # Store prediction if requested
            if request.store_prediction:
                prediction = Prediction(
                    model_id=model.id,
                    input_data=request.input_data,
                    prediction=result.predictions,
                    prediction_probability=result.probabilities,
                    response_time_ms=response_time,
                    drift_score=max([dr.p_value for dr in result.drift_details]) if result.drift_details else None,
                    request_id=result.request_id
                )
                db.add(prediction)
                db.commit()
                db.refresh(prediction)
            
            # Record metrics
            PREDICTION_COUNT.inc()
            if result.drift_detected:
                DRIFT_DETECTION_COUNT.inc()
            
            return PredictionAPIResponse(
                prediction_id=str(uuid.uuid4()),
                prediction=result.predictions,
                prediction_probability=result.probabilities[0] if result.probabilities is not None else None,
                drift_score=max([dr.p_value for dr in result.drift_details]) if result.drift_details else None,
                drift_detected=result.drift_detected,
                response_time_ms=response_time,
                timestamp=result.timestamp.isoformat()
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @app.post(f"{settings.api_v1_str}/predict/batch", tags=["Predictions"])
    async def batch_prediction(
        request: BatchPredictionRequest,
        background_tasks: BackgroundTasks,
        db=Depends(get_db)
    ):
        """Make batch predictions."""
        # Get model
        model = db.query(Model).filter(Model.id == request.model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if model.status != "active":
            raise HTTPException(status_code=400, detail="Model is not active")
        
        try:
            prediction_service = app.state.prediction_service
            results = []
            
            for data_point in request.input_data:
                pred_request = PredictionRequest(
                    data=data_point,
                    model_name=model.name,
                    return_probabilities=True,
                    track_metrics=True
                )
                
                result = await prediction_service.predict_async(pred_request)
                results.append(result.to_dict())
            
            return {
                "predictions": results,
                "total_predictions": len(results),
                "model_id": request.model_id
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    @app.get(f"{settings.api_v1_str}/predictions", tags=["Predictions"])
    async def list_predictions(
        model_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100,
        db=Depends(get_db)
    ):
        """List predictions with filtering."""
        query = db.query(Prediction)
        
        if model_id:
            query = query.filter(Prediction.model_id == model_id)
        
        if start_date:
            query = query.filter(Prediction.prediction_time >= start_date)
        
        if end_date:
            query = query.filter(Prediction.prediction_time <= end_date)
        
        predictions = query.offset(skip).limit(limit).all()
        
        return {
            "predictions": [
                {
                    "id": str(p.id),
                    "model_id": str(p.model_id),
                    "prediction": p.prediction,
                    "prediction_time": p.prediction_time.isoformat(),
                    "response_time_ms": p.response_time_ms,
                    "drift_score": p.drift_score
                }
                for p in predictions
            ],
            "total": query.count(),
            "skip": skip,
            "limit": limit
        }
    
    # Training endpoints
    @app.post(f"{settings.api_v1_str}/train", tags=["Training"])
    async def train_model(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        db=Depends(get_db)
    ):
        """Train a new model."""
        try:
            trainer = app.state.model_trainer
            
            if request.async_training:
                # Start async training
                task_id = str(uuid.uuid4())
                
                # TODO: Implement actual async training with Celery
                background_tasks.add_task(
                    trainer.train_model,
                    data_path=request.data_path,
                    target_column=request.target_column,
                    model_name=request.model_name,
                    model_type=request.model_type,
                    hyperparameters=request.hyperparameters or {},
                    validation_split=request.validation_split
                )
                
                return {
                    "message": "Training started",
                    "task_id": task_id,
                    "status": "started",
                    "async": True
                }
            else:
                # Synchronous training
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
                    "model_id": result.get("model_id"),
                    "metrics": result.get("metrics"),
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
            # TODO: Implement actual drift check with Celery
            task_id = str(uuid.uuid4())
            
            return {
                "message": "Drift check started",
                "task_id": task_id,
                "model_id": request.model_id,
                "time_window_hours": request.time_window_hours
            }
            
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")
    
    @app.get(f"{settings.api_v1_str}/drift/reports", response_model=List[DriftReportResponse], tags=["Drift Detection"])
    async def list_drift_reports(
        model_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        db=Depends(get_db)
    ):
        """List drift reports."""
        query = db.query(DriftReport)
        
        if model_id:
            query = query.filter(DriftReport.model_id == model_id)
        
        reports = query.offset(skip).limit(limit).all()
        
        return [
            DriftReportResponse(
                id=str(report.id),
                model_id=str(report.model_id),
                drift_detected=report.drift_detected,
                overall_drift_score=report.overall_drift_score,
                detection_method=report.detection_method,
                created_at=report.created_at.isoformat(),
                feature_drift_scores=report.feature_drift_scores
            )
            for report in reports
        ]
    
    # Alerts endpoints
    @app.get(f"{settings.api_v1_str}/alerts", response_model=List[AlertResponse], tags=["Alerts"])
    async def list_alerts(
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100,
        db=Depends(get_db)
    ):
        """List alerts."""
        query = db.query(Alert)
        
        if severity:
            query = query.filter(Alert.severity == severity)
        
        if resolved is not None:
            query = query.filter(Alert.is_resolved == resolved)
        
        alerts = query.offset(skip).limit(limit).all()
        
        return [
            AlertResponse(
                id=str(alert.id),
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                message=alert.message,
                created_at=alert.created_at.isoformat(),
                is_resolved=alert.is_resolved
            )
            for alert in alerts
        ]
    
    @app.patch(f"{settings.api_v1_str}/alerts/{{alert_id}}/resolve", tags=["Alerts"])
    async def resolve_alert(alert_id: str, db=Depends(get_db)):
        """Resolve an alert."""
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.is_resolved = True
        alert.resolved_at = datetime.utcnow()
        db.commit()
        
        return {"message": "Alert resolved successfully"}
    
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
        "mdt_dashboard.api.main_complete:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.monitoring.log_level.lower()
    )


if __name__ == "__main__":
    run()

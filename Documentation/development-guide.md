# 🛠️ Development Guide

## 📋 Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Strategy](#testing-strategy)
6. [Debugging Guide](#debugging-guide)
7. [Performance Optimization](#performance-optimization)
8. [Contributing Guidelines](#contributing-guidelines)

---

## 🚀 Development Environment Setup

### 📋 Prerequisites

```bash
# Required Software
- Python 3.9+
- Docker & Docker Compose
- Git
- Poetry (recommended) or pip
- Node.js 16+ (for frontend development)

# Optional but Recommended
- VS Code with Python extension
- PostgreSQL client (psql)
- Redis client (redis-cli)
- Kubernetes client (kubectl)
```

### ⚡ Quick Setup

```bash
# 1. Clone and navigate to project
git clone <repository-url>
cd MLOPS/Project/MDT-Dashboard

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install poetry
poetry install --with dev

# 4. Setup pre-commit hooks
pre-commit install

# 5. Copy environment file
cp .env.example .env

# 6. Start development services
docker-compose -f docker-compose.dev.yml up -d

# 7. Run database migrations
poetry run alembic upgrade head

# 8. Start the application
poetry run streamlit run src/mdt_dashboard/dashboard/main.py
```

### 🔧 Environment Configuration

```bash
# .env file structure
DATABASE_URL=postgresql://user:password@localhost:5432/mdt_db
REDIS_URL=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5000
SECRET_KEY=your-secret-key-here
ENVIRONMENT=development
LOG_LEVEL=DEBUG
```

---

## 📁 Project Structure

### 🏗️ Directory Organization

```
MDT-Dashboard/
├── 📁 src/                         # Source code
│   └── 📁 mdt_dashboard/
│       ├── 📁 api/                 # FastAPI backend
│       │   ├── 📄 main.py          # API entry point
│       │   ├── 📁 routers/         # API route handlers
│       │   └── 📁 dependencies/    # API dependencies
│       ├── 📁 core/                # Core functionality
│       │   ├── 📄 config.py        # Configuration
│       │   ├── 📄 database.py      # Database setup
│       │   └── 📄 models.py        # SQLAlchemy models
│       ├── 📁 dashboard/           # Streamlit frontend
│       │   ├── 📄 main.py          # Dashboard entry point
│       │   ├── 📁 pages/           # Dashboard pages
│       │   └── 📁 components/      # Reusable components
│       ├── 📁 ml_pipeline/         # ML pipeline
│       │   ├── 📄 pipeline.py      # Training pipeline
│       │   └── 📄 models/          # ML models
│       ├── 📁 drift_detection/     # Drift detection
│       │   ├── 📄 algorithms.py    # Detection algorithms
│       │   └── 📄 metrics.py       # Drift metrics
│       ├── 📁 monitoring/          # System monitoring
│       │   ├── 📄 metrics.py       # Prometheus metrics
│       │   └── 📄 alerts.py        # Alerting system
│       └── 📁 utils/               # Utilities
│           ├── 📄 logging.py       # Logging configuration
│           └── 📄 helpers.py       # Helper functions
├── 📁 tests/                       # Test suites
│   ├── 📁 unit/                    # Unit tests
│   ├── 📁 integration/             # Integration tests
│   ├── 📁 e2e/                     # End-to-end tests
│   └── 📁 performance/             # Performance tests
├── 📁 docker/                      # Docker configurations
├── 📁 k8s/                         # Kubernetes manifests
├── 📁 docs/                        # Documentation
├── 📁 scripts/                     # Utility scripts
└── 📁 notebooks/                   # Jupyter notebooks
```

### 🔧 Key Files

| File | Purpose | Description |
|------|---------|-------------|
| `pyproject.toml` | Dependency management | Poetry configuration |
| `docker-compose.yml` | Local development | Development services |
| `Dockerfile` | Container build | Production container |
| `alembic.ini` | Database migrations | Alembic configuration |
| `.env.example` | Environment template | Environment variables |
| `pytest.ini` | Test configuration | PyTest settings |

---

## 🔄 Development Workflow

### 🌿 Git Workflow

```bash
# 1. Create feature branch
git checkout -b feature/awesome-feature

# 2. Make changes and commit
git add .
git commit -m "feat: add awesome feature"

# 3. Push and create PR
git push origin feature/awesome-feature
# Create Pull Request on GitHub/GitLab

# 4. After review and merge
git checkout main
git pull origin main
git branch -d feature/awesome-feature
```

### 📝 Commit Convention

```bash
# Format: <type>(<scope>): <description>

feat(api): add new drift detection endpoint
fix(dashboard): resolve chart rendering issue
docs(readme): update installation instructions
test(unit): add tests for drift algorithms
refactor(core): improve database connection handling
style(lint): fix code formatting issues
perf(query): optimize database queries
chore(deps): update dependencies
```

### 🚀 Development Commands

```bash
# Start development environment
make dev-start

# Stop development environment
make dev-stop

# Run tests
make test

# Run linting
make lint

# Run formatting
make format

# Build Docker image
make build

# Deploy to staging
make deploy-staging
```

---

## 📏 Coding Standards

### 🐍 Python Style Guide

```python
# Follow PEP 8 guidelines
# Use type hints
def calculate_drift_score(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_columns: List[str]
) -> Dict[str, float]:
    """Calculate drift scores for specified features.
    
    Args:
        reference_data: Historical reference dataset
        current_data: Current dataset to compare
        feature_columns: List of features to analyze
        
    Returns:
        Dictionary mapping feature names to drift scores
    """
    drift_scores = {}
    
    for feature in feature_columns:
        score = compute_ks_statistic(
            reference_data[feature],
            current_data[feature]
        )
        drift_scores[feature] = score
    
    return drift_scores

# Use descriptive variable names
is_drift_detected = drift_score > threshold
model_accuracy_threshold = 0.85
prediction_confidence_score = model.predict_proba(X)

# Add comprehensive docstrings
class DriftDetector:
    """Advanced drift detection using multiple statistical methods.
    
    This class implements various statistical tests to detect
    distribution drift between reference and current datasets.
    
    Attributes:
        threshold: Drift detection threshold
        algorithms: List of detection algorithms to use
        
    Example:
        detector = DriftDetector(threshold=0.05)
        is_drift = detector.detect_drift(ref_data, current_data)
    """
```

### 🎨 Code Organization

```python
# Group imports logically
# Standard library
import os
import logging
from typing import List, Dict, Optional

# Third-party
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

# Local imports
from mdt_dashboard.core.config import settings
from mdt_dashboard.core.database import get_db
from mdt_dashboard.drift_detection.algorithms import KSTest

# Constants at module level
DEFAULT_DRIFT_THRESHOLD = 0.05
MAX_RETRY_ATTEMPTS = 3
SUPPORTED_ALGORITHMS = ["ks_test", "psi", "js_distance"]
```

### 📊 Database Models

```python
# SQLAlchemy models with proper relationships
class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    version = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model")
    metrics = relationship("ModelMetric", back_populates="model")
    alerts = relationship("Alert", back_populates="model")
    
    def __repr__(self):
        return f"<Model(name='{self.name}', version='{self.version}')>"
```

### 🌐 API Design

```python
# RESTful API design with proper error handling
@router.post("/models/{model_id}/predictions", response_model=PredictionResponse)
async def create_prediction(
    model_id: int,
    prediction_data: PredictionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new prediction for the specified model."""
    try:
        model = get_model_or_404(db, model_id)
        
        # Validate input data
        validated_data = validate_prediction_input(prediction_data)
        
        # Make prediction
        prediction_result = await model.predict(validated_data)
        
        # Store in database
        prediction = create_prediction_record(
            db=db,
            model_id=model_id,
            input_data=validated_data,
            prediction=prediction_result,
            user_id=current_user.id
        )
        
        # Trigger async drift detection
        trigger_drift_detection.delay(model_id, prediction.id)
        
        return PredictionResponse.from_orm(prediction)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

---

## 🧪 Testing Strategy

### 📊 Test Pyramid

```mermaid
pyramid
    title Test Pyramid
    
    top "E2E Tests (5%)"
    middle "Integration Tests (15%)"
    bottom "Unit Tests (80%)"
```

### 🔬 Unit Tests

```python
# tests/unit/test_drift_detection.py
import pytest
import pandas as pd
from unittest.mock import Mock, patch

from mdt_dashboard.drift_detection.algorithms import KSTest, PSICalculator


class TestKSTest:
    """Test suite for Kolmogorov-Smirnov drift detection."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        reference = pd.Series([1, 2, 3, 4, 5] * 100)
        current = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1] * 100)
        return reference, current
    
    def test_ks_test_no_drift(self, sample_data):
        """Test KS test when no drift is present."""
        reference, _ = sample_data
        current = reference.copy()  # No drift
        
        ks_test = KSTest(threshold=0.05)
        result = ks_test.detect_drift(reference, current)
        
        assert not result.is_drift_detected
        assert result.p_value > 0.05
        assert result.statistic < ks_test.threshold
    
    def test_ks_test_with_drift(self, sample_data):
        """Test KS test when drift is present."""
        reference, current = sample_data
        
        ks_test = KSTest(threshold=0.05)
        result = ks_test.detect_drift(reference, current)
        
        assert result.is_drift_detected
        assert result.p_value < 0.05
        assert result.statistic > ks_test.threshold
    
    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1])
    def test_ks_test_different_thresholds(self, sample_data, threshold):
        """Test KS test with different thresholds."""
        reference, current = sample_data
        
        ks_test = KSTest(threshold=threshold)
        result = ks_test.detect_drift(reference, current)
        
        assert isinstance(result.is_drift_detected, bool)
        assert 0 <= result.statistic <= 1
```

### 🔗 Integration Tests

```python
# tests/integration/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mdt_dashboard.api.main import app
from mdt_dashboard.core.database import get_db, Base


@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///./test.db")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db):
    """Create test client."""
    return TestClient(app)


class TestModelAPI:
    """Integration tests for model API endpoints."""
    
    def test_create_model(self, client, test_db):
        """Test creating a new model."""
        model_data = {
            "name": "test_model",
            "version": "1.0.0",
            "model_type": "classification",
            "description": "Test model"
        }
        
        response = client.post("/api/v1/models/", json=model_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == model_data["name"]
        assert data["version"] == model_data["version"]
        assert "id" in data
    
    def test_get_model_predictions(self, client, test_db):
        """Test retrieving model predictions."""
        # First create a model
        model_response = client.post("/api/v1/models/", json={
            "name": "test_model",
            "version": "1.0.0",
            "model_type": "classification"
        })
        model_id = model_response.json()["id"]
        
        # Then get predictions
        response = client.get(f"/api/v1/models/{model_id}/predictions")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
```

### 🎯 End-to-End Tests

```python
# tests/e2e/test_system_e2e.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@pytest.fixture
def browser():
    """Setup browser for E2E tests."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()


class TestDashboardE2E:
    """End-to-end tests for the dashboard."""
    
    def test_dashboard_loads(self, browser):
        """Test that dashboard loads successfully."""
        browser.get("http://localhost:8501")
        
        # Wait for page to load
        wait = WebDriverWait(browser, 10)
        title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        
        assert "MDT Dashboard" in title.text
    
    def test_model_monitoring_workflow(self, browser):
        """Test complete model monitoring workflow."""
        browser.get("http://localhost:8501")
        
        # Navigate to model page
        model_link = browser.find_element(By.LINK_TEXT, "Models")
        model_link.click()
        
        # Select a model
        model_dropdown = browser.find_element(By.CSS_SELECTOR, "select")
        model_dropdown.send_keys("test_model")
        
        # Verify metrics are displayed
        metrics_section = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "metrics-container"))
        )
        
        assert metrics_section.is_displayed()
```

### ⚡ Performance Tests

```python
# tests/performance/test_api_performance.py
import pytest
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor


class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self):
        """Test API performance under concurrent load."""
        url = "http://localhost:8000/api/v1/models/1/predict"
        payload = {"features": [1, 2, 3, 4, 5]}
        
        async def make_request(session):
            async with session.post(url, json=payload) as response:
                return await response.json()
        
        # Test with 100 concurrent requests
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session) for _ in range(100)]
            responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Assertions
        assert len(responses) == 100
        assert duration < 10  # Should complete within 10 seconds
        assert all("prediction" in response for response in responses)
```

---

## 🐛 Debugging Guide

### 📊 Logging Configuration

```python
# src/mdt_dashboard/utils/logging.py
import logging
from loguru import logger
import sys

def setup_logging(log_level: str = "INFO"):
    """Configure application logging."""
    # Remove default handler
    logger.remove()
    
    # Add console handler with formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler for production
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=log_level
    )

# Usage in modules
from mdt_dashboard.utils.logging import logger

def detect_drift(reference_data, current_data):
    logger.info(f"Starting drift detection for {len(current_data)} samples")
    
    try:
        result = perform_ks_test(reference_data, current_data)
        logger.info(f"Drift detection completed: drift={result.is_drift_detected}")
        return result
    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        raise
```

### 🔍 Debug Techniques

```python
# Using debugger
import pdb; pdb.set_trace()  # Python debugger
import ipdb; ipdb.set_trace()  # IPython debugger

# VS Code debugging configuration
# .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "program": "src/mdt_dashboard/api/main.py",
            "args": ["--reload"],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "Python: Streamlit",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "src/mdt_dashboard/dashboard/main.py"],
            "console": "integratedTerminal"
        }
    ]
}
```

### 📈 Monitoring Debug Information

```python
# Add debug endpoints for development
@router.get("/debug/health")
async def debug_health():
    """Debug endpoint to check system health."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "database": check_database_connection(),
        "redis": check_redis_connection(),
        "memory_usage": get_memory_usage(),
        "active_workers": get_active_worker_count()
    }

@router.get("/debug/metrics")
async def debug_metrics():
    """Debug endpoint to view system metrics."""
    return {
        "api_requests": get_request_count(),
        "response_times": get_average_response_time(),
        "error_rate": get_error_rate(),
        "drift_detections": get_drift_detection_count()
    }
```

---

## 🚀 Performance Optimization

### 📊 Database Optimization

```python
# Connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=300
)

# Query optimization
from sqlalchemy.orm import joinedload

# Eager loading to avoid N+1 queries
models = db.query(Model).options(
    joinedload(Model.metrics),
    joinedload(Model.predictions)
).all()

# Pagination for large datasets
def get_paginated_predictions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Prediction).offset(skip).limit(limit).all()
```

### ⚡ Caching Strategy

```python
# Redis caching decorator
import functools
import json
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration: int = 300):
    """Cache function result in Redis."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key, 
                expiration, 
                json.dumps(result, default=str)
            )
            return result
        return wrapper
    return decorator

# Usage
@cache_result(expiration=600)  # Cache for 10 minutes
def get_model_metrics(model_id: int):
    """Get model metrics with caching."""
    return compute_expensive_metrics(model_id)
```

### 🔄 Async Processing

```python
# Async FastAPI endpoints
@router.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: int, db: AsyncSession = Depends(get_async_db)):
    """Get model metrics asynchronously."""
    metrics = await db.execute(
        select(ModelMetric).where(ModelMetric.model_id == model_id)
    )
    return metrics.scalars().all()

# Background tasks with Celery
from celery import Celery

celery_app = Celery('mdt_dashboard')

@celery_app.task
def compute_drift_metrics(model_id: int):
    """Compute drift metrics in background."""
    model = get_model(model_id)
    recent_predictions = get_recent_predictions(model_id)
    
    drift_score = calculate_drift(model.reference_data, recent_predictions)
    store_drift_metrics(model_id, drift_score)
    
    if drift_score > model.drift_threshold:
        send_drift_alert(model_id, drift_score)
```

---

## 🤝 Contributing Guidelines

### 📝 Pull Request Process

1. **Create Issue**: Describe the problem or feature
2. **Fork Repository**: Create your own fork
3. **Create Branch**: Use descriptive branch names
4. **Make Changes**: Follow coding standards
5. **Add Tests**: Ensure adequate test coverage
6. **Update Docs**: Update relevant documentation
7. **Submit PR**: Provide clear description

### ✅ PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests are added/updated
- [ ] Documentation is updated
- [ ] CI/CD pipeline passes
- [ ] Code is reviewed by maintainer
- [ ] Breaking changes are documented

### 🔍 Code Review Guidelines

```python
# Good code review comments
# ❌ Bad: "This is wrong"
# ✅ Good: "Consider using async/await here for better performance"

# ❌ Bad: "Fix this"
# ✅ Good: "This function could benefit from error handling for the case where model_id doesn't exist"

# ❌ Bad: "Style issue"
# ✅ Good: "Please add type hints to improve code readability: def process_data(data: pd.DataFrame) -> Dict[str, float]"
```

---

## 📚 Additional Resources

### 🛠️ Development Tools

- **IDE**: VS Code with Python extension
- **Database**: DBeaver or pgAdmin for PostgreSQL
- **API Testing**: Postman or Insomnia
- **Performance**: py-spy for profiling
- **Memory**: memory_profiler for memory analysis

### 📖 Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Celery Documentation](https://docs.celeryproject.org/)

### 🎓 Learning Resources

- [Clean Code in Python](https://realpython.com/python-code-quality/)
- [Testing Best Practices](https://realpython.com/python-testing/)
- [Async Python](https://realpython.com/async-io-python/)
- [Database Optimization](https://use-the-index-luke.com/)

---

*Happy coding! 🚀*

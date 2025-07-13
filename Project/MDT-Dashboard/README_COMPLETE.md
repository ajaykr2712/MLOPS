# MDT Dashboard - Model Drift Detection & Telemetry Platform

## 🚀 Enterprise-Grade ML Operations Platform

The **Model Drift Detection & Telemetry (MDT) Dashboard** is a comprehensive, production-ready platform for monitoring machine learning models in real-time. It provides advanced drift detection, performance monitoring, alerting, and a modern web interface for managing your ML infrastructure.

![MDT Dashboard](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │   PostgreSQL    │
│   Dashboard     │◄──►│   REST API      │◄──►│   Database      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                      ┌─────────────────┐    ┌─────────────────┐
                      │     Celery      │◄──►│     Redis       │
                      │   Task Queue    │    │   Message       │
                      │                 │    │   Broker        │
                      └─────────────────┘    └─────────────────┘
                              │
                              ▼
                      ┌─────────────────┐
                      │   ML Models     │
                      │   Registry      │
                      │                 │
                      └─────────────────┘
```

## ✨ Key Features

### 🔍 **Advanced Drift Detection**
- **Statistical Methods**: Kolmogorov-Smirnov, Population Stability Index (PSI), Chi-Square tests
- **Multivariate Analysis**: Detects complex relationships between features
- **Real-time Monitoring**: Continuous monitoring of prediction inputs
- **Configurable Thresholds**: Custom sensitivity settings per model

### 📊 **Comprehensive Monitoring**
- **Model Performance**: Accuracy, precision, recall, F1-score tracking
- **System Metrics**: CPU, memory, response times, error rates
- **Business Metrics**: Prediction volume, user engagement, revenue impact
- **Data Quality**: Missing values, outliers, schema violations

### 🚨 **Intelligent Alerting**
- **Multi-Channel Notifications**: Email, Slack, webhooks
- **Severity Levels**: Low, medium, high, critical alerts
- **Smart Thresholds**: Adaptive alerting based on historical patterns
- **Root Cause Analysis**: Automated insights into performance issues

### 🌐 **Modern Web Interface**
- **Real-time Dashboard**: Live metrics and visualizations
- **Interactive Charts**: Plotly-powered analytics
- **Model Management**: Deploy, monitor, and version models
- **Prediction Testing**: Interactive model testing interface

### 🏭 **Production Ready**
- **Scalable Architecture**: Horizontal scaling with Redis and Celery
- **Multi-Cloud Support**: AWS, GCP, Azure deployment options
- **CI/CD Pipeline**: Automated testing and deployment
- **Security**: Authentication, authorization, audit logging

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PostgreSQL 13+
- Redis 6+
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd MDT-Dashboard
```

2. **Install dependencies**
```bash
# Using pip
pip install -r requirements.txt

# Using poetry (recommended)
poetry install
```

3. **Setup environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Initialize the database**
```bash
python cli.py setup
```

5. **Start the platform**
```bash
python cli.py run
```

The platform will be available at:
- 📊 **Dashboard**: http://localhost:8501
- 🔧 **API Docs**: http://localhost:8000/docs
- 📈 **Metrics**: http://localhost:8000/metrics

## 🔧 Configuration

### Environment Variables

```bash
# Application
APP_NAME="MDT Dashboard"
ENVIRONMENT=development  # development, staging, production
DEBUG=true

# Database
DATABASE_URL=postgresql://mdt_user:mdt_pass@localhost:5432/mdt_db
DATABASE_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Security
SECRET_KEY=your-secret-key-here
ENABLE_AUTH=false

# Drift Detection
KS_TEST_THRESHOLD=0.05
PSI_THRESHOLD=0.2
DRIFT_DETECTION_ENABLED=true

# Alerts
ALERT_ENABLE_EMAIL=true
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

ALERT_ENABLE_SLACK=false
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Cloud Storage (optional)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
S3_BUCKET=your-mdt-bucket
```

## 📖 Usage Guide

### 1. Model Registration

```python
from mdt_dashboard.train import ModelTrainer

# Train and register a model
trainer = ModelTrainer()
result = trainer.train_model(
    data_path="data/training_data.csv",
    target_column="target",
    model_name="customer_churn_v1",
    model_type="xgboost"
)
```

### 2. Making Predictions

```python
import requests

# API prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "model_id": "customer_churn_v1",
        "input_data": {
            "age": 35,
            "income": 50000,
            "usage_days": 180
        },
        "detect_drift": True
    }
)

prediction = response.json()
```

### 3. Python SDK

```python
from mdt_dashboard.predict import PredictionService

# Initialize service
service = PredictionService()

# Make prediction
result = service.predict({
    "age": 35,
    "income": 50000,
    "usage_days": 180
}, model_name="customer_churn_v1")

print(f"Prediction: {result.predictions[0]}")
print(f"Drift detected: {result.drift_detected}")
```

### 4. Monitoring Setup

```python
from mdt_dashboard.monitoring.alerts import AlertManager

# Setup alerting
alert_manager = AlertManager()

# Custom drift alert
alert_manager.create_and_send_drift_alert(
    model=model,
    drift_score=0.15,
    affected_features=["age", "income"],
    severity="high"
)
```

## 🐳 Docker Deployment

### Local Development

```bash
docker-compose up -d
```

### Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## ☸️ Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods -n mdt-dashboard
```

## 🧪 Testing

### Run Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# Performance tests
k6 run tests/performance/api_load_test.js
```

### Test Coverage

```bash
pytest --cov=src tests/
```

## 📊 Monitoring & Observability

### Metrics

The platform exposes Prometheus metrics at `/metrics`:

- `http_requests_total`: Total HTTP requests
- `predictions_total`: Total predictions made
- `drift_detections_total`: Total drift detections
- `model_performance_score`: Model performance metrics

### Logging

Structured JSON logging with configurable levels:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "mdt_dashboard.predict",
  "message": "Prediction completed",
  "model_name": "customer_churn_v1",
  "processing_time_ms": 25.4,
  "drift_detected": false
}
```

## 🚀 Advanced Features

### Custom Drift Algorithms

```python
from mdt_dashboard.drift_detection.algorithms import BaseDriftDetector

class CustomDriftDetector(BaseDriftDetector):
    def detect_drift(self, reference_data, current_data):
        # Implement custom logic
        return DriftResult(...)

# Register custom detector
service.register_drift_detector("custom", CustomDriftDetector())
```

### Model Pipelines

```python
from mdt_dashboard.ml_pipeline import MLPipeline

pipeline = MLPipeline()
pipeline.add_step("preprocessing", CustomPreprocessor())
pipeline.add_step("model", XGBoostModel())
pipeline.add_step("postprocessing", CustomPostprocessor())

result = pipeline.run(input_data)
```

### Custom Metrics

```python
from mdt_dashboard.monitoring.metrics import MetricsCollector

collector = MetricsCollector()
collector.record_custom_metric("business_revenue", 1250.50, {
    "model": "customer_churn_v1",
    "segment": "premium"
})
```

## 🔐 Security

### Authentication

```bash
# Enable authentication
export ENABLE_AUTH=true
export SECRET_KEY=your-strong-secret-key

# Create API key
python -c "from mdt_dashboard.auth import create_api_key; print(create_api_key('user@company.com'))"
```

### Rate Limiting

```python
# Configure rate limiting
RATE_LIMIT_PER_MINUTE=100
```

### Audit Logging

All API calls and model operations are logged for compliance:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "user_id": "user@company.com",
  "action": "model_prediction",
  "resource": "customer_churn_v1",
  "ip_address": "192.168.1.100",
  "user_agent": "Python/requests"
}
```

## 🔧 CLI Reference

```bash
# Setup and run
python cli.py setup              # Initialize database
python cli.py run                # Run all services
python cli.py run --dev          # Run in development mode

# Individual services
python cli.py api                # Run API server only
python cli.py dashboard          # Run dashboard only
python cli.py worker             # Run Celery worker
python cli.py beat               # Run Celery scheduler

# Utilities
python cli.py check              # Check dependencies
```

## 📚 API Reference

### Core Endpoints

- `POST /api/v1/predict` - Make predictions
- `GET /api/v1/models` - List models
- `POST /api/v1/train` - Train models
- `POST /api/v1/drift/check` - Check drift
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

### Webhook Integration

```python
# Setup webhook for alerts
{
  "url": "https://your-webhook-endpoint.com/alerts",
  "events": ["drift_detected", "performance_degradation"],
  "secret": "webhook-secret"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 **Email**: support@mdt-dashboard.com
- 💬 **Slack**: [Join our community](https://slack.mdt-dashboard.com)
- 📖 **Documentation**: [docs.mdt-dashboard.com](https://docs.mdt-dashboard.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/mdt-dashboard/issues)

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Streamlit](https://streamlit.io/)
- Inspired by MLOps best practices and production ML systems
- Special thanks to the open-source ML community

---

**⭐ Star this repository if you find it useful!**

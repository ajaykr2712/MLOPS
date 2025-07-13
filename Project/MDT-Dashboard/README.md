# 🔬 MDT Dashboard - Model Drift Detection & Telemetry Platform

<div align="center">

![MDT Dashboard](https://img.shields.io/badge/MDT-Dashboard-blue?style=for-the-badge&logo=python)
![Version](https://img.shields.io/badge/version-0.1.0-green?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-orange?style=for-the-badge)

**Enterprise-grade solution for automated model performance monitoring, drift detection, and ML telemetry**

</div>

## 🚀 **Overview**

MDT Dashboard is a comprehensive MLOps platform designed to solve the critical challenge of **Automated Model Performance Degradation Detection** as outlined in Project A. This enterprise-grade solution provides:

- **Real-time drift detection** using multiple statistical algorithms
- **Comprehensive model monitoring** with performance tracking
- **Automated alerting** and root cause analysis
- **Multi-cloud deployment** support
- **Production-ready** architecture with scalability in mind

## 🎯 **Key Features**

### 🔍 **Advanced Drift Detection**
- **Statistical Tests**: KS test, PSI, Chi-square, Jensen-Shannon distance
- **Multi-variate analysis** with feature-level granularity
- **Adaptive thresholds** and early warning systems
- **Real-time** and batch processing modes

### 📊 **Comprehensive Monitoring**
- **System metrics**: CPU, memory, network, disk usage
- **Model performance**: Accuracy, latency, throughput
- **Business metrics**: Request volume, error rates
- **Custom alerting** with configurable channels

### 🤖 **ML Pipeline Integration**
- **Automated model training** with hyperparameter optimization
- **Model versioning** and artifact management
- **A/B testing** support for model comparison
- **MLflow integration** for experiment tracking

### 🎨 **Interactive Dashboard**
- **Real-time visualizations** with Plotly and Streamlit
- **Model management** interface
- **Prediction playground** for testing
- **Comprehensive analytics** and reporting

### 🏗️ **Production Architecture**
- **FastAPI** backend with async support
- **PostgreSQL** database with Redis caching
- **Docker containerization** for easy deployment
- **Prometheus/Grafana** monitoring stack

## 🛠️ **Technical Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │   PostgreSQL    │
│   Dashboard     │◄──►│      API        │◄──►│    Database     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Redis       │◄──►│  Drift Detection│    │     MLflow      │
│     Cache       │    │     Engine      │◄──►│   Tracking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │◄──►│   Monitoring    │    │    Celery       │
│   Metrics       │    │    System       │◄──►│    Workers      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 **Installation**

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd MDT-Dashboard
```

2. **Install dependencies**
```bash
# Using Poetry (recommended)
pip install poetry
poetry install

# Or using pip
pip install -e .
```

3. **Start infrastructure**
```bash
docker-compose up -d postgres redis mlflow
```

4. **Configure environment**
```bash
cp config/.env.example .env
# Edit .env with your settings
```

5. **Run database migrations**
```bash
mdt-migrate
```

6. **Start services**
```bash
# Terminal 1: Start API server
mdt-server

# Terminal 2: Start dashboard
mdt-dashboard

# Terminal 3: Start background worker
mdt-worker
```

7. **Access the application**
- Dashboard: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Grafana: http://localhost:3000

## 🔧 **Configuration**

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=postgresql://mdt_user:mdt_pass@localhost:5432/mdt_db

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard
DASHBOARD_PORT=8501

# Drift Detection
KS_TEST_THRESHOLD=0.05
PSI_THRESHOLD=0.2
REFERENCE_WINDOW_SIZE=1000

# Monitoring
LOG_LEVEL=INFO
PROMETHEUS_PORT=9090

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Advanced Configuration

For production deployments, see `config/` directory for:
- Docker configurations
- Kubernetes manifests
- Prometheus rules
- Grafana dashboards

## 🎮 **Usage**

### 1. **Model Training**

```python
from mdt_dashboard.train import ModelTrainer, ModelConfig
from mdt_dashboard.data_processing import ComprehensiveDataProcessor
import pandas as pd

# Load and process data
data = pd.read_csv("your_data.csv")
processor = ComprehensiveDataProcessor()
X_processed = processor.fit_transform(data.drop('target', axis=1))
y = data['target']

# Configure and train models
config = ModelConfig(
    problem_type="regression",
    algorithms=["random_forest", "xgboost", "lightgbm"],
    cv_folds=5
)

trainer = ModelTrainer(config)
results = trainer.train_all_models(X_train, X_test, y_train, y_test)
```

### 2. **Real-time Predictions**

```python
from mdt_dashboard.predict import PredictionService, PredictionRequest

# Initialize service
service = PredictionService()

# Make prediction
request = PredictionRequest(
    data={"feature1": 1.0, "feature2": 2.0},
    model_name="best_model",
    return_probabilities=True,
    return_feature_importance=True
)

response = service.predict(request)
print(f"Prediction: {response.predictions}")
print(f"Drift detected: {response.drift_detected}")
```

### 3. **API Usage**

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {"feature1": 1.0, "feature2": 2.0},
    "model_name": "best_model",
    "return_probabilities": true
  }'

# Get metrics
curl "http://localhost:8000/metrics?time_window_minutes=60"
```

### 4. **Drift Detection**

```python
from mdt_dashboard.drift_detection.algorithms import MultivariateDriftDetector

# Initialize detector
detector = MultivariateDriftDetector()

# Detect drift
results = detector.detect_multivariate(reference_data, current_data)

# Get summary
summary = detector.get_summary(results)
print(f"Drift detected in {summary['drift_features_count']} features")
```

## 📊 **Dashboard Features**

### Overview Page
- System health status
- Real-time metrics
- Alert summaries
- Performance indicators

### Models Page
- Model listing and comparison
- Performance metrics
- Model details and metadata
- Training history

### Predictions Page
- Interactive prediction interface
- Feature importance visualization
- Drift detection results
- Confidence scores

### Monitoring Page
- Time-series visualizations
- System resource usage
- Model performance trends
- Custom dashboards

## 🔄 **Production Deployment**

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Scale workers
docker-compose up -d --scale mdt-worker=3

# View logs
docker-compose logs -f mdt-api
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -n mdt-dashboard

# Port forward for local access
kubectl port-forward svc/mdt-dashboard 8501:8501
```

### Cloud Deployment

Supports deployment on:
- **AWS**: ECS, EKS, Lambda
- **GCP**: GKE, Cloud Run, App Engine
- **Azure**: AKS, Container Instances

## 📈 **Monitoring & Observability**

### Metrics Collected
- **Request metrics**: Volume, latency, error rates
- **Model metrics**: Accuracy, drift scores, confidence
- **System metrics**: CPU, memory, disk, network
- **Business metrics**: User activity, feature usage

### Alerting
- **Threshold-based** alerts
- **Anomaly detection** alerts
- **Multi-channel** notifications (Email, Slack, Webhook)
- **Alert suppression** and escalation

### Dashboards
- **Grafana** dashboards for infrastructure
- **Streamlit** dashboard for model analytics
- **Custom** business dashboards

## 🧪 **Testing**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mdt_dashboard --cov-report=html

# Run specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Setup pre-commit hooks
pre-commit install

# Run quality checks
black src/
isort src/
flake8 src/
mypy src/
```

## 📚 **Documentation**

- **API Documentation**: http://localhost:8000/docs
- **Architecture Guide**: [docs/architecture.md](docs/architecture.md)
- **Deployment Guide**: [docs/deployment.md](docs/deployment.md)
- **API Reference**: [docs/api.md](docs/api.md)
- **Troubleshooting**: [docs/troubleshooting.md](docs/troubleshooting.md)

## 🔐 **Security**

- **Authentication**: JWT-based API authentication
- **Authorization**: Role-based access control
- **Data encryption**: At rest and in transit
- **Audit logging**: Complete activity tracking
- **Vulnerability scanning**: Regular security scans

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 **Acknowledgments**

- Built to address **Project A** requirements for automated model performance degradation detection
- Inspired by industry best practices from Databricks, AWS SageMaker, and Google Vertex AI
- Designed for enterprise use cases across financial services, healthcare, retail, and manufacturing

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: aponduga@cisco.com

---

<div align="center">

**Built with ❤️ by [Ajay Kumar Pondugala](mailto:aponduga@cisco.com)**

*Empowering ML teams with production-ready monitoring and drift detection*

</div>
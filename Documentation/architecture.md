# 🏗️ System Architecture Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Deployment Architecture](#deployment-architecture)
6. [Security Architecture](#security-architecture)
7. [Scalability Considerations](#scalability-considerations)

---

## 🎯 Overview

The MLOPS ecosystem is designed as a **modular, microservices-based architecture** that supports multiple MLOps use cases. The current implementation focuses on **Project A: Automated Model Performance Degradation Detection** through the MDT Dashboard, with a foundation that can be extended for all other projects.

### 🎨 Design Principles

- **Microservices**: Loosely coupled, independently deployable services
- **Event-Driven**: Asynchronous communication using message queues
- **Cloud-Native**: Container-first with Kubernetes orchestration
- **Observability**: Comprehensive monitoring and logging
- **Security**: Zero-trust security model with end-to-end encryption

---

## 🏗️ High-Level Architecture

```mermaid
C4Context
    title System Context Diagram - MLOPS Platform
    
    Person(user, "ML Engineer", "Monitors models and manages ML pipelines")
    Person(business, "Business User", "Views dashboards and reports")
    Person(admin, "System Admin", "Manages infrastructure and deployments")
    
    System(mlops, "MLOPS Platform", "Comprehensive MLOps solution for model monitoring and management")
    
    System_Ext(cloud, "Cloud Providers", "AWS, GCP, Azure for compute and storage")
    System_Ext(ml_systems, "ML Systems", "External ML models and data sources")
    System_Ext(monitoring, "External Monitoring", "Prometheus, Grafana, ELK Stack")
    
    Rel(user, mlops, "Manages models")
    Rel(business, mlops, "Views analytics")
    Rel(admin, mlops, "Administers system")
    Rel(mlops, cloud, "Deploys to")
    Rel(mlops, ml_systems, "Monitors")
    Rel(mlops, monitoring, "Sends metrics to")
```

### 🔧 Component Architecture

```mermaid
C4Container
    title Container Diagram - MDT Dashboard
    
    Container(webapp, "Streamlit Dashboard", "Python, Streamlit", "Interactive web interface for model monitoring")
    Container(api, "FastAPI Backend", "Python, FastAPI", "REST API for model operations and data access")
    Container(worker, "Celery Workers", "Python, Celery", "Background task processing for model training and monitoring")
    Container(scheduler, "Task Scheduler", "Python, APScheduler", "Scheduled tasks for periodic model evaluation")
    
    ContainerDb(postgres, "PostgreSQL", "Relational Database", "Stores model metadata, metrics, and configurations")
    ContainerDb(redis, "Redis", "In-Memory Cache", "Caching and message broker for Celery")
    ContainerDb(timeseries, "InfluxDB", "Time Series Database", "Stores time-series metrics and monitoring data")
    
    Container(mlflow, "MLflow Server", "Python, MLflow", "Model registry and experiment tracking")
    Container(prometheus, "Prometheus", "Monitoring", "Metrics collection and alerting")
    Container(grafana, "Grafana", "Visualization", "Infrastructure monitoring dashboards")
    
    Rel(webapp, api, "Makes API calls to", "HTTPS")
    Rel(api, postgres, "Reads from and writes to", "SQL")
    Rel(api, redis, "Caches data in", "Redis Protocol")
    Rel(api, mlflow, "Tracks models with", "HTTP")
    Rel(worker, postgres, "Updates model data in", "SQL")
    Rel(worker, timeseries, "Stores metrics in", "InfluxDB Protocol")
    Rel(scheduler, worker, "Triggers tasks in", "Celery")
    Rel(prometheus, api, "Scrapes metrics from", "HTTP")
    Rel(grafana, prometheus, "Queries metrics from", "PromQL")
```

---

## 🔧 Component Details

### 🎨 Frontend Layer

#### Streamlit Dashboard
- **Purpose**: Interactive web interface for model monitoring
- **Technology**: Python, Streamlit, Plotly
- **Features**:
  - Real-time model performance dashboards
  - Drift detection visualizations
  - Model comparison tools
  - Prediction playground
  - Alert management interface

```python
# Key Components
- src/mdt_dashboard/dashboard/main.py          # Main dashboard application
- src/mdt_dashboard/dashboard/components/      # Reusable UI components
- src/mdt_dashboard/plotting.py               # Visualization utilities
```

### 🚀 Backend Layer

#### FastAPI Application
- **Purpose**: RESTful API for all backend operations
- **Technology**: Python, FastAPI, SQLAlchemy
- **Features**:
  - Model CRUD operations
  - Prediction endpoints
  - Drift detection APIs
  - Metrics collection
  - Authentication & authorization

```python
# Key Components
- src/mdt_dashboard/api/main.py               # API application
- src/mdt_dashboard/api/routers/              # API route handlers
- src/mdt_dashboard/core/models.py            # Database models
- src/mdt_dashboard/core/config.py            # Configuration management
```

#### Celery Workers
- **Purpose**: Background task processing
- **Technology**: Python, Celery, Redis
- **Features**:
  - Model training and retraining
  - Drift detection computations
  - Data quality checks
  - Report generation
  - Alert notifications

```python
# Key Components
- src/mdt_dashboard/worker.py                 # Celery worker definitions
- src/mdt_dashboard/ml_pipeline/pipeline.py  # ML pipeline tasks
- src/mdt_dashboard/drift_detection/         # Drift detection algorithms
```

### 💾 Data Layer

#### PostgreSQL Database
- **Purpose**: Primary data store for structured data
- **Schema**:
  - `models`: Model metadata and versions
  - `predictions`: Model predictions and actuals
  - `metrics`: Performance metrics
  - `alerts`: Alert configurations and history
  - `users`: User management

#### Redis Cache
- **Purpose**: Caching and message broker
- **Usage**:
  - API response caching
  - Session storage
  - Celery message broker
  - Real-time data caching

#### InfluxDB (Optional)
- **Purpose**: Time-series metrics storage
- **Usage**:
  - High-frequency metrics
  - Real-time monitoring data
  - Performance analytics

### 🤖 ML Layer

#### MLflow Integration
- **Purpose**: Model lifecycle management
- **Features**:
  - Model registry
  - Experiment tracking
  - Model versioning
  - Artifact storage

#### Drift Detection Engine
- **Purpose**: Automated drift detection
- **Algorithms**:
  - Kolmogorov-Smirnov test
  - Population Stability Index (PSI)
  - Jensen-Shannon Distance
  - Chi-square test
  - Custom statistical tests

---

## 🌊 Data Flow

### 📊 Model Monitoring Flow

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant API
    participant Worker
    participant DB
    participant MLflow
    
    User->>Dashboard: Access monitoring page
    Dashboard->>API: Request model metrics
    API->>DB: Query model data
    DB-->>API: Return metrics
    API-->>Dashboard: Return formatted data
    Dashboard-->>User: Display visualizations
    
    Note over Worker: Background Process
    Worker->>DB: Fetch new predictions
    Worker->>Worker: Compute drift metrics
    Worker->>DB: Store drift scores
    Worker->>MLflow: Log metrics
    
    alt Drift Detected
        Worker->>API: Trigger alert
        API->>User: Send notification
    end
```

### 🔄 Model Training Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Worker
    participant MLflow
    participant Storage
    
    User->>API: Request model training
    API->>Worker: Queue training task
    Worker->>Storage: Load training data
    Worker->>Worker: Train model
    Worker->>MLflow: Log experiment
    Worker->>MLflow: Register model
    Worker->>API: Update status
    API-->>User: Training complete
```

### 📈 Prediction Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Cache
    participant Model
    participant DB
    participant Worker
    
    Client->>API: Send prediction request
    API->>Cache: Check cache
    alt Cache Hit
        Cache-->>API: Return cached result
    else Cache Miss
        API->>Model: Make prediction
        Model-->>API: Return prediction
        API->>Cache: Store in cache
    end
    API->>DB: Log prediction
    API-->>Client: Return result
    
    Note over Worker: Async Process
    Worker->>DB: Fetch recent predictions
    Worker->>Worker: Analyze for drift
    Worker->>DB: Update drift metrics
```

---

## 🚀 Deployment Architecture

### 🐳 Docker Containerization

```yaml
# docker-compose.yml structure
services:
  mdt-dashboard:     # Streamlit frontend
  mdt-api:          # FastAPI backend
  mdt-worker:       # Celery workers
  mdt-scheduler:    # Task scheduler
  postgres:         # PostgreSQL database
  redis:            # Redis cache/broker
  mlflow:           # MLflow server
  prometheus:       # Metrics collection
  grafana:          # Monitoring dashboards
```

### ☸️ Kubernetes Deployment

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Application Namespace"
            subgraph "Frontend"
                DASH[Dashboard Pod]
            end
            
            subgraph "Backend"
                API[API Pods]
                WORKER[Worker Pods]
                SCHED[Scheduler Pod]
            end
            
            subgraph "Data Services"
                PG[(PostgreSQL)]
                REDIS[(Redis)]
                MLFLOW[MLflow Server]
            end
        end
        
        subgraph "Monitoring Namespace"
            PROM[Prometheus]
            GRAF[Grafana]
            ALERT[AlertManager]
        end
        
        subgraph "Ingress"
            NGINX[NGINX Ingress]
            CERT[Cert Manager]
        end
    end
    
    NGINX --> DASH
    NGINX --> API
    DASH --> API
    API --> PG
    API --> REDIS
    API --> MLFLOW
    WORKER --> PG
    WORKER --> REDIS
    PROM --> API
    GRAF --> PROM
```

### ☁️ Cloud Architecture

#### AWS Deployment
```mermaid
graph LR
    subgraph "AWS"
        subgraph "EKS Cluster"
            POD[Application Pods]
        end
        
        subgraph "Data Services"
            RDS[(RDS PostgreSQL)]
            ELASTICACHE[(ElastiCache Redis)]
            S3[(S3 Storage)]
        end
        
        subgraph "Monitoring"
            CW[CloudWatch]
            ALB[Application Load Balancer]
        end
        
        subgraph "Security"
            IAM[IAM Roles]
            SECRETS[Secrets Manager]
        end
    end
    
    POD --> RDS
    POD --> ELASTICACHE
    POD --> S3
    ALB --> POD
    CW --> POD
```

---

## 🔐 Security Architecture

### 🛡️ Security Layers

```mermaid
graph TB
    subgraph "Network Security"
        VPC[Virtual Private Cloud]
        SG[Security Groups]
        NACL[Network ACLs]
    end
    
    subgraph "Application Security"
        AUTH[JWT Authentication]
        RBAC[Role-Based Access Control]
        OAUTH[OAuth 2.0]
    end
    
    subgraph "Data Security"
        ENCRYPT[Encryption at Rest]
        TLS[TLS in Transit]
        VAULT[Secret Management]
    end
    
    subgraph "Infrastructure Security"
        KUBE[Kubernetes RBAC]
        PSP[Pod Security Policies]
        NETWORK[Network Policies]
    end
```

### 🔑 Authentication & Authorization

- **JWT Tokens**: Stateless authentication
- **OAuth 2.0**: Third-party integration
- **RBAC**: Role-based permissions
- **API Keys**: Service-to-service auth

### 🔒 Data Protection

- **Encryption at Rest**: AES-256 for databases
- **Encryption in Transit**: TLS 1.3 for all communications
- **Secret Management**: HashiCorp Vault or cloud KMS
- **Data Masking**: PII protection in logs

---

## 📈 Scalability Considerations

### 🔄 Horizontal Scaling

| Component | Scaling Strategy | Considerations |
|-----------|------------------|----------------|
| **Dashboard** | Load balancer + multiple instances | Stateless design |
| **API** | Auto-scaling based on CPU/memory | Connection pooling |
| **Workers** | Queue-based scaling | Task distribution |
| **Database** | Read replicas + connection pooling | Data partitioning |

### 📊 Performance Optimization

```mermaid
graph LR
    subgraph "Frontend Optimization"
        CACHE[Browser Caching]
        CDN[Content Delivery Network]
        LAZY[Lazy Loading]
    end
    
    subgraph "Backend Optimization"
        POOL[Connection Pooling]
        REDIS_CACHE[Redis Caching]
        ASYNC[Async Processing]
    end
    
    subgraph "Database Optimization"
        INDEX[Database Indexing]
        PARTITION[Data Partitioning]
        REPLICA[Read Replicas]
    end
```

### 📈 Monitoring & Alerting

- **Application Metrics**: Response time, error rate, throughput
- **Infrastructure Metrics**: CPU, memory, disk, network
- **Business Metrics**: Model accuracy, drift detection, SLA compliance
- **Custom Alerts**: Threshold-based and anomaly detection

---

## 🔧 Technology Stack Summary

| Layer | Production | Development | Testing |
|-------|------------|-------------|---------|
| **Frontend** | Streamlit, Plotly | Jupyter, Streamlit | Selenium, PyTest |
| **Backend** | FastAPI, SQLAlchemy | FastAPI, SQLite | PyTest, TestClient |
| **Database** | PostgreSQL, Redis | PostgreSQL, Redis | PyTest fixtures |
| **ML** | Scikit-learn, MLflow | Jupyter, MLflow | ML test frameworks |
| **Deployment** | Kubernetes, Docker | Docker Compose | Local testing |
| **Monitoring** | Prometheus, Grafana | Local monitoring | Mock services |

---

## 📚 Next Steps

1. **Phase 1**: Complete MDT Dashboard implementation
2. **Phase 2**: Implement Project B (Multi-Model Deployment)
3. **Phase 3**: Add Projects C-E (Feature Store, Explainability, Data Quality)
4. **Phase 4**: Implement Projects F-J (Advanced MLOps features)

---

*For detailed implementation guides, see the [Development Documentation](development-guide.md)*

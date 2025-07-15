# 🚀 Deployment Guide

## 📋 Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Environment Configuration](#environment-configuration)
7. [Monitoring & Logging](#monitoring--logging)
8. [Security Configuration](#security-configuration)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Deployment Overview

The MDT Dashboard supports multiple deployment strategies to accommodate different environments and requirements:

| Environment | Method | Use Case | Complexity |
|-------------|--------|----------|------------|
| **Development** | Local + Docker Compose | Development and testing | Low |
| **Staging** | Kubernetes | Pre-production testing | Medium |
| **Production** | Cloud + Kubernetes | Production workloads | High |
| **Edge** | Single Container | Edge computing scenarios | Low |

### 🏗️ Deployment Architecture

```mermaid
graph TB
    subgraph "Development"
        DEV[Local Development]
        COMPOSE[Docker Compose]
    end
    
    subgraph "Staging"
        K8S_STAGE[Kubernetes Staging]
        HELM_STAGE[Helm Charts]
    end
    
    subgraph "Production"
        subgraph "Cloud Providers"
            AWS[AWS EKS]
            GCP[GCP GKE]
            AZURE[Azure AKS]
        end
    end
    
    DEV --> COMPOSE
    COMPOSE --> K8S_STAGE
    K8S_STAGE --> AWS
    K8S_STAGE --> GCP
    K8S_STAGE --> AZURE
```

---

## 💻 Local Development

### 🚀 Quick Start

```bash
# Clone repository
git clone <repository-url>
cd MLOPS/Project/MDT-Dashboard

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install poetry
poetry install

# Start local services
docker-compose -f docker-compose.dev.yml up -d

# Run migrations
poetry run alembic upgrade head

# Start application
poetry run streamlit run src/mdt_dashboard/dashboard/main.py
```

### 🔧 Development Services

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mdt_db
      POSTGRES_USER: mdt_user
      POSTGRES_PASSWORD: mdt_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  mlflow:
    image: python:3.9-slim
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000 
               --backend-store-uri postgresql://mdt_user:mdt_password@postgres:5432/mdt_db
               --default-artifact-root ./mlruns"
    ports:
      - "5000:5000"
    depends_on:
      - postgres
    volumes:
      - mlflow_data:/mlflow

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
```

---

## 🐳 Docker Deployment

### 📦 Container Build

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash mdt
USER mdt

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "src/mdt_dashboard/dashboard/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 🚀 Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  mdt-dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=postgresql://mdt_user:${POSTGRES_PASSWORD}@postgres:5432/mdt_db
      - REDIS_URL=redis://redis:6379/0
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - postgres
      - redis
      - mlflow
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mdt-api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://mdt_user:${POSTGRES_PASSWORD}@postgres:5432/mdt_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  mdt-worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - DATABASE_URL=postgresql://mdt_user:${POSTGRES_PASSWORD}@postgres:5432/mdt_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mdt_db
      POSTGRES_USER: mdt_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  mlflow:
    image: python:3.9-slim
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000 
               --backend-store-uri postgresql://mdt_user:${POSTGRES_PASSWORD}@postgres:5432/mdt_db
               --default-artifact-root s3://mlflow-artifacts"
    ports:
      - "5000:5000"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    depends_on:
      - postgres
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - mdt-dashboard
      - mdt-api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
  prometheus_data:
  grafana_data:
```

### 🏃 Running Production Stack

```bash
# Set environment variables
export POSTGRES_PASSWORD=secure_password
export GRAFANA_PASSWORD=admin_password
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f mdt-dashboard
```

---

## ☸️ Kubernetes Deployment

### 📊 Namespace Configuration

```yaml
# k8s/00-namespace-config.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mdt-dashboard
  labels:
    name: mdt-dashboard
    environment: production

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: mdt-config
  namespace: mdt-dashboard
data:
  DATABASE_HOST: postgres-service
  DATABASE_PORT: "5432"
  DATABASE_NAME: mdt_db
  REDIS_HOST: redis-service
  REDIS_PORT: "6379"
  MLFLOW_HOST: mlflow-service
  MLFLOW_PORT: "5000"
  LOG_LEVEL: INFO
  ENVIRONMENT: production
```

### 🗄️ Database Deployment

```yaml
# k8s/01-postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: mdt-dashboard
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: mdt_db
        - name: POSTGRES_USER
          value: mdt_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - mdt_user
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - mdt_user
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: mdt-dashboard
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP

---
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: mdt-dashboard
type: Opaque
data:
  password: <base64-encoded-password>
```

### 🎯 Application Deployment

```yaml
# k8s/03-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mdt-api
  namespace: mdt-dashboard
  labels:
    app: mdt-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mdt-api
  template:
    metadata:
      labels:
        app: mdt-api
    spec:
      containers:
      - name: mdt-api
        image: mdt-dashboard:latest
        command: ["uvicorn", "src.mdt_dashboard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: postgresql://mdt_user:$(POSTGRES_PASSWORD)@postgres-service:5432/mdt_db
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: REDIS_URL
          value: redis://redis-service:6379/0
        envFrom:
        - configMapRef:
            name: mdt-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: mdt-api-service
  namespace: mdt-dashboard
spec:
  selector:
    app: mdt-api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mdt-dashboard
  namespace: mdt-dashboard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mdt-dashboard
  template:
    metadata:
      labels:
        app: mdt-dashboard
    spec:
      containers:
      - name: mdt-dashboard
        image: mdt-dashboard:latest
        ports:
        - containerPort: 8501
        env:
        - name: API_URL
          value: http://mdt-api-service:8000
        envFrom:
        - configMapRef:
            name: mdt-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---
apiVersion: v1
kind: Service
metadata:
  name: mdt-dashboard-service
  namespace: mdt-dashboard
spec:
  selector:
    app: mdt-dashboard
  ports:
  - port: 8501
    targetPort: 8501
  type: LoadBalancer
```

### 🔄 Ingress Configuration

```yaml
# k8s/04-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mdt-ingress
  namespace: mdt-dashboard
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
spec:
  tls:
  - hosts:
    - mdt.yourdomain.com
    secretName: mdt-tls-secret
  rules:
  - host: mdt.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: mdt-api-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mdt-dashboard-service
            port:
              number: 8501
```

### 🚀 Deployment Commands

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n mdt-dashboard

# View logs
kubectl logs -f deployment/mdt-dashboard -n mdt-dashboard

# Port forward for local access
kubectl port-forward svc/mdt-dashboard-service 8501:8501 -n mdt-dashboard

# Scale deployment
kubectl scale deployment/mdt-api --replicas=5 -n mdt-dashboard

# Update deployment
kubectl set image deployment/mdt-dashboard mdt-dashboard=mdt-dashboard:v2.0.0 -n mdt-dashboard
```

---

## ☁️ Cloud Deployment

### ☁️ AWS Deployment

#### EKS Cluster Setup

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name mdt-dashboard \
  --version 1.24 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name mdt-dashboard
```

#### RDS Database

```yaml
# terraform/aws/rds.tf
resource "aws_db_instance" "mdt_db" {
  identifier = "mdt-dashboard-db"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true
  
  db_name  = "mdt_db"
  username = "mdt_user"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.mdt.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  deletion_protection = false
  
  tags = {
    Name        = "MDT Dashboard Database"
    Environment = "production"
  }
}
```

### 🌐 Google Cloud Deployment

#### GKE Cluster

```bash
# Set project and region
gcloud config set project your-project-id
gcloud config set compute/region us-central1

# Create GKE cluster
gcloud container clusters create mdt-dashboard \
  --num-nodes=3 \
  --machine-type=e2-medium \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=5 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials mdt-dashboard
```

#### Cloud SQL

```yaml
# terraform/gcp/cloudsql.tf
resource "google_sql_database_instance" "mdt_db" {
  name             = "mdt-dashboard-db"
  database_version = "POSTGRES_15"
  region           = "us-central1"
  
  settings {
    tier = "db-f1-micro"
    
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
    }
    
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        value = "0.0.0.0/0"
        name  = "all"
      }
    }
  }
  
  deletion_protection = false
}

resource "google_sql_database" "mdt_database" {
  name     = "mdt_db"
  instance = google_sql_database_instance.mdt_db.name
}

resource "google_sql_user" "mdt_user" {
  name     = "mdt_user"
  instance = google_sql_database_instance.mdt_db.name
  password = var.db_password
}
```

### 🔷 Azure Deployment

#### AKS Cluster

```bash
# Create resource group
az group create --name mdt-dashboard-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group mdt-dashboard-rg \
  --name mdt-dashboard \
  --node-count 3 \
  --node-vm-size Standard_B2s \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group mdt-dashboard-rg --name mdt-dashboard
```

---

## 🔧 Environment Configuration

### 📊 Configuration Management

```python
# src/mdt_dashboard/core/config.py
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    app_name: str = "MDT Dashboard"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # Database
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 300
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "mdt_dashboard"
    
    # Security
    secret_key: str
    access_token_expire_minutes: int = 30
    
    # Monitoring
    prometheus_multiproc_dir: Optional[str] = None
    log_level: str = "INFO"
    
    # Feature flags
    enable_drift_detection: bool = True
    enable_model_training: bool = True
    enable_alerting: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### 🔐 Secret Management

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: mdt-secrets
  namespace: mdt-dashboard
type: Opaque
data:
  database-password: <base64-encoded-password>
  secret-key: <base64-encoded-secret>
  mlflow-s3-access-key: <base64-encoded-key>
  mlflow-s3-secret-key: <base64-encoded-secret>
```

### 🌍 Environment-Specific Configs

```yaml
# environments/development.yaml
environment: development
debug: true
log_level: DEBUG
database_pool_size: 5
redis_cache_ttl: 60

# environments/staging.yaml
environment: staging
debug: false
log_level: INFO
database_pool_size: 10
redis_cache_ttl: 300

# environments/production.yaml
environment: production
debug: false
log_level: WARNING
database_pool_size: 20
redis_cache_ttl: 600
enable_model_training: false  # Disable in production
```

---

## 📊 Monitoring & Logging

### 📈 Prometheus Configuration

```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'mdt-api'
    static_configs:
      - targets: ['mdt-api-service:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'mdt-dashboard'
    static_configs:
      - targets: ['mdt-dashboard-service:8501']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 📊 Grafana Dashboards

```json
{
  "dashboard": {
    "title": "MDT Dashboard - System Overview",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Model Drift Detections",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(drift_detections_total[1h]))",
            "legendFormat": "Drift Detections/hour"
          }
        ]
      }
    ]
  }
}
```

### 📝 Centralized Logging

```yaml
# ELK Stack configuration
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./config/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

---

## 🔒 Security Configuration

### 🛡️ Network Security

```yaml
# k8s/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mdt-network-policy
  namespace: mdt-dashboard
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  - from:
    - podSelector:
        matchLabels:
          app: mdt-api
    ports:
    - protocol: TCP
      port: 5432
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 80
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 5432
    - protocol: TCP
      port: 6379
```

### 🔐 RBAC Configuration

```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mdt-service-account
  namespace: mdt-dashboard

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mdt-role
  namespace: mdt-dashboard
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mdt-role-binding
  namespace: mdt-dashboard
subjects:
- kind: ServiceAccount
  name: mdt-service-account
  namespace: mdt-dashboard
roleRef:
  kind: Role
  name: mdt-role
  apiGroup: rbac.authorization.k8s.io
```

### 🔑 SSL/TLS Configuration

```nginx
# config/nginx.conf
server {
    listen 80;
    server_name mdt.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name mdt.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;

    location /api/ {
        proxy_pass http://mdt-api-service:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://mdt-dashboard-service:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 🔧 Troubleshooting

### 📊 Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Database Connection** | Connection refused errors | Check database credentials and network connectivity |
| **Redis Connection** | Cache misses, slow response | Verify Redis is running and accessible |
| **Pod CrashLoopBackOff** | Pods keep restarting | Check logs for application errors |
| **Ingress Not Working** | 404 errors | Verify ingress controller and DNS configuration |
| **High Memory Usage** | Out of memory errors | Increase resource limits or optimize queries |

### 📝 Debugging Commands

```bash
# Kubernetes debugging
kubectl describe pod <pod-name> -n mdt-dashboard
kubectl logs -f <pod-name> -n mdt-dashboard
kubectl exec -it <pod-name> -n mdt-dashboard -- /bin/bash

# Docker debugging
docker logs <container-id>
docker exec -it <container-id> /bin/bash
docker stats

# Database debugging
kubectl port-forward svc/postgres-service 5432:5432 -n mdt-dashboard
psql -h localhost -p 5432 -U mdt_user -d mdt_db

# Application debugging
kubectl port-forward svc/mdt-api-service 8000:8000 -n mdt-dashboard
curl http://localhost:8000/health
```

### 🔍 Health Checks

```python
# Health check endpoints
@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check with dependencies."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "checks": {}
    }
    
    # Database check
    try:
        db.execute("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Redis check
    try:
        redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return health_status
```

---

## 📚 Additional Resources

### 🔗 Useful Links

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Helm Charts](https://helm.sh/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)

### 🛠️ Tools

- **kubectl**: Kubernetes CLI
- **helm**: Package manager for Kubernetes
- **docker**: Container runtime
- **terraform**: Infrastructure as code
- **ansible**: Configuration management

---

*For additional support, please refer to the [troubleshooting guide](troubleshooting.md) or contact the development team.*

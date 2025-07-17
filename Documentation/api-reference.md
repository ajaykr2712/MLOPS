# 📖 API Reference Documentation

## 📋 Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Models API](#models-api)
4. [Predictions API](#predictions-api)
5. [Drift Detection API](#drift-detection-api)
6. [Monitoring API](#monitoring-api)
7. [User Management API](#user-management-api)
8. [Webhooks API](#webhooks-api)
9. [Error Handling](#error-handling)
10. [Rate Limiting](#rate-limiting)

---

## 🎯 API Overview

The MDT Dashboard API is built using **FastAPI** and follows **RESTful** design principles. All endpoints return JSON responses and support standard HTTP methods.

### 🔗 Base URL

```
Production:  https://api.mdt-dashboard.com/api/v1
Staging:     https://staging-api.mdt-dashboard.com/api/v1
Development: http://localhost:8000/api/v1
```

### 📊 API Specification

- **Version**: 1.0.0
- **Protocol**: HTTPS (HTTP in development)
- **Format**: JSON
- **Authentication**: JWT Bearer Token
- **Documentation**: OpenAPI 3.0 (Swagger)

### 🔧 Interactive Documentation

- **Swagger UI**: `{BASE_URL}/docs`
- **ReDoc**: `{BASE_URL}/redoc`
- **OpenAPI Schema**: `{BASE_URL}/openapi.json`

---

## 🔐 Authentication

### 🎫 JWT Token Authentication

All protected endpoints require a valid JWT token in the Authorization header.

```http
Authorization: Bearer <jwt_token>
```

### 🔑 Login Endpoint

```http
POST /auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": 1,
    "username": "user@example.com",
    "role": "data_scientist"
  }
}
```

### 🔄 Token Refresh

```http
POST /auth/refresh
Authorization: Bearer <refresh_token>
```

### 🚪 Logout

```http
POST /auth/logout
Authorization: Bearer <jwt_token>
```

---

## 🤖 Models API

### 📋 List Models

```http
GET /models
Authorization: Bearer <jwt_token>
```

**Query Parameters:**
- `skip` (int): Number of records to skip (default: 0)
- `limit` (int): Maximum number of records (default: 100)
- `model_type` (str): Filter by model type
- `status` (str): Filter by status (active, inactive, training)

**Response:**
```json
{
  "models": [
    {
      "id": 1,
      "name": "fraud_detection_v1",
      "version": "1.0.0",
      "model_type": "classification",
      "status": "active",
      "accuracy": 0.95,
      "created_at": "2025-07-01T10:00:00Z",
      "updated_at": "2025-07-15T14:30:00Z",
      "metadata": {
        "algorithm": "random_forest",
        "features": 25,
        "training_samples": 100000
      }
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 100
}
```

### 📝 Create Model

```http
POST /models
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "name": "fraud_detection_v2",
  "version": "2.0.0",
  "model_type": "classification",
  "description": "Updated fraud detection model",
  "algorithm": "xgboost",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1
  },
  "training_config": {
    "data_source": "s3://data-bucket/fraud-training.csv",
    "validation_split": 0.2,
    "random_state": 42
  }
}
```

**Response:**
```json
{
  "id": 2,
  "name": "fraud_detection_v2",
  "version": "2.0.0",
  "model_type": "classification",
  "status": "created",
  "created_at": "2025-07-15T15:00:00Z",
  "training_job_id": "job_123456"
}
```

### 🔍 Get Model Details

```http
GET /models/{model_id}
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "id": 1,
  "name": "fraud_detection_v1",
  "version": "1.0.0",
  "model_type": "classification",
  "status": "active",
  "description": "Production fraud detection model",
  "algorithm": "random_forest",
  "accuracy": 0.95,
  "precision": 0.92,
  "recall": 0.89,
  "f1_score": 0.91,
  "created_at": "2025-07-01T10:00:00Z",
  "updated_at": "2025-07-15T14:30:00Z",
  "artifact_path": "s3://models/fraud_detection_v1.pkl",
  "metadata": {
    "algorithm": "random_forest",
    "features": 25,
    "training_samples": 100000,
    "feature_importance": {
      "transaction_amount": 0.25,
      "merchant_category": 0.18,
      "time_since_last_transaction": 0.15
    }
  },
  "drift_threshold": 0.05,
  "monitoring_enabled": true
}
```

### ✏️ Update Model

```http
PUT /models/{model_id}
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "description": "Updated description",
  "drift_threshold": 0.03,
  "monitoring_enabled": false
}
```

### 🗑️ Delete Model

```http
DELETE /models/{model_id}
Authorization: Bearer <jwt_token>
```

### 🚀 Deploy Model

```http
POST /models/{model_id}/deploy
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "environment": "production",
  "scaling_config": {
    "min_instances": 2,
    "max_instances": 10,
    "target_cpu_utilization": 70
  },
  "resource_config": {
    "cpu": "1000m",
    "memory": "2Gi"
  }
}
```

---

## 🎯 Predictions API

### 🔮 Make Prediction

```http
POST /models/{model_id}/predict
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "features": {
    "transaction_amount": 150.00,
    "merchant_category": "grocery",
    "time_since_last_transaction": 3600,
    "user_age": 35,
    "account_balance": 2500.00
  },
  "prediction_id": "pred_123456",
  "metadata": {
    "request_source": "mobile_app",
    "user_id": "user_789"
  }
}
```

**Response:**
```json
{
  "prediction_id": "pred_123456",
  "model_id": 1,
  "model_version": "1.0.0",
  "prediction": {
    "class": "legitimate",
    "probability": 0.92,
    "confidence": "high"
  },
  "feature_importance": {
    "transaction_amount": 0.25,
    "merchant_category": 0.18,
    "time_since_last_transaction": 0.15
  },
  "timestamp": "2025-07-15T15:30:00Z",
  "latency_ms": 45
}
```

### 📊 Batch Predictions

```http
POST /models/{model_id}/predict/batch
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "instances": [
    {
      "prediction_id": "pred_001",
      "features": {
        "transaction_amount": 150.00,
        "merchant_category": "grocery"
      }
    },
    {
      "prediction_id": "pred_002",
      "features": {
        "transaction_amount": 5000.00,
        "merchant_category": "electronics"
      }
    }
  ]
}
```

**Response:**
```json
{
  "batch_id": "batch_789",
  "model_id": 1,
  "predictions": [
    {
      "prediction_id": "pred_001",
      "prediction": {
        "class": "legitimate",
        "probability": 0.92
      }
    },
    {
      "prediction_id": "pred_002",
      "prediction": {
        "class": "fraudulent",
        "probability": 0.89
      }
    }
  ],
  "total_predictions": 2,
  "avg_latency_ms": 38,
  "timestamp": "2025-07-15T15:35:00Z"
}
```

### 📈 Get Predictions

```http
GET /models/{model_id}/predictions
Authorization: Bearer <jwt_token>
```

**Query Parameters:**
- `start_date` (ISO 8601): Start date filter
- `end_date` (ISO 8601): End date filter
- `limit` (int): Maximum number of records
- `skip` (int): Number of records to skip

**Response:**
```json
{
  "predictions": [
    {
      "id": 1,
      "prediction_id": "pred_123456",
      "model_id": 1,
      "features": {
        "transaction_amount": 150.00,
        "merchant_category": "grocery"
      },
      "prediction": {
        "class": "legitimate",
        "probability": 0.92
      },
      "actual": "legitimate",
      "timestamp": "2025-07-15T15:30:00Z"
    }
  ],
  "total": 1000,
  "accuracy": 0.95
}
```

### ✅ Update Prediction with Actual

```http
PATCH /predictions/{prediction_id}
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "actual": "fraudulent",
  "feedback": "False negative - transaction was actually fraudulent"
}
```

---

## 📊 Drift Detection API

### 🔍 Detect Drift

```http
POST /models/{model_id}/drift/detect
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "current_data": [
    {
      "transaction_amount": 200.00,
      "merchant_category": "online",
      "user_age": 25
    }
  ],
  "reference_window": "7d",
  "algorithms": ["ks_test", "psi", "js_distance"],
  "features": ["transaction_amount", "user_age"]
}
```

**Response:**
```json
{
  "drift_detection_id": "drift_123",
  "model_id": 1,
  "timestamp": "2025-07-15T16:00:00Z",
  "overall_drift": {
    "is_drift_detected": true,
    "drift_score": 0.08,
    "threshold": 0.05
  },
  "feature_drift": {
    "transaction_amount": {
      "ks_test": {
        "statistic": 0.12,
        "p_value": 0.03,
        "is_drift": true
      },
      "psi": {
        "score": 0.15,
        "is_drift": true
      }
    },
    "user_age": {
      "ks_test": {
        "statistic": 0.04,
        "p_value": 0.45,
        "is_drift": false
      }
    }
  },
  "recommendations": [
    "Consider retraining the model",
    "Investigate data source changes",
    "Review feature engineering pipeline"
  ]
}
```

### 📈 Get Drift History

```http
GET /models/{model_id}/drift/history
Authorization: Bearer <jwt_token>
```

**Query Parameters:**
- `start_date` (ISO 8601): Start date
- `end_date` (ISO 8601): End date
- `granularity` (str): hour, day, week (default: day)

**Response:**
```json
{
  "drift_history": [
    {
      "timestamp": "2025-07-15T00:00:00Z",
      "drift_score": 0.03,
      "is_drift_detected": false,
      "features_with_drift": []
    },
    {
      "timestamp": "2025-07-16T00:00:00Z",
      "drift_score": 0.08,
      "is_drift_detected": true,
      "features_with_drift": ["transaction_amount"]
    }
  ],
  "summary": {
    "total_detections": 50,
    "drift_detections": 3,
    "drift_rate": 0.06,
    "avg_drift_score": 0.025
  }
}
```

### ⚙️ Configure Drift Detection

```http
PUT /models/{model_id}/drift/config
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "threshold": 0.05,
  "algorithms": ["ks_test", "psi"],
  "features": ["transaction_amount", "user_age"],
  "detection_frequency": "hourly",
  "alert_channels": ["email", "slack"],
  "auto_retrain": true,
  "min_samples": 1000
}
```

---

## 📈 Monitoring API

### 📊 Get Model Metrics

```http
GET /models/{model_id}/metrics
Authorization: Bearer <jwt_token>
```

**Query Parameters:**
- `metric_type` (str): accuracy, latency, throughput, drift
- `time_range` (str): 1h, 24h, 7d, 30d
- `granularity` (str): minute, hour, day

**Response:**
```json
{
  "metrics": {
    "accuracy": {
      "current": 0.95,
      "trend": "stable",
      "history": [
        {
          "timestamp": "2025-07-15T15:00:00Z",
          "value": 0.95
        }
      ]
    },
    "latency": {
      "p50": 25,
      "p95": 85,
      "p99": 150,
      "unit": "ms"
    },
    "throughput": {
      "current": 1200,
      "unit": "requests_per_minute"
    },
    "drift_score": {
      "current": 0.03,
      "threshold": 0.05,
      "status": "normal"
    }
  },
  "alerts": [
    {
      "id": "alert_123",
      "type": "drift_detected",
      "severity": "warning",
      "message": "Drift detected in feature: transaction_amount",
      "timestamp": "2025-07-15T16:00:00Z"
    }
  ]
}
```

### 🎯 Get System Health

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-15T16:00:00Z",
  "version": "1.0.0",
  "checks": {
    "database": "healthy",
    "redis": "healthy",
    "mlflow": "healthy",
    "celery_workers": "healthy"
  },
  "metrics": {
    "uptime": "7d 12h 30m",
    "cpu_usage": 65.2,
    "memory_usage": 78.5,
    "disk_usage": 45.3
  }
}
```

### 📈 Get Performance Metrics

```http
GET /metrics/performance
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "api": {
    "total_requests": 150000,
    "avg_response_time": 145.5,
    "error_rate": 0.02,
    "requests_per_second": 85.3
  },
  "models": {
    "total_models": 15,
    "active_models": 12,
    "total_predictions": 500000,
    "avg_prediction_latency": 35.2
  },
  "drift_detection": {
    "total_detections": 1200,
    "drift_events": 15,
    "false_positive_rate": 0.05
  }
}
```

---

## 👥 User Management API

### 📋 List Users

```http
GET /users
Authorization: Bearer <admin_jwt_token>
```

**Response:**
```json
{
  "users": [
    {
      "id": 1,
      "username": "john.doe@company.com",
      "role": "data_scientist",
      "is_active": true,
      "created_at": "2025-06-01T10:00:00Z",
      "last_login": "2025-07-15T14:30:00Z"
    }
  ],
  "total": 25
}
```

### 👤 Create User

```http
POST /users
Authorization: Bearer <admin_jwt_token>
Content-Type: application/json

{
  "username": "jane.smith@company.com",
  "password": "secure_password123",
  "role": "ml_engineer",
  "permissions": ["read_models", "create_predictions", "view_metrics"]
}
```

### ✏️ Update User

```http
PUT /users/{user_id}
Authorization: Bearer <admin_jwt_token>
Content-Type: application/json

{
  "role": "senior_data_scientist",
  "permissions": ["admin_access"],
  "is_active": true
}
```

---

## 🔗 Webhooks API

### 📝 Create Webhook

```http
POST /webhooks
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "url": "https://your-system.com/webhook",
  "events": ["drift_detected", "model_deployed", "prediction_accuracy_low"],
  "secret": "webhook_secret_key",
  "active": true,
  "retry_config": {
    "max_retries": 3,
    "retry_delay": 5
  }
}
```

### 📋 List Webhooks

```http
GET /webhooks
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "webhooks": [
    {
      "id": 1,
      "url": "https://your-system.com/webhook",
      "events": ["drift_detected"],
      "active": true,
      "created_at": "2025-07-01T10:00:00Z",
      "last_triggered": "2025-07-15T16:00:00Z",
      "success_rate": 0.98
    }
  ]
}
```

### 🧪 Test Webhook

```http
POST /webhooks/{webhook_id}/test
Authorization: Bearer <jwt_token>
```

---

## ❌ Error Handling

### 📊 Standard Error Response

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model with ID 999 not found",
    "details": {
      "model_id": 999,
      "available_models": [1, 2, 3, 4, 5]
    },
    "timestamp": "2025-07-15T16:00:00Z",
    "request_id": "req_123456"
  }
}
```

### 🔢 HTTP Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful GET, PUT, PATCH requests |
| 201 | Created | Successful POST requests |
| 202 | Accepted | Async operations accepted |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource already exists |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side errors |

### 🔍 Common Error Codes

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `INVALID_TOKEN` | 401 | JWT token is invalid or expired |
| `MODEL_NOT_FOUND` | 404 | Specified model does not exist |
| `PREDICTION_FAILED` | 500 | Model prediction failed |
| `DRIFT_DETECTION_ERROR` | 500 | Drift detection algorithm failed |
| `INSUFFICIENT_DATA` | 422 | Not enough data for operation |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `VALIDATION_ERROR` | 422 | Request validation failed |

---

## ⏱️ Rate Limiting

### 📊 Rate Limits

| Endpoint Type | Limit | Window | Header |
|---------------|-------|--------|---------|
| Authentication | 10 requests | 1 minute | `X-RateLimit-Auth` |
| Model Operations | 100 requests | 1 minute | `X-RateLimit-Models` |
| Predictions | 1000 requests | 1 minute | `X-RateLimit-Predictions` |
| Drift Detection | 50 requests | 1 minute | `X-RateLimit-Drift` |

### 📝 Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1626364800
X-RateLimit-Window: 60
```

### ⚠️ Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded for predictions endpoint",
    "details": {
      "limit": 1000,
      "window": 60,
      "reset_time": "2025-07-15T16:01:00Z"
    }
  }
}
```

---

## 🛠️ SDK Examples and Integrations

### Python SDK

```python
from mdt_dashboard import MDTClient
import pandas as pd
import numpy as np

# Initialize client with authentication
client = MDTClient(
    base_url="https://api.mdt-dashboard.com/api/v1",
    api_key="your-api-key",
    timeout=30,
    retry_attempts=3
)

# Advanced model registration with validation
model_config = {
    "name": "fraud_detection_v4",
    "description": "Advanced fraud detection with ensemble methods",
    "framework": "scikit-learn",
    "version": "4.0.0",
    "algorithm": "ensemble",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 15,
        "learning_rate": 0.1
    },
    "training_config": {
        "dataset_size": 1000000,
        "features": ["amount", "merchant_category", "time_of_day", "location"],
        "target": "is_fraud",
        "cross_validation_folds": 5,
        "test_size": 0.2
    },
    "deployment_config": {
        "environment": "production",
        "auto_scaling": True,
        "min_replicas": 2,
        "max_replicas": 10,
        "cpu_threshold": 70,
        "memory_threshold": 80
    },
    "monitoring_config": {
        "drift_detection": True,
        "performance_monitoring": True,
        "alert_thresholds": {
            "accuracy_drop": 0.05,
            "latency_increase": 100
        }
    }
}

# Register model with comprehensive configuration
model = client.models.register(**model_config)
print(f"Model registered with ID: {model.id}")

# Batch prediction with error handling
def batch_predict_with_retry(model_id, data_file, max_retries=3):
    """
    Perform batch predictions with automatic retry logic.
    """
    for attempt in range(max_retries):
        try:
            batch_job = client.predictions.submit_batch(
                model_id=model_id,
                input_file=data_file,
                output_format="csv",
                batch_size=1000,
                parallel_workers=4
            )
            
            # Monitor batch job progress
            while batch_job.status in ['queued', 'running']:
                time.sleep(30)
                batch_job = client.predictions.get_batch_status(batch_job.id)
                print(f"Batch progress: {batch_job.progress}%")
            
            if batch_job.status == 'completed':
                return batch_job.output_file
            else:
                raise Exception(f"Batch job failed: {batch_job.error_message}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e

# Advanced drift monitoring setup
drift_config = client.drift.configure(
    model_id=model.id,
    detection_methods=["ks_test", "psi", "jensen_shannon"],
    monitoring_frequency="hourly",
    reference_data_config={
        "type": "rolling_window",
        "window_size": "30d",
        "min_samples": 1000
    },
    thresholds={
        "ks_test": {"p_value": 0.05, "bonferroni_correction": True},
        "psi": {"threshold": 0.2, "num_bins": 10},
        "jensen_shannon": {"threshold": 0.1}
    },
    alert_config={
        "escalation_policy": "immediate",
        "notification_channels": ["slack", "email", "pagerduty"],
        "suppression_window": "1h"
    }
)

# Real-time monitoring dashboard
class ModelMonitor:
    def __init__(self, client, model_id):
        self.client = client
        self.model_id = model_id
        self.metrics_cache = {}
    
    def get_real_time_metrics(self):
        """Get real-time model performance metrics."""
        return self.client.monitoring.get_metrics(
            model_id=self.model_id,
            time_range="1h",
            metrics=["latency", "throughput", "error_rate", "accuracy"],
            aggregation="avg"
        )
    
    def detect_anomalies(self):
        """Detect performance anomalies using statistical methods."""
        metrics = self.get_real_time_metrics()
        anomalies = []
        
        for metric_name, values in metrics.items():
            if len(values) > 10:  # Need sufficient data
                mean = np.mean(values)
                std = np.std(values)
                threshold = mean + 3 * std  # 3-sigma rule
                
                current_value = values[-1]
                if current_value > threshold:
                    anomalies.append({
                        "metric": metric_name,
                        "current_value": current_value,
                        "threshold": threshold,
                        "severity": "high" if current_value > mean + 4 * std else "medium"
                    })
        
        return anomalies
    
    def auto_remediation(self, anomaly):
        """Automatic remediation actions for common issues."""
        if anomaly["metric"] == "latency" and anomaly["severity"] == "high":
            # Scale up replicas
            self.client.models.scale(
                model_id=self.model_id,
                target_replicas=min(self.client.models.get(self.model_id).max_replicas, 
                                  self.client.models.get(self.model_id).current_replicas + 2)
            )
            return "Scaled up model replicas"
        
        elif anomaly["metric"] == "error_rate":
            # Switch to backup model version
            backup_version = self.client.models.get_backup_version(self.model_id)
            if backup_version:
                self.client.models.rollback(self.model_id, backup_version)
                return f"Rolled back to version {backup_version}"
        
        return "No automatic remediation available"

# Usage example
monitor = ModelMonitor(client, model.id)
anomalies = monitor.detect_anomalies()

for anomaly in anomalies:
    action_taken = monitor.auto_remediation(anomaly)
    print(f"Anomaly detected in {anomaly['metric']}: {action_taken}")
```

### JavaScript/TypeScript SDK

```typescript
import { MDTClient, Model, PredictionRequest, DriftAnalysis } from '@mdt/dashboard-js';

interface MLOpsConfig {
  apiKey: string;
  baseUrl: string;
  timeout?: number;
  retryConfig?: {
    attempts: number;
    backoff: 'linear' | 'exponential';
    delay: number;
  };
}

class AdvancedMLOpsClient {
  private client: MDTClient;
  private eventHandlers: Map<string, Function[]> = new Map();

  constructor(config: MLOpsConfig) {
    this.client = new MDTClient({
      baseUrl: config.baseUrl,
      apiKey: config.apiKey,
      timeout: config.timeout || 30000,
      retry: config.retryConfig || {
        attempts: 3,
        backoff: 'exponential',
        delay: 1000
      }
    });

    // Set up WebSocket connection for real-time updates
    this.setupWebSocketConnection();
  }

  private setupWebSocketConnection(): void {
    const ws = new WebSocket(`${this.client.baseUrl.replace('http', 'ws')}/ws`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleRealtimeEvent(data);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  private handleRealtimeEvent(event: any): void {
    const handlers = this.eventHandlers.get(event.type) || [];
    handlers.forEach(handler => handler(event.data));
  }

  public on(eventType: string, handler: Function): void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, []);
    }
    this.eventHandlers.get(eventType)!.push(handler);
  }

  // Advanced model deployment with A/B testing
  public async deployWithABTesting(
    modelConfig: any,
    abTestConfig: {
      trafficSplit: number;
      duration: string;
      successMetrics: string[];
      failureThresholds: Record<string, number>;
    }
  ): Promise<{ primaryModel: Model; candidateModel: Model; testId: string }> {
    
    // Deploy candidate model
    const candidateModel = await this.client.models.register({
      ...modelConfig,
      deployment: {
        ...modelConfig.deployment,
        trafficPercent: abTestConfig.trafficSplit
      }
    });

    // Set up A/B test monitoring
    const testId = await this.client.experiments.createABTest({
      modelId: candidateModel.id,
      trafficSplit: abTestConfig.trafficSplit,
      duration: abTestConfig.duration,
      successMetrics: abTestConfig.successMetrics,
      failureThresholds: abTestConfig.failureThresholds,
      autoPromote: true,
      autoRollback: true
    });

    // Get primary model for comparison
    const primaryModel = await this.client.models.getCurrent(modelConfig.name);

    return { primaryModel, candidateModel, testId };
  }

  // Real-time prediction with circuit breaker pattern
  public async predictWithCircuitBreaker(
    modelId: string,
    request: PredictionRequest,
    circuitBreakerConfig = {
      failureThreshold: 5,
      timeout: 10000,
      resetTimeout: 60000
    }
  ): Promise<any> {
    
    // Check circuit breaker state
    const circuitState = await this.client.monitoring.getCircuitBreakerState(modelId);
    
    if (circuitState === 'OPEN') {
      // Use fallback model or cached prediction
      return this.getFallbackPrediction(request);
    }

    try {
      const prediction = await this.client.models.predict(modelId, request);
      
      // Reset circuit breaker on success
      await this.client.monitoring.resetCircuitBreaker(modelId);
      
      return prediction;
    } catch (error) {
      // Record failure and potentially open circuit
      await this.client.monitoring.recordCircuitBreakerFailure(modelId);
      throw error;
    }
  }

  private async getFallbackPrediction(request: PredictionRequest): Promise<any> {
    // Implement fallback logic (e.g., use cached predictions, default model, etc.)
    return {
      prediction: null,
      confidence: 0,
      fallback: true,
      message: "Prediction service temporarily unavailable"
    };
  }

  // Comprehensive drift analysis with automated responses
  public async setupAdvancedDriftMonitoring(
    modelId: string,
    config: {
      monitoringSchedule: string;
      driftThresholds: Record<string, number>;
      autoRetrainThreshold: number;
      notificationChannels: string[];
      customActions: Array<{
        condition: string;
        action: string;
        parameters: Record<string, any>;
      }>;
    }
  ): Promise<void> {
    
    // Configure drift detection
    await this.client.drift.configure(modelId, {
      schedule: config.monitoringSchedule,
      thresholds: config.driftThresholds,
      notifications: config.notificationChannels
    });

    // Set up automated responses
    for (const customAction of config.customActions) {
      await this.client.automation.addRule({
        modelId,
        trigger: customAction.condition,
        action: customAction.action,
        parameters: customAction.parameters
      });
    }

    // Monitor for drift events
    this.on('drift.detected', async (driftEvent: any) => {
      console.log(`Drift detected for model ${modelId}:`, driftEvent);
      
      if (driftEvent.score > config.autoRetrainThreshold) {
        // Trigger automatic retraining
        await this.triggerAutoRetrain(modelId, driftEvent);
      }
    });
  }

  private async triggerAutoRetrain(modelId: string, driftEvent: any): Promise<void> {
    const retrainJob = await this.client.training.retrain({
      baseModelId: modelId,
      includeRecentData: true,
      dataWindow: '30d',
      hyperparameterTuning: true,
      validationStrategy: 'time_series_split',
      deployOnSuccess: false  // Manual approval required
    });

    console.log(`Retraining job ${retrainJob.id} started for model ${modelId}`);
  }

  // Model performance dashboard integration
  public async createPerformanceDashboard(modelIds: string[]): Promise<string> {
    const dashboardConfig = {
      name: `Model Performance Dashboard - ${new Date().toISOString()}`,
      models: modelIds,
      metrics: [
        'accuracy', 'precision', 'recall', 'f1_score',
        'latency', 'throughput', 'error_rate',
        'drift_score', 'data_quality_score'
      ],
      timeRanges: ['1h', '24h', '7d', '30d'],
      visualizations: [
        'time_series_plots',
        'distribution_comparisons',
        'correlation_matrices',
        'performance_heatmaps'
      ],
      alerts: {
        thresholds: {
          accuracy_drop: 0.05,
          latency_increase: 100,
          error_rate_spike: 0.02
        },
        notifications: ['email', 'slack']
      }
    };

    const dashboard = await this.client.dashboards.create(dashboardConfig);
    return dashboard.url;
  }
}

// Usage example
const mlopsClient = new AdvancedMLOpsClient({
  apiKey: process.env.MDT_API_KEY!,
  baseUrl: 'https://api.mdt-dashboard.com/api/v1',
  timeout: 30000,
  retryConfig: {
    attempts: 3,
    backoff: 'exponential',
    delay: 1000
  }
});

// Set up real-time monitoring
mlopsClient.on('model.deployed', (model) => {
  console.log(`Model ${model.name} deployed successfully`);
});

mlopsClient.on('alert.triggered', (alert) => {
  console.log(`Alert triggered: ${alert.name} - ${alert.description}`);
});

// Deploy model with A/B testing
const abTestResult = await mlopsClient.deployWithABTesting(
  modelConfig,
  {
    trafficSplit: 10,  // 10% traffic to new model
    duration: '7d',
    successMetrics: ['accuracy', 'precision', 'user_satisfaction'],
    failureThresholds: {
      accuracy_drop: 0.02,
      error_rate_increase: 0.01,
      latency_increase: 50
    }
  }
);

console.log(`A/B test created with ID: ${abTestResult.testId}`);
```

### CLI Tool Integration

```bash
#!/bin/bash

# MDT Dashboard CLI Tool for Advanced Operations

# Install the CLI tool
pip install mdt-dashboard-cli

# Configure authentication
mdt auth login --api-key $MDT_API_KEY --base-url https://api.mdt-dashboard.com/api/v1

# Advanced model deployment pipeline
mdt models deploy \
  --config deployment-config.yaml \
  --environment production \
  --strategy blue-green \
  --health-check-endpoint /health \
  --rollback-threshold 0.02 \
  --monitoring-enabled \
  --auto-scaling \
  --min-replicas 2 \
  --max-replicas 10

# Automated testing pipeline
mdt models test \
  --model-id model_123 \
  --test-suite integration \
  --test-data test_data.csv \
  --performance-thresholds performance_thresholds.json \
  --generate-report \
  --output-format html

# Drift detection with custom rules
mdt drift monitor \
  --model-id model_123 \
  --detection-methods ks_test,psi,jensen_shannon \
  --reference-data-window 30d \
  --alert-channels slack,email,pagerduty \
  --custom-rules drift_rules.yaml

# Automated model maintenance
mdt maintenance schedule \
  --model-id model_123 \
  --retrain-schedule "0 2 * * 0" \
  --data-quality-checks \
  --feature-drift-monitoring \
  --performance-regression-detection \
  --auto-approve-improvements

# Comprehensive monitoring setup
mdt monitoring setup \
  --models model_123,model_456 \
  --metrics accuracy,latency,throughput \
  --dashboards grafana,custom \
  --alert-rules alert_rules.yaml \
  --sla-definitions sla.yaml

# Batch operations for multiple models
mdt batch-operations \
  --operation update-config \
  --model-pattern "fraud_detection_*" \
  --config-updates config_updates.json \
  --dry-run \
  --parallel-jobs 5

# Export model artifacts and metadata
mdt export \
  --model-id model_123 \
  --include-artifacts \
  --include-metadata \
  --include-training-data \
  --include-performance-history \
  --format mlflow \
  --output-path ./exports/
```

---

## 🚀 Advanced Integration Patterns

### Infrastructure as Code with Terraform

```hcl
# terraform/main.tf

# MDT Dashboard model deployment
resource "mdt_model" "fraud_detection" {
  name        = "fraud_detection_v5"
  framework   = "tensorflow"
  version     = "5.0.0"
  description = "Advanced fraud detection with deep learning"
  
  model_config = jsonencode({
    algorithm = "deep_neural_network"
    architecture = {
      layers = [
        { type = "dense", units = 128, activation = "relu" },
        { type = "dropout", rate = 0.3 },
        { type = "dense", units = 64, activation = "relu" },
        { type = "dropout", rate = 0.2 },
        { type = "dense", units = 1, activation = "sigmoid" }
      ]
    }
    hyperparameters = {
      learning_rate = 0.001
      batch_size = 32
      epochs = 100
      early_stopping_patience = 10
    }
  })
  
  deployment_config = jsonencode({
    environment = "production"
    auto_scaling = {
      enabled = true
      min_replicas = 3
      max_replicas = 20
      target_cpu_utilization = 70
      target_memory_utilization = 80
    }
    health_check = {
      endpoint = "/health"
      interval = "30s"
      timeout = "10s"
      failure_threshold = 3
    }
    resource_limits = {
      cpu = "2000m"
      memory = "4Gi"
    }
    resource_requests = {
      cpu = "500m"
      memory = "1Gi"
    }
  })
  
  monitoring_config = jsonencode({
    drift_detection = {
      enabled = true
      methods = ["ks_test", "psi", "jensen_shannon"]
      schedule = "hourly"
      thresholds = {
        ks_test = {
          p_value = 0.05
          bonferroni_correction = true
        }
        psi = {
          threshold = 0.2
          num_bins = 10
        }
        jensen_shannon = {
          threshold = 0.1
        }
      }
    }
    performance_monitoring = {
      enabled = true
      metrics = ["accuracy", "precision", "recall", "f1_score", "latency"]
      alert_thresholds = {
        accuracy_drop = 0.05
        latency_increase = 100
        error_rate_spike = 0.02
      }
    }
    alerts = {
      channels = ["slack", "email"]
      escalation_policy = "immediate"
    }
  })
  
  tags = {
    Environment = "production"
    Team = "ml-engineering"
    Project = "fraud-detection"
    Version = "5.0.0"
  }
}

# Drift detection configuration
resource "mdt_drift_config" "fraud_detection_drift" {
  model_id = mdt_model.fraud_detection.id
  
  detection_methods = ["ks_test", "psi", "jensen_shannon"]
  monitoring_schedule = "hourly"
  
  reference_data_config = jsonencode({
    type = "rolling_window"
    window_size = "30d"
    min_samples = 1000
    refresh_interval = "daily"
  })
  
  thresholds = jsonencode({
    ks_test = {
      p_value = 0.05
      bonferroni_correction = true
    }
    psi = {
      threshold = 0.2
      num_bins = 10
    }
    jensen_shannon = {
      threshold = 0.1
    }
  })
  
  alert_config = jsonencode({
    enabled = true
    severity_mapping = {
      low = "0.1-0.3"
      medium = "0.3-0.6"
      high = "0.6-0.8"
      critical = "0.8-1.0"
    }
    notification_channels = ["slack", "email"]
    suppression_window = "1h"
    escalation_policy = "immediate"
  })
}

# Monitoring dashboard
resource "mdt_dashboard" "fraud_detection_dashboard" {
  name = "Fraud Detection Model Dashboard"
  description = "Comprehensive monitoring dashboard for fraud detection models"
  
  models = [mdt_model.fraud_detection.id]
  
  metrics = [
    "accuracy", "precision", "recall", "f1_score",
    "latency", "throughput", "error_rate",
    "drift_score", "data_quality_score"
  ]
  
  time_ranges = ["1h", "24h", "7d", "30d"]
  
  visualizations = jsonencode([
    {
      type = "time_series"
      metrics = ["accuracy", "latency"]
      title = "Model Performance Over Time"
    },
    {
      type = "distribution_comparison"
      features = ["amount", "merchant_category"]
      title = "Feature Distribution Drift"
    },
    {
      type = "confusion_matrix"
      title = "Current Model Predictions"
    },
    {
      type = "feature_importance"
      title = "Feature Importance Analysis"
    }
  ])
  
  alerts = jsonencode({
    thresholds = {
      accuracy_drop = 0.05
      latency_increase = 100
      error_rate_spike = 0.02
      drift_score_high = 0.7
    }
    notifications = ["email", "slack"]
  })
  
  access_control = jsonencode({
    public = false
    teams = ["ml-engineering", "data-science", "operations"]
    permissions = {
      view = ["ml-engineering", "data-science", "operations", "management"]
      edit = ["ml-engineering"]
      admin = ["ml-engineering-leads"]
    }
  })
}

# Automated retraining pipeline
resource "mdt_automation_rule" "auto_retrain" {
  name = "Auto Retrain on Drift"
  description = "Automatically retrain model when significant drift is detected"
  
  model_id = mdt_model.fraud_detection.id
  
  trigger = jsonencode({
    type = "drift_detected"
    conditions = {
      drift_score = { gt = 0.7 }
      duration = "2h"  # Drift must persist for 2 hours
    }
  })
  
  action = jsonencode({
    type = "retrain_model"
    parameters = {
      include_recent_data = true
      data_window = "60d"
      hyperparameter_tuning = true
      validation_strategy = "time_series_split"
      auto_deploy = false  # Require manual approval
      notification_channels = ["slack", "email"]
    }
  })
  
  enabled = true
}

# Output important information
output "model_id" {
  value = mdt_model.fraud_detection.id
}

output "model_endpoint" {
  value = mdt_model.fraud_detection.endpoint_url
}

output "dashboard_url" {
  value = mdt_dashboard.fraud_detection_dashboard.url
}
```

### Kubernetes Integration

```yaml
# k8s/model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-model
  namespace: ml-models
  labels:
    app: fraud-detection
    version: v5.0.0
    component: model-server
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: fraud-detection
      version: v5.0.0
  template:
    metadata:
      labels:
        app: fraud-detection
        version: v5.0.0
        component: model-server
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: model-server
      containers:
      - name: model-server
        image: mdt-dashboard/fraud-detection:v5.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: MODEL_ID
          value: "model_123"
        - name: MDT_API_KEY
          valueFrom:
            secretKeyRef:
              name: mdt-credentials
              key: api-key
        - name: MDT_BASE_URL
          value: "https://api.mdt-dashboard.com/api/v1"
        - name: MONITORING_ENABLED
          value: "true"
        - name: DRIFT_DETECTION_ENABLED
          value: "true"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: model-artifacts
          mountPath: /app/models
          readOnly: true
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: model-artifacts
        persistentVolumeClaim:
          claimName: model-artifacts-pvc
      - name: config
        configMap:
          name: model-config
      imagePullSecrets:
      - name: docker-registry-secret

---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection-service
  namespace: ml-models
  labels:
    app: fraud-detection
spec:
  selector:
    app: fraud-detection
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-hpa
  namespace: ml-models
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detection-model
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: prediction_latency_p95
      target:
        type: AverageValue
        averageValue: "100m"  # 100ms
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 30

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fraud-detection-ingress
  namespace: ml-models
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - models.mdt-dashboard.com
    secretName: model-tls-secret
  rules:
  - host: models.mdt-dashboard.com
    http:
      paths:
      - path: /fraud-detection
        pathType: Prefix
        backend:
          service:
            name: fraud-detection-service
            port:
              number: 80
```

---

## 📊 Monitoring and Observability Integration

### Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "mdt_rules.yml"

scrape_configs:
  - job_name: 'mdt-dashboard-api'
    static_configs:
      - targets: ['api.mdt-dashboard.com:9090']
    metrics_path: /metrics
    scrape_interval: 30s
    
  - job_name: 'model-servers'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ['ml-models']
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
        
  - job_name: 'drift-detection'
    static_configs:
      - targets: ['drift-detection:9090']
    scrape_interval: 60s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Custom Prometheus Rules

```yaml
# prometheus/mdt_rules.yml
groups:
  - name: mdt_model_performance
    interval: 30s
    rules:
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m])) > 0.1
        for: 2m
        labels:
          severity: warning
          team: ml-engineering
        annotations:
          summary: "High prediction latency detected"
          description: "Model {{ $labels.model_id }} has 95th percentile latency of {{ $value }}s"
          
      - alert: ModelAccuracyDrop
        expr: model_accuracy < 0.9
        for: 5m
        labels:
          severity: critical
          team: ml-engineering
        annotations:
          summary: "Model accuracy dropped below threshold"
          description: "Model {{ $labels.model_id }} accuracy is {{ $value }}"
          
      - alert: DriftDetected
        expr: model_drift_score > 0.7
        for: 10m
        labels:
          severity: high
          team: data-science
        annotations:
          summary: "Significant drift detected"
          description: "Model {{ $labels.model_id }} drift score is {{ $value }}"
          
      - alert: HighErrorRate
        expr: rate(model_prediction_errors_total[5m]) > 0.02
        for: 2m
        labels:
          severity: critical
          team: ml-engineering
        annotations:
          summary: "High model error rate"
          description: "Model {{ $labels.model_id }} error rate is {{ $value }}"

  - name: mdt_infrastructure
    interval: 30s
    rules:
      - alert: ModelServerDown
        expr: up{job="model-servers"} == 0
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Model server is down"
          description: "Model server {{ $labels.instance }} is not responding"
          
      - alert: HighCPUUsage
        expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High CPU usage on model server"
          description: "Container {{ $labels.container }} CPU usage is {{ $value }}"
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "MDT Dashboard - Model Performance",
    "tags": ["mdt", "ml", "monitoring"],
    "timezone": "browser",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "title": "Prediction Latency",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 0.05},
                {"color": "red", "value": 0.1}
              ]
            }
          }
        }
      },
      {
        "title": "Model Accuracy",
        "type": "timeseries",
        "targets": [
          {
            "expr": "model_accuracy",
            "legendFormat": "Accuracy - {{model_id}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "unit": "percentunit"
          }
        }
      },
      {
        "title": "Drift Scores",
        "type": "timeseries",
        "targets": [
          {
            "expr": "model_drift_score",
            "legendFormat": "Drift Score - {{model_id}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 0.3},
                {"color": "orange", "value": 0.6},
                {"color": "red", "value": 0.8}
              ]
            }
          }
        }
      },
      {
        "title": "Prediction Volume",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(model_predictions_total[5m])",
            "legendFormat": "Predictions/sec - {{model_id}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps"
          }
        }
      }
    ]
  }
}
```

---

## 🔄 CI/CD Integration Examples

### GitHub Actions Workflow

```yaml
# .github/workflows/model-deployment.yml
name: Model Deployment Pipeline

on:
  push:
    branches: [main]
    paths: ['models/**', 'config/**']
  pull_request:
    branches: [main]
    paths: ['models/**', 'config/**']

env:
  MDT_API_KEY: ${{ secrets.MDT_API_KEY }}
  MDT_BASE_URL: ${{ secrets.MDT_BASE_URL }}

jobs:
  model-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install mdt-dashboard-cli
        
    - name: Validate model configuration
      run: |
        mdt models validate --config models/fraud_detection/config.yaml
        
    - name: Run model tests
      run: |
        python -m pytest tests/model_tests/ -v
        
    - name: Generate model documentation
      run: |
        mdt models document --config models/fraud_detection/config.yaml --output docs/

  performance-testing:
    runs-on: ubuntu-latest
    needs: model-validation
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up test environment
      run: |
        docker-compose -f docker-compose.test.yml up -d
        
    - name: Load test data
      run: |
        python scripts/load_test_data.py
        
    - name: Run performance tests
      run: |
        mdt models test \
          --model-config models/fraud_detection/config.yaml \
          --test-suite performance \
          --test-data data/test_data.csv \
          --performance-thresholds config/performance_thresholds.json \
          --output-format junit \
          --output-file test-results.xml
          
    - name: Publish test results
      uses: dorny/test-reporter@v1
      if: always()
      with:
        name: Performance Tests
        path: test-results.xml
        reporter: java-junit

  deploy-staging:
    runs-on: ubuntu-latest
    needs: [model-validation, performance-testing]
    if: github.ref == 'refs/heads/main'
    environment: staging
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        mdt models deploy \
          --config models/fraud_detection/config.yaml \
          --environment staging \
          --strategy blue-green \
          --wait-for-ready \
          --health-check-retries 10
          
    - name: Run integration tests
      run: |
        mdt models test \
          --environment staging \
          --test-suite integration \
          --test-data data/integration_test_data.csv
          
    - name: Configure monitoring
      run: |
        mdt monitoring setup \
          --environment staging \
          --config config/monitoring_staging.yaml

  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        mdt models deploy \
          --config models/fraud_detection/config.yaml \
          --environment production \
          --strategy canary \
          --canary-traffic-percent 10 \
          --canary-duration 24h \
          --success-threshold 0.95 \
          --auto-promote \
          --rollback-on-failure
          
    - name: Configure production monitoring
      run: |
        mdt monitoring setup \
          --environment production \
          --config config/monitoring_production.yaml \
          --alerts-enabled \
          --drift-detection-enabled
          
    - name: Update documentation
      run: |
        mdt models document \
          --environment production \
          --include-performance-metrics \
          --include-deployment-info \
          --output docs/production/
          
    - name: Send deployment notification
      run: |
        mdt notifications send \
          --channel slack \
          --message "Model deployment to production completed successfully" \
          --include-deployment-summary
```

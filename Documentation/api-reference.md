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

## 📚 SDK and Examples

### 🐍 Python SDK Example

```python
from mdt_dashboard_sdk import MDTClient

# Initialize client
client = MDTClient(
    base_url="https://api.mdt-dashboard.com",
    api_key="your_api_key"
)

# Make prediction
result = client.predict(
    model_id=1,
    features={
        "transaction_amount": 150.00,
        "merchant_category": "grocery"
    }
)

print(f"Prediction: {result.prediction.class}")
print(f"Confidence: {result.prediction.probability}")

# Check for drift
drift_result = client.detect_drift(
    model_id=1,
    current_data=df_current,
    algorithms=["ks_test", "psi"]
)

if drift_result.overall_drift.is_drift_detected:
    print("Drift detected! Consider retraining the model.")
```

### 📊 JavaScript/Node.js Example

```javascript
const { MDTClient } = require('mdt-dashboard-sdk');

const client = new MDTClient({
  baseURL: 'https://api.mdt-dashboard.com',
  apiKey: 'your_api_key'
});

// Make prediction
const prediction = await client.predict(1, {
  transaction_amount: 150.00,
  merchant_category: 'grocery'
});

console.log(`Prediction: ${prediction.class}`);
console.log(`Confidence: ${prediction.probability}`);

// Get model metrics
const metrics = await client.getModelMetrics(1, {
  timeRange: '24h',
  metricType: 'accuracy'
});

console.log(`Current accuracy: ${metrics.accuracy.current}`);
```

### 🔧 cURL Examples

```bash
# Login
curl -X POST "https://api.mdt-dashboard.com/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "password"
  }'

# Make prediction
curl -X POST "https://api.mdt-dashboard.com/api/v1/models/1/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "transaction_amount": 150.00,
      "merchant_category": "grocery"
    }
  }'

# Get drift detection results
curl -X GET "https://api.mdt-dashboard.com/api/v1/models/1/drift/history?start_date=2025-07-01&end_date=2025-07-15" \
  -H "Authorization: Bearer $TOKEN"
```

---

## 📝 Changelog

### Version 1.0.0 (2025-07-15)
- Initial API release
- Model management endpoints
- Prediction API
- Drift detection API
- User authentication
- Rate limiting
- Webhook support

---

*For more examples and detailed implementation guides, visit our [GitHub repository](https://github.com/your-org/mdt-dashboard) or check the [interactive API documentation](https://api.mdt-dashboard.com/docs).*

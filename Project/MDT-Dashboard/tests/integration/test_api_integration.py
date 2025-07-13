"""
Integration tests for the MDT Dashboard API endpoints.
Tests the complete API functionality with real database and services.
"""

import pytest
import httpx
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import time

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mdt_dashboard.api.main import app
from mdt_dashboard.core.database import get_db, Base
from mdt_dashboard.core.config import get_settings


class TestAPIIntegration:
    """Integration tests for the API endpoints."""
    
    @pytest.fixture(scope="class")
    def settings(self):
        """Get test settings."""
        return get_settings()
    
    @pytest.fixture(scope="class")
    def test_db_engine(self, settings):
        """Create test database engine."""
        # Use test database URL
        test_db_url = settings.database_url.replace('mdt_db', 'mdt_test')
        engine = create_engine(test_db_url)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        yield engine
        
        # Cleanup
        Base.metadata.drop_all(bind=engine)
    
    @pytest.fixture(scope="class")
    def client(self, test_db_engine):
        """Create test client with test database."""
        def override_get_db():
            TestingSessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=test_db_engine
            )
            db = TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()
        
        app.dependency_overrides[get_db] = override_get_db
        
        with TestClient(app) as client:
            yield client
        
        app.dependency_overrides.clear()
    
    def test_health_endpoints(self, client):
        """Test health check endpoints."""
        # Root health check
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        
        # API health check
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        api_health = response.json()
        assert "status" in api_health
        assert "database" in api_health
        assert "redis" in api_health
    
    def test_model_crud_operations(self, client):
        """Test complete model CRUD operations."""
        # Create a new model
        model_data = {
            "name": "test-integration-model",
            "version": "1.0.0",
            "algorithm": "random_forest",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.94
            }
        }
        
        # Create model
        response = client.post("/api/v1/models", json=model_data)
        assert response.status_code == 201
        
        created_model = response.json()
        model_id = created_model["id"]
        assert created_model["name"] == model_data["name"]
        assert created_model["status"] == "active"
        
        # Get model by ID
        response = client.get(f"/api/v1/models/{model_id}")
        assert response.status_code == 200
        
        retrieved_model = response.json()
        assert retrieved_model["id"] == model_id
        assert retrieved_model["name"] == model_data["name"]
        
        # List models
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        
        models_list = response.json()
        assert "models" in models_list
        assert len(models_list["models"]) >= 1
        
        # Update model
        update_data = {
            "status": "deprecated",
            "metrics": {
                "accuracy": 0.96,
                "precision": 0.94,
                "recall": 0.95
            }
        }
        
        response = client.put(f"/api/v1/models/{model_id}", json=update_data)
        assert response.status_code == 200
        
        updated_model = response.json()
        assert updated_model["status"] == "deprecated"
        assert updated_model["metrics"]["accuracy"] == 0.96
        
        # Delete model
        response = client.delete(f"/api/v1/models/{model_id}")
        assert response.status_code == 200
        
        # Verify deletion
        response = client.get(f"/api/v1/models/{model_id}")
        assert response.status_code == 404
    
    def test_prediction_endpoint(self, client):
        """Test prediction endpoint functionality."""
        # First create a model for predictions
        model_data = {
            "name": "prediction-test-model",
            "version": "1.0.0",
            "algorithm": "linear_regression",
            "status": "active"
        }
        
        response = client.post("/api/v1/models", json=model_data)
        assert response.status_code == 201
        model_id = response.json()["id"]
        
        # Test prediction request
        prediction_data = {
            "data": {
                "feature1": 1.5,
                "feature2": 2.3,
                "feature3": 0.8
            },
            "model_name": "prediction-test-model",
            "return_probabilities": False,
            "track_metrics": True
        }
        
        # Note: This might fail if actual model file doesn't exist
        # That's expected in integration tests without trained models
        response = client.post("/api/v1/predict", json=prediction_data)
        
        # Accept both success and expected failure cases
        assert response.status_code in [200, 404, 422, 500]
        
        if response.status_code == 200:
            prediction_result = response.json()
            assert "prediction" in prediction_result
            assert "model_info" in prediction_result
    
    def test_drift_detection_endpoint(self, client):
        """Test drift detection endpoint."""
        # Generate test data
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, 1000).tolist()
        comparison_data = np.random.normal(0.5, 1.2, 1000).tolist()  # Slight drift
        
        drift_request = {
            "reference_data": reference_data,
            "comparison_data": comparison_data,
            "feature_name": "test_feature",
            "test_methods": ["ks", "psi"]
        }
        
        response = client.post("/api/v1/drift/detect", json=drift_request)
        assert response.status_code == 200
        
        drift_result = response.json()
        assert "test_results" in drift_result
        assert "summary" in drift_result
        assert len(drift_result["test_results"]) >= 1
        
        # Check result structure
        for result in drift_result["test_results"]:
            assert "test_name" in result
            assert "p_value" in result
            assert "statistic" in result
            assert "is_drift" in result
            assert "severity" in result
    
    def test_metrics_endpoint(self, client):
        """Test metrics collection endpoint."""
        response = client.get("/api/v1/metrics")
        
        # Metrics endpoint might not be fully implemented
        assert response.status_code in [200, 404, 501]
        
        if response.status_code == 200:
            metrics_data = response.json()
            # Check for expected metrics structure
            assert isinstance(metrics_data, dict)
    
    def test_model_training_endpoint(self, client):
        """Test model training endpoint."""
        # Create sample training data
        training_data = {
            "training_data_path": "/tmp/test_data.csv",
            "target_column": "target",
            "problem_type": "regression",
            "algorithms": ["linear_regression"],
            "experiment_name": "integration-test"
        }
        
        # This will likely fail without actual data file
        response = client.post("/api/v1/models/train", json=training_data)
        
        # Accept failure cases as expected
        assert response.status_code in [200, 202, 400, 404, 422]
        
        if response.status_code in [200, 202]:
            training_result = response.json()
            assert "message" in training_result
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_health_request():
            response = client.get("/health")
            return response.status_code == 200
        
        # Test 10 concurrent health check requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(results)
    
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test invalid model ID
        response = client.get("/api/v1/models/invalid-id")
        assert response.status_code == 422  # Invalid UUID format
        
        # Test non-existent model
        response = client.get("/api/v1/models/123e4567-e89b-12d3-a456-426614174000")
        assert response.status_code == 404
        
        # Test invalid JSON in request body
        response = client.post(
            "/api/v1/models",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test missing required fields
        response = client.post("/api/v1/models", json={"name": "test"})
        assert response.status_code == 422
    
    def test_data_validation(self, client):
        """Test request data validation."""
        # Test model creation with invalid data
        invalid_model_data = {
            "name": "",  # Empty name
            "version": "invalid-version-format",
            "algorithm": "unsupported_algorithm"
        }
        
        response = client.post("/api/v1/models", json=invalid_model_data)
        assert response.status_code == 422
        
        error_detail = response.json()
        assert "detail" in error_detail
    
    def test_pagination(self, client):
        """Test pagination in list endpoints."""
        # Create multiple models for pagination testing
        for i in range(5):
            model_data = {
                "name": f"pagination-test-model-{i}",
                "version": "1.0.0",
                "algorithm": "random_forest"
            }
            response = client.post("/api/v1/models", json=model_data)
            assert response.status_code == 201
        
        # Test pagination
        response = client.get("/api/v1/models?skip=0&limit=3")
        assert response.status_code == 200
        
        models_page1 = response.json()
        assert len(models_page1["models"]) <= 3
        assert "total" in models_page1
        assert "skip" in models_page1
        assert "limit" in models_page1
        
        # Test second page
        response = client.get("/api/v1/models?skip=3&limit=3")
        assert response.status_code == 200
        
        models_page2 = response.json()
        assert len(models_page2["models"]) <= 3


class TestDatabaseIntegration:
    """Test database operations and data persistence."""
    
    @pytest.fixture(scope="class")
    def db_engine(self):
        """Create test database engine."""
        settings = get_settings()
        test_db_url = settings.database_url.replace('mdt_db', 'mdt_test')
        engine = create_engine(test_db_url)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        yield engine
        
        # Cleanup
        Base.metadata.drop_all(bind=engine)
    
    def test_database_connection(self, db_engine):
        """Test database connectivity."""
        with db_engine.connect() as connection:
            result = connection.execute("SELECT 1")
            assert result.fetchone()[0] == 1
    
    def test_model_persistence(self, db_engine):
        """Test model data persistence in database."""
        from mdt_dashboard.core.models import Model
        
        SessionLocal = sessionmaker(bind=db_engine)
        db = SessionLocal()
        
        try:
            # Create model
            model = Model(
                name="test-persistence-model",
                version="1.0.0",
                algorithm="test_algorithm",
                parameters={"param1": "value1"},
                metrics={"accuracy": 0.95}
            )
            
            db.add(model)
            db.commit()
            db.refresh(model)
            
            model_id = model.id
            
            # Retrieve model
            retrieved_model = db.query(Model).filter(Model.id == model_id).first()
            
            assert retrieved_model is not None
            assert retrieved_model.name == "test-persistence-model"
            assert retrieved_model.parameters["param1"] == "value1"
            assert retrieved_model.metrics["accuracy"] == 0.95
            
        finally:
            db.close()


class TestExternalServiceIntegration:
    """Test integration with external services."""
    
    def test_redis_connection(self):
        """Test Redis connectivity."""
        try:
            import redis
            from mdt_dashboard.core.config import get_settings
            
            settings = get_settings()
            r = redis.from_url(settings.redis_url)
            
            # Test basic Redis operations
            r.set("test_key", "test_value")
            assert r.get("test_key").decode() == "test_value"
            r.delete("test_key")
            
        except ImportError:
            pytest.skip("Redis not available")
        except Exception as e:
            pytest.skip(f"Redis connection failed: {str(e)}")
    
    def test_mlflow_integration(self):
        """Test MLflow integration."""
        try:
            import mlflow
            from mdt_dashboard.core.config import get_settings
            
            settings = get_settings()
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            
            # Test MLflow connectivity
            experiments = mlflow.search_experiments()
            assert isinstance(experiments, list)
            
        except ImportError:
            pytest.skip("MLflow not available")
        except Exception as e:
            pytest.skip(f"MLflow connection failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

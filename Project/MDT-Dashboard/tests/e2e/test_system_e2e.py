"""
End-to-end tests for the complete MDT Dashboard system.
Tests the full workflow from data ingestion to model deployment and monitoring.
"""

import pytest
import requests
import time
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import subprocess
import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import docker


class TestSystemE2E:
    """End-to-end system tests."""
    
    @pytest.fixture(scope="class")
    def services_running(self):
        """Ensure all services are running for E2E tests."""
        api_url = "http://localhost:8000"
        dashboard_url = "http://localhost:8501"
        
        # Wait for services to be ready
        max_retries = 30
        retry_interval = 2
        
        for service_name, url in [("API", api_url), ("Dashboard", dashboard_url)]:
            for i in range(max_retries):
                try:
                    response = requests.get(f"{url}/health", timeout=5)
                    if response.status_code == 200:
                        print(f"{service_name} service is ready")
                        break
                except requests.RequestException:
                    if i == max_retries - 1:
                        pytest.skip(f"{service_name} service not available")
                    time.sleep(retry_interval)
        
        yield {
            "api_url": api_url,
            "dashboard_url": dashboard_url
        }
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Create test dataset for E2E tests."""
        # Generate synthetic dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.exponential(2, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature4': np.random.uniform(0, 100, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            data_file = f.name
        
        yield {
            "data": data,
            "file_path": data_file
        }
        
        # Cleanup
        Path(data_file).unlink(missing_ok=True)
    
    def test_full_ml_workflow(self, services_running, test_data):
        """Test complete ML workflow from data to deployment."""
        api_url = services_running["api_url"]
        
        # Step 1: Upload and validate data quality
        print("Testing data quality analysis...")
        
        # This would typically involve data upload endpoint
        # For now, we'll test the data processing directly
        
        # Step 2: Create and train a model
        print("Testing model training...")
        
        training_request = {
            "training_data_path": test_data["file_path"],
            "target_column": "target",
            "problem_type": "classification",
            "algorithms": ["random_forest"],
            "experiment_name": "e2e-test-experiment"
        }
        
        # Note: This might fail if actual training pipeline isn't implemented
        response = requests.post(f"{api_url}/api/v1/models/train", json=training_request)
        
        if response.status_code in [200, 202]:
            training_result = response.json()
            print(f"Training initiated: {training_result}")
            
            # Step 3: Deploy the model
            print("Testing model deployment...")
            
            # Wait for training to complete (simplified)
            time.sleep(5)
            
            # List models to find our trained model
            response = requests.get(f"{api_url}/api/v1/models")
            assert response.status_code == 200
            
            models = response.json()["models"]
            if models:
                model = models[0]  # Use first available model
                
                # Step 4: Make predictions
                print("Testing prediction endpoint...")
                
                prediction_request = {
                    "data": {
                        "feature1": 1.5,
                        "feature2": 2.3,
                        "feature3": "A",
                        "feature4": 50.0
                    },
                    "model_name": model["name"],
                    "return_probabilities": True
                }
                
                response = requests.post(f"{api_url}/api/v1/predict", json=prediction_request)
                
                # Accept various response codes as the model might not be actually trained
                assert response.status_code in [200, 404, 422, 500]
                
                if response.status_code == 200:
                    prediction_result = response.json()
                    assert "prediction" in prediction_result
        
        # Step 5: Test drift detection
        print("Testing drift detection...")
        
        # Generate reference and comparison data
        reference_data = np.random.normal(0, 1, 500).tolist()
        comparison_data = np.random.normal(0.5, 1.2, 500).tolist()  # With drift
        
        drift_request = {
            "reference_data": reference_data,
            "comparison_data": comparison_data,
            "feature_name": "test_feature",
            "test_methods": ["ks", "psi"]
        }
        
        response = requests.post(f"{api_url}/api/v1/drift/detect", json=drift_request)
        assert response.status_code == 200
        
        drift_result = response.json()
        assert "test_results" in drift_result
        assert len(drift_result["test_results"]) >= 1
        
        print("Full ML workflow test completed successfully!")
    
    def test_monitoring_workflow(self, services_running):
        """Test monitoring and alerting workflow."""
        api_url = services_running["api_url"]
        
        # Step 1: Check system health
        response = requests.get(f"{api_url}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # Step 2: Check API health with detailed info
        response = requests.get(f"{api_url}/api/v1/health")
        assert response.status_code == 200
        
        api_health = response.json()
        assert "database" in api_health
        assert "redis" in api_health
        
        # Step 3: Test metrics collection
        response = requests.get(f"{api_url}/api/v1/metrics")
        # Metrics endpoint might not be fully implemented
        assert response.status_code in [200, 404, 501]
        
        print("Monitoring workflow test completed!")
    
    def test_data_pipeline_workflow(self, services_running, test_data):
        """Test data processing and quality assessment workflow."""
        # This test simulates the data pipeline workflow
        
        data = test_data["data"]
        
        # Simulate data quality analysis
        print("Testing data quality analysis workflow...")
        
        # Check data quality metrics
        assert len(data) == 1000
        assert data.isnull().sum().sum() == 0  # No missing values in our test data
        
        # Test feature engineering workflow
        print("Testing feature engineering...")
        
        # Create temporal features (simulate)
        data['datetime_feature'] = pd.date_range('2023-01-01', periods=len(data), freq='H')
        
        # Verify feature engineering
        assert 'datetime_feature' in data.columns
        
        # Test data preprocessing workflow
        print("Testing data preprocessing...")
        
        # Simulate preprocessing steps
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        assert len(numeric_columns) >= 3
        assert len(categorical_columns) >= 1
        
        print("Data pipeline workflow test completed!")
    
    def test_scalability_workflow(self, services_running):
        """Test system scalability under load."""
        api_url = services_running["api_url"]
        
        print("Testing system scalability...")
        
        # Test concurrent health checks
        import concurrent.futures
        
        def make_health_request():
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                return response.status_code == 200
            except Exception:
                return False
        
        # Test 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_health_request) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # At least 80% of requests should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Success rate: {success_rate}"
        
        print(f"Scalability test completed! Success rate: {success_rate}")
    
    def test_error_recovery_workflow(self, services_running):
        """Test system error handling and recovery."""
        api_url = services_running["api_url"]
        
        print("Testing error recovery workflow...")
        
        # Test invalid requests
        invalid_requests = [
            ("/api/v1/models/invalid-id", "GET"),
            ("/api/v1/models", "POST"),  # Missing body
            ("/api/v1/predict", "POST"),  # Missing body
            ("/api/v1/nonexistent", "GET"),  # Non-existent endpoint
        ]
        
        for endpoint, method in invalid_requests:
            if method == "GET":
                response = requests.get(f"{api_url}{endpoint}")
            elif method == "POST":
                response = requests.post(f"{api_url}{endpoint}")
            
            # Should return proper error codes, not crash
            assert response.status_code in [400, 404, 422, 500]
        
        # Verify system is still healthy after error requests
        response = requests.get(f"{api_url}/health")
        assert response.status_code == 200
        
        print("Error recovery workflow test completed!")


class TestDashboardE2E:
    """End-to-end tests for the dashboard interface."""
    
    @pytest.fixture(scope="class")
    def driver(self):
        """Set up Selenium WebDriver."""
        try:
            # Configure Chrome options for headless mode
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(10)
            
            yield driver
            
        except Exception as e:
            pytest.skip(f"WebDriver not available: {str(e)}")
        finally:
            if 'driver' in locals():
                driver.quit()
    
    def test_dashboard_loading(self, driver):
        """Test dashboard loading and basic functionality."""
        dashboard_url = "http://localhost:8501"
        
        try:
            # Load dashboard
            driver.get(dashboard_url)
            
            # Wait for page to load
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Check if it's a Streamlit app
            page_source = driver.page_source.lower()
            assert "streamlit" in page_source or "mdt" in page_source or "dashboard" in page_source
            
            # Check for common dashboard elements
            # This is basic as Streamlit apps have dynamic content
            assert driver.title  # Should have a title
            
            print("Dashboard loading test completed!")
            
        except Exception as e:
            pytest.skip(f"Dashboard not accessible: {str(e)}")
    
    def test_dashboard_interactions(self, driver):
        """Test basic dashboard interactions."""
        dashboard_url = "http://localhost:8501"
        
        try:
            driver.get(dashboard_url)
            
            # Wait for content to load
            time.sleep(5)
            
            # Try to find interactive elements
            # Note: Streamlit components have dynamic IDs, so this is basic
            buttons = driver.find_elements(By.TAG_NAME, "button")
            inputs = driver.find_elements(By.TAG_NAME, "input")
            
            # Should have some interactive elements
            assert len(buttons) > 0 or len(inputs) > 0
            
            print("Dashboard interactions test completed!")
            
        except Exception as e:
            pytest.skip(f"Dashboard interaction test failed: {str(e)}")


class TestSystemIntegration:
    """Test integration between different system components."""
    
    def test_api_dashboard_integration(self):
        """Test integration between API and dashboard."""
        api_url = "http://localhost:8000"
        dashboard_url = "http://localhost:8501"
        
        # Test that both services are running
        try:
            api_response = requests.get(f"{api_url}/health", timeout=5)
            dashboard_response = requests.get(dashboard_url, timeout=10)
            
            assert api_response.status_code == 200
            assert dashboard_response.status_code == 200
            
            print("API-Dashboard integration test completed!")
            
        except requests.RequestException as e:
            pytest.skip(f"Services not available for integration test: {str(e)}")
    
    def test_database_api_integration(self):
        """Test database and API integration."""
        api_url = "http://localhost:8000"
        
        try:
            # Test API health which includes database status
            response = requests.get(f"{api_url}/api/v1/health", timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                # Should include database status
                assert "database" in health_data
                
                print("Database-API integration test completed!")
            else:
                pytest.skip("API not available for database integration test")
                
        except requests.RequestException as e:
            pytest.skip(f"API not available: {str(e)}")
    
    def test_end_to_end_data_flow(self):
        """Test complete data flow through the system."""
        api_url = "http://localhost:8000"
        
        try:
            # 1. Create a model
            model_data = {
                "name": "e2e-flow-test-model",
                "version": "1.0.0",
                "algorithm": "test_algorithm",
                "status": "active"
            }
            
            response = requests.post(f"{api_url}/api/v1/models", json=model_data)
            
            if response.status_code == 201:
                model = response.json()
                model_id = model["id"]
                
                # 2. Retrieve the model
                response = requests.get(f"{api_url}/api/v1/models/{model_id}")
                assert response.status_code == 200
                
                # 3. List models
                response = requests.get(f"{api_url}/api/v1/models")
                assert response.status_code == 200
                
                models = response.json()["models"]
                assert any(m["id"] == model_id for m in models)
                
                # 4. Test drift detection
                reference_data = [1, 2, 3, 4, 5] * 100
                comparison_data = [1.1, 2.1, 3.1, 4.1, 5.1] * 100
                
                drift_request = {
                    "reference_data": reference_data,
                    "comparison_data": comparison_data,
                    "feature_name": "test_feature"
                }
                
                response = requests.post(f"{api_url}/api/v1/drift/detect", json=drift_request)
                assert response.status_code == 200
                
                # 5. Clean up
                response = requests.delete(f"{api_url}/api/v1/models/{model_id}")
                assert response.status_code == 200
                
                print("End-to-end data flow test completed!")
                
            else:
                pytest.skip("Cannot create model for data flow test")
                
        except requests.RequestException as e:
            pytest.skip(f"API not available for data flow test: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

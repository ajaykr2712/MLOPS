#!/usr/bin/env python
"""
Demo script for MDT Dashboard platform.
Creates sample models, data, and demonstrates key features.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import requests
import json
import logging
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configuration
API_BASE_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8501"

def create_sample_data():
    """Create sample training and test data."""
    print("📊 Creating sample datasets...")
    
    # Create data directory
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Generate synthetic customer churn dataset
    np.random.seed(42)
    n_samples = 10000
    
    # Features
    age = np.random.normal(35, 12, n_samples).clip(18, 80)
    income = np.random.normal(50000, 15000, n_samples).clip(20000, 150000)
    tenure_months = np.random.exponential(24, n_samples).clip(1, 120)
    monthly_charges = income * 0.002 + np.random.normal(0, 20, n_samples)
    total_charges = monthly_charges * tenure_months
    
    # Create churn probability based on features
    churn_prob = (
        0.1 +  # base probability
        (age < 25) * 0.3 +  # young customers more likely to churn
        (income < 30000) * 0.2 +  # low income more likely to churn
        (tenure_months < 6) * 0.4 +  # new customers more likely to churn
        (monthly_charges > 100) * 0.2  # high charges more likely to churn
    )
    
    # Add noise
    churn_prob += np.random.normal(0, 0.1, n_samples)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate binary target
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age.round().astype(int),
        'income': income.round(2),
        'tenure_months': tenure_months.round().astype(int),
        'monthly_charges': monthly_charges.round(2),
        'total_charges': total_charges.round(2),
        'churn': churn
    })
    
    # Split into train/test
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Save datasets
    train_df.to_csv(data_dir / "customer_churn_train.csv", index=False)
    test_df.to_csv(data_dir / "customer_churn_test.csv", index=False)
    
    print(f"✅ Created training data: {len(train_df)} samples")
    print(f"✅ Created test data: {len(test_df)} samples")
    
    return train_df, test_df


def train_sample_model():
    """Train a sample model using the MDT platform."""
    print("🤖 Training sample model...")
    
    try:
        from src.mdt_dashboard.train import ModelTrainer
        
        trainer = ModelTrainer()
        
        result = trainer.train_model(
            data_path="data/customer_churn_train.csv",
            target_column="churn",
            model_name="customer_churn_demo",
            model_type="random_forest",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        )
        
        print(f"✅ Model trained successfully!")
        print(f"   Model ID: {result['model_id']}")
        print(f"   Performance: {result['metrics']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return None


def create_sample_predictions():
    """Create sample predictions to demonstrate monitoring."""
    print("🔮 Creating sample predictions...")
    
    try:
        from src.mdt_dashboard.predict import PredictionService, PredictionRequest
        
        # Load test data
        test_df = pd.read_csv("data/customer_churn_test.csv")
        
        # Initialize prediction service
        service = PredictionService()
        
        # Set reference data for drift detection
        train_df = pd.read_csv("data/customer_churn_train.csv")
        service.set_reference_data("customer_churn_demo", train_df.drop('churn', axis=1))
        
        # Make sample predictions
        sample_size = 100
        sample_data = test_df.sample(sample_size).drop('churn', axis=1)
        
        predictions = []
        for _, row in sample_data.iterrows():
            request = PredictionRequest(
                data=row.to_dict(),
                model_name="customer_churn_demo",
                return_probabilities=True,
                track_metrics=True
            )
            
            result = service.predict(request)
            predictions.append(result)
            
            # Small delay to simulate real usage
            time.sleep(0.01)
        
        print(f"✅ Created {len(predictions)} sample predictions")
        
        # Show drift status
        drift_count = sum(1 for p in predictions if p.drift_detected)
        print(f"   Drift detected in {drift_count} predictions")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Prediction creation failed: {e}")
        return []


def test_api_connection(base_url=API_BASE_URL):
    """Test connection to the API server."""
    print(f"🔗 Testing API connection to {base_url}")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API server is healthy!")
            print(f"   Version: {health_data.get('version', 'Unknown')}")
            print(f"   Environment: {health_data.get('environment', 'Unknown')}")
            print(f"   Database: {health_data.get('database', 'Unknown')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        print("💡 Make sure the API server is running: python cli.py api")
        return False


def test_model_endpoints(base_url=API_BASE_URL):
    """Test model-related API endpoints."""
    print("🤖 Testing model endpoints...")
    
    try:
        # Test models list endpoint
        response = requests.get(f"{base_url}/api/v1/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])
            print(f"✅ Found {len(models)} models in registry")
            
            if models:
                for model in models[:3]:  # Show first 3 models
                    print(f"   📦 {model.get('name', 'Unknown')} - {model.get('status', 'Unknown')}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")


def test_prediction_api(base_url=API_BASE_URL):
    """Test prediction API with sample data."""
    print("🔮 Testing prediction API...")
    
    # Sample prediction data
    sample_data = {
        "model_id": "customer_churn_demo",
        "input_data": {
            "age": 35,
            "income": 55000.0,
            "tenure_months": 18,
            "monthly_charges": 75.50,
            "total_charges": 1359.0
        },
        "detect_drift": True,
        "store_prediction": True
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   🎯 Prediction: {result.get('prediction')}")
            print(f"   📊 Probability: {result.get('prediction_probability')}")
            print(f"   ⚠️  Drift detected: {result.get('drift_detected', 'N/A')}")
            print(f"   ⏱️  Response time: {result.get('response_time_ms', 0):.1f}ms")
            return result
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Prediction API error: {e}")
        return None


def test_batch_predictions(base_url=API_BASE_URL, num_samples=5):
    """Test batch prediction performance."""
    print(f"📊 Testing batch predictions ({num_samples} samples)...")
    
    # Generate batch data
    batch_data = []
    for i in range(num_samples):
        batch_data.append({
            "age": np.random.randint(20, 70),
            "income": np.random.uniform(30000, 100000),
            "tenure_months": np.random.randint(1, 60),
            "monthly_charges": np.random.uniform(30, 150),
            "total_charges": np.random.uniform(100, 5000)
        })
    
    start_time = time.time()
    successful_predictions = 0
    
    for i, data in enumerate(batch_data):
        try:
            response = requests.post(
                f"{base_url}/api/v1/predict",
                json={
                    "model_id": "customer_churn_demo",
                    "input_data": data,
                    "detect_drift": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                successful_predictions += 1
                if i == 0:  # Log first prediction
                    result = response.json()
                    print(f"   Sample prediction: {result.get('prediction')}")
            else:
                print(f"   ❌ Prediction {i+1} failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Prediction {i+1} error: {e}")
    
    total_time = time.time() - start_time
    avg_time = (total_time / num_samples) * 1000  # ms per prediction
    
    print(f"✅ Batch prediction results:")
    print(f"   📈 Success rate: {successful_predictions}/{num_samples} ({100*successful_predictions/num_samples:.1f}%)")
    print(f"   ⏱️  Average time: {avg_time:.1f}ms per prediction")
    print(f"   🔄 Total time: {total_time:.2f}s")


def test_drift_detection_api(base_url=API_BASE_URL):
    """Test drift detection API endpoint."""
    print("🔍 Testing drift detection API...")
    
    try:
        drift_request = {
            "model_id": "customer_churn_demo",
            "time_window_hours": 24,
            "drift_threshold": 0.05
        }
        
        response = requests.post(
            f"{base_url}/api/v1/drift/check",
            json=drift_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Drift check initiated!")
            print(f"   📝 Task ID: {result.get('task_id', 'N/A')}")
            print(f"   🕒 Time window: {result.get('time_window_hours', 0)} hours")
        else:
            print(f"❌ Drift check failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Drift detection error: {e}")


def test_metrics_endpoint(base_url=API_BASE_URL):
    """Test Prometheus metrics endpoint."""
    print("📊 Testing metrics endpoint...")
    
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        if response.status_code == 200:
            metrics_text = response.text
            lines = metrics_text.split('\n')
            
            # Count different metric types
            counter_metrics = [line for line in lines if '_total' in line and not line.startswith('#')]
            histogram_metrics = [line for line in lines if '_bucket' in line and not line.startswith('#')]
            
            print("✅ Metrics endpoint accessible!")
            print(f"   📊 Counter metrics: {len(counter_metrics)}")
            print(f"   📈 Histogram metrics: {len(histogram_metrics)}")
            
            # Show sample metrics
            print("   📋 Sample metrics:")
            for line in lines[:10]:
                if line and not line.startswith('#') and '=' in line:
                    print(f"      {line[:60]}...")
                    break
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics endpoint error: {e}")


def demonstrate_local_services():
    """Demonstrate local SDK functionality."""
    print("🔧 Testing local services...")
    
    try:
        # Test data processing
        from src.mdt_dashboard.data_processing import ComprehensiveDataProcessor
        
        processor = ComprehensiveDataProcessor()
        
        # Create sample data for processing
        sample_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.exponential(2, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Test data quality analysis
        quality_report = processor.analyze_data_quality(sample_df)
        print(f"✅ Data quality analysis completed")
        print(f"   📊 Overall score: {quality_report.get('overall_score', 0):.2f}")
        
        # Test drift detection
        from src.mdt_dashboard.drift_detection import DriftDetectionSuite
        
        drift_suite = DriftDetectionSuite()
        
        # Create reference and current data
        reference_data = sample_df[:50]
        current_data = sample_df[50:] * 1.2  # Add some drift
        
        drift_results = drift_suite.detect_multivariate_drift(
            reference_data.select_dtypes(include=[np.number]),
            current_data.select_dtypes(include=[np.number])
        )
        
        print(f"✅ Drift detection completed")
        print(f"   🔍 Features analyzed: {len(drift_results)}")
        
        # Test prediction service
        from src.mdt_dashboard.predict import PredictionService
        
        pred_service = PredictionService()
        health = pred_service.get_health_status()
        print(f"✅ Prediction service healthy")
        print(f"   📦 Available models: {health.get('available_models', 0)}")
        
    except Exception as e:
        print(f"❌ Local services error: {e}")
        print(f"   💡 Make sure all dependencies are installed")


def run_performance_benchmark(base_url=API_BASE_URL):
    """Run a comprehensive performance benchmark."""
    print("🚀 Running performance benchmark...")
    
    # Test different payload sizes
    payload_sizes = [1, 5, 10, 20]
    results = {}
    
    for size in payload_sizes:
        print(f"   Testing {size} concurrent predictions...")
        
        start_time = time.time()
        successful = 0
        
        # Create batch requests
        import concurrent.futures
        import threading
        
        def make_prediction():
            try:
                data = {
                    "model_id": "customer_churn_demo",
                    "input_data": {
                        "age": np.random.randint(20, 70),
                        "income": np.random.uniform(30000, 100000),
                        "tenure_months": np.random.randint(1, 60),
                        "monthly_charges": np.random.uniform(30, 150),
                        "total_charges": np.random.uniform(100, 5000)
                    },
                    "detect_drift": True
                }
                
                response = requests.post(
                    f"{base_url}/api/v1/predict",
                    json=data,
                    timeout=15
                )
                return response.status_code == 200
            except:
                return False
        
        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=size) as executor:
            futures = [executor.submit(make_prediction) for _ in range(size)]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    successful += 1
        
        total_time = time.time() - start_time
        results[size] = {
            'successful': successful,
            'total': size,
            'time': total_time,
            'rps': size / total_time if total_time > 0 else 0
        }
        
        print(f"      ✅ {successful}/{size} successful ({100*successful/size:.1f}%)")
        print(f"      ⏱️  {total_time:.2f}s total, {results[size]['rps']:.1f} RPS")
    
    print("📊 Performance benchmark summary:")
    for size, result in results.items():
        print(f"   {size:2d} concurrent: {result['rps']:5.1f} RPS, {result['successful']:2d}/{result['total']:2d} success")


def check_dashboard():
    """Check if the dashboard is accessible."""
    print("📊 Checking dashboard accessibility...")
    
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is accessible at http://localhost:8501")
        else:
            print(f"❌ Dashboard not accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard not accessible: {e}")


def show_usage_examples():
    """Show usage examples for the platform."""
    print("\n" + "="*60)
    print("🚀 MDT Dashboard Demo Complete!")
    print("="*60)
    
    print("\n📖 Usage Examples:")
    print("="*40)
    
    print("\n1. 🌐 Web Interface:")
    print("   Dashboard: http://localhost:8501")
    print("   API Docs:  http://localhost:8000/docs")
    print("   Metrics:   http://localhost:8000/metrics")
    
    print("\n2. 🐍 Python SDK:")
    print("""
   from mdt_dashboard.predict import PredictionService
   
   service = PredictionService()
   result = service.predict({
       "age": 35,
       "income": 50000,
       "tenure_months": 24
   }, model_name="customer_churn_demo")
   
   print(f"Prediction: {result.predictions[0]}")
   print(f"Drift detected: {result.drift_detected}")
   """)
    
    print("\n3. 🔗 REST API:")
    print("""
   curl -X POST "http://localhost:8000/api/v1/predict" \\
        -H "Content-Type: application/json" \\
        -d '{
          "model_id": "customer_churn_demo",
          "input_data": {
            "age": 35,
            "income": 50000,
            "tenure_months": 24,
            "monthly_charges": 75.0,
            "total_charges": 1800.0
          },
          "detect_drift": true
        }'
   """)
    
    print("\n4. 🔍 Model Information:")
    print("""
   from mdt_dashboard.predict import PredictionService
   
   service = PredictionService()
   models = service.list_available_models()
   
   for model in models:
       print(f"Model: {model['model_name']}")
       print(f"Performance: {model['performance']}")
   """)
    
    print("\n📚 Next Steps:")
    print("="*40)
    print("• Explore the dashboard at http://localhost:8501")
    print("• Check API documentation at http://localhost:8000/docs")
    print("• Train your own models using the training interface")
    print("• Set up alerts and monitoring for production use")
    print("• Configure cloud storage and deployment")
    
    print("\n💡 Pro Tips:")
    print("="*40)
    print("• Use the CLI: python cli.py --help")
    print("• Monitor with: curl http://localhost:8000/metrics")
    print("• Check logs in: logs/mdt_dashboard.log")
    print("• Configure alerts in: .env file")


def main():
    """Main demo function."""
    print("🔬 MDT Dashboard Demo")
    print("="*40)
    print("This script demonstrates the key features of the MDT platform")
    print()
    
    # Create sample data
    create_sample_data()
    print()
    
    # Train model
    model_result = train_sample_model()
    print()
    
    if model_result:
        # Create predictions
        create_sample_predictions()
        print()
        
        # Test API (if running)
        test_api_connection()
        test_model_endpoints()
        test_prediction_api()
        test_batch_predictions()
        test_drift_detection_api()
        test_metrics_endpoint()
        print()
        
        # Check dashboard
        check_dashboard()
        
        # Show usage examples
        show_usage_examples()
    else:
        print("❌ Demo incomplete due to model training failure")
        print("Please check the error messages above and ensure all dependencies are installed")


if __name__ == "__main__":
    main()

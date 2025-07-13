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

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

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


def test_api_endpoints():
    """Test API endpoints to ensure they're working."""
    print("🔧 Testing API endpoints...")
    
    api_base = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{api_base}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
    
    # Test models endpoint
    try:
        response = requests.get(f"{api_base}/api/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models endpoint working ({len(models.get('models', []))} models)")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
    
    # Test prediction endpoint
    try:
        sample_data = {
            "age": 35,
            "income": 50000,
            "tenure_months": 24,
            "monthly_charges": 75.0,
            "total_charges": 1800.0
        }
        
        payload = {
            "model_id": "customer_churn_demo",
            "input_data": sample_data,
            "detect_drift": True
        }
        
        response = requests.post(f"{api_base}/api/v1/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction endpoint working (prediction: {result.get('prediction')})")
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prediction endpoint failed: {e}")


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
        test_api_endpoints()
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

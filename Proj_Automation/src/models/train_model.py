import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os
import joblib

def load_data():
    """Load processed training data"""
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    return X_train, y_train

def train_model(X_train, y_train):
    """Train linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def save_model(model, model_path):
    """Save trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

if __name__ == "__main__":
    # Start MLflow experiment
    mlflow.set_experiment("salary_prediction")
    
    with mlflow.start_run():
        # Load data
        X_train, y_train = load_data()
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        mse, r2 = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Save model
        model_path = "models/salary_model.joblib"
        save_model(model, model_path)
        mlflow.log_artifact(model_path)
        
        print(f"Model trained and saved. MSE: {mse:.2f}, R2: {r2:.2f}")
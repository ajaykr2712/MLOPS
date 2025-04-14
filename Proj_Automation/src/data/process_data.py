import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import dvc.api

# Load data from DVC
data_path = 'data/raw/salary_data.csv'
data_url = dvc.api.get_url(data_path)

def load_data():
    """Load raw data from DVC storage"""
    return pd.read_csv(data_url)

def clean_data(df):
    """Clean and preprocess raw data"""
    # Handle missing values
    df = df.dropna()
    
    # Remove outliers
    q1 = df['Salary'].quantile(0.25)
    q3 = df['Salary'].quantile(0.75)
    iqr = q3 - q1
    df = df[~((df['Salary'] < (q1 - 1.5 * iqr)) | 
              (df['Salary'] > (q3 + 1.5 * iqr)))]
    
    return df

def split_data(df):
    """Split data into train and test sets"""
    X = df['YearsExperience'].values.reshape(-1, 1)
    y = df['Salary'].values
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Create data directory if not exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Process data
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save processed data
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
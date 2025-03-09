# End-to-End Deployment of a Machine Learning Model for Salary Prediction

## Project Overview
This project demonstrates an end-to-end pipeline for deploying a simple Machine Learning (ML) model that predicts software engineers' salaries based on their years of experience. The model will be deployed using Streamlit, and DevOps best practices will be followed for containerization, CI/CD, and cloud deployment.

## Tech Stack
ML Framework: Scikit-learn (Simple Linear Regression)
Frontend: Streamlit
Backend: Flask (Optional, if using API-based architecture)
Containerization: Docker
CI/CD: GitHub Actions
Deployment Options: AWS (using Docker or Kubernetes)



# HLD
```mermaid
graph TD;
    A[Start: Data Collection] --> B[Data Preprocessing]
    B --> C[Train Model using Scikit-learn]
    C --> D[Evaluate Model Performance]
    D --> E[Save Trained Model]
    E -->|Containerization| F[Create Docker Container]
    F -->|Deploy| G[Deploy on AWS using Docker/Kubernetes]
    G -->|Serve Predictions| H[Frontend with Streamlit]
    H -->|User Input| I[Model Prediction]
    I -->|Monitor| J[Logging & Monitoring with Prometheus/AWS CloudWatch]
    J -->|Automate| K[CI/CD using GitHub Actions]
    K --> L[Continuous Integration & Deployment]
    L --> M[End-to-End Deployment Complete]

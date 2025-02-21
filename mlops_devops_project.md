# 🚀 End-to-End ML Model Deployment with DevOps & MLOps

This project demonstrates an end-to-end pipeline for deploying a Machine Learning (ML) model that predicts software engineers' salaries based on years of experience. It follows DevOps best practices, including containerization, CI/CD, cloud deployment, and monitoring.

---

## **✅ Phase 1: Model Development & Training**  

### **1.1 Setup the Environment**
```bash
pip install numpy pandas scikit-learn streamlit joblib
```

### **1.2 Code the Salary Prediction Model (`train_model.py`)**  
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
np.random.seed(42)
X = np.random.randint(1, 20, size=(50, 1))
y = 5000 * X + np.random.randint(10000, 20000, size=(50, 1))

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "salary_model.pkl")
print("Model trained and saved!")
```

---

## **✅ Phase 2: Streamlit App & Containerization**  

### **2.1 Create the Streamlit App (`app.py`)**  
```python
import streamlit as st
import joblib
import numpy as np

model = joblib.load("salary_model.pkl")

st.title("Software Engineer Salary Predictor 💰")

experience = st.number_input("Enter Years of Experience:", min_value=0, max_value=50, step=1)

if st.button("Predict Salary"):
    salary = model.predict(np.array([[experience]]))[0][0]
    st.success(f"Estimated Salary: ${salary:,.2f}")
```

### **2.2 Dockerizing the App (`Dockerfile`)**  
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **2.3 Build & Run Docker Container Locally**  
```bash
docker build -t salary-predictor .
docker run -p 8501:8501 salary-predictor
```

---

## **✅ Phase 3: CI/CD with GitHub Actions**  

### **3.1 Setup GitHub Actions (`.github/workflows/deploy.yml`)**  
```yaml
name: Deploy Streamlit App

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint code
        run: pylint app.py train_model.py

      - name: Build Docker Image
        run: docker build -t salary-predictor .

      - name: Push Docker Image
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker tag salary-predictor your-dockerhub-username/salary-predictor:latest
          docker push your-dockerhub-username/salary-predictor:latest
```

---

## **✅ Phase 4: Cloud Deployment (AWS/GCP/Azure)**  

### **Option 1: Deploy to AWS Elastic Beanstalk**
```bash
eb init -p docker salary-predictor-app
eb create salary-env
```

### **Option 2: Deploy to Kubernetes**  
#### **Create Kubernetes Deployment (`k8s-deployment.yaml`)**  
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: salary-predictor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: salary-predictor
  template:
    metadata:
      labels:
        app: salary-predictor
    spec:
      containers:
        - name: salary-predictor
          image: your-dockerhub-username/salary-predictor:latest
          ports:
            - containerPort: 8501
---
apiVersion: v1
kind: Service
metadata:
  name: salary-service
spec:
  type: LoadBalancer
  selector:
    app: salary-predictor
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
```

#### **Deploy to Kubernetes**
```bash
kubectl apply -f k8s-deployment.yaml
```

---

## **✅ Phase 5: Monitoring & Scaling**  

### **5.1 Monitoring with Prometheus & Grafana**  
#### **Deploy Prometheus to Kubernetes**
```bash
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/setup/
```

#### **Visualize Logs & Metrics**
```bash
kubectl port-forward svc/prometheus 9090:9090
kubectl port-forward svc/grafana 3000:3000
```

---

# **🔥 Final Outcome**  
✅ **Fully automated pipeline** from training to deployment.  
✅ **Scalable infrastructure** (Docker, Kubernetes, CI/CD).  
✅ **Monitored & optimized** using Prometheus/Grafana.  
✅ **Live in production!** 🚀  

## **Next Steps & Enhancements**
🔹 Convert Streamlit to FastAPI for API-based architecture  
🔹 Add feature store for real-time salary data (Feast)  
🔹 Integrate drift detection with Evidently AI  

---

🚀 **Which cloud platform do you prefer? AWS, GCP, or Kubernetes? Let’s finalize the deployment!** 🔥

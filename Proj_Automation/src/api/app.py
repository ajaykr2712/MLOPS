from flask import Flask, request, jsonify
import joblib
import numpy as np
from prometheus_client import make_wsgi_app, Counter, Histogram
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), '../../models/salary_model.joblib')
model = joblib.load(model_path)

# Prometheus metrics
PREDICTION_COUNTER = Counter('prediction_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')

@app.route('/predict', methods=['POST'])
@PREDICTION_LATENCY.time()
def predict():
    """API endpoint for salary predictions"""
    data = request.get_json()
    years_exp = float(data['years_experience'])
    prediction = model.predict(np.array([[years_exp]]))[0]
    
    PREDICTION_COUNTER.inc()
    return jsonify({
        'predicted_salary': float(prediction),
        'years_experience': years_exp
    })

# Add prometheus wsgi middleware
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
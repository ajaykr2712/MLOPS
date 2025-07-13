/**
 * API Load Testing Script using K6
 * Tests the MDT Dashboard API endpoints under various load conditions
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
export let errorRate = new Rate('errors');
export let apiResponseTime = new Trend('api_response_time');

// Test configuration
export let options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 10 },   // Steady state with 10 users
    { duration: '2m', target: 20 },   // Ramp up to 20 users
    { duration: '5m', target: 20 },   // Steady state with 20 users
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },   // Steady state with 50 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],   // 95% of requests under 500ms
    http_req_failed: ['rate<0.05'],     // Error rate under 5%
    checks: ['rate>0.9'],               // 90% of checks should pass
  },
};

// Base configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

// Test data
const testModel = {
  name: `test-model-${Date.now()}`,
  version: '1.0.0',
  algorithm: 'random_forest',
  parameters: {
    n_estimators: 100,
    max_depth: 10
  }
};

const testPredictionData = {
  data: {
    feature1: Math.random() * 100,
    feature2: Math.random() * 50,
    feature3: Math.random() * 200
  },
  model_name: 'test-model',
  return_probabilities: false
};

export function setup() {
  console.log('Setting up load test...');
  
  // Health check before starting
  let healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'API is healthy': (r) => r.status === 200,
  });
  
  return { testModel, testPredictionData };
}

export default function(data) {
  let responses = {};
  
  // Test 1: Health check endpoint
  responses.health = http.get(`${BASE_URL}/health`);
  check(responses.health, {
    'Health check status is 200': (r) => r.status === 200,
    'Health check response time < 100ms': (r) => r.timings.duration < 100,
  });
  
  // Test 2: API health check
  responses.apiHealth = http.get(`${BASE_URL}${API_VERSION}/health`);
  check(responses.apiHealth, {
    'API health status is 200': (r) => r.status === 200,
    'API health response time < 200ms': (r) => r.timings.duration < 200,
  });
  
  // Test 3: List models endpoint
  responses.listModels = http.get(`${BASE_URL}${API_VERSION}/models`);
  check(responses.listModels, {
    'List models status is 200': (r) => r.status === 200,
    'List models has models array': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Array.isArray(body.models);
      } catch (e) {
        return false;
      }
    },
  });
  
  // Test 4: Model metrics endpoint
  responses.metrics = http.get(`${BASE_URL}${API_VERSION}/metrics`);
  check(responses.metrics, {
    'Metrics endpoint accessible': (r) => r.status === 200 || r.status === 404,
  });
  
  // Test 5: Create prediction (simulated)
  const predictionPayload = JSON.stringify(data.testPredictionData);
  const predictionParams = {
    headers: { 'Content-Type': 'application/json' },
  };
  
  responses.prediction = http.post(
    `${BASE_URL}${API_VERSION}/predict`, 
    predictionPayload, 
    predictionParams
  );
  
  check(responses.prediction, {
    'Prediction request processed': (r) => r.status === 200 || r.status === 422 || r.status === 404,
    'Prediction response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  
  // Test 6: Drift detection endpoint
  const driftPayload = JSON.stringify({
    reference_data: Array.from({length: 100}, () => Math.random()),
    comparison_data: Array.from({length: 100}, () => Math.random()),
    feature_name: 'test_feature'
  });
  
  responses.drift = http.post(
    `${BASE_URL}${API_VERSION}/drift/detect`, 
    driftPayload, 
    predictionParams
  );
  
  check(responses.drift, {
    'Drift detection request processed': (r) => r.status === 200 || r.status === 422 || r.status === 404,
    'Drift detection response time < 2000ms': (r) => r.timings.duration < 2000,
  });
  
  // Track error rates and response times
  for (let endpoint in responses) {
    let response = responses[endpoint];
    errorRate.add(response.status >= 400);
    apiResponseTime.add(response.timings.duration);
  }
  
  // Think time between requests
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

export function teardown(data) {
  console.log('Cleaning up after load test...');
  
  // Final health check
  let finalHealthCheck = http.get(`${BASE_URL}/health`);
  check(finalHealthCheck, {
    'API still healthy after load test': (r) => r.status === 200,
  });
}

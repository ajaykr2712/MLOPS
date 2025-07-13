/**
 * Dashboard Load Testing Script using K6
 * Tests the MDT Dashboard Streamlit application under load
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
export let errorRate = new Rate('dashboard_errors');
export let pageLoadTime = new Trend('page_load_time');

// Test configuration
export let options = {
  stages: [
    { duration: '1m', target: 5 },    // Ramp up to 5 users
    { duration: '3m', target: 5 },    // Steady state with 5 users
    { duration: '1m', target: 10 },   // Ramp up to 10 users
    { duration: '3m', target: 10 },   // Steady state with 10 users
    { duration: '1m', target: 20 },   // Ramp up to 20 users
    { duration: '3m', target: 20 },   // Steady state with 20 users
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],  // 95% of requests under 2s
    http_req_failed: ['rate<0.1'],      // Error rate under 10%
    checks: ['rate>0.85'],              // 85% of checks should pass
  },
};

// Base configuration
const DASHBOARD_URL = __ENV.DASHBOARD_URL || 'http://localhost:8501';

export function setup() {
  console.log('Setting up dashboard load test...');
  
  // Check if dashboard is accessible
  let healthCheck = http.get(DASHBOARD_URL);
  check(healthCheck, {
    'Dashboard is accessible': (r) => r.status === 200,
  });
  
  return {};
}

export default function() {
  let responses = {};
  
  // Test 1: Load main dashboard page
  responses.mainPage = http.get(DASHBOARD_URL);
  check(responses.mainPage, {
    'Main page loads successfully': (r) => r.status === 200,
    'Main page contains Streamlit': (r) => r.body.includes('streamlit') || r.body.includes('Streamlit'),
    'Main page load time < 3s': (r) => r.timings.duration < 3000,
  });
  
  // Test 2: Load static assets (if any)
  const staticRequests = [
    '/_stcore/static/css/bootstrap.min.css',
    '/_stcore/static/js/bootstrap.bundle.min.js',
    '/_stcore/health'
  ];
  
  staticRequests.forEach((path, index) => {
    responses[`static_${index}`] = http.get(`${DASHBOARD_URL}${path}`);
    check(responses[`static_${index}`], {
      [`Static asset ${path} loads`]: (r) => r.status === 200 || r.status === 404,
    });
  });
  
  // Test 3: Simulate user interactions with dashboard
  // This simulates WebSocket connections that Streamlit uses
  const wsParams = {
    headers: {
      'Connection': 'Upgrade',
      'Upgrade': 'websocket',
      'Sec-WebSocket-Key': 'dGhlIHNhbXBsZSBub25jZQ==',
      'Sec-WebSocket-Version': '13'
    }
  };
  
  responses.websocket = http.get(`${DASHBOARD_URL}/_stcore/stream`, wsParams);
  check(responses.websocket, {
    'WebSocket connection attempt': (r) => r.status === 101 || r.status === 400 || r.status === 404,
  });
  
  // Test 4: Test different dashboard pages/tabs (if implemented)
  const dashboardPaths = [
    '/?tab=overview',
    '/?tab=models',
    '/?tab=drift',
    '/?tab=monitoring'
  ];
  
  // Randomly select a dashboard path to test
  const randomPath = dashboardPaths[Math.floor(Math.random() * dashboardPaths.length)];
  responses.tabPage = http.get(`${DASHBOARD_URL}${randomPath}`);
  check(responses.tabPage, {
    'Dashboard tab loads': (r) => r.status === 200,
    'Tab page response time < 2s': (r) => r.timings.duration < 2000,
  });
  
  // Test 5: Health check endpoint (if available)
  responses.health = http.get(`${DASHBOARD_URL}/health`);
  check(responses.health, {
    'Health endpoint accessible': (r) => r.status === 200 || r.status === 404,
  });
  
  // Track metrics
  for (let endpoint in responses) {
    let response = responses[endpoint];
    errorRate.add(response.status >= 400);
    pageLoadTime.add(response.timings.duration);
  }
  
  // Simulate user reading/interaction time
  sleep(Math.random() * 5 + 2); // 2-7 seconds
}

export function teardown(data) {
  console.log('Cleaning up dashboard load test...');
  
  // Final check
  let finalCheck = http.get(DASHBOARD_URL);
  check(finalCheck, {
    'Dashboard still accessible after load test': (r) => r.status === 200,
  });
}

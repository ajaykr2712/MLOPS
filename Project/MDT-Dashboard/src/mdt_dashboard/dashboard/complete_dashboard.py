"""
Complete Streamlit Dashboard for MDT Platform.
Enterprise-grade dashboard with comprehensive monitoring and management features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import time

# Page configuration
st.set_page_config(
    page_title="MDT Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin-bottom: 1rem;
    }
    .info-card {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
    .stSelectbox > label {
        font-weight: bold;
    }
    .stMetric > label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)


class DashboardAPI:
    """API client for dashboard interactions."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = 10
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        try:
            url = f"{self.api_base_url}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {}
    
    def get_health(self) -> Dict[str, Any]:
        """Get API health status."""
        return self._make_request("GET", "/health")
    
    def get_models(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """Get list of available models."""
        return self._make_request("GET", "/api/v1/models", params={"skip": skip, "limit": limit})
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get model by ID."""
        return self._make_request("GET", f"/api/v1/models/{model_id}")
    
    def get_predictions(self, model_id: Optional[str] = None, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """Get predictions with filtering."""
        params = {"skip": skip, "limit": limit}
        if model_id:
            params["model_id"] = model_id
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        return self._make_request("GET", "/api/v1/predictions", params=params)
    
    def make_prediction(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction."""
        payload = {
            "model_id": model_id,
            "input_data": input_data,
            "detect_drift": True,
            "store_prediction": True
        }
        return self._make_request("POST", "/api/v1/predict", json=payload)
    
    def get_drift_reports(self, model_id: Optional[str] = None, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get drift reports."""
        params = {"skip": skip, "limit": limit}
        if model_id:
            params["model_id"] = model_id
        
        result = self._make_request("GET", "/api/v1/drift/reports", params=params)
        return result if isinstance(result, list) else []
    
    def get_alerts(self, severity: Optional[str] = None, resolved: Optional[bool] = None, 
                   skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts."""
        params = {"skip": skip, "limit": limit}
        if severity:
            params["severity"] = severity
        if resolved is not None:
            params["resolved"] = resolved
        
        result = self._make_request("GET", "/api/v1/alerts", params=params)
        return result if isinstance(result, list) else []
    
    def resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        """Resolve an alert."""
        return self._make_request("PATCH", f"/api/v1/alerts/{alert_id}/resolve")


@st.cache_data(ttl=30)
def get_cached_data(api_client: DashboardAPI, data_type: str, **kwargs):
    """Cache data with TTL."""
    if data_type == "health":
        return api_client.get_health()
    elif data_type == "models":
        return api_client.get_models(**kwargs)
    elif data_type == "predictions":
        return api_client.get_predictions(**kwargs)
    elif data_type == "drift_reports":
        return api_client.get_drift_reports(**kwargs)
    elif data_type == "alerts":
        return api_client.get_alerts(**kwargs)
    return {}


def create_metric_card(title: str, value: Any, delta: Optional[str] = None, card_type: str = "metric"):
    """Create a styled metric card."""
    card_class = f"{card_type}-card"
    
    delta_html = ""
    if delta:
        delta_color = "green" if "↑" in delta or "+" in delta else "red" if "↓" in delta or "-" in delta else "gray"
        delta_html = f'<small style="color: {delta_color};">{delta}</small>'
    
    st.markdown(f"""
    <div class="{card_class}">
        <h4>{title}</h4>
        <h2>{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def plot_time_series_metrics(data: List[Dict[str, Any]], title: str, y_column: str, x_column: str = "timestamp"):
    """Create time series plot for metrics."""
    if not data:
        st.warning(f"No data available for {title}")
        return
    
    df = pd.DataFrame(data)
    if x_column in df.columns:
        df[x_column] = pd.to_datetime(df[x_column])
    
    fig = px.line(
        df, 
        x=x_column, 
        y=y_column,
        title=title,
        labels={x_column: 'Time', y_column: title}
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title=title,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_model_performance_comparison(models_data: List[Dict[str, Any]]):
    """Create model performance comparison chart."""
    if not models_data:
        st.warning("No models available for comparison")
        return
    
    df = pd.DataFrame(models_data)
    
    # Create bar chart for model performance
    fig = px.bar(
        df,
        x='name',
        y='performance_score',
        title="Model Performance Comparison",
        labels={'name': 'Model', 'performance_score': 'Performance Score'},
        color='performance_score',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_drift_trends(drift_reports: List[Dict[str, Any]]):
    """Plot drift trends over time."""
    if not drift_reports:
        st.warning("No drift reports available")
        return
    
    df = pd.DataFrame(drift_reports)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Group by date and calculate drift rate
    daily_drift = df.groupby(df['created_at'].dt.date).agg({
        'drift_detected': 'mean',
        'overall_drift_score': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Drift Detection Rate', 'Average Drift Score'),
        vertical_spacing=0.1
    )
    
    # Drift rate
    fig.add_trace(
        go.Scatter(
            x=daily_drift['created_at'],
            y=daily_drift['drift_detected'],
            mode='lines+markers',
            name='Drift Rate'
        ),
        row=1, col=1
    )
    
    # Drift score
    fig.add_trace(
        go.Scatter(
            x=daily_drift['created_at'],
            y=daily_drift['overall_drift_score'],
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=500, title_text="Drift Trends Over Time")
    st.plotly_chart(fig, use_container_width=True)


def plot_alert_distribution(alerts: List[Dict[str, Any]]):
    """Plot alert distribution by severity."""
    if not alerts:
        st.warning("No alerts available")
        return
    
    df = pd.DataFrame(alerts)
    
    # Count by severity
    severity_counts = df['severity'].value_counts()
    
    fig = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        title="Alert Distribution by Severity"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_overview_page(api_client: DashboardAPI):
    """Render the overview dashboard page."""
    st.markdown('<h1 class="main-header">🔬 MDT Dashboard Overview</h1>', unsafe_allow_html=True)
    
    # Health check
    health_data = get_cached_data(api_client, "health")
    
    # Health status indicators
    st.subheader("🏥 System Health")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        api_status = health_data.get("status", "unknown")
        card_type = "success" if api_status == "healthy" else "error"
        create_metric_card("API Status", api_status.title(), card_type=card_type)
    
    with col2:
        db_status = health_data.get("database", "unknown")
        card_type = "success" if db_status == "healthy" else "error"
        create_metric_card("Database", db_status.title(), card_type=card_type)
    
    with col3:
        redis_status = health_data.get("redis", "unknown")
        card_type = "success" if redis_status == "healthy" else "error"
        create_metric_card("Redis", redis_status.title(), card_type=card_type)
    
    with col4:
        env = health_data.get("environment", "unknown")
        create_metric_card("Environment", env.title(), card_type="info")
    
    # Get models data
    models_data = get_cached_data(api_client, "models")
    models_list = models_data.get("models", [])
    
    # Model metrics
    st.subheader("🤖 Model Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_models = len(models_list)
        create_metric_card("Total Models", total_models)
    
    with col2:
        active_models = len([m for m in models_list if m.get("status") == "active"])
        create_metric_card("Active Models", active_models)
    
    with col3:
        training_models = len([m for m in models_list if m.get("status") == "training"])
        card_type = "alert" if training_models > 0 else "success"
        create_metric_card("Training", training_models, card_type=card_type)
    
    with col4:
        failed_models = len([m for m in models_list if m.get("status") == "failed"])
        card_type = "error" if failed_models > 0 else "success"
        create_metric_card("Failed", failed_models, card_type=card_type)
    
    # Get predictions data
    predictions_data = get_cached_data(api_client, "predictions", limit=1000)
    predictions_list = predictions_data.get("predictions", [])
    
    # Prediction metrics
    st.subheader("📊 Prediction Metrics")
    if predictions_list:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predictions = len(predictions_list)
            create_metric_card("Total Predictions", total_predictions)
        
        with col2:
            # Calculate average response time
            response_times = [p.get("response_time_ms", 0) for p in predictions_list if p.get("response_time_ms")]
            avg_response_time = np.mean(response_times) if response_times else 0
            create_metric_card("Avg Response Time", f"{avg_response_time:.1f}ms")
        
        with col3:
            # Calculate drift rate
            drift_predictions = [p for p in predictions_list if p.get("drift_score") is not None]
            drift_rate = len([p for p in drift_predictions if p.get("drift_score", 0) > 0.05]) / len(drift_predictions) * 100 if drift_predictions else 0
            card_type = "error" if drift_rate > 20 else "alert" if drift_rate > 10 else "success"
            create_metric_card("Drift Rate", f"{drift_rate:.1f}%", card_type=card_type)
        
        with col4:
            # Predictions today
            today = datetime.now().date()
            today_predictions = [p for p in predictions_list if datetime.fromisoformat(p.get("prediction_time", "")).date() == today]
            create_metric_card("Today's Predictions", len(today_predictions))
    
    # Get alerts data
    alerts_data = get_cached_data(api_client, "alerts", resolved=False)
    
    # Alerts overview
    st.subheader("🚨 Active Alerts")
    if alerts_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_alerts = len(alerts_data)
            card_type = "error" if total_alerts > 10 else "alert" if total_alerts > 5 else "success"
            create_metric_card("Active Alerts", total_alerts, card_type=card_type)
        
        with col2:
            critical_alerts = len([a for a in alerts_data if a.get("severity") == "critical"])
            card_type = "error" if critical_alerts > 0 else "success"
            create_metric_card("Critical", critical_alerts, card_type=card_type)
        
        with col3:
            high_alerts = len([a for a in alerts_data if a.get("severity") == "high"])
            card_type = "error" if high_alerts > 3 else "alert" if high_alerts > 0 else "success"
            create_metric_card("High Severity", high_alerts, card_type=card_type)
        
        with col4:
            medium_alerts = len([a for a in alerts_data if a.get("severity") == "medium"])
            create_metric_card("Medium Severity", medium_alerts, card_type="info")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if models_list:
            # Add performance score for demo
            for model in models_list:
                model['performance_score'] = np.random.uniform(0.7, 0.95)
            plot_model_performance_comparison(models_list)
    
    with col2:
        if alerts_data:
            plot_alert_distribution(alerts_data)


def render_models_page(api_client: DashboardAPI):
    """Render the models management page."""
    st.markdown('<h1 class="main-header">🤖 Model Management</h1>', unsafe_allow_html=True)
    
    # Get models list
    models_data = get_cached_data(api_client, "models")
    models_list = models_data.get("models", [])
    
    if not models_list:
        st.warning("No models available or unable to connect to API")
        return
    
    # Models overview
    st.subheader("📋 Available Models")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status:",
            options=["All", "active", "training", "deprecated", "failed"],
            index=0
        )
    
    with col2:
        algorithm_filter = st.selectbox(
            "Filter by Algorithm:",
            options=["All"] + list(set([m.get("algorithm", "unknown") for m in models_list])),
            index=0
        )
    
    with col3:
        # Sort is for future implementation
        st.selectbox(
            "Sort by:",
            options=["Name", "Created Date", "Status"],
            index=1
        )
    
    # Apply filters
    filtered_models = models_list
    if status_filter != "All":
        filtered_models = [m for m in filtered_models if m.get("status") == status_filter]
    if algorithm_filter != "All":
        filtered_models = [m for m in filtered_models if m.get("algorithm") == algorithm_filter]
    
    # Create DataFrame for display
    if filtered_models:
        models_df = pd.DataFrame(filtered_models)
        
        # Display models table
        st.dataframe(
            models_df[['name', 'version', 'algorithm', 'status', 'created_at']],
            use_container_width=True
        )
        
        # Model details section
        st.subheader("🔍 Model Details")
        
        selected_model_name = st.selectbox(
            "Select a model for detailed information:",
            options=[model['name'] for model in filtered_models],
            index=0
        )
        
        if selected_model_name:
            selected_model = next((m for m in filtered_models if m['name'] == selected_model_name), None)
            
            if selected_model:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json(selected_model)
                
                with col2:
                    # Model metrics
                    metrics = selected_model.get('metrics', {})
                    if metrics:
                        st.subheader("Model Metrics")
                        for metric_name, metric_value in metrics.items():
                            st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.4f}")
                    
                    # Model info
                    st.subheader("Model Information")
                    st.write(f"**Algorithm:** {selected_model.get('algorithm', 'N/A')}")
                    st.write(f"**Framework:** {selected_model.get('framework', 'N/A')}")
                    st.write(f"**Status:** {selected_model.get('status', 'N/A')}")
                    st.write(f"**Created:** {selected_model.get('created_at', 'N/A')}")


def render_predictions_page(api_client: DashboardAPI):
    """Render the predictions page."""
    st.markdown('<h1 class="main-header">📊 Predictions</h1>', unsafe_allow_html=True)
    
    # Get models for selection
    models_data = get_cached_data(api_client, "models")
    models_list = models_data.get("models", [])
    
    if not models_list:
        st.warning("No models available")
        return
    
    # Prediction interface
    st.subheader("🔮 Make Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Select Model:",
            options=[m['name'] for m in models_list if m.get('status') == 'active'],
            index=0 if any(m.get('status') == 'active' for m in models_list) else None
        )
    
    with col2:
        if selected_model:
            model_info = next((m for m in models_list if m['name'] == selected_model), None)
            if model_info:
                st.info(f"Model: {model_info['name']} ({model_info['algorithm']})")
    
    # Input form for prediction
    if selected_model:
        st.subheader("Input Data")
        
        # Simple input form (can be customized based on model features)
        input_method = st.radio("Input Method:", ["Manual Entry", "JSON Upload"], index=0)
        
        if input_method == "Manual Entry":
            col1, col2 = st.columns(2)
            
            with col1:
                feature1 = st.number_input("Feature 1:", value=0.0)
                feature3 = st.number_input("Feature 3:", value=0.0)
            
            with col2:
                feature2 = st.number_input("Feature 2:", value=0.0)
                feature4 = st.number_input("Feature 4:", value=0.0)
            
            input_data = {
                "feature1": feature1,
                "feature2": feature2,
                "feature3": feature3,
                "feature4": feature4
            }
        
        else:
            json_input = st.text_area("JSON Input:", value='{"feature1": 1.0, "feature2": 2.0}', height=150)
            try:
                input_data = json.loads(json_input)
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
                input_data = None
        
        # Make prediction
        if st.button("Make Prediction", type="primary") and input_data:
            with st.spinner("Making prediction..."):
                model_id = model_info.get('id')
                result = api_client.make_prediction(model_id, input_data)
                
                if result:
                    st.subheader("Prediction Result")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Prediction", result.get('prediction', 'N/A'))
                    
                    with col2:
                        response_time = result.get('response_time_ms', 0)
                        st.metric("Response Time", f"{response_time:.2f} ms")
                    
                    with col3:
                        drift_detected = result.get('drift_detected', False)
                        drift_status = "Yes" if drift_detected else "No"
                        st.metric("Drift Detected", drift_status)
                    
                    # Display full result
                    with st.expander("Full Result"):
                        st.json(result)
                else:
                    st.error("Prediction failed. Please check the API connection.")
    
    # Recent predictions
    st.subheader("📈 Recent Predictions")
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date:", value=datetime.now().date() - timedelta(days=7))
    
    with col2:
        end_date = st.date_input("End Date:", value=datetime.now().date())
    
    # Get predictions
    predictions_data = get_cached_data(
        api_client, 
        "predictions", 
        start_date=datetime.combine(start_date, datetime.min.time()),
        end_date=datetime.combine(end_date, datetime.max.time()),
        limit=1000
    )
    
    predictions_list = predictions_data.get("predictions", [])
    
    if predictions_list:
        # Display predictions table
        predictions_df = pd.DataFrame(predictions_list)
        st.dataframe(predictions_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time over time
            plot_time_series_metrics(predictions_list, "Response Time (ms)", "response_time_ms", "prediction_time")
        
        with col2:
            # Drift scores over time
            drift_predictions = [p for p in predictions_list if p.get("drift_score") is not None]
            if drift_predictions:
                plot_time_series_metrics(drift_predictions, "Drift Score", "drift_score", "prediction_time")
    else:
        st.info("No predictions found for the selected date range.")


def render_drift_page(api_client: DashboardAPI):
    """Render the drift detection page."""
    st.markdown('<h1 class="main-header">🌊 Drift Detection</h1>', unsafe_allow_html=True)
    
    # Get drift reports
    drift_reports = get_cached_data(api_client, "drift_reports", limit=1000)
    
    if not drift_reports:
        st.warning("No drift reports available")
        return
    
    # Drift overview
    st.subheader("📊 Drift Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_reports = len(drift_reports)
        create_metric_card("Total Reports", total_reports)
    
    with col2:
        drift_detected_count = len([r for r in drift_reports if r.get("drift_detected", False)])
        drift_rate = (drift_detected_count / total_reports * 100) if total_reports > 0 else 0
        card_type = "error" if drift_rate > 20 else "alert" if drift_rate > 10 else "success"
        create_metric_card("Drift Rate", f"{drift_rate:.1f}%", card_type=card_type)
    
    with col3:
        avg_drift_score = np.mean([r.get("overall_drift_score", 0) for r in drift_reports])
        create_metric_card("Avg Drift Score", f"{avg_drift_score:.4f}")
    
    with col4:
        recent_reports = [r for r in drift_reports if 
                         datetime.fromisoformat(r.get("created_at", "")).date() >= datetime.now().date() - timedelta(days=1)]
        create_metric_card("Reports (24h)", len(recent_reports))
    
    # Drift trends
    st.subheader("📈 Drift Trends")
    plot_drift_trends(drift_reports)
    
    # Drift reports table
    st.subheader("📋 Drift Reports")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        drift_filter = st.selectbox(
            "Filter by Drift Status:",
            options=["All", "Drift Detected", "No Drift"],
            index=0
        )
    
    with col2:
        method_filter = st.selectbox(
            "Filter by Detection Method:",
            options=["All"] + list(set([r.get("detection_method", "unknown") for r in drift_reports])),
            index=0
        )
    
    # Apply filters
    filtered_reports = drift_reports
    if drift_filter == "Drift Detected":
        filtered_reports = [r for r in filtered_reports if r.get("drift_detected", False)]
    elif drift_filter == "No Drift":
        filtered_reports = [r for r in filtered_reports if not r.get("drift_detected", False)]
    
    if method_filter != "All":
        filtered_reports = [r for r in filtered_reports if r.get("detection_method") == method_filter]
    
    if filtered_reports:
        reports_df = pd.DataFrame(filtered_reports)
        st.dataframe(reports_df, use_container_width=True)
    else:
        st.info("No drift reports match the selected filters.")


def render_alerts_page(api_client: DashboardAPI):
    """Render the alerts page."""
    st.markdown('<h1 class="main-header">🚨 Alerts & Notifications</h1>', unsafe_allow_html=True)
    
    # Get alerts
    alerts_data = get_cached_data(api_client, "alerts", limit=1000)
    
    if not alerts_data:
        st.warning("No alerts available")
        return
    
    # Alerts overview
    st.subheader("📊 Alerts Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alerts = len(alerts_data)
        create_metric_card("Total Alerts", total_alerts)
    
    with col2:
        active_alerts = len([a for a in alerts_data if not a.get("is_resolved", False)])
        card_type = "error" if active_alerts > 10 else "alert" if active_alerts > 5 else "success"
        create_metric_card("Active Alerts", active_alerts, card_type=card_type)
    
    with col3:
        critical_alerts = len([a for a in alerts_data if a.get("severity") == "critical" and not a.get("is_resolved", False)])
        card_type = "error" if critical_alerts > 0 else "success"
        create_metric_card("Critical Active", critical_alerts, card_type=card_type)
    
    with col4:
        resolved_today = len([a for a in alerts_data if 
                             a.get("is_resolved", False) and 
                             datetime.fromisoformat(a.get("created_at", "")).date() == datetime.now().date()])
        create_metric_card("Resolved Today", resolved_today, card_type="success")
    
    # Alert distribution
    col1, col2 = st.columns(2)
    
    with col1:
        plot_alert_distribution(alerts_data)
    
    with col2:
        # Alert trends over time
        if alerts_data:
            alerts_df = pd.DataFrame(alerts_data)
            alerts_df['created_at'] = pd.to_datetime(alerts_df['created_at'])
            daily_alerts = alerts_df.groupby(alerts_df['created_at'].dt.date).size().reset_index()
            daily_alerts.columns = ['date', 'count']
            
            fig = px.line(
                daily_alerts,
                x='date',
                y='count',
                title="Daily Alert Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Alerts management
    st.subheader("📋 Alert Management")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity:",
            options=["All", "critical", "high", "medium", "low"],
            index=0
        )
    
    with col2:
        status_filter = st.selectbox(
            "Filter by Status:",
            options=["All", "Active", "Resolved"],
            index=0
        )
    
    with col3:
        alert_type_filter = st.selectbox(
            "Filter by Type:",
            options=["All"] + list(set([a.get("alert_type", "unknown") for a in alerts_data])),
            index=0
        )
    
    # Apply filters
    filtered_alerts = alerts_data
    if severity_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a.get("severity") == severity_filter]
    if status_filter == "Active":
        filtered_alerts = [a for a in filtered_alerts if not a.get("is_resolved", False)]
    elif status_filter == "Resolved":
        filtered_alerts = [a for a in filtered_alerts if a.get("is_resolved", False)]
    if alert_type_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a.get("alert_type") == alert_type_filter]
    
    # Display alerts
    if filtered_alerts:
        for alert in filtered_alerts:
            with st.container():
                severity = alert.get("severity", "unknown")
                is_resolved = alert.get("is_resolved", False)
                
                # Determine card type based on severity and status
                if is_resolved:
                    card_type = "success"
                elif severity == "critical":
                    card_type = "error"
                elif severity == "high":
                    card_type = "alert"
                else:
                    card_type = "info"
                
                st.markdown(f"""
                <div class="{card_type}-card">
                    <h4>{alert.get('title', 'Untitled Alert')} 
                        <span style="float: right;">
                            <small>{severity.upper()}</small>
                            {'✅ RESOLVED' if is_resolved else '🔴 ACTIVE'}
                        </span>
                    </h4>
                    <p>{alert.get('message', 'No message')}</p>
                    <small>Created: {alert.get('created_at', 'Unknown')} | Type: {alert.get('alert_type', 'Unknown')}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Resolve button for active alerts
                if not is_resolved:
                    if st.button("Resolve Alert", key=f"resolve_{alert.get('id')}"):
                        result = api_client.resolve_alert(alert.get('id'))
                        if result:
                            st.success("Alert resolved successfully!")
                            st.experimental_rerun()
                        else:
                            st.error("Failed to resolve alert")
    else:
        st.info("No alerts match the selected filters.")


def main():
    """Main dashboard application."""
    # Initialize API client
    api_base_url = st.sidebar.text_input("API Base URL", value="http://localhost:8000")
    api_client = DashboardAPI(api_base_url)
    
    # Sidebar navigation
    st.sidebar.title("🔬 MDT Dashboard")
    
    pages = {
        "📊 Overview": render_overview_page,
        "🤖 Models": render_models_page,
        "📈 Predictions": render_predictions_page,
        "🌊 Drift Detection": render_drift_page,
        "🚨 Alerts": render_alerts_page
    }
    
    selected_page = st.sidebar.selectbox("Navigate to:", list(pages.keys()))
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Connection status
    st.sidebar.subheader("Connection Status")
    health_status = api_client.get_health()
    if health_status:
        status = health_status.get("status", "unknown")
        color = "🟢" if status == "healthy" else "🔴"
        st.sidebar.markdown(f"{color} API: {status.title()}")
    else:
        st.sidebar.markdown("🔴 API: Disconnected")
    
    # Render selected page
    pages[selected_page](api_client)


if __name__ == "__main__":
    main()

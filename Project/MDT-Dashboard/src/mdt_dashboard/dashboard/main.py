"""
Advanced Streamlit Dashboard for MDT Platform.
Provides comprehensive visualization and management interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import time
import requests
from pathlib import Path

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
    }
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)


class DashboardAPI:
    """API client for dashboard interactions."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip("/")
        self.session = requests.Session()
    
    def get_health(self) -> Dict[str, Any]:
        """Get API health status."""
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.api_base_url}/models", timeout=10)
            return response.json() if response.status_code == 200 else []
        except:
            return []
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information."""
        try:
            response = self.session.get(f"{self.api_base_url}/models/{model_name}", timeout=10)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def get_metrics(self, time_window_minutes: int = 60, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics summary."""
        try:
            params = {"time_window_minutes": time_window_minutes}
            if model_name:
                params["model_name"] = model_name
            
            response = self.session.get(f"{self.api_base_url}/metrics", params=params, timeout=10)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def predict(self, data: Dict[str, Any], model_name: str = "best_model") -> Dict[str, Any]:
        """Make a prediction."""
        try:
            payload = {
                "data": data,
                "model_name": model_name,
                "return_probabilities": True,
                "return_feature_importance": True
            }
            response = self.session.post(f"{self.api_base_url}/predict", json=payload, timeout=30)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}


@st.cache_data(ttl=30)
def get_cached_data(api_client: DashboardAPI, data_type: str, **kwargs):
    """Cache data with TTL."""
    if data_type == "health":
        return api_client.get_health()
    elif data_type == "models":
        return api_client.get_models()
    elif data_type == "metrics":
        return api_client.get_metrics(**kwargs)
    elif data_type == "model_info":
        return api_client.get_model_info(kwargs.get("model_name", ""))
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


def plot_time_series_metrics(metrics_data: List[Dict[str, Any]], title: str, y_column: str):
    """Create time series plot for metrics."""
    if not metrics_data:
        st.warning(f"No data available for {title}")
        return
    
    df = pd.DataFrame(metrics_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = px.line(
        df, 
        x='timestamp', 
        y=y_column,
        title=title,
        labels={'timestamp': 'Time', y_column: title}
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title=title,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_model_comparison(models_data: List[Dict[str, Any]]):
    """Create model comparison chart."""
    if not models_data:
        st.warning("No models available for comparison")
        return
    
    df = pd.DataFrame(models_data)
    
    # Create bar chart for model performance
    fig = px.bar(
        df,
        x='model_name',
        y='performance',
        title="Model Performance Comparison",
        labels={'model_name': 'Model', 'performance': 'Performance Score'},
        color='performance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_drift_detection_results(drift_results: List[Dict[str, Any]]):
    """Plot drift detection results."""
    if not drift_results:
        st.info("No drift detection results available")
        return
    
    # Create dataframe
    df = pd.DataFrame(drift_results)
    
    # Severity distribution
    severity_counts = df['severity'].value_counts()
    
    fig_severity = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        title="Drift Severity Distribution",
        color_discrete_map={
            'none': '#28a745',
            'low': '#ffc107',
            'medium': '#fd7e14',
            'high': '#dc3545'
        }
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        # Feature-wise drift detection
        feature_drift = df.groupby('feature_name')['is_drift'].sum().sort_values(ascending=False).head(10)
        
        if not feature_drift.empty:
            fig_features = px.bar(
                x=feature_drift.values,
                y=feature_drift.index,
                orientation='h',
                title="Top Features with Drift",
                labels={'x': 'Drift Count', 'y': 'Feature'}
            )
            st.plotly_chart(fig_features, use_container_width=True)


def render_overview_page(api_client: DashboardAPI):
    """Render the overview page."""
    st.markdown('<h1 class="main-header">🔬 MDT Dashboard - Overview</h1>', unsafe_allow_html=True)
    
    # Get health status
    health_data = get_cached_data(api_client, "health")
    
    if not health_data:
        st.error("❌ Cannot connect to API server. Please ensure the server is running.")
        return
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = health_data.get("status", "unknown")
        status_emoji = "✅" if status == "healthy" else "❌"
        create_metric_card("API Status", f"{status_emoji} {status.title()}")
    
    with col2:
        uptime = health_data.get("uptime_seconds", 0)
        uptime_str = f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m"
        create_metric_card("Uptime", uptime_str)
    
    with col3:
        version = health_data.get("version", "Unknown")
        create_metric_card("Version", version)
    
    with col4:
        pred_service = health_data.get("prediction_service", {})
        models_count = pred_service.get("available_models", 0)
        create_metric_card("Available Models", models_count)
    
    st.markdown("---")
    
    # Get metrics
    metrics_data = get_cached_data(api_client, "metrics", time_window_minutes=60)
    
    if metrics_data:
        prediction_summary = metrics_data.get("prediction_summary", {})
        system_summary = metrics_data.get("system_summary", {})
        
        # Prediction metrics
        st.subheader("📊 Prediction Metrics (Last Hour)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_requests = prediction_summary.get("total_requests", 0)
            create_metric_card("Total Requests", total_requests)
        
        with col2:
            avg_time = prediction_summary.get("avg_processing_time_ms", 0)
            create_metric_card("Avg Response Time", f"{avg_time:.1f}ms")
        
        with col3:
            error_rate = prediction_summary.get("error_rate", 0) * 100
            card_type = "error" if error_rate > 5 else "success" if error_rate == 0 else "alert"
            create_metric_card("Error Rate", f"{error_rate:.1f}%", card_type=card_type)
        
        with col4:
            drift_rate = prediction_summary.get("drift_detection_rate", 0) * 100
            card_type = "error" if drift_rate > 20 else "alert" if drift_rate > 10 else "success"
            create_metric_card("Drift Rate", f"{drift_rate:.1f}%", card_type=card_type)
        
        # System metrics
        st.subheader("🖥️ System Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = system_summary.get("avg_cpu_usage", 0)
            card_type = "error" if cpu_usage > 80 else "alert" if cpu_usage > 60 else "success"
            create_metric_card("CPU Usage", f"{cpu_usage:.1f}%", card_type=card_type)
        
        with col2:
            memory_usage = system_summary.get("avg_memory_usage", 0)
            card_type = "error" if memory_usage > 80 else "alert" if memory_usage > 60 else "success"
            create_metric_card("Memory Usage", f"{memory_usage:.1f}%", card_type=card_type)
        
        with col3:
            network_sent = system_summary.get("total_network_sent_mb", 0)
            create_metric_card("Network Sent", f"{network_sent:.1f} MB")
        
        with col4:
            active_threads = system_summary.get("avg_active_threads", 0)
            create_metric_card("Active Threads", f"{active_threads:.0f}")


def render_models_page(api_client: DashboardAPI):
    """Render the models page."""
    st.markdown('<h1 class="main-header">🤖 Model Management</h1>', unsafe_allow_html=True)
    
    # Get models list
    models_data = get_cached_data(api_client, "models")
    
    if not models_data:
        st.warning("No models available or unable to connect to API")
        return
    
    # Models overview
    st.subheader("📋 Available Models")
    
    # Create DataFrame for display
    models_df = pd.DataFrame(models_data)
    
    if not models_df.empty:
        # Display models table
        st.dataframe(
            models_df[['model_name', 'algorithm', 'performance', 'size_mb', 'registered_at']],
            use_container_width=True
        )
        
        # Model comparison chart
        plot_model_comparison(models_data)
        
        # Model details section
        st.subheader("🔍 Model Details")
        
        selected_model = st.selectbox(
            "Select a model for detailed information:",
            options=[model['model_name'] for model in models_data],
            index=0
        )
        
        if selected_model:
            model_info = get_cached_data(api_client, "model_info", model_name=selected_model)
            
            if model_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json(model_info)
                
                with col2:
                    # Model performance metrics if available
                    if 'performance' in model_info:
                        st.metric(
                            "Performance Score",
                            f"{model_info['performance']:.4f}"
                        )
                    
                    if 'features' in model_info:
                        st.write(f"**Number of Features:** {len(model_info['features'])}")
                    
                    if 'training_date' in model_info:
                        st.write(f"**Training Date:** {model_info['training_date']}")


def render_predictions_page(api_client: DashboardAPI):
    """Render the predictions page."""
    st.markdown('<h1 class="main-header">🎯 Model Predictions</h1>', unsafe_allow_html=True)
    
    # Get available models
    models_data = get_cached_data(api_client, "models")
    
    if not models_data:
        st.error("No models available")
        return
    
    model_names = [model['model_name'] for model in models_data]
    
    # Prediction interface
    st.subheader("🔮 Make Predictions")
    
    selected_model = st.selectbox("Select Model:", model_names)
    
    # Get model info to understand features
    model_info = get_cached_data(api_client, "model_info", model_name=selected_model)
    
    if model_info and 'features' in model_info:
        features = model_info['features']
        
        st.write("**Enter feature values:**")
        
        # Create input fields for each feature
        feature_inputs = {}
        cols = st.columns(min(3, len(features)))
        
        for i, feature in enumerate(features):
            with cols[i % len(cols)]:
                # Try to infer input type based on feature name
                if any(keyword in feature.lower() for keyword in ['count', 'number', 'age', 'size']):
                    feature_inputs[feature] = st.number_input(f"{feature}:", value=0.0)
                elif any(keyword in feature.lower() for keyword in ['category', 'type', 'class']):
                    feature_inputs[feature] = st.text_input(f"{feature}:", value="")
                else:
                    feature_inputs[feature] = st.number_input(f"{feature}:", value=0.0)
        
        if st.button("🚀 Make Prediction", type="primary"):
            # Make prediction
            result = api_client.predict(feature_inputs, selected_model)
            
            if result:
                st.success("✅ Prediction completed!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Results")
                    
                    predictions = result.get('predictions', [])
                    if predictions:
                        st.metric("Prediction", f"{predictions[0]:.4f}")
                    
                    # Probabilities if available
                    probabilities = result.get('probabilities')
                    if probabilities:
                        st.write("**Class Probabilities:**")
                        for i, prob in enumerate(probabilities[0]):
                            st.write(f"Class {i}: {prob:.4f}")
                    
                    # Confidence scores
                    confidence = result.get('confidence_scores', [])
                    if confidence:
                        st.metric("Confidence", f"{confidence[0]:.4f}")
                
                with col2:
                    st.subheader("🔍 Analysis")
                    
                    # Feature importance
                    feature_importance = result.get('feature_importance')
                    if feature_importance:
                        importance_df = pd.DataFrame(
                            list(feature_importance.items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            importance_df.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 10 Feature Importance"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Drift detection
                    drift_detected = result.get('drift_detected', False)
                    if drift_detected:
                        st.warning("⚠️ Drift detected in input data!")
                        
                        drift_details = result.get('drift_details', [])
                        if drift_details:
                            plot_drift_detection_results(drift_details)
                    else:
                        st.success("✅ No drift detected")
            else:
                st.error("❌ Prediction failed. Please check your inputs.")
    else:
        st.warning("Model features information not available. You can still try manual input.")
        
        # Manual JSON input
        st.subheader("📝 Manual Input")
        json_input = st.text_area(
            "Enter feature data as JSON:",
            value='{"feature1": 1.0, "feature2": 2.0}',
            height=150
        )
        
        if st.button("🚀 Predict with JSON"):
            try:
                data = json.loads(json_input)
                result = api_client.predict(data, selected_model)
                
                if result:
                    st.success("✅ Prediction completed!")
                    st.json(result)
                else:
                    st.error("❌ Prediction failed")
                    
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format")


def render_monitoring_page(api_client: DashboardAPI):
    """Render the monitoring page."""
    st.markdown('<h1 class="main-header">📈 Monitoring & Analytics</h1>', unsafe_allow_html=True)
    
    # Time window selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Metrics Dashboard")
    with col2:
        time_window = st.selectbox(
            "Time Window:",
            [60, 180, 360, 720, 1440],
            format_func=lambda x: f"{x//60}h {x%60}m" if x >= 60 else f"{x}m"
        )
    
    # Get metrics
    metrics_data = get_cached_data(api_client, "metrics", time_window_minutes=time_window)
    
    if not metrics_data:
        st.warning("No metrics data available")
        return
    
    prediction_summary = metrics_data.get("prediction_summary", {})
    system_summary = metrics_data.get("system_summary", {})
    model_stats = metrics_data.get("model_stats", {})
    
    # Key metrics overview
    st.subheader("🎯 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_requests = prediction_summary.get("total_requests", 0)
        requests_per_min = prediction_summary.get("requests_per_minute", 0)
        create_metric_card("Total Requests", total_requests, f"{requests_per_min:.1f}/min")
    
    with col2:
        avg_response = prediction_summary.get("avg_processing_time_ms", 0)
        create_metric_card("Avg Response", f"{avg_response:.1f}ms")
    
    with col3:
        error_rate = prediction_summary.get("error_rate", 0) * 100
        create_metric_card("Error Rate", f"{error_rate:.1f}%")
    
    with col4:
        avg_confidence = prediction_summary.get("avg_confidence", 0)
        create_metric_card("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col5:
        drift_rate = prediction_summary.get("drift_detection_rate", 0) * 100
        create_metric_card("Drift Rate", f"{drift_rate:.1f}%")
    
    # System health metrics
    st.subheader("🖥️ System Health")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = system_summary.get("avg_cpu_usage", 0)
        max_cpu = system_summary.get("max_cpu_usage", 0)
        create_metric_card("CPU Usage", f"{cpu_usage:.1f}%", f"Max: {max_cpu:.1f}%")
    
    with col2:
        memory_usage = system_summary.get("avg_memory_usage", 0)
        max_memory = system_summary.get("max_memory_usage", 0)
        create_metric_card("Memory Usage", f"{memory_usage:.1f}%", f"Max: {max_memory:.1f}%")
    
    with col3:
        network_sent = system_summary.get("total_network_sent_mb", 0)
        network_recv = system_summary.get("total_network_received_mb", 0)
        create_metric_card("Network I/O", f"{network_sent:.1f}MB", f"↓{network_recv:.1f}MB")
    
    with col4:
        avg_threads = system_summary.get("avg_active_threads", 0)
        max_threads = system_summary.get("max_active_threads", 0)
        create_metric_card("Active Threads", f"{avg_threads:.0f}", f"Max: {max_threads:.0f}")
    
    # Model usage statistics
    if model_stats:
        st.subheader("🤖 Model Usage Statistics")
        
        # Create model usage dataframe
        model_usage_data = []
        for model_name, stats in model_stats.items():
            model_usage_data.append({
                "Model": model_name,
                "Total Requests": stats.get("total_requests", 0),
                "Total Predictions": stats.get("total_predictions", 0),
                "Error Count": stats.get("total_errors", 0),
                "Avg Processing Time (ms)": stats.get("avg_processing_time", 0),
                "Drift Detections": stats.get("drift_detection_count", 0),
                "Last Seen": stats.get("last_seen", "Never")
            })
        
        if model_usage_data:
            model_usage_df = pd.DataFrame(model_usage_data)
            st.dataframe(model_usage_df, use_container_width=True)
            
            # Model usage charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Requests by model
                fig_requests = px.bar(
                    model_usage_df,
                    x="Model",
                    y="Total Requests",
                    title="Requests by Model",
                    color="Total Requests",
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig_requests, use_container_width=True)
            
            with col2:
                # Processing time by model
                fig_time = px.bar(
                    model_usage_df,
                    x="Model",
                    y="Avg Processing Time (ms)",
                    title="Average Processing Time by Model",
                    color="Avg Processing Time (ms)",
                    color_continuous_scale="plasma"
                )
                st.plotly_chart(fig_time, use_container_width=True)


def main():
    """Main dashboard application."""
    
    # Initialize API client
    api_client = DashboardAPI()
    
    # Sidebar navigation
    st.sidebar.title("🔬 MDT Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Overview", "Models", "Predictions", "Monitoring"],
        index=0
    )
    
    # Settings section
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    
    # API endpoint configuration
    api_url = st.sidebar.text_input(
        "API Endpoint:",
        value="http://localhost:8000",
        help="URL of the MDT API server"
    )
    api_client.api_base_url = api_url.rstrip("/")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    
    if auto_refresh:
        # Add refresh timer
        time.sleep(0.1)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Connection status
    st.sidebar.markdown("---")
    health = api_client.get_health()
    if health:
        st.sidebar.success("🟢 API Connected")
        st.sidebar.write(f"Version: {health.get('version', 'Unknown')}")
    else:
        st.sidebar.error("🔴 API Disconnected")
    
    # Render selected page
    if page == "Overview":
        render_overview_page(api_client)
    elif page == "Models":
        render_models_page(api_client)
    elif page == "Predictions":
        render_predictions_page(api_client)
    elif page == "Monitoring":
        render_monitoring_page(api_client)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**MDT Dashboard** - Model Drift Detection & Telemetry Platform | "
        f"Built with ❤️ using Streamlit | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()

"""
Advanced Streamlit Dashboard for MDT Platform.
Provides comprehensive visualization and management interface.

Refactored for improved code quality, type safety, and maintainability.
"""

import asyncio
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    STREAMLIT_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    
    def ConfigDict(**kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    def Field(default=None, **kwargs):
        return default
    
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class CardType(Enum):
    """Types of dashboard cards."""
    METRIC = "metric"
    ALERT = "alert"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class PageType(Enum):
    """Available dashboard pages."""
    OVERVIEW = "Overview"
    MODELS = "Models"
    PREDICTIONS = "Predictions"
    MONITORING = "Monitoring"
    DRIFT_ANALYSIS = "Drift Analysis"
    ALERTS = "Alerts"


class ChartType(Enum):
    """Available chart types."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"


# Pydantic models for data validation
if PYDANTIC_AVAILABLE:
    class DashboardConfig(BaseModel):
        """Dashboard configuration."""
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
        
        api_base_url: str = Field(default="http://localhost:8000", description="API base URL")
        auto_refresh: bool = Field(default=True, description="Enable auto-refresh")
        refresh_interval: int = Field(default=30, description="Refresh interval in seconds")
        default_time_window: int = Field(default=60, description="Default time window in minutes")
        max_models_display: int = Field(default=10, description="Maximum models to display")
        theme: str = Field(default="plotly", description="Chart theme")
        
        @field_validator('refresh_interval')
        @classmethod
        def validate_refresh_interval(cls, v):
            if v < 10 or v > 300:
                raise ValueError("Refresh interval must be between 10 and 300 seconds")
            return v
    
    class ModelInfo(BaseModel):
        """Model information."""
        model_config = ConfigDict(arbitrary_types_allowed=True)
        
        model_name: str
        algorithm: str
        performance: float
        size_mb: float
        features: List[str] = Field(default_factory=list)
        training_date: Optional[str] = None
        status: str = "active"
        
    class PredictionRequest(BaseModel):
        """Prediction request model."""
        model_config = ConfigDict(arbitrary_types_allowed=True)
        
        data: Dict[str, Any]
        model_name: str = "best_model"
        return_probabilities: bool = True
        return_feature_importance: bool = True
        
    class MetricsSummary(BaseModel):
        """Metrics summary model."""
        model_config = ConfigDict(arbitrary_types_allowed=True)
        
        total_requests: int = 0
        avg_processing_time_ms: float = 0.0
        error_rate: float = 0.0
        drift_detection_rate: float = 0.0
        avg_confidence: float = 0.0
        requests_per_minute: float = 0.0

else:
    # Fallback classes when Pydantic is not available
    @dataclass
    class DashboardConfig:
        api_base_url: str = "http://localhost:8000"
        auto_refresh: bool = True
        refresh_interval: int = 30
        default_time_window: int = 60
        max_models_display: int = 10
        theme: str = "plotly"
    
    @dataclass
    class ModelInfo:
        model_name: str
        algorithm: str
        performance: float
        size_mb: float
        features: List[str] = None
        training_date: Optional[str] = None
        status: str = "active"
        
        def __post_init__(self):
            if self.features is None:
                self.features = []
    
    @dataclass
    class PredictionRequest:
        data: Dict[str, Any]
        model_name: str = "best_model"
        return_probabilities: bool = True
        return_feature_importance: bool = True
    
    @dataclass
    class MetricsSummary:
        total_requests: int = 0
        avg_processing_time_ms: float = 0.0
        error_rate: float = 0.0
        drift_detection_rate: float = 0.0
        avg_confidence: float = 0.0
        requests_per_minute: float = 0.0


class DashboardAPIClient:
    """Enhanced API client for dashboard interactions."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.session = None
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize HTTP session."""
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, API client disabled")
            return
        
        self.session = requests.Session()
        self.session.timeout = 10
        
        # Add retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    async def get_health(self) -> Dict[str, Any]:
        """Get API health status."""
        if not self.session:
            return {}
        
        try:
            response = self.session.get(
                f"{self.config.api_base_url}/health",
                timeout=5
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {}
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        if not self.session:
            return []
        
        try:
            response = self.session.get(
                f"{self.config.api_base_url}/models",
                timeout=10
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information."""
        if not self.session:
            return {}
        
        try:
            response = self.session.get(
                f"{self.config.api_base_url}/models/{model_name}",
                timeout=10
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {}
    
    async def get_metrics(
        self, 
        time_window_minutes: int = 60, 
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.session:
            return {}
        
        try:
            params = {"time_window_minutes": time_window_minutes}
            if model_name:
                params["model_name"] = model_name
            
            response = self.session.get(
                f"{self.config.api_base_url}/metrics",
                params=params,
                timeout=10
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    async def predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """Make a prediction."""
        if not self.session:
            return {}
        
        try:
            payload = {
                "data": request.data,
                "model_name": request.model_name,
                "return_probabilities": request.return_probabilities,
                "return_feature_importance": request.return_feature_importance
            }
            
            response = self.session.post(
                f"{self.config.api_base_url}/predict",
                json=payload,
                timeout=30
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            return {}
    
    async def get_drift_reports(
        self, 
        model_name: Optional[str] = None,
        time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get drift detection reports."""
        if not self.session:
            return []
        
        try:
            params = {"time_window_hours": time_window_hours}
            if model_name:
                params["model_name"] = model_name
            
            response = self.session.get(
                f"{self.config.api_base_url}/drift/reports",
                params=params,
                timeout=10
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            logger.error(f"Failed to get drift reports: {e}")
            return []
    
    async def get_alerts(
        self, 
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get alerts."""
        if not self.session:
            return []
        
        try:
            params = {"limit": limit}
            if severity:
                params["severity"] = severity
            
            response = self.session.get(
                f"{self.config.api_base_url}/alerts",
                params=params,
                timeout=10
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []


class ChartRenderer:
    """Chart rendering utilities."""
    
    def __init__(self, theme: str = "plotly"):
        self.theme = theme
    
    def create_time_series_chart(
        self, 
        data: List[Dict[str, Any]], 
        title: str, 
        y_column: str,
        x_column: str = "timestamp"
    ) -> Optional[Any]:
        """Create time series chart."""
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE or not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            df[x_column] = pd.to_datetime(df[x_column])
            
            fig = px.line(
                df, 
                x=x_column, 
                y=y_column,
                title=title,
                labels={x_column: 'Time', y_column: title},
                template=self.theme
            )
            
            fig.update_layout(
                height=400,
                xaxis_title="Time",
                yaxis_title=title,
                showlegend=False
            )
            
            return fig
        except Exception as e:
            logger.error(f"Failed to create time series chart: {e}")
            return None
    
    def create_bar_chart(
        self, 
        data: List[Dict[str, Any]], 
        x_column: str,
        y_column: str,
        title: str,
        color_column: Optional[str] = None
    ) -> Optional[Any]:
        """Create bar chart."""
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE or not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            
            fig = px.bar(
                df,
                x=x_column,
                y=y_column,
                title=title,
                color=color_column or y_column,
                color_continuous_scale='viridis',
                template=self.theme
            )
            
            fig.update_layout(height=400)
            return fig
        except Exception as e:
            logger.error(f"Failed to create bar chart: {e}")
            return None
    
    def create_pie_chart(
        self, 
        data: List[Dict[str, Any]], 
        values_column: str,
        names_column: str,
        title: str,
        color_map: Optional[Dict[str, str]] = None
    ) -> Optional[Any]:
        """Create pie chart."""
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE or not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            
            fig = px.pie(
                df,
                values=values_column,
                names=names_column,
                title=title,
                color_discrete_map=color_map,
                template=self.theme
            )
            
            return fig
        except Exception as e:
            logger.error(f"Failed to create pie chart: {e}")
            return None
    
    def create_scatter_plot(
        self, 
        data: List[Dict[str, Any]], 
        x_column: str,
        y_column: str,
        title: str,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None
    ) -> Optional[Any]:
        """Create scatter plot."""
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE or not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                title=title,
                color=color_column,
                size=size_column,
                template=self.theme
            )
            
            fig.update_layout(height=400)
            return fig
        except Exception as e:
            logger.error(f"Failed to create scatter plot: {e}")
            return None


class DashboardComponents:
    """Reusable dashboard components."""
    
    @staticmethod
    def create_metric_card(
        title: str, 
        value: Any, 
        delta: Optional[str] = None, 
        card_type: CardType = CardType.METRIC
    ):
        """Create a styled metric card."""
        if not STREAMLIT_AVAILABLE:
            return
        
        card_class = f"{card_type.value}-card"
        
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
    
    @staticmethod
    def create_status_indicator(
        status: str, 
        label: str = "Status"
    ):
        """Create status indicator."""
        if not STREAMLIT_AVAILABLE:
            return
        
        status_colors = {
            "healthy": "🟢",
            "warning": "🟡", 
            "error": "🔴",
            "unknown": "⚪"
        }
        
        status_emoji = status_colors.get(status.lower(), "⚪")
        st.markdown(f"**{label}:** {status_emoji} {status.title()}")
    
    @staticmethod
    def create_data_table(
        data: List[Dict[str, Any]], 
        title: str,
        columns: Optional[List[str]] = None
    ):
        """Create data table."""
        if not STREAMLIT_AVAILABLE or not PANDAS_AVAILABLE or not data:
            return
        
        try:
            df = pd.DataFrame(data)
            
            if columns:
                available_columns = [col for col in columns if col in df.columns]
                if available_columns:
                    df = df[available_columns]
            
            st.subheader(title)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            logger.error(f"Failed to create data table: {e}")
            st.error(f"Failed to display table: {e}")
    
    @staticmethod
    def create_alert_banner(
        message: str, 
        alert_type: CardType = CardType.ALERT
    ):
        """Create alert banner."""
        if not STREAMLIT_AVAILABLE:
            return
        
        if alert_type == CardType.SUCCESS:
            st.success(message)
        elif alert_type == CardType.ERROR:
            st.error(message)
        elif alert_type == CardType.WARNING:
            st.warning(message)
        else:
            st.info(message)


class BasePage(ABC):
    """Base class for dashboard pages."""
    
    def __init__(self, api_client: DashboardAPIClient, chart_renderer: ChartRenderer):
        self.api_client = api_client
        self.chart_renderer = chart_renderer
        self.components = DashboardComponents()
    
    @abstractmethod
    async def render(self):
        """Render the page."""
        pass
    
    def safe_render(self):
        """Safely render the page with error handling."""
        try:
            # Run async render in sync context
            if hasattr(asyncio, 'run'):
                asyncio.run(self.render())
            else:
                # Fallback for older Python versions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.render())
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"Error rendering page {self.__class__.__name__}: {e}")
            if STREAMLIT_AVAILABLE:
                st.error(f"Error loading page: {e}")


class OverviewPage(BasePage):
    """Overview dashboard page."""
    
    async def render(self):
        """Render the overview page."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.markdown('<h1 class="main-header">🔬 MDT Dashboard - Overview</h1>', unsafe_allow_html=True)
        
        # Get health status
        health_data = await self.api_client.get_health()
        
        if not health_data:
            self.components.create_alert_banner(
                "❌ Cannot connect to API server. Please ensure the server is running.",
                CardType.ERROR
            )
            return
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = health_data.get("status", "unknown")
            status_emoji = "✅" if status == "healthy" else "❌"
            self.components.create_metric_card("API Status", f"{status_emoji} {status.title()}")
        
        with col2:
            uptime = health_data.get("uptime_seconds", 0)
            uptime_str = f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m"
            self.components.create_metric_card("Uptime", uptime_str)
        
        with col3:
            version = health_data.get("version", "Unknown")
            self.components.create_metric_card("Version", version)
        
        with col4:
            pred_service = health_data.get("prediction_service", {})
            models_count = pred_service.get("available_models", 0)
            self.components.create_metric_card("Available Models", models_count)
        
        st.markdown("---")
        
        # Get metrics
        metrics_data = await self.api_client.get_metrics(time_window_minutes=60)
        
        if metrics_data:
            await self._render_metrics_overview(metrics_data)
    
    async def _render_metrics_overview(self, metrics_data: Dict[str, Any]):
        """Render metrics overview section."""
        prediction_summary = metrics_data.get("prediction_summary", {})
        system_summary = metrics_data.get("system_summary", {})
        
        # Prediction metrics
        st.subheader("📊 Prediction Metrics (Last Hour)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_requests = prediction_summary.get("total_requests", 0)
            self.components.create_metric_card("Total Requests", total_requests)
        
        with col2:
            avg_time = prediction_summary.get("avg_processing_time_ms", 0)
            self.components.create_metric_card("Avg Response Time", f"{avg_time:.1f}ms")
        
        with col3:
            error_rate = prediction_summary.get("error_rate", 0) * 100
            card_type = CardType.ERROR if error_rate > 5 else CardType.SUCCESS if error_rate == 0 else CardType.ALERT
            self.components.create_metric_card("Error Rate", f"{error_rate:.1f}%", card_type=card_type)
        
        with col4:
            drift_rate = prediction_summary.get("drift_detection_rate", 0) * 100
            card_type = CardType.ERROR if drift_rate > 20 else CardType.ALERT if drift_rate > 10 else CardType.SUCCESS
            self.components.create_metric_card("Drift Rate", f"{drift_rate:.1f}%", card_type=card_type)
        
        # System metrics
        st.subheader("🖥️ System Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = system_summary.get("avg_cpu_usage", 0)
            card_type = CardType.ERROR if cpu_usage > 80 else CardType.ALERT if cpu_usage > 60 else CardType.SUCCESS
            self.components.create_metric_card("CPU Usage", f"{cpu_usage:.1f}%", card_type=card_type)
        
        with col2:
            memory_usage = system_summary.get("avg_memory_usage", 0)
            card_type = CardType.ERROR if memory_usage > 80 else CardType.ALERT if memory_usage > 60 else CardType.SUCCESS
            self.components.create_metric_card("Memory Usage", f"{memory_usage:.1f}%", card_type=card_type)
        
        with col3:
            network_sent = system_summary.get("total_network_sent_mb", 0)
            self.components.create_metric_card("Network Sent", f"{network_sent:.1f} MB")
        
        with col4:
            active_threads = system_summary.get("avg_active_threads", 0)
            self.components.create_metric_card("Active Threads", f"{active_threads:.0f}")


class ModelsPage(BasePage):
    """Models management page."""
    
    async def render(self):
        """Render the models page."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.markdown('<h1 class="main-header">🤖 Model Management</h1>', unsafe_allow_html=True)
        
        # Get models list
        models_data = await self.api_client.get_models()
        
        if not models_data:
            self.components.create_alert_banner(
                "No models available or unable to connect to API",
                CardType.WARNING
            )
            return
        
        # Models overview
        st.subheader("📋 Available Models")
        
        # Display models table
        self.components.create_data_table(
            models_data,
            "Models Overview",
            ['model_name', 'algorithm', 'performance', 'size_mb', 'registered_at']
        )
        
        # Model comparison chart
        if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
            fig = self.chart_renderer.create_bar_chart(
                models_data,
                'model_name',
                'performance',
                "Model Performance Comparison",
                'performance'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Model details section
        await self._render_model_details(models_data)
    
    async def _render_model_details(self, models_data: List[Dict[str, Any]]):
        """Render model details section."""
        st.subheader("🔍 Model Details")
        
        selected_model = st.selectbox(
            "Select a model for detailed information:",
            options=[model['model_name'] for model in models_data],
            index=0
        )
        
        if selected_model:
            model_info = await self.api_client.get_model_info(selected_model)
            
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


class PredictionsPage(BasePage):
    """Predictions interface page."""
    
    async def render(self):
        """Render the predictions page."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.markdown('<h1 class="main-header">🎯 Model Predictions</h1>', unsafe_allow_html=True)
        
        # Get available models
        models_data = await self.api_client.get_models()
        
        if not models_data:
            self.components.create_alert_banner("No models available", CardType.ERROR)
            return
        
        model_names = [model['model_name'] for model in models_data]
        
        # Prediction interface
        st.subheader("🔮 Make Predictions")
        
        selected_model = st.selectbox("Select Model:", model_names)
        
        # Get model info to understand features
        model_info = await self.api_client.get_model_info(selected_model)
        
        if model_info and 'features' in model_info:
            await self._render_feature_input_form(model_info, selected_model)
        else:
            await self._render_json_input_form(selected_model)
    
    async def _render_feature_input_form(self, model_info: Dict[str, Any], selected_model: str):
        """Render feature input form."""
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
            await self._make_prediction(feature_inputs, selected_model)
    
    async def _render_json_input_form(self, selected_model: str):
        """Render JSON input form."""
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
                await self._make_prediction(data, selected_model)
            except json.JSONDecodeError:
                self.components.create_alert_banner("❌ Invalid JSON format", CardType.ERROR)
    
    async def _make_prediction(self, data: Dict[str, Any], model_name: str):
        """Make prediction and display results."""
        request = PredictionRequest(data=data, model_name=model_name)
        result = await self.api_client.predict(request)
        
        if result:
            self.components.create_alert_banner("✅ Prediction completed!", CardType.SUCCESS)
            await self._render_prediction_results(result)
        else:
            self.components.create_alert_banner(
                "❌ Prediction failed. Please check your inputs.",
                CardType.ERROR
            )
    
    async def _render_prediction_results(self, result: Dict[str, Any]):
        """Render prediction results."""
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
            if feature_importance and PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
                importance_df = pd.DataFrame(
                    list(feature_importance.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = self.chart_renderer.create_bar_chart(
                    importance_df.head(10).to_dict('records'),
                    'Feature',
                    'Importance',
                    "Top 10 Feature Importance"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Drift detection
            drift_detected = result.get('drift_detected', False)
            if drift_detected:
                self.components.create_alert_banner("⚠️ Drift detected in input data!", CardType.WARNING)
            else:
                self.components.create_alert_banner("✅ No drift detected", CardType.SUCCESS)


class MonitoringPage(BasePage):
    """Monitoring and analytics page."""
    
    async def render(self):
        """Render the monitoring page."""
        if not STREAMLIT_AVAILABLE:
            return
        
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
        metrics_data = await self.api_client.get_metrics(time_window_minutes=time_window)
        
        if not metrics_data:
            self.components.create_alert_banner("No metrics data available", CardType.WARNING)
            return
        
        await self._render_kpi_overview(metrics_data)
        await self._render_system_health(metrics_data)
        await self._render_model_usage_stats(metrics_data)
    
    async def _render_kpi_overview(self, metrics_data: Dict[str, Any]):
        """Render KPI overview."""
        prediction_summary = metrics_data.get("prediction_summary", {})
        
        st.subheader("🎯 Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_requests = prediction_summary.get("total_requests", 0)
            requests_per_min = prediction_summary.get("requests_per_minute", 0)
            self.components.create_metric_card("Total Requests", total_requests, f"{requests_per_min:.1f}/min")
        
        with col2:
            avg_response = prediction_summary.get("avg_processing_time_ms", 0)
            self.components.create_metric_card("Avg Response", f"{avg_response:.1f}ms")
        
        with col3:
            error_rate = prediction_summary.get("error_rate", 0) * 100
            self.components.create_metric_card("Error Rate", f"{error_rate:.1f}%")
        
        with col4:
            avg_confidence = prediction_summary.get("avg_confidence", 0)
            self.components.create_metric_card("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col5:
            drift_rate = prediction_summary.get("drift_detection_rate", 0) * 100
            self.components.create_metric_card("Drift Rate", f"{drift_rate:.1f}%")
    
    async def _render_system_health(self, metrics_data: Dict[str, Any]):
        """Render system health metrics."""
        system_summary = metrics_data.get("system_summary", {})
        
        st.subheader("🖥️ System Health")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = system_summary.get("avg_cpu_usage", 0)
            max_cpu = system_summary.get("max_cpu_usage", 0)
            self.components.create_metric_card("CPU Usage", f"{cpu_usage:.1f}%", f"Max: {max_cpu:.1f}%")
        
        with col2:
            memory_usage = system_summary.get("avg_memory_usage", 0)
            max_memory = system_summary.get("max_memory_usage", 0)
            self.components.create_metric_card("Memory Usage", f"{memory_usage:.1f}%", f"Max: {max_memory:.1f}%")
        
        with col3:
            network_sent = system_summary.get("total_network_sent_mb", 0)
            network_recv = system_summary.get("total_network_received_mb", 0)
            self.components.create_metric_card("Network I/O", f"{network_sent:.1f}MB", f"↓{network_recv:.1f}MB")
        
        with col4:
            avg_threads = system_summary.get("avg_active_threads", 0)
            max_threads = system_summary.get("max_active_threads", 0)
            self.components.create_metric_card("Active Threads", f"{avg_threads:.0f}", f"Max: {max_threads:.0f}")
    
    async def _render_model_usage_stats(self, metrics_data: Dict[str, Any]):
        """Render model usage statistics."""
        model_stats = metrics_data.get("model_stats", {})
        
        if not model_stats:
            return
        
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
            self.components.create_data_table(model_usage_data, "Model Usage Statistics")
            
            # Model usage charts
            if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Requests by model
                    fig_requests = self.chart_renderer.create_bar_chart(
                        model_usage_data,
                        "Model",
                        "Total Requests",
                        "Requests by Model"
                    )
                    if fig_requests:
                        st.plotly_chart(fig_requests, use_container_width=True)
                
                with col2:
                    # Processing time by model
                    fig_time = self.chart_renderer.create_bar_chart(
                        model_usage_data,
                        "Model",
                        "Avg Processing Time (ms)",
                        "Average Processing Time by Model"
                    )
                    if fig_time:
                        st.plotly_chart(fig_time, use_container_width=True)


class DriftAnalysisPage(BasePage):
    """Drift analysis page."""
    
    async def render(self):
        """Render the drift analysis page."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.markdown('<h1 class="main-header">🌊 Drift Analysis</h1>', unsafe_allow_html=True)
        
        # Get drift reports
        drift_reports = await self.api_client.get_drift_reports(time_window_hours=24)
        
        if not drift_reports:
            self.components.create_alert_banner("No drift reports available", CardType.WARNING)
            return
        
        await self._render_drift_overview(drift_reports)
        await self._render_drift_details(drift_reports)
    
    async def _render_drift_overview(self, drift_reports: List[Dict[str, Any]]):
        """Render drift overview."""
        st.subheader("📊 Drift Detection Overview")
        
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(drift_reports)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_reports = len(drift_reports)
                self.components.create_metric_card("Total Reports", total_reports)
            
            with col2:
                drift_detected_count = df['drift_detected'].sum() if 'drift_detected' in df.columns else 0
                self.components.create_metric_card("Drift Detected", drift_detected_count)
            
            with col3:
                avg_drift_score = df['overall_drift_score'].mean() if 'overall_drift_score' in df.columns else 0
                self.components.create_metric_card("Avg Drift Score", f"{avg_drift_score:.3f}")
            
            with col4:
                models_affected = df['model_id'].nunique() if 'model_id' in df.columns else 0
                self.components.create_metric_card("Models Affected", models_affected)
        
        # Drift timeline chart
        if PLOTLY_AVAILABLE:
            fig = self.chart_renderer.create_time_series_chart(
                drift_reports,
                "Drift Score Over Time",
                "overall_drift_score",
                "start_time"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    async def _render_drift_details(self, drift_reports: List[Dict[str, Any]]):
        """Render drift details."""
        st.subheader("🔍 Drift Report Details")
        
        self.components.create_data_table(
            drift_reports,
            "Recent Drift Reports",
            ['model_id', 'drift_detected', 'overall_drift_score', 'p_value', 'start_time']
        )


class AlertsPage(BasePage):
    """Alerts management page."""
    
    async def render(self):
        """Render the alerts page."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.markdown('<h1 class="main-header">🚨 Alerts & Notifications</h1>', unsafe_allow_html=True)
        
        # Get alerts
        alerts = await self.api_client.get_alerts(limit=100)
        
        if not alerts:
            self.components.create_alert_banner("No alerts available", CardType.SUCCESS)
            return
        
        await self._render_alerts_overview(alerts)
        await self._render_alerts_details(alerts)
    
    async def _render_alerts_overview(self, alerts: List[Dict[str, Any]]):
        """Render alerts overview."""
        st.subheader("📊 Alerts Overview")
        
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(alerts)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_alerts = len(alerts)
                self.components.create_metric_card("Total Alerts", total_alerts)
            
            with col2:
                critical_alerts = len(df[df['severity'] == 'critical']) if 'severity' in df.columns else 0
                self.components.create_metric_card("Critical", critical_alerts, card_type=CardType.ERROR)
            
            with col3:
                warning_alerts = len(df[df['severity'] == 'warning']) if 'severity' in df.columns else 0
                self.components.create_metric_card("Warning", warning_alerts, card_type=CardType.WARNING)
            
            with col4:
                info_alerts = len(df[df['severity'] == 'info']) if 'severity' in df.columns else 0
                self.components.create_metric_card("Info", info_alerts)
        
        # Alerts distribution chart
        if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
            severity_counts = pd.DataFrame(alerts)['severity'].value_counts()
            severity_data = [{"severity": k, "count": v} for k, v in severity_counts.items()]
            
            fig = self.chart_renderer.create_pie_chart(
                severity_data,
                "count",
                "severity",
                "Alert Severity Distribution",
                color_map={
                    'info': '#17a2b8',
                    'warning': '#ffc107',
                    'error': '#dc3545',
                    'critical': '#6f42c1'
                }
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    async def _render_alerts_details(self, alerts: List[Dict[str, Any]]):
        """Render alerts details."""
        st.subheader("🔍 Alert Details")
        
        self.components.create_data_table(
            alerts,
            "Recent Alerts",
            ['alert_type', 'severity', 'message', 'model_id', 'created_at']
        )


class DashboardApp:
    """Main dashboard application."""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.api_client = DashboardAPIClient(self.config)
        self.chart_renderer = ChartRenderer(self.config.theme)
        self.components = DashboardComponents()
        
        # Initialize pages
        self.pages = {
            PageType.OVERVIEW: OverviewPage(self.api_client, self.chart_renderer),
            PageType.MODELS: ModelsPage(self.api_client, self.chart_renderer),
            PageType.PREDICTIONS: PredictionsPage(self.api_client, self.chart_renderer),
            PageType.MONITORING: MonitoringPage(self.api_client, self.chart_renderer),
            PageType.DRIFT_ANALYSIS: DriftAnalysisPage(self.api_client, self.chart_renderer),
            PageType.ALERTS: AlertsPage(self.api_client, self.chart_renderer)
        }
    
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        if not STREAMLIT_AVAILABLE:
            return
        
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
            .warning-card {
                background-color: #fff3cd;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #ffc107;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self) -> PageType:
        """Render sidebar and return selected page."""
        if not STREAMLIT_AVAILABLE:
            return PageType.OVERVIEW
        
        # Sidebar navigation
        st.sidebar.title("🔬 MDT Dashboard")
        st.sidebar.markdown("---")
        
        # Navigation
        page_name = st.sidebar.selectbox(
            "Navigate to:",
            [page.value for page in PageType],
            index=0
        )
        
        # Settings section
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Settings")
        
        # API endpoint configuration
        api_url = st.sidebar.text_input(
            "API Endpoint:",
            value=self.config.api_base_url,
            help="URL of the MDT API server"
        )
        self.config.api_base_url = api_url.rstrip("/")
        self.api_client.config.api_base_url = self.config.api_base_url
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=self.config.auto_refresh)
        self.config.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh interval (seconds):",
                min_value=10,
                max_value=300,
                value=self.config.refresh_interval,
                step=10
            )
            self.config.refresh_interval = refresh_interval
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            st.rerun()
        
        # Connection status
        st.sidebar.markdown("---")
        
        # Get health status asynchronously
        health = {}
        try:
            if hasattr(asyncio, 'run'):
                health = asyncio.run(self.api_client.get_health())
            else:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    health = loop.run_until_complete(self.api_client.get_health())
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
        
        if health:
            st.sidebar.success("🟢 API Connected")
            st.sidebar.write(f"Version: {health.get('version', 'Unknown')}")
        else:
            st.sidebar.error("🔴 API Disconnected")
        
        # Convert page name back to enum
        for page_type in PageType:
            if page_type.value == page_name:
                return page_type
        
        return PageType.OVERVIEW
    
    def render_footer(self):
        """Render footer."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.markdown("---")
        st.markdown(
            "**MDT Dashboard** - Model Drift Detection & Telemetry Platform | "
            f"Built with ❤️ using Streamlit | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    def run(self):
        """Run the dashboard application."""
        if not STREAMLIT_AVAILABLE:
            logger.error("Streamlit is not available. Cannot run dashboard.")
            return
        
        # Setup page configuration
        self.setup_page_config()
        
        # Render sidebar and get selected page
        selected_page = self.render_sidebar()
        
        # Render selected page
        if selected_page in self.pages:
            self.pages[selected_page].safe_render()
        else:
            st.error(f"Page {selected_page} not found")
        
        # Render footer
        self.render_footer()
        
        # Auto-refresh logic
        if self.config.auto_refresh:
            time.sleep(1)  # Small delay to prevent excessive refresh
            st.rerun()


def main():
    """Main entry point for the dashboard."""
    try:
        config = DashboardConfig()
        app = DashboardApp(config)
        app.run()
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}")
        if STREAMLIT_AVAILABLE:
            st.error(f"Dashboard startup failed: {e}")


if __name__ == "__main__":
    main()

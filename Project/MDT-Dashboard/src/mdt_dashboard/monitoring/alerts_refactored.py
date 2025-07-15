"""
Alert management and notification system for MDT Dashboard.
Handles email, Slack, webhook, and other notification channels.

Refactored for improved code quality, type safety, and maintainability.
"""

import asyncio
import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

try:
    from jinja2 import Template
except ImportError:
    Template = None

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    """Types of alerts."""
    
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_HEALTH = "system_health"
    DATA_QUALITY = "data_quality"
    MODEL_ERROR = "model_error"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    """Available notification channels."""
    
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class AlertData:
    """Alert data structure."""
    
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class NotificationResult:
    """Result of a notification attempt."""
    
    def __init__(self, channel: NotificationChannel, success: bool, error: Optional[str] = None):
        self.channel = channel
        self.success = success
        self.error = error
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel": self.channel.value,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""
    
    @abstractmethod
    async def send_notification(self, alert: AlertData) -> NotificationResult:
        """Send notification for the given alert."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider."""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
        default_recipients: Optional[List[str]] = None,
        enable_tls: bool = True
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.default_recipients = default_recipients or []
        self.enable_tls = enable_tls
        
        # Email templates
        self.templates = {
            AlertType.DRIFT_DETECTED: self._get_drift_template(),
            AlertType.PERFORMANCE_DEGRADATION: self._get_performance_template(),
            AlertType.SYSTEM_HEALTH: self._get_system_health_template(),
        }
    
    def is_configured(self) -> bool:
        """Check if email provider is configured."""
        return bool(
            self.smtp_server and
            self.smtp_port and
            self.default_recipients
        )
    
    def _get_drift_template(self) -> str:
        """Get drift detection email template."""
        return """
        <h2>🚨 Data Drift Detected</h2>
        <p><strong>Model:</strong> {{ model_name }}</p>
        <p><strong>Drift Score:</strong> {{ drift_score }}</p>
        <p><strong>Severity:</strong> {{ severity }}</p>
        <p><strong>Time:</strong> {{ timestamp }}</p>
        <p><strong>Message:</strong> {{ message }}</p>
        
        {% if affected_features %}
        <h3>Affected Features:</h3>
        <ul>
        {% for feature in affected_features %}
            <li>{{ feature }}</li>
        {% endfor %}
        </ul>
        {% endif %}
        
        <p>Please investigate the model performance and consider retraining if necessary.</p>
        """
    
    def _get_performance_template(self) -> str:
        """Get performance degradation email template."""
        return """
        <h2>📉 Model Performance Degradation</h2>
        <p><strong>Model:</strong> {{ model_name }}</p>
        <p><strong>Performance Score:</strong> {{ performance_score }}</p>
        <p><strong>Severity:</strong> {{ severity }}</p>
        <p><strong>Time:</strong> {{ timestamp }}</p>
        <p><strong>Message:</strong> {{ message }}</p>
        
        <p>The model performance has dropped below acceptable thresholds.</p>
        """
    
    def _get_system_health_template(self) -> str:
        """Get system health email template."""
        return """
        <h2>⚠️ System Health Alert</h2>
        <p><strong>Component:</strong> {{ component }}</p>
        <p><strong>Status:</strong> {{ status }}</p>
        <p><strong>Severity:</strong> {{ severity }}</p>
        <p><strong>Time:</strong> {{ timestamp }}</p>
        <p><strong>Message:</strong> {{ message }}</p>
        
        <p>Please check the system health dashboard for more details.</p>
        """
    
    async def send_notification(self, alert: AlertData) -> NotificationResult:
        """Send email notification."""
        if not self.is_configured():
            return NotificationResult(
                NotificationChannel.EMAIL,
                False,
                "Email provider not configured"
            )
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[MDT Alert] {alert.title}"
            msg['From'] = self.username or "noreply@mdt-dashboard.com"
            msg['To'] = ", ".join(self.default_recipients)
            
            # Get template
            template_str = self.templates.get(
                alert.alert_type,
                self.templates[AlertType.SYSTEM_HEALTH]
            )
            
            # Prepare template variables
            template_vars = {
                "model_name": alert.model_name or "Unknown",
                "severity": alert.severity.value,
                "timestamp": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "message": alert.message,
                **alert.metadata
            }
            
            # Render template
            if Template is not None:
                template = Template(template_str)
                html_content = template.render(**template_vars)
            else:
                # Fallback without jinja2
                html_content = f"""
                <h2>Alert: {alert.title}</h2>
                <p><strong>Severity:</strong> {alert.severity.value}</p>
                <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                """
            
            # Create HTML part
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._send_email_sync,
                msg
            )
            
            logger.info(f"Email alert sent successfully for alert {alert.alert_id}")
            return NotificationResult(NotificationChannel.EMAIL, True)
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return NotificationResult(NotificationChannel.EMAIL, False, str(e))
    
    def _send_email_sync(self, msg: MIMEMultipart) -> None:
        """Send email synchronously."""
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            if self.enable_tls:
                server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            server.send_message(msg)


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.color_map = {
            AlertSeverity.LOW: "#36a64f",      # Green
            AlertSeverity.MEDIUM: "#ff9900",   # Orange
            AlertSeverity.HIGH: "#ff0000",     # Red
            AlertSeverity.CRITICAL: "#8B0000"  # Dark Red
        }
    
    def is_configured(self) -> bool:
        """Check if Slack provider is configured."""
        return bool(self.webhook_url and requests is not None)
    
    async def send_notification(self, alert: AlertData) -> NotificationResult:
        """Send Slack notification."""
        if not self.is_configured():
            return NotificationResult(
                NotificationChannel.SLACK,
                False,
                "Slack provider not configured or requests library not available"
            )
        
        try:
            color = self.color_map.get(alert.severity, "#808080")
            
            # Create Slack message
            payload = {
                "username": "MDT Dashboard",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Model",
                                "value": alert.model_name or "N/A",
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            },
                            {
                                "title": "Alert Type",
                                "value": alert.alert_type.value.replace("_", " ").title(),
                                "short": True
                            }
                        ],
                        "footer": "MDT Dashboard",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            # Add metadata fields if available
            if alert.metadata:
                for key, value in alert.metadata.items():
                    if key not in ["drift_score", "affected_features"]:
                        payload["attachments"][0]["fields"].append({
                            "title": key.replace("_", " ").title(),
                            "value": str(value),
                            "short": True
                        })
            
            # Send webhook in executor
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._send_slack_sync,
                payload
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent successfully for alert {alert.alert_id}")
                return NotificationResult(NotificationChannel.SLACK, True)
            else:
                error_msg = f"Slack webhook failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return NotificationResult(NotificationChannel.SLACK, False, error_msg)
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return NotificationResult(NotificationChannel.SLACK, False, str(e))
    
    def _send_slack_sync(self, payload: Dict[str, Any]):
        """Send Slack webhook synchronously."""
        return requests.post(
            self.webhook_url,
            json=payload,
            timeout=10
        )


class WebhookNotificationProvider(NotificationProvider):
    """Generic webhook notification provider."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
    
    def is_configured(self) -> bool:
        """Check if webhook provider is configured."""
        return bool(self.webhook_url and requests is not None)
    
    async def send_notification(self, alert: AlertData) -> NotificationResult:
        """Send webhook notification."""
        if not self.is_configured():
            return NotificationResult(
                NotificationChannel.WEBHOOK,
                False,
                "Webhook provider not configured or requests library not available"
            )
        
        try:
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "model_id": alert.model_id,
                "model_name": alert.model_name,
                "created_at": alert.created_at.isoformat(),
                "metadata": alert.metadata,
                "source": "mdt_dashboard"
            }
            
            # Send webhook in executor
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._send_webhook_sync,
                payload
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Webhook alert sent successfully for alert {alert.alert_id}")
                return NotificationResult(NotificationChannel.WEBHOOK, True)
            else:
                error_msg = f"Webhook failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return NotificationResult(NotificationChannel.WEBHOOK, False, error_msg)
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return NotificationResult(NotificationChannel.WEBHOOK, False, str(e))
    
    def _send_webhook_sync(self, payload: Dict[str, Any]):
        """Send webhook synchronously."""
        return requests.post(
            self.webhook_url,
            json=payload,
            headers=self.headers,
            timeout=10
        )


class AlertManager:
    """Manages alert notifications across multiple channels."""
    
    def __init__(self):
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.enabled_channels: List[NotificationChannel] = []
        
    def add_provider(self, channel: NotificationChannel, provider: NotificationProvider) -> None:
        """Add a notification provider."""
        self.providers[channel] = provider
        if provider.is_configured() and channel not in self.enabled_channels:
            self.enabled_channels.append(channel)
            logger.info(f"Added {channel.value} notification provider")
    
    def remove_provider(self, channel: NotificationChannel) -> None:
        """Remove a notification provider."""
        if channel in self.providers:
            del self.providers[channel]
        if channel in self.enabled_channels:
            self.enabled_channels.remove(channel)
        logger.info(f"Removed {channel.value} notification provider")
    
    async def send_alert(
        self,
        alert: AlertData,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[NotificationChannel, NotificationResult]:
        """Send alert via specified channels or all enabled channels."""
        target_channels = channels or self.enabled_channels
        results = {}
        
        # Send notifications concurrently
        tasks = []
        for channel in target_channels:
            if channel in self.providers:
                provider = self.providers[channel]
                task = asyncio.create_task(provider.send_notification(alert))
                tasks.append((channel, task))
        
        # Wait for all tasks to complete
        for channel, task in tasks:
            try:
                result = await task
                results[channel] = result
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
                results[channel] = NotificationResult(channel, False, str(e))
        
        # Log overall result
        successful_channels = [
            channel.value for channel, result in results.items() if result.success
        ]
        if successful_channels:
            logger.info(f"Alert {alert.alert_id} sent via: {', '.join(successful_channels)}")
        else:
            logger.warning(f"Failed to send alert {alert.alert_id} via any channel")
        
        return results
    
    async def create_and_send_drift_alert(
        self,
        model_id: str,
        model_name: str,
        drift_score: float,
        affected_features: List[str],
        severity: AlertSeverity = AlertSeverity.MEDIUM
    ) -> Dict[NotificationChannel, NotificationResult]:
        """Create and send a drift detection alert."""
        alert = AlertData(
            alert_id=f"drift_{model_id}_{int(datetime.now().timestamp())}",
            alert_type=AlertType.DRIFT_DETECTED,
            severity=severity,
            title=f"Data Drift Detected: {model_name}",
            message=f"Data drift detected with score {drift_score:.4f}. "
                   f"Affected features: {', '.join(affected_features[:5])}",
            model_id=model_id,
            model_name=model_name,
            metadata={
                "drift_score": drift_score,
                "affected_features": affected_features
            }
        )
        
        return await self.send_alert(alert)
    
    async def create_and_send_performance_alert(
        self,
        model_id: str,
        model_name: str,
        performance_metric: str,
        current_value: float,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.MEDIUM
    ) -> Dict[NotificationChannel, NotificationResult]:
        """Create and send a performance degradation alert."""
        alert = AlertData(
            alert_id=f"perf_{model_id}_{int(datetime.now().timestamp())}",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=severity,
            title=f"Performance Degradation: {model_name}",
            message=f"Model {performance_metric} has degraded to {current_value:.4f}, "
                   f"below threshold {threshold:.4f}",
            model_id=model_id,
            model_name=model_name,
            metadata={
                "performance_metric": performance_metric,
                "current_value": current_value,
                "threshold": threshold,
                "degradation_percent": ((threshold - current_value) / threshold) * 100
            }
        )
        
        return await self.send_alert(alert)
    
    async def create_and_send_system_alert(
        self,
        component: str,
        status: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[NotificationChannel, NotificationResult]:
        """Create and send a system health alert."""
        alert = AlertData(
            alert_id=f"system_{component}_{int(datetime.now().timestamp())}",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=severity,
            title=f"System Alert: {component}",
            message=message,
            metadata={
                "component": component,
                "status": status,
                **(metadata or {})
            }
        )
        
        return await self.send_alert(alert)
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert manager status."""
        provider_status = {}
        for channel, provider in self.providers.items():
            provider_status[channel.value] = {
                "configured": provider.is_configured(),
                "enabled": channel in self.enabled_channels
            }
        
        return {
            "total_providers": len(self.providers),
            "enabled_channels": [c.value for c in self.enabled_channels],
            "provider_status": provider_status
        }


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def configure_alert_manager(settings) -> AlertManager:
    """Configure alert manager with settings."""
    manager = get_alert_manager()
    
    # Configure email provider
    if hasattr(settings, 'alerts') and settings.alerts.enable_email:
        email_provider = EmailNotificationProvider(
            smtp_server=settings.alerts.smtp_server,
            smtp_port=settings.alerts.smtp_port,
            username=settings.alerts.smtp_username,
            password=settings.alerts.smtp_password,
            default_recipients=settings.alerts.default_recipients
        )
        manager.add_provider(NotificationChannel.EMAIL, email_provider)
    
    # Configure Slack provider
    if hasattr(settings, 'alerts') and settings.alerts.enable_slack:
        slack_provider = SlackNotificationProvider(
            webhook_url=settings.alerts.slack_webhook_url
        )
        manager.add_provider(NotificationChannel.SLACK, slack_provider)
    
    # Configure webhook provider
    if hasattr(settings, 'alerts') and settings.alerts.enable_webhook:
        webhook_provider = WebhookNotificationProvider(
            webhook_url=settings.alerts.webhook_url
        )
        manager.add_provider(NotificationChannel.WEBHOOK, webhook_provider)
    
    return manager

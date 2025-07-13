"""
Alert management and notification system for MDT Dashboard.
Handles email, Slack, webhook, and other notification channels.
"""

import logging
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Any
from jinja2 import Template

from ..core.config import get_settings
from ..core.models import Alert, Model

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alert notifications across multiple channels."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Email templates
        self.email_templates = {
            "drift_detected": """
            <h2>🚨 Data Drift Detected</h2>
            <p><strong>Model:</strong> {{ model_name }}</p>
            <p><strong>Drift Score:</strong> {{ drift_score }}</p>
            <p><strong>Severity:</strong> {{ severity }}</p>
            <p><strong>Time:</strong> {{ timestamp }}</p>
            <p><strong>Message:</strong> {{ message }}</p>
            
            <h3>Details:</h3>
            <ul>
            {% for feature in affected_features %}
                <li>{{ feature }}</li>
            {% endfor %}
            </ul>
            
            <p>Please investigate the model performance and consider retraining if necessary.</p>
            """,
            
            "performance_degradation": """
            <h2>📉 Model Performance Degradation</h2>
            <p><strong>Model:</strong> {{ model_name }}</p>
            <p><strong>Performance Score:</strong> {{ performance_score }}</p>
            <p><strong>Severity:</strong> {{ severity }}</p>
            <p><strong>Time:</strong> {{ timestamp }}</p>
            <p><strong>Message:</strong> {{ message }}</p>
            
            <p>The model performance has dropped below acceptable thresholds.</p>
            """,
            
            "system_health": """
            <h2>⚠️ System Health Alert</h2>
            <p><strong>Component:</strong> {{ component }}</p>
            <p><strong>Status:</strong> {{ status }}</p>
            <p><strong>Severity:</strong> {{ severity }}</p>
            <p><strong>Time:</strong> {{ timestamp }}</p>
            <p><strong>Message:</strong> {{ message }}</p>
            
            <p>Please check the system health dashboard for more details.</p>
            """
        }
    
    def send_email_alert(self, alert: Alert) -> bool:
        """
        Send email alert.
        
        Args:
            alert: Alert object to send
            
        Returns:
            Success status
        """
        if not self.settings.alerts.enable_email:
            logger.info("Email alerts disabled")
            return False
            
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[MDT Alert] {alert.title}"
            msg['From'] = self.settings.alerts.smtp_username
            msg['To'] = ", ".join(self.settings.alerts.default_recipients)
            
            # Get template
            template_name = alert.alert_type
            if template_name not in self.email_templates:
                template_name = "system_health"
            
            template = Template(self.email_templates[template_name])
            
            # Prepare template variables
            template_vars = {
                "model_name": alert.model.name if alert.model else "Unknown",
                "severity": alert.severity,
                "timestamp": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "message": alert.message,
            }
            
            # Add metadata if available
            if alert.metadata:
                template_vars.update(alert.metadata)
            
            # Render template
            html_content = template.render(**template_vars)
            
            # Create HTML part
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.settings.alerts.smtp_server, self.settings.alerts.smtp_port) as server:
                server.starttls()
                if self.settings.alerts.smtp_username and self.settings.alerts.smtp_password:
                    server.login(self.settings.alerts.smtp_username, self.settings.alerts.smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_slack_alert(self, alert: Alert) -> bool:
        """
        Send Slack alert via webhook.
        
        Args:
            alert: Alert object to send
            
        Returns:
            Success status
        """
        if not self.settings.alerts.enable_slack or not self.settings.alerts.slack_webhook_url:
            logger.info("Slack alerts disabled or webhook URL not configured")
            return False
            
        try:
            # Determine color based on severity
            color_map = {
                "low": "#36a64f",     # Green
                "medium": "#ff9900",  # Orange
                "high": "#ff0000",    # Red
                "critical": "#8B0000" # Dark Red
            }
            
            color = color_map.get(alert.severity, "#808080")
            
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
                                "value": alert.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Model",
                                "value": alert.model.name if alert.model else "N/A",
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            },
                            {
                                "title": "Alert Type",
                                "value": alert.alert_type.replace("_", " ").title(),
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
            
            # Send webhook
            response = requests.post(
                self.settings.alerts.slack_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent successfully for alert {alert.id}")
                return True
            else:
                logger.error(f"Slack webhook failed with status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def send_webhook_alert(self, alert: Alert) -> bool:
        """
        Send generic webhook alert.
        
        Args:
            alert: Alert object to send
            
        Returns:
            Success status
        """
        if not self.settings.alerts.enable_webhook or not self.settings.alerts.webhook_url:
            logger.info("Webhook alerts disabled or URL not configured")
            return False
            
        try:
            # Create webhook payload
            payload = {
                "alert_id": str(alert.id),
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "model_id": str(alert.model_id) if alert.model_id else None,
                "model_name": alert.model.name if alert.model else None,
                "created_at": alert.created_at.isoformat(),
                "metadata": alert.metadata,
                "source": "mdt_dashboard"
            }
            
            # Send webhook
            response = requests.post(
                self.settings.alerts.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Webhook alert sent successfully for alert {alert.id}")
                return True
            else:
                logger.error(f"Webhook failed with status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """
        Send alert via all enabled channels.
        
        Args:
            alert: Alert object to send
            
        Returns:
            Dictionary of channel success statuses
        """
        results = {}
        
        if self.settings.alerts.enable_email:
            results["email"] = self.send_email_alert(alert)
        
        if self.settings.alerts.enable_slack:
            results["slack"] = self.send_slack_alert(alert)
        
        if self.settings.alerts.enable_webhook:
            results["webhook"] = self.send_webhook_alert(alert)
        
        # Log overall result
        successful_channels = [channel for channel, success in results.items() if success]
        if successful_channels:
            logger.info(f"Alert {alert.id} sent via: {', '.join(successful_channels)}")
        else:
            logger.warning(f"Failed to send alert {alert.id} via any channel")
        
        return results
    
    def create_and_send_drift_alert(
        self,
        model: Model,
        drift_score: float,
        affected_features: List[str],
        severity: str = "medium"
    ) -> Alert:
        """
        Create and send a drift detection alert.
        
        Args:
            model: Model object
            drift_score: Detected drift score
            affected_features: List of affected features
            severity: Alert severity
            
        Returns:
            Created alert object
        """
        from ..core.database import db_manager
        
        # Create alert
        alert = Alert(
            model_id=model.id,
            alert_type="drift_detected",
            severity=severity,
            title=f"Data Drift Detected: {model.name}",
            message=f"Data drift detected with score {drift_score:.4f}. "
                   f"Affected features: {', '.join(affected_features[:5])}",
            metadata={
                "drift_score": drift_score,
                "affected_features": affected_features,
                "threshold": self.settings.drift_detection.ks_test_threshold
            }
        )
        
        # Save to database
        with db_manager.session_scope() as session:
            session.add(alert)
            session.commit()
            session.refresh(alert)
        
        # Send notifications
        self.send_alert(alert)
        
        return alert
    
    def create_and_send_performance_alert(
        self,
        model: Model,
        performance_metric: str,
        current_value: float,
        threshold: float,
        severity: str = "medium"
    ) -> Alert:
        """
        Create and send a performance degradation alert.
        
        Args:
            model: Model object
            performance_metric: Name of the degraded metric
            current_value: Current metric value
            threshold: Alert threshold
            severity: Alert severity
            
        Returns:
            Created alert object
        """
        from ..core.database import db_manager
        
        # Create alert
        alert = Alert(
            model_id=model.id,
            alert_type="performance_degradation",
            severity=severity,
            title=f"Performance Degradation: {model.name}",
            message=f"Model {performance_metric} has degraded to {current_value:.4f}, "
                   f"below threshold {threshold:.4f}",
            metadata={
                "performance_metric": performance_metric,
                "current_value": current_value,
                "threshold": threshold,
                "degradation_percent": ((threshold - current_value) / threshold) * 100
            }
        )
        
        # Save to database
        with db_manager.session_scope() as session:
            session.add(alert)
            session.commit()
            session.refresh(alert)
        
        # Send notifications
        self.send_alert(alert)
        
        return alert
    
    def create_and_send_system_alert(
        self,
        component: str,
        status: str,
        message: str,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Create and send a system health alert.
        
        Args:
            component: System component name
            status: Component status
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
            
        Returns:
            Created alert object
        """
        from ..core.database import db_manager
        
        # Create alert
        alert = Alert(
            alert_type="system_health",
            severity=severity,
            title=f"System Alert: {component}",
            message=message,
            metadata={
                "component": component,
                "status": status,
                **(metadata or {})
            }
        )
        
        # Save to database
        with db_manager.session_scope() as session:
            session.add(alert)
            session.commit()
            session.refresh(alert)
        
        # Send notifications
        self.send_alert(alert)
        
        return alert

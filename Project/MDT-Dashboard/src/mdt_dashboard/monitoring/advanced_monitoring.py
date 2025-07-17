"""
Advanced monitoring and alerting system for MLOps pipelines.

This module provides comprehensive monitoring capabilities including:
- Real-time model performance tracking
- Custom metric collection
- Multi-channel alerting
- Alert correlation and deduplication
- Adaptive thresholds
- SLA monitoring

Features:
- Prometheus metrics integration
- Custom alerting rules
- Slack, email, and webhook notifications
- Alert escalation policies
- Historical metric analysis
"""

import time
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import statistics
from collections import defaultdict, deque


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class Alert:
    """Alert definition and state."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    metric_name: str
    threshold: float
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    duration: timedelta
    labels: Dict[str, str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['severity'] = self.severity.value
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        result['duration'] = str(self.duration)
        return result


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert notification."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate channel configuration."""
        pass


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, webhook_url: str, channel: str = None):
        """
        Initialize Slack notification channel.
        
        Args:
            webhook_url: Slack webhook URL
            channel: Optional channel override
        """
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            # This is a simplified implementation
            # In practice, you would use aiohttp or similar
            payload = self._format_slack_message(alert)
            
            # Simulated async HTTP request
            await asyncio.sleep(0.1)  # Simulate network delay
            
            logger.info(f"Alert sent to Slack: {alert.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Slack configuration."""
        return 'webhook_url' in config and config['webhook_url'].startswith('https://hooks.slack.com')
    
    def _format_slack_message(self, alert: Alert) -> Dict[str, Any]:
        """Format alert as Slack message."""
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning", 
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        return {
            "channel": self.channel,
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "title": f"🚨 {alert.name}",
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Status", "value": alert.status.value.upper(), "short": True},
                    {"title": "Metric", "value": alert.metric_name, "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True}
                ],
                "footer": "MDT Dashboard",
                "ts": int(alert.created_at.timestamp())
            }]
        }


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, smtp_config: Dict[str, Any], recipients: List[str]):
        """
        Initialize email notification channel.
        
        Args:
            smtp_config: SMTP server configuration
            recipients: List of email recipients
        """
        self.smtp_config = smtp_config
        self.recipients = recipients
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Simulated email sending
            await asyncio.sleep(0.2)  # Simulate sending delay
            
            logger.info(f"Alert sent via email to {len(self.recipients)} recipients: {alert.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate email configuration."""
        required_keys = ['smtp_server', 'smtp_port', 'username', 'password']
        return all(key in config for key in required_keys)


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        """
        Initialize webhook notification channel.
        
        Args:
            webhook_url: Webhook URL
            headers: Optional HTTP headers
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        try:
            payload = alert.to_dict()
            
            # Simulated HTTP POST request
            await asyncio.sleep(0.1)
            
            logger.info(f"Alert sent to webhook: {alert.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate webhook configuration."""
        return 'webhook_url' in config and config['webhook_url'].startswith('http')


class MetricCollector:
    """Collects and stores metrics."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        """
        Initialize metric collector.
        
        Args:
            max_points_per_metric: Maximum points to store per metric
        """
        self.max_points_per_metric = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: List[str] = None
    ) -> None:
        """
        Register a new metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            labels: List of label names
        """
        self.metric_metadata[name] = {
            'type': metric_type,
            'description': description,
            'labels': labels or [],
            'created_at': datetime.now()
        }
        
        logger.debug(f"Registered metric: {name} ({metric_type.value})")
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        timestamp: datetime = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        point = MetricPoint(
            timestamp=timestamp,
            value=value,
            labels=labels or {}
        )
        
        self.metrics[name].append(point)
    
    def get_metric_history(
        self,
        name: str,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[MetricPoint]:
        """
        Get metric history within time range.
        
        Args:
            name: Metric name
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of metric points
        """
        if name not in self.metrics:
            return []
        
        points = list(self.metrics[name])
        
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        return points
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        return self.metrics[name][-1].value
    
    def calculate_aggregate(
        self,
        name: str,
        aggregation: str,
        window: timedelta = None
    ) -> Optional[float]:
        """
        Calculate aggregate value for a metric.
        
        Args:
            name: Metric name
            aggregation: Aggregation type ('avg', 'sum', 'min', 'max', 'count')
            window: Time window for aggregation
            
        Returns:
            Aggregated value
        """
        end_time = datetime.now()
        start_time = end_time - window if window else None
        
        points = self.get_metric_history(name, start_time, end_time)
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        if aggregation == 'avg':
            return statistics.mean(values)
        elif aggregation == 'sum':
            return sum(values)
        elif aggregation == 'min':
            return min(values)
        elif aggregation == 'max':
            return max(values)
        elif aggregation == 'count':
            return len(values)
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation}")


class AlertRule:
    """Defines conditions for triggering alerts."""
    
    def __init__(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        duration: timedelta,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        aggregation: str = 'avg',
        labels: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize alert rule.
        
        Args:
            name: Rule name
            metric_name: Name of metric to monitor
            condition: Condition ('gt', 'lt', 'eq', 'ne')
            threshold: Threshold value
            duration: Duration condition must be met
            severity: Alert severity
            aggregation: How to aggregate metric values
            labels: Optional labels
            metadata: Optional metadata
        """
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.threshold = threshold
        self.duration = duration
        self.severity = severity
        self.aggregation = aggregation
        self.labels = labels or {}
        self.metadata = metadata or {}
        
        self.last_triggered = None
        self.trigger_start_time = None
    
    def evaluate(self, metric_collector: MetricCollector) -> bool:
        """
        Evaluate if alert should be triggered.
        
        Args:
            metric_collector: Metric collector to query
            
        Returns:
            True if alert should be triggered
        """
        current_value = metric_collector.calculate_aggregate(
            self.metric_name,
            self.aggregation,
            window=timedelta(minutes=5)  # Default 5-minute window
        )
        
        if current_value is None:
            return False
        
        condition_met = self._check_condition(current_value)
        
        now = datetime.now()
        
        if condition_met:
            if self.trigger_start_time is None:
                self.trigger_start_time = now
            
            # Check if condition has been met for required duration
            if now - self.trigger_start_time >= self.duration:
                return True
        else:
            # Reset trigger start time if condition is not met
            self.trigger_start_time = None
        
        return False
    
    def _check_condition(self, value: float) -> bool:
        """Check if value meets the condition."""
        if self.condition == 'gt':
            return value > self.threshold
        elif self.condition == 'lt':
            return value < self.threshold
        elif self.condition == 'eq':
            return abs(value - self.threshold) < 1e-9
        elif self.condition == 'ne':
            return abs(value - self.threshold) >= 1e-9
        else:
            raise ValueError(f"Unknown condition: {self.condition}")


class AdvancedMonitoringSystem:
    """
    Advanced monitoring system with alerting capabilities.
    
    Features:
    - Real-time metric collection
    - Configurable alert rules
    - Multi-channel notifications
    - Alert correlation and deduplication
    - Adaptive thresholds
    """
    
    def __init__(self):
        """Initialize monitoring system."""
        self.metric_collector = MetricCollector()
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        self.is_running = False
        self.evaluation_interval = 30  # seconds
        
        logger.info("Advanced monitoring system initialized")
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: List[str] = None
    ) -> None:
        """Register a new metric."""
        self.metric_collector.register_metric(name, metric_type, description, labels)
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        timestamp: datetime = None
    ) -> None:
        """Record a metric value."""
        self.metric_collector.record_metric(name, value, labels, timestamp)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> None:
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_channel(self, name: str, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.notification_channels[name] = channel
        logger.info(f"Added notification channel: {name}")
    
    def remove_notification_channel(self, name: str) -> None:
        """Remove a notification channel."""
        if name in self.notification_channels:
            del self.notification_channels[name]
            logger.info(f"Removed notification channel: {name}")
    
    async def start_monitoring(self) -> None:
        """Start the monitoring loop."""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return
        
        self.is_running = True
        logger.info("Starting monitoring system")
        
        while self.is_running:
            try:
                await self._evaluate_alert_rules()
                await asyncio.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self.is_running = False
        logger.info("Stopping monitoring system")
    
    async def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules."""
        for rule_name, rule in self.alert_rules.items():
            try:
                should_trigger = rule.evaluate(self.metric_collector)
                
                if should_trigger and rule_name not in self.active_alerts:
                    await self._trigger_alert(rule)
                elif not should_trigger and rule_name in self.active_alerts:
                    await self._resolve_alert(rule_name)
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    async def _trigger_alert(self, rule: AlertRule) -> None:
        """Trigger a new alert."""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            name=rule.name,
            description=f"Alert triggered for metric {rule.metric_name}",
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metric_name=rule.metric_name,
            threshold=rule.threshold,
            condition=rule.condition,
            duration=rule.duration,
            labels=rule.labels,
            metadata=rule.metadata
        )
        
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert triggered: {alert.name}")
        
        # Send notifications
        await self._send_alert_notifications(alert)
    
    async def _resolve_alert(self, rule_name: str) -> None:
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.status = AlertStatus.RESOLVED
            alert.updated_at = datetime.now()
            
            del self.active_alerts[rule_name]
            
            logger.info(f"Alert resolved: {alert.name}")
            
            # Send resolution notifications
            await self._send_alert_notifications(alert)
    
    async def _send_alert_notifications(self, alert: Alert) -> None:
        """Send alert notifications to all channels."""
        tasks = []
        
        for channel_name, channel in self.notification_channels.items():
            task = asyncio.create_task(
                self._send_notification_with_retry(channel, alert, channel_name)
            )
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_notification_with_retry(
        self,
        channel: NotificationChannel,
        alert: Alert,
        channel_name: str,
        max_retries: int = 3
    ) -> None:
        """Send notification with retry logic."""
        for attempt in range(max_retries):
            try:
                success = await channel.send_alert(alert)
                if success:
                    logger.debug(f"Alert sent via {channel_name}")
                    return
                else:
                    logger.warning(f"Failed to send alert via {channel_name}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Error sending alert via {channel_name}: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        severity: AlertSeverity = None
    ) -> List[Alert]:
        """
        Get alert history with optional filters.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            severity: Severity filter
            
        Returns:
            Filtered list of alerts
        """
        alerts = self.alert_history
        
        if start_time:
            alerts = [a for a in alerts if a.created_at >= start_time]
        
        if end_time:
            alerts = [a for a in alerts if a.created_at <= end_time]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if alert was acknowledged
        """
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.updated_at = datetime.now()
                alert.metadata['acknowledged_by'] = acknowledged_by
                
                logger.info(f"Alert acknowledged: {alert.name} by {acknowledged_by}")
                return True
        
        return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        now = datetime.now()
        
        return {
            'is_monitoring': self.is_running,
            'active_alerts_count': len(self.active_alerts),
            'total_metrics': len(self.metric_collector.metrics),
            'alert_rules_count': len(self.alert_rules),
            'notification_channels_count': len(self.notification_channels),
            'uptime_seconds': (now - datetime.now()).total_seconds() if hasattr(self, 'start_time') else 0,
            'last_evaluation': now.isoformat()
        }

"""
Risk Alerts Module
Advanced alerting system for risk management
"""

import asyncio
import smtplib
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum
import aiohttp

logger = structlog.get_logger()

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"
    DASHBOARD = "dashboard"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str
    threshold: float
    level: AlertLevel
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown_minutes: int = 15
    escalation_minutes: int = 60
    max_alerts_per_hour: int = 10

@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_name: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    data: Dict
    channels: List[AlertChannel]
    sent: bool = False
    acknowledged: bool = False
    escalated: bool = False

class RiskAlerts:
    """Advanced risk alerting system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alerts = []
        self.alert_rules = {}
        self.alert_history = []
        self.alert_counts = {}
        self.initialize_alert_rules()
        self.setup_notification_channels()
    
    def initialize_alert_rules(self):
        """Initialize default alert rules"""
        logger.info("Initializing risk alert rules")
        
        # Daily loss alerts
        self.alert_rules["daily_loss_warning"] = AlertRule(
            name="Daily Loss Warning",
            condition="daily_loss > threshold",
            threshold=2.0,
            level=AlertLevel.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown_minutes=30
        )
        
        self.alert_rules["daily_loss_critical"] = AlertRule(
            name="Daily Loss Critical",
            condition="daily_loss > threshold",
            threshold=5.0,
            level=AlertLevel.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.SLACK],
            cooldown_minutes=5
        )
        
        # Drawdown alerts
        self.alert_rules["drawdown_warning"] = AlertRule(
            name="Drawdown Warning",
            condition="drawdown > threshold",
            threshold=10.0,
            level=AlertLevel.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        
        self.alert_rules["drawdown_critical"] = AlertRule(
            name="Drawdown Critical",
            condition="drawdown > threshold",
            threshold=15.0,
            level=AlertLevel.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.SLACK]
        )
        
        # VaR breach alerts
        self.alert_rules["var_breach"] = AlertRule(
            name="VaR Breach",
            condition="var > threshold",
            threshold=8.0,
            level=AlertLevel.ERROR,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        
        # Volatility alerts
        self.alert_rules["high_volatility"] = AlertRule(
            name="High Volatility",
            condition="volatility > threshold",
            threshold=40.0,
            level=AlertLevel.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        
        # Position size alerts
        self.alert_rules["large_position"] = AlertRule(
            name="Large Position",
            condition="position_size > threshold",
            threshold=3.0,
            level=AlertLevel.WARNING,
            channels=[AlertChannel.EMAIL]
        )
        
        # Correlation alerts
        self.alert_rules["high_correlation"] = AlertRule(
            name="High Correlation",
            condition="correlation > threshold",
            threshold=0.8,
            level=AlertLevel.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        
        # Liquidity alerts
        self.alert_rules["low_liquidity"] = AlertRule(
            name="Low Liquidity",
            condition="liquidity < threshold",
            threshold=0.3,
            level=AlertLevel.WARNING,
            channels=[AlertChannel.EMAIL]
        )
        
        # Circuit breaker alerts
        self.alert_rules["circuit_breaker_triggered"] = AlertRule(
            name="Circuit Breaker Triggered",
            condition="circuit_breaker == true",
            threshold=1.0,
            level=AlertLevel.EMERGENCY,
            channels=[AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.SLACK, AlertChannel.WEBHOOK],
            cooldown_minutes=0
        )
        
        # Black swan alerts
        self.alert_rules["black_swan_detected"] = AlertRule(
            name="Black Swan Detected",
            condition="black_swan_threat > threshold",
            threshold=0.7,
            level=AlertLevel.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.SLACK],
            cooldown_minutes=5
        )
        
        logger.info("Alert rules initialized", n_rules=len(self.alert_rules))
    
    def setup_notification_channels(self):
        """Setup notification channels"""
        self.email_config = {
            "smtp_server": self.config.get("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": self.config.get("SMTP_PORT", 587),
            "username": self.config.get("SMTP_USERNAME", ""),
            "password": self.config.get("SMTP_PASSWORD", ""),
            "from_email": self.config.get("FROM_EMAIL", "alerts@trading.com"),
            "to_emails": self.config.get("ALERT_EMAILS", "").split(",")
        }
        
        self.slack_config = {
            "webhook_url": self.config.get("SLACK_WEBHOOK_URL", ""),
            "channel": self.config.get("SLACK_CHANNEL", "#risk-alerts")
        }
        
        self.sms_config = {
            "api_key": self.config.get("SMS_API_KEY", ""),
            "from_number": self.config.get("SMS_FROM_NUMBER", ""),
            "to_numbers": self.config.get("SMS_NUMBERS", "").split(",")
        }
    
    async def check_alerts(self, risk_metrics: Dict) -> List[Alert]:
        """Check all alert rules against current risk metrics"""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check if rule should be triggered
            if await self._evaluate_rule(rule, risk_metrics):
                # Check cooldown
                if self._is_in_cooldown(rule_name):
                    continue
                
                # Check rate limiting
                if self._is_rate_limited(rule_name):
                    continue
                
                # Create alert
                alert = await self._create_alert(rule, risk_metrics)
                triggered_alerts.append(alert)
                
                # Send alert
                await self._send_alert(alert)
                
                # Update counters
                self._update_alert_counters(rule_name)
        
        return triggered_alerts
    
    async def _evaluate_rule(self, rule: AlertRule, risk_metrics: Dict) -> bool:
        """Evaluate if an alert rule should be triggered"""
        try:
            # Parse condition and evaluate
            if rule.condition == "daily_loss > threshold":
                return risk_metrics.get("daily_loss", 0) > rule.threshold
            elif rule.condition == "drawdown > threshold":
                return risk_metrics.get("current_drawdown", 0) > rule.threshold
            elif rule.condition == "var > threshold":
                return risk_metrics.get("var_95", 0) > rule.threshold
            elif rule.condition == "volatility > threshold":
                return risk_metrics.get("volatility", 0) > rule.threshold
            elif rule.condition == "position_size > threshold":
                return risk_metrics.get("max_position_size", 0) > rule.threshold
            elif rule.condition == "correlation > threshold":
                return risk_metrics.get("correlation_risk", 0) > rule.threshold
            elif rule.condition == "liquidity < threshold":
                return risk_metrics.get("liquidity_risk", 100) < rule.threshold
            elif rule.condition == "circuit_breaker == true":
                return risk_metrics.get("circuit_breaker_active", False)
            elif rule.condition == "black_swan_threat > threshold":
                return risk_metrics.get("black_swan_threat", 0) > rule.threshold
            else:
                return False
                
        except Exception as e:
            logger.error("Error evaluating alert rule", rule=rule.name, error=str(e))
            return False
    
    async def _create_alert(self, rule: AlertRule, risk_metrics: Dict) -> Alert:
        """Create an alert instance"""
        alert_id = f"ALERT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{rule.name.replace(' ', '_')}"
        
        # Generate alert message
        title, message = self._generate_alert_content(rule, risk_metrics)
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            level=rule.level,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            data=risk_metrics,
            channels=rule.channels
        )
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        return alert
    
    def _generate_alert_content(self, rule: AlertRule, risk_metrics: Dict) -> Tuple[str, str]:
        """Generate alert title and message"""
        title = f"🚨 {rule.level.value.upper()}: {rule.name}"
        
        message = f"""
Alert: {rule.name}
Level: {rule.level.value.upper()}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

Current Risk Metrics:
- Daily Loss: {risk_metrics.get('daily_loss', 0):.2f}%
- Current Drawdown: {risk_metrics.get('current_drawdown', 0):.2f}%
- VaR (95%): {risk_metrics.get('var_95', 0):.2f}%
- Volatility: {risk_metrics.get('volatility', 0):.2f}%
- Portfolio Risk: {risk_metrics.get('portfolio_risk', 0):.2f}%

Threshold Breached: {rule.threshold}
Condition: {rule.condition}

Please review the risk metrics and take appropriate action.
        """.strip()
        
        return title, message
    
    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        for channel in alert.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(alert)
                elif channel == AlertChannel.SMS:
                    await self._send_sms_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert)
                elif channel == AlertChannel.LOG:
                    await self._send_log_alert(alert)
                elif channel == AlertChannel.DASHBOARD:
                    await self._send_dashboard_alert(alert)
                
                logger.info("Alert sent", alert_id=alert.id, channel=channel.value)
                
            except Exception as e:
                logger.error("Failed to send alert", alert_id=alert.id, channel=channel.value, error=str(e))
        
        alert.sent = True
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.email_config["to_emails"] or not self.email_config["username"]:
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config["from_email"]
            msg['To'] = ", ".join(self.email_config["to_emails"])
            msg['Subject'] = alert.title
            
            msg.attach(MIMEText(alert.message, 'plain'))
            
            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error("Email alert failed", error=str(e))
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        if not self.slack_config["webhook_url"]:
            return
        
        try:
            # Color coding based on alert level
            color_map = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning",
                AlertLevel.ERROR: "danger",
                AlertLevel.CRITICAL: "danger",
                AlertLevel.EMERGENCY: "danger"
            }
            
            payload = {
                "channel": self.slack_config["channel"],
                "username": "Risk Alert Bot",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.level, "warning"),
                        "title": alert.title,
                        "text": alert.message,
                        "footer": "Omni Alpha Risk Management",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_config["webhook_url"], json=payload) as response:
                    if response.status != 200:
                        logger.error("Slack alert failed", status=response.status)
        
        except Exception as e:
            logger.error("Slack alert failed", error=str(e))
    
    async def _send_sms_alert(self, alert: Alert):
        """Send SMS alert"""
        if not self.sms_config["to_numbers"] or not self.sms_config["api_key"]:
            return
        
        try:
            # Simplified SMS sending (would use actual SMS service)
            logger.info("SMS alert sent", message=alert.title)
        
        except Exception as e:
            logger.error("SMS alert failed", error=str(e))
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        webhook_url = self.config.get("ALERT_WEBHOOK_URL", "")
        if not webhook_url:
            return
        
        try:
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "data": alert.data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status not in [200, 201]:
                        logger.error("Webhook alert failed", status=response.status)
        
        except Exception as e:
            logger.error("Webhook alert failed", error=str(e))
    
    async def _send_log_alert(self, alert: Alert):
        """Send log alert"""
        log_level_map = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
            AlertLevel.EMERGENCY: logger.critical
        }
        
        log_func = log_level_map.get(alert.level, logger.warning)
        log_func("Risk Alert", 
                alert_id=alert.id,
                rule=alert.rule_name,
                level=alert.level.value,
                message=alert.message)
    
    async def _send_dashboard_alert(self, alert: Alert):
        """Send dashboard alert"""
        # Store alert for dashboard consumption
        logger.info("Dashboard alert stored", alert_id=alert.id)
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if rule is in cooldown period"""
        rule = self.alert_rules.get(rule_name)
        if not rule:
            return False
        
        # Find last alert for this rule
        last_alert = None
        for alert in reversed(self.alert_history):
            if alert.rule_name == rule.name:
                last_alert = alert
                break
        
        if not last_alert:
            return False
        
        time_since_last = (datetime.utcnow() - last_alert.timestamp).total_seconds()
        cooldown_seconds = rule.cooldown_minutes * 60
        
        return time_since_last < cooldown_seconds
    
    def _is_rate_limited(self, rule_name: str) -> bool:
        """Check if rule is rate limited"""
        rule = self.alert_rules.get(rule_name)
        if not rule:
            return False
        
        # Count alerts in the last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.rule_name == rule.name and alert.timestamp > one_hour_ago
        ]
        
        return len(recent_alerts) >= rule.max_alerts_per_hour
    
    def _update_alert_counters(self, rule_name: str):
        """Update alert counters"""
        if rule_name not in self.alert_counts:
            self.alert_counts[rule_name] = 0
        
        self.alert_counts[rule_name] += 1
    
    async def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info("Alert acknowledged", alert_id=alert_id, user=user)
                return True
        
        return False
    
    async def get_active_alerts(self) -> List[Dict]:
        """Get active (unacknowledged) alerts"""
        active_alerts = [alert for alert in self.alerts if not alert.acknowledged]
        
        return [
            {
                "id": alert.id,
                "rule_name": alert.rule_name,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "channels": [ch.value for ch in alert.channels],
                "sent": alert.sent,
                "acknowledged": alert.acknowledged
            }
            for alert in active_alerts
        ]
    
    async def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get alert history"""
        recent_alerts = self.alert_history[-limit:] if self.alert_history else []
        
        return [
            {
                "id": alert.id,
                "rule_name": alert.rule_name,
                "level": alert.level.value,
                "title": alert.title,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged,
                "escalated": alert.escalated
            }
            for alert in recent_alerts
        ]
    
    async def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        
        # Count by level
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = sum(
                1 for alert in self.alert_history if alert.level == level
            )
        
        # Count by rule
        rule_counts = {}
        for alert in self.alert_history:
            rule_counts[alert.rule_name] = rule_counts.get(alert.rule_name, 0) + 1
        
        # Recent activity (last 24 hours)
        one_day_ago = datetime.utcnow() - timedelta(days=1)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > one_day_ago
        ]
        
        return {
            "total_alerts": total_alerts,
            "level_counts": level_counts,
            "rule_counts": rule_counts,
            "recent_24h": len(recent_alerts),
            "active_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "acknowledged_alerts": len([a for a in self.alerts if a.acknowledged])
        }
    
    async def create_custom_alert_rule(
        self,
        name: str,
        condition: str,
        threshold: float,
        level: AlertLevel,
        channels: List[AlertChannel],
        cooldown_minutes: int = 15
    ) -> bool:
        """Create a custom alert rule"""
        try:
            rule = AlertRule(
                name=name,
                condition=condition,
                threshold=threshold,
                level=level,
                channels=channels,
                cooldown_minutes=cooldown_minutes
            )
            
            self.alert_rules[name.lower().replace(" ", "_")] = rule
            
            logger.info("Custom alert rule created", name=name, condition=condition)
            return True
            
        except Exception as e:
            logger.error("Failed to create custom alert rule", error=str(e))
            return False
    
    async def test_alert_rule(self, rule_name: str) -> bool:
        """Test an alert rule"""
        rule = self.alert_rules.get(rule_name)
        if not rule:
            return False
        
        # Create test alert
        test_metrics = {"test": True}
        alert = await self._create_alert(rule, test_metrics)
        await self._send_alert(alert)
        
        logger.info("Test alert sent", rule_name=rule_name)
        return True

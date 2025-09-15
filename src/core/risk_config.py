"""
Risk Configuration Module
Central configuration for all risk parameters
"""

import os
from typing import Dict, Any, List
import json

def load_risk_config() -> Dict[str, Any]:
    """Load risk configuration from environment variables"""
    
    config = {
        # Core Risk Parameters
        "RISK_MANAGEMENT_ENABLED": os.getenv("RISK_MANAGEMENT_ENABLED", "true").lower() == "true",
        "RISK_ENGINE_MODE": os.getenv("RISK_ENGINE_MODE", "moderate"),
        "RISK_CHECK_INTERVAL_MS": int(os.getenv("RISK_CHECK_INTERVAL_MS", "100")),
        
        # Position Sizing
        "MAX_POSITION_SIZE_PERCENT": float(os.getenv("MAX_POSITION_SIZE_PERCENT", "2.0")),
        "MIN_POSITION_SIZE_USD": float(os.getenv("MIN_POSITION_SIZE_USD", "100")),
        "MAX_POSITION_SIZE_USD": float(os.getenv("MAX_POSITION_SIZE_USD", "100000")),
        "POSITION_SCALING_METHOD": os.getenv("POSITION_SCALING_METHOD", "kelly_criterion"),
        "KELLY_FRACTION": float(os.getenv("KELLY_FRACTION", "0.25")),
        
        # Portfolio Risk Limits
        "MAX_PORTFOLIO_RISK_PERCENT": float(os.getenv("MAX_PORTFOLIO_RISK_PERCENT", "6.0")),
        "MAX_DAILY_LOSS_PERCENT": float(os.getenv("MAX_DAILY_LOSS_PERCENT", "5.0")),
        "MAX_DRAWDOWN_PERCENT": float(os.getenv("MAX_DRAWDOWN_PERCENT", "20.0")),
        "MAX_CORRELATION_EXPOSURE": float(os.getenv("MAX_CORRELATION_EXPOSURE", "0.7")),
        
        # Risk Per Trade
        "MAX_RISK_PER_TRADE_PERCENT": float(os.getenv("MAX_RISK_PER_TRADE_PERCENT", "1.0")),
        "DEFAULT_STOP_LOSS_PERCENT": float(os.getenv("DEFAULT_STOP_LOSS_PERCENT", "2.0")),
        "TRAILING_STOP_ENABLED": os.getenv("TRAILING_STOP_ENABLED", "true").lower() == "true",
        
        # VaR Settings
        "VAR_ENABLED": os.getenv("VAR_ENABLED", "true").lower() == "true",
        "VAR_CONFIDENCE_LEVEL": float(os.getenv("VAR_CONFIDENCE_LEVEL", "0.95")),
        "VAR_CALCULATION_METHOD": os.getenv("VAR_CALCULATION_METHOD", "monte_carlo"),
        "VAR_SIMULATIONS": int(os.getenv("VAR_SIMULATIONS", "10000")),
        "MAX_VAR_PERCENT": float(os.getenv("MAX_VAR_PERCENT", "10.0")),
        
        # Stress Testing
        "STRESS_TESTING_ENABLED": os.getenv("STRESS_TESTING_ENABLED", "true").lower() == "true",
        "MAX_STRESS_LOSS_PERCENT": float(os.getenv("MAX_STRESS_LOSS_PERCENT", "30.0")),
        
        # Liquidity Risk
        "MIN_LIQUIDITY_RATIO": float(os.getenv("MIN_LIQUIDITY_RATIO", "0.01")),
        "MAX_POSITION_OF_ADV_PERCENT": float(os.getenv("MAX_POSITION_OF_ADV_PERCENT", "5.0")),
        
        # Correlation Risk
        "MAX_POSITIVE_CORRELATION": float(os.getenv("MAX_POSITIVE_CORRELATION", "0.8")),
        
        # Volatility Risk
        "MAX_VOLATILITY_PERCENT": float(os.getenv("MAX_VOLATILITY_PERCENT", "50.0")),
        "HIGH_VOLATILITY_THRESHOLD": float(os.getenv("HIGH_VOLATILITY_THRESHOLD", "40.0")),
        "VOLATILITY_CALCULATION_METHOD": os.getenv("VOLATILITY_CALCULATION_METHOD", "ewma"),
        
        # Black Swan Protection
        "BLACK_SWAN_PROTECTION_ENABLED": os.getenv("BLACK_SWAN_PROTECTION_ENABLED", "true").lower() == "true",
        "CRISIS_MODE_TRIGGER_VIX": float(os.getenv("CRISIS_MODE_TRIGGER_VIX", "30.0")),
        "BLACK_SWAN_THREAT_THRESHOLD": float(os.getenv("BLACK_SWAN_THREAT_THRESHOLD", "0.7")),
        
        # Circuit Breakers
        "CIRCUIT_BREAKER_ENABLED": os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true",
        "DAILY_LOSS_CIRCUIT_BREAKER_PERCENT": float(os.getenv("DAILY_LOSS_CIRCUIT_BREAKER_PERCENT", "3.0")),
        
        # Emergency Controls
        "EMERGENCY_SHUTDOWN_ENABLED": os.getenv("EMERGENCY_SHUTDOWN_ENABLED", "true").lower() == "true",
        "EMERGENCY_LIQUIDATION_ENABLED": os.getenv("EMERGENCY_LIQUIDATION_ENABLED", "false").lower() == "true",
        "EMERGENCY_HEDGE_ENABLED": os.getenv("EMERGENCY_HEDGE_ENABLED", "true").lower() == "true",
        
        # Risk Limits
        "MAX_TOTAL_RISK_SCORE": float(os.getenv("MAX_TOTAL_RISK_SCORE", "100.0")),
        "MAX_CONCENTRATION_PERCENT": float(os.getenv("MAX_CONCENTRATION_PERCENT", "20.0")),
        
        # Alerting Configuration
        "ALERT_EMAILS": os.getenv("ALERT_EMAILS", ""),
        "SLACK_WEBHOOK_URL": os.getenv("SLACK_WEBHOOK_URL", ""),
        "SLACK_CHANNEL": os.getenv("SLACK_CHANNEL", "#risk-alerts"),
        "SMS_NUMBERS": os.getenv("SMS_NUMBERS", ""),
        "ALERT_WEBHOOK_URL": os.getenv("ALERT_WEBHOOK_URL", ""),
        
        # Email Configuration
        "SMTP_SERVER": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "SMTP_PORT": int(os.getenv("SMTP_PORT", "587")),
        "SMTP_USERNAME": os.getenv("SMTP_USERNAME", ""),
        "SMTP_PASSWORD": os.getenv("SMTP_PASSWORD", ""),
        "FROM_EMAIL": os.getenv("FROM_EMAIL", "alerts@trading.com"),
        
        # SMS Configuration
        "SMS_API_KEY": os.getenv("SMS_API_KEY", ""),
        "SMS_FROM_NUMBER": os.getenv("SMS_FROM_NUMBER", ""),
        
        # Database Configuration
        "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql://omni:omni@localhost:5432/omni_alpha"),
        
        # Performance Settings
        "RISK_CACHE_TTL": int(os.getenv("RISK_CACHE_TTL", "60")),
        "RISK_CALCULATION_TIMEOUT": int(os.getenv("RISK_CALCULATION_TIMEOUT", "30")),
        "MAX_RISK_CALCULATIONS_PER_MINUTE": int(os.getenv("MAX_RISK_CALCULATIONS_PER_MINUTE", "100")),
        
        # Monitoring
        "RISK_METRICS_RETENTION_DAYS": int(os.getenv("RISK_METRICS_RETENTION_DAYS", "90")),
        "ALERT_RETENTION_DAYS": int(os.getenv("ALERT_RETENTION_DAYS", "30")),
        "CIRCUIT_BREAKER_RETENTION_DAYS": int(os.getenv("CIRCUIT_BREAKER_RETENTION_DAYS", "90")),
        
        # Advanced Settings
        "ENABLE_RISK_ATTRIBUTION": os.getenv("ENABLE_RISK_ATTRIBUTION", "true").lower() == "true",
        "ENABLE_TAIL_RISK_ANALYSIS": os.getenv("ENABLE_TAIL_RISK_ANALYSIS", "true").lower() == "true",
        "ENABLE_LIQUIDITY_ANALYSIS": os.getenv("ENABLE_LIQUIDITY_ANALYSIS", "true").lower() == "true",
        "ENABLE_CORRELATION_ANALYSIS": os.getenv("ENABLE_CORRELATION_ANALYSIS", "true").lower() == "true",
        
        # Model Settings
        "GARCH_MODEL_ENABLED": os.getenv("GARCH_MODEL_ENABLED", "true").lower() == "true",
        "EWMA_LAMBDA": float(os.getenv("EWMA_LAMBDA", "0.94")),
        "REGIME_SWITCHING_ENABLED": os.getenv("REGIME_SWITCHING_ENABLED", "true").lower() == "true",
        "JUMP_DIFFUSION_ENABLED": os.getenv("JUMP_DIFFUSION_ENABLED", "true").lower() == "true",
        
        # Backtesting
        "ENABLE_VAR_BACKTESTING": os.getenv("ENABLE_VAR_BACKTESTING", "true").lower() == "true",
        "BACKTEST_WINDOW_SIZE": int(os.getenv("BACKTEST_WINDOW_SIZE", "252")),
        "BACKTEST_CONFIDENCE_LEVEL": float(os.getenv("BACKTEST_CONFIDENCE_LEVEL", "0.95")),
        
        # Real-time Settings
        "REAL_TIME_RISK_MONITORING": os.getenv("REAL_TIME_RISK_MONITORING", "true").lower() == "true",
        "RISK_UPDATE_FREQUENCY_SECONDS": int(os.getenv("RISK_UPDATE_FREQUENCY_SECONDS", "60")),
        "ALERT_CHECK_FREQUENCY_SECONDS": int(os.getenv("ALERT_CHECK_FREQUENCY_SECONDS", "30")),
        
        # API Settings
        "RISK_API_RATE_LIMIT": int(os.getenv("RISK_API_RATE_LIMIT", "100")),
        "RISK_API_TIMEOUT": int(os.getenv("RISK_API_TIMEOUT", "30")),
        
        # Logging
        "RISK_LOG_LEVEL": os.getenv("RISK_LOG_LEVEL", "INFO"),
        "ENABLE_RISK_AUDIT_LOG": os.getenv("ENABLE_RISK_AUDIT_LOG", "true").lower() == "true",
        
        # Security
        "RISK_API_AUTHENTICATION": os.getenv("RISK_API_AUTHENTICATION", "true").lower() == "true",
        "RISK_DATA_ENCRYPTION": os.getenv("RISK_DATA_ENCRYPTION", "true").lower() == "true",
        
        # Compliance
        "ENABLE_RISK_REPORTING": os.getenv("ENABLE_RISK_REPORTING", "true").lower() == "true",
        "RISK_REPORT_FREQUENCY": os.getenv("RISK_REPORT_FREQUENCY", "daily"),
        "COMPLIANCE_MODE": os.getenv("COMPLIANCE_MODE", "false").lower() == "true",
        
        # Development/Testing
        "RISK_SIMULATION_MODE": os.getenv("RISK_SIMULATION_MODE", "false").lower() == "true",
        "RISK_TEST_DATA_ENABLED": os.getenv("RISK_TEST_DATA_ENABLED", "false").lower() == "true",
        "RISK_DEBUG_MODE": os.getenv("RISK_DEBUG_MODE", "false").lower() == "true"
    }
    
    return config

# Risk presets for different trading styles
RISK_PRESETS = {
    "conservative": {
        "MAX_POSITION_SIZE_PERCENT": 1.0,
        "MAX_DAILY_LOSS_PERCENT": 2.0,
        "MAX_DRAWDOWN_PERCENT": 10.0,
        "MAX_RISK_PER_TRADE_PERCENT": 0.5,
        "KELLY_FRACTION": 0.15,
        "MAX_PORTFOLIO_RISK_PERCENT": 4.0,
        "MAX_VAR_PERCENT": 5.0,
        "DAILY_LOSS_CIRCUIT_BREAKER_PERCENT": 1.5,
        "MAX_VOLATILITY_PERCENT": 30.0,
        "MAX_POSITIVE_CORRELATION": 0.6
    },
    "moderate": {
        "MAX_POSITION_SIZE_PERCENT": 2.0,
        "MAX_DAILY_LOSS_PERCENT": 5.0,
        "MAX_DRAWDOWN_PERCENT": 20.0,
        "MAX_RISK_PER_TRADE_PERCENT": 1.0,
        "KELLY_FRACTION": 0.25,
        "MAX_PORTFOLIO_RISK_PERCENT": 6.0,
        "MAX_VAR_PERCENT": 10.0,
        "DAILY_LOSS_CIRCUIT_BREAKER_PERCENT": 3.0,
        "MAX_VOLATILITY_PERCENT": 50.0,
        "MAX_POSITIVE_CORRELATION": 0.8
    },
    "aggressive": {
        "MAX_POSITION_SIZE_PERCENT": 5.0,
        "MAX_DAILY_LOSS_PERCENT": 10.0,
        "MAX_DRAWDOWN_PERCENT": 30.0,
        "MAX_RISK_PER_TRADE_PERCENT": 2.0,
        "KELLY_FRACTION": 0.40,
        "MAX_PORTFOLIO_RISK_PERCENT": 12.0,
        "MAX_VAR_PERCENT": 20.0,
        "DAILY_LOSS_CIRCUIT_BREAKER_PERCENT": 5.0,
        "MAX_VOLATILITY_PERCENT": 80.0,
        "MAX_POSITIVE_CORRELATION": 0.9
    },
    "institutional": {
        "MAX_POSITION_SIZE_PERCENT": 0.5,
        "MAX_DAILY_LOSS_PERCENT": 1.0,
        "MAX_DRAWDOWN_PERCENT": 5.0,
        "MAX_RISK_PER_TRADE_PERCENT": 0.25,
        "KELLY_FRACTION": 0.10,
        "MAX_PORTFOLIO_RISK_PERCENT": 2.0,
        "MAX_VAR_PERCENT": 3.0,
        "DAILY_LOSS_CIRCUIT_BREAKER_PERCENT": 0.5,
        "MAX_VOLATILITY_PERCENT": 20.0,
        "MAX_POSITIVE_CORRELATION": 0.5,
        "COMPLIANCE_MODE": True,
        "ENABLE_RISK_REPORTING": True,
        "RISK_REPORT_FREQUENCY": "hourly"
    },
    "hedge_fund": {
        "MAX_POSITION_SIZE_PERCENT": 3.0,
        "MAX_DAILY_LOSS_PERCENT": 7.0,
        "MAX_DRAWDOWN_PERCENT": 25.0,
        "MAX_RISK_PER_TRADE_PERCENT": 1.5,
        "KELLY_FRACTION": 0.30,
        "MAX_PORTFOLIO_RISK_PERCENT": 8.0,
        "MAX_VAR_PERCENT": 15.0,
        "DAILY_LOSS_CIRCUIT_BREAKER_PERCENT": 4.0,
        "MAX_VOLATILITY_PERCENT": 60.0,
        "MAX_POSITIVE_CORRELATION": 0.7,
        "ENABLE_TAIL_RISK_ANALYSIS": True,
        "ENABLE_LIQUIDITY_ANALYSIS": True
    },
    "prop_trading": {
        "MAX_POSITION_SIZE_PERCENT": 10.0,
        "MAX_DAILY_LOSS_PERCENT": 15.0,
        "MAX_DRAWDOWN_PERCENT": 40.0,
        "MAX_RISK_PER_TRADE_PERCENT": 3.0,
        "KELLY_FRACTION": 0.50,
        "MAX_PORTFOLIO_RISK_PERCENT": 20.0,
        "MAX_VAR_PERCENT": 30.0,
        "DAILY_LOSS_CIRCUIT_BREAKER_PERCENT": 8.0,
        "MAX_VOLATILITY_PERCENT": 100.0,
        "MAX_POSITIVE_CORRELATION": 0.95,
        "REAL_TIME_RISK_MONITORING": True,
        "RISK_UPDATE_FREQUENCY_SECONDS": 10
    }
}

def apply_risk_preset(config: Dict[str, Any], preset: str) -> Dict[str, Any]:
    """Apply a risk preset to the configuration"""
    if preset in RISK_PRESETS:
        config.update(RISK_PRESETS[preset])
        config["RISK_ENGINE_MODE"] = preset
    return config

def validate_risk_config(config: Dict[str, Any]) -> List[str]:
    """Validate risk configuration and return any issues"""
    issues = []
    
    # Check required settings
    required_settings = [
        "MAX_POSITION_SIZE_PERCENT",
        "MAX_DAILY_LOSS_PERCENT",
        "MAX_DRAWDOWN_PERCENT",
        "MAX_RISK_PER_TRADE_PERCENT",
        "MAX_PORTFOLIO_RISK_PERCENT"
    ]
    
    for setting in required_settings:
        if setting not in config:
            issues.append(f"Missing required setting: {setting}")
        elif config[setting] <= 0:
            issues.append(f"Invalid value for {setting}: must be positive")
    
    # Check logical consistency
    if config.get("MAX_POSITION_SIZE_PERCENT", 0) > config.get("MAX_PORTFOLIO_RISK_PERCENT", 0):
        issues.append("MAX_POSITION_SIZE_PERCENT cannot be greater than MAX_PORTFOLIO_RISK_PERCENT")
    
    if config.get("MAX_DAILY_LOSS_PERCENT", 0) > config.get("MAX_DRAWDOWN_PERCENT", 0):
        issues.append("MAX_DAILY_LOSS_PERCENT cannot be greater than MAX_DRAWDOWN_PERCENT")
    
    if config.get("KELLY_FRACTION", 0) > 1.0:
        issues.append("KELLY_FRACTION cannot be greater than 1.0")
    
    # Check email configuration if alerts are enabled
    if config.get("ALERT_EMAILS") and not config.get("SMTP_USERNAME"):
        issues.append("SMTP_USERNAME required when ALERT_EMAILS is configured")
    
    # Check Slack configuration
    if config.get("SLACK_WEBHOOK_URL") and not config.get("SLACK_CHANNEL"):
        issues.append("SLACK_CHANNEL required when SLACK_WEBHOOK_URL is configured")
    
    return issues

def get_risk_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of risk configuration"""
    return {
        "risk_management_enabled": config.get("RISK_MANAGEMENT_ENABLED", False),
        "risk_engine_mode": config.get("RISK_ENGINE_MODE", "moderate"),
        "position_limits": {
            "max_position_size_percent": config.get("MAX_POSITION_SIZE_PERCENT", 0),
            "max_portfolio_risk_percent": config.get("MAX_PORTFOLIO_RISK_PERCENT", 0),
            "max_daily_loss_percent": config.get("MAX_DAILY_LOSS_PERCENT", 0),
            "max_drawdown_percent": config.get("MAX_DRAWDOWN_PERCENT", 0)
        },
        "risk_per_trade": {
            "max_risk_per_trade_percent": config.get("MAX_RISK_PER_TRADE_PERCENT", 0),
            "kelly_fraction": config.get("KELLY_FRACTION", 0),
            "position_scaling_method": config.get("POSITION_SCALING_METHOD", "")
        },
        "var_settings": {
            "var_enabled": config.get("VAR_ENABLED", False),
            "var_confidence_level": config.get("VAR_CONFIDENCE_LEVEL", 0),
            "var_calculation_method": config.get("VAR_CALCULATION_METHOD", ""),
            "max_var_percent": config.get("MAX_VAR_PERCENT", 0)
        },
        "circuit_breakers": {
            "circuit_breaker_enabled": config.get("CIRCUIT_BREAKER_ENABLED", False),
            "daily_loss_circuit_breaker_percent": config.get("DAILY_LOSS_CIRCUIT_BREAKER_PERCENT", 0)
        },
        "alerting": {
            "alert_emails": bool(config.get("ALERT_EMAILS")),
            "slack_webhook": bool(config.get("SLACK_WEBHOOK_URL")),
            "sms_numbers": bool(config.get("SMS_NUMBERS"))
        },
        "monitoring": {
            "real_time_monitoring": config.get("REAL_TIME_RISK_MONITORING", False),
            "risk_update_frequency_seconds": config.get("RISK_UPDATE_FREQUENCY_SECONDS", 0),
            "alert_check_frequency_seconds": config.get("ALERT_CHECK_FREQUENCY_SECONDS", 0)
        }
    }

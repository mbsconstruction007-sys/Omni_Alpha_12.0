"""
Portfolio Configuration Module
Central configuration for all portfolio management parameters
"""

import os
from typing import Dict, Any, List
import json

def load_portfolio_config() -> Dict[str, Any]:
    """Load portfolio configuration from environment variables"""
    
    config = {
        # Core Portfolio Settings
        "PORTFOLIO_MANAGEMENT_ENABLED": os.getenv("PORTFOLIO_MANAGEMENT_ENABLED", "true").lower() == "true",
        "PORTFOLIO_UPDATE_INTERVAL_MS": int(os.getenv("PORTFOLIO_UPDATE_INTERVAL_MS", "1000")),
        "METRICS_CACHE_TTL_SECONDS": int(os.getenv("METRICS_CACHE_TTL_SECONDS", "60")),
        
        # Optimization Settings
        "OPTIMIZATION_METHOD": os.getenv("OPTIMIZATION_METHOD", "ensemble"),
        "ML_PORTFOLIO_OPTIMIZATION": os.getenv("ML_PORTFOLIO_OPTIMIZATION", "false").lower() == "true",
        "ML_PREDICTION_HORIZON_DAYS": int(os.getenv("ML_PREDICTION_HORIZON_DAYS", "30")),
        
        # Position Constraints
        "MAX_POSITION_WEIGHT": float(os.getenv("MAX_POSITION_WEIGHT", "0.10")),
        "MIN_POSITION_WEIGHT": float(os.getenv("MIN_POSITION_WEIGHT", "0.01")),
        "MAX_SECTOR_WEIGHT": float(os.getenv("MAX_SECTOR_WEIGHT", "0.30")),
        "MAX_CORRELATION_SUM": float(os.getenv("MAX_CORRELATION_SUM", "0.50")),
        "TARGET_PORTFOLIO_VOLATILITY": float(os.getenv("TARGET_PORTFOLIO_VOLATILITY", "0.15")),
        "MAX_LEVERAGE": float(os.getenv("MAX_LEVERAGE", "1.0")),
        
        # Rebalancing Settings
        "REBALANCING_ENABLED": os.getenv("REBALANCING_ENABLED", "true").lower() == "true",
        "REBALANCING_METHOD": os.getenv("REBALANCING_METHOD", "threshold"),
        "REBALANCING_FREQUENCY": os.getenv("REBALANCING_FREQUENCY", "daily"),
        "REBALANCING_THRESHOLD_PERCENT": float(os.getenv("REBALANCING_THRESHOLD_PERCENT", "0.05")),
        "EMERGENCY_REBALANCE_THRESHOLD": float(os.getenv("EMERGENCY_REBALANCE_THRESHOLD", "0.20")),
        
        # Tax Optimization
        "TAX_OPTIMIZATION_ENABLED": os.getenv("TAX_OPTIMIZATION_ENABLED", "true").lower() == "true",
        "SHORT_TERM_CAPITAL_GAINS_RATE": float(os.getenv("SHORT_TERM_CAPITAL_GAINS_RATE", "0.37")),
        "LONG_TERM_CAPITAL_GAINS_RATE": float(os.getenv("LONG_TERM_CAPITAL_GAINS_RATE", "0.20")),
        "STATE_TAX_RATE": float(os.getenv("STATE_TAX_RATE", "0.05")),
        "WASH_SALE_PERIOD_DAYS": int(os.getenv("WASH_SALE_PERIOD_DAYS", "30")),
        "TAX_HARVEST_THRESHOLD_USD": float(os.getenv("TAX_HARVEST_THRESHOLD_USD", "1000")),
        
        # Risk Management
        "RISK_BUDGETING_ENABLED": os.getenv("RISK_BUDGETING_ENABLED", "true").lower() == "true",
        "MAX_PORTFOLIO_VAR": float(os.getenv("MAX_PORTFOLIO_VAR", "0.05")),
        "MAX_POSITION_VAR": float(os.getenv("MAX_POSITION_VAR", "0.02")),
        "VAR_CONFIDENCE_LEVEL": float(os.getenv("VAR_CONFIDENCE_LEVEL", "0.95")),
        
        # Regime Detection
        "REGIME_DETECTION_ENABLED": os.getenv("REGIME_DETECTION_ENABLED", "true").lower() == "true",
        "REGIME_UPDATE_FREQUENCY_HOURS": int(os.getenv("REGIME_UPDATE_FREQUENCY_HOURS", "24")),
        
        # Regime-Specific Parameters
        "REGIME_SPECIFIC_PARAMS": {
            "bull_quiet": {
                "leverage": 1.0,
                "position_count": 20,
                "stop_loss": 0.10
            },
            "bull_volatile": {
                "leverage": 0.8,
                "position_count": 15,
                "stop_loss": 0.08
            },
            "bear_quiet": {
                "leverage": 0.6,
                "position_count": 10,
                "stop_loss": 0.06
            },
            "bear_volatile": {
                "leverage": 0.4,
                "position_count": 8,
                "stop_loss": 0.05
            },
            "transition": {
                "leverage": 0.7,
                "position_count": 12,
                "stop_loss": 0.07
            },
            "crisis": {
                "leverage": 0.2,
                "position_count": 5,
                "stop_loss": 0.03
            }
        },
        
        # Performance Tracking
        "PERFORMANCE_TRACKING_ENABLED": os.getenv("PERFORMANCE_TRACKING_ENABLED", "true").lower() == "true",
        "ATTRIBUTION_ANALYSIS_ENABLED": os.getenv("ATTRIBUTION_ANALYSIS_ENABLED", "true").lower() == "true",
        
        # Backtesting
        "BACKTESTING_ENABLED": os.getenv("BACKTESTING_ENABLED", "true").lower() == "true",
        "BACKTEST_START_DATE": os.getenv("BACKTEST_START_DATE", "2020-01-01"),
        "BACKTEST_END_DATE": os.getenv("BACKTEST_END_DATE", "2024-01-01"),
        
        # Data Sources
        "MARKET_DATA_PROVIDER": os.getenv("MARKET_DATA_PROVIDER", "alpaca"),
        "FUNDAMENTAL_DATA_PROVIDER": os.getenv("FUNDAMENTAL_DATA_PROVIDER", "alpha_vantage"),
        "NEWS_DATA_PROVIDER": os.getenv("NEWS_DATA_PROVIDER", "newsapi"),
        
        # Universe Selection
        "UNIVERSE_SIZE": int(os.getenv("UNIVERSE_SIZE", "100")),
        "MIN_MARKET_CAP": float(os.getenv("MIN_MARKET_CAP", "1e9")),  # $1B
        "MIN_DAILY_VOLUME": float(os.getenv("MIN_DAILY_VOLUME", "1e6")),  # $1M
        "MAX_PRICE": float(os.getenv("MAX_PRICE", "1000")),
        "MIN_PRICE": float(os.getenv("MIN_PRICE", "5")),
        
        # Signal Generation
        "SIGNAL_GENERATION_ENABLED": os.getenv("SIGNAL_GENERATION_ENABLED", "true").lower() == "true",
        "SIGNAL_UPDATE_FREQUENCY_MINUTES": int(os.getenv("SIGNAL_UPDATE_FREQUENCY_MINUTES", "15")),
        "MIN_SIGNAL_STRENGTH": float(os.getenv("MIN_SIGNAL_STRENGTH", "0.1")),
        "MAX_SIGNAL_STRENGTH": float(os.getenv("MAX_SIGNAL_STRENGTH", "1.0")),
        
        # Transaction Costs
        "COMMISSION_PER_TRADE": float(os.getenv("COMMISSION_PER_TRADE", "0.0")),
        "SPREAD_COST_BPS": float(os.getenv("SPREAD_COST_BPS", "5.0")),  # 5 basis points
        "MARKET_IMPACT_MODEL": os.getenv("MARKET_IMPACT_MODEL", "linear"),
        
        # Monitoring and Alerts
        "PORTFOLIO_MONITORING_ENABLED": os.getenv("PORTFOLIO_MONITORING_ENABLED", "true").lower() == "true",
        "ALERT_EMAIL_ENABLED": os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true",
        "ALERT_SLACK_ENABLED": os.getenv("ALERT_SLACK_ENABLED", "false").lower() == "true",
        "ALERT_THRESHOLD_PERCENT": float(os.getenv("ALERT_THRESHOLD_PERCENT", "0.05")),
        
        # Database Settings
        "PORTFOLIO_DB_URL": os.getenv("PORTFOLIO_DB_URL", "postgresql://localhost/portfolio"),
        "PORTFOLIO_DB_POOL_SIZE": int(os.getenv("PORTFOLIO_DB_POOL_SIZE", "10")),
        "PORTFOLIO_DB_TIMEOUT": int(os.getenv("PORTFOLIO_DB_TIMEOUT", "30")),
        
        # Caching
        "CACHE_ENABLED": os.getenv("CACHE_ENABLED", "true").lower() == "true",
        "CACHE_TTL_SECONDS": int(os.getenv("CACHE_TTL_SECONDS", "300")),
        "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379/1"),
        
        # Logging
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "LOG_PORTFOLIO_DECISIONS": os.getenv("LOG_PORTFOLIO_DECISIONS", "true").lower() == "true",
        "LOG_TRADE_EXECUTION": os.getenv("LOG_TRADE_EXECUTION", "true").lower() == "true",
        
        # Advanced Features
        "FACTOR_MODEL_ENABLED": os.getenv("FACTOR_MODEL_ENABLED", "true").lower() == "true",
        "ESG_INTEGRATION_ENABLED": os.getenv("ESG_INTEGRATION_ENABLED", "false").lower() == "true",
        "OPTIONS_OVERLAY_ENABLED": os.getenv("OPTIONS_OVERLAY_ENABLED", "false").lower() == "true",
        "CRYPTO_INTEGRATION_ENABLED": os.getenv("CRYPTO_INTEGRATION_ENABLED", "false").lower() == "true",
        
        # Performance Optimization
        "PARALLEL_OPTIMIZATION": os.getenv("PARALLEL_OPTIMIZATION", "true").lower() == "true",
        "MAX_WORKERS": int(os.getenv("MAX_WORKERS", "8")),
        "OPTIMIZATION_TIMEOUT_SECONDS": int(os.getenv("OPTIMIZATION_TIMEOUT_SECONDS", "300")),
        
        # Compliance
        "COMPLIANCE_CHECKS_ENABLED": os.getenv("COMPLIANCE_CHECKS_ENABLED", "true").lower() == "true",
        "POSITION_LIMITS_ENABLED": os.getenv("POSITION_LIMITS_ENABLED", "true").lower() == "true",
        "CONCENTRATION_LIMITS_ENABLED": os.getenv("CONCENTRATION_LIMITS_ENABLED", "true").lower() == "true",
        
        # Reporting
        "DAILY_REPORTS_ENABLED": os.getenv("DAILY_REPORTS_ENABLED", "true").lower() == "true",
        "WEEKLY_REPORTS_ENABLED": os.getenv("WEEKLY_REPORTS_ENABLED", "true").lower() == "true",
        "MONTHLY_REPORTS_ENABLED": os.getenv("MONTHLY_REPORTS_ENABLED", "true").lower() == "true",
        "REPORT_FORMAT": os.getenv("REPORT_FORMAT", "pdf"),  # pdf, html, json
    }
    
    return config

def apply_portfolio_preset(config: Dict[str, Any], preset: str) -> Dict[str, Any]:
    """Apply portfolio management preset"""
    
    presets = {
        "conservative": {
            "TARGET_PORTFOLIO_VOLATILITY": 0.10,
            "MAX_LEVERAGE": 0.5,
            "REBALANCING_THRESHOLD_PERCENT": 0.03,
            "MAX_POSITION_WEIGHT": 0.05,
            "REGIME_SPECIFIC_PARAMS": {
                "bull_quiet": {"leverage": 0.5, "position_count": 15},
                "bull_volatile": {"leverage": 0.4, "position_count": 12},
                "bear_quiet": {"leverage": 0.3, "position_count": 10},
                "bear_volatile": {"leverage": 0.2, "position_count": 8},
                "transition": {"leverage": 0.35, "position_count": 10},
                "crisis": {"leverage": 0.1, "position_count": 5}
            }
        },
        
        "moderate": {
            "TARGET_PORTFOLIO_VOLATILITY": 0.15,
            "MAX_LEVERAGE": 1.0,
            "REBALANCING_THRESHOLD_PERCENT": 0.05,
            "MAX_POSITION_WEIGHT": 0.10,
            "REGIME_SPECIFIC_PARAMS": {
                "bull_quiet": {"leverage": 1.0, "position_count": 20},
                "bull_volatile": {"leverage": 0.8, "position_count": 15},
                "bear_quiet": {"leverage": 0.6, "position_count": 10},
                "bear_volatile": {"leverage": 0.4, "position_count": 8},
                "transition": {"leverage": 0.7, "position_count": 12},
                "crisis": {"leverage": 0.2, "position_count": 5}
            }
        },
        
        "aggressive": {
            "TARGET_PORTFOLIO_VOLATILITY": 0.25,
            "MAX_LEVERAGE": 2.0,
            "REBALANCING_THRESHOLD_PERCENT": 0.08,
            "MAX_POSITION_WEIGHT": 0.15,
            "REGIME_SPECIFIC_PARAMS": {
                "bull_quiet": {"leverage": 2.0, "position_count": 25},
                "bull_volatile": {"leverage": 1.5, "position_count": 20},
                "bear_quiet": {"leverage": 1.0, "position_count": 15},
                "bear_volatile": {"leverage": 0.8, "position_count": 10},
                "transition": {"leverage": 1.2, "position_count": 15},
                "crisis": {"leverage": 0.5, "position_count": 8}
            }
        },
        
        "institutional": {
            "TARGET_PORTFOLIO_VOLATILITY": 0.12,
            "MAX_LEVERAGE": 1.5,
            "REBALANCING_THRESHOLD_PERCENT": 0.02,
            "MAX_POSITION_WEIGHT": 0.08,
            "TAX_OPTIMIZATION_ENABLED": True,
            "FACTOR_MODEL_ENABLED": True,
            "ESG_INTEGRATION_ENABLED": True,
            "COMPLIANCE_CHECKS_ENABLED": True,
            "REGIME_SPECIFIC_PARAMS": {
                "bull_quiet": {"leverage": 1.5, "position_count": 30},
                "bull_volatile": {"leverage": 1.2, "position_count": 25},
                "bear_quiet": {"leverage": 0.8, "position_count": 20},
                "bear_volatile": {"leverage": 0.6, "position_count": 15},
                "transition": {"leverage": 1.0, "position_count": 20},
                "crisis": {"leverage": 0.3, "position_count": 10}
            }
        }
    }
    
    if preset in presets:
        config.update(presets[preset])
        print(f"Applied {preset} portfolio preset")
    else:
        print(f"Unknown preset: {preset}. Available presets: {list(presets.keys())}")
    
    return config

def validate_portfolio_config(config: Dict[str, Any]) -> List[str]:
    """Validate portfolio configuration"""
    errors = []
    
    # Check required settings
    required_settings = [
        "PORTFOLIO_MANAGEMENT_ENABLED",
        "OPTIMIZATION_METHOD",
        "MAX_POSITION_WEIGHT",
        "MIN_POSITION_WEIGHT",
        "TARGET_PORTFOLIO_VOLATILITY"
    ]
    
    for setting in required_settings:
        if setting not in config:
            errors.append(f"Missing required setting: {setting}")
    
    # Validate numeric ranges
    if config.get("MAX_POSITION_WEIGHT", 0) <= 0 or config.get("MAX_POSITION_WEIGHT", 0) > 1:
        errors.append("MAX_POSITION_WEIGHT must be between 0 and 1")
    
    if config.get("MIN_POSITION_WEIGHT", 0) < 0 or config.get("MIN_POSITION_WEIGHT", 0) >= config.get("MAX_POSITION_WEIGHT", 1):
        errors.append("MIN_POSITION_WEIGHT must be >= 0 and < MAX_POSITION_WEIGHT")
    
    if config.get("TARGET_PORTFOLIO_VOLATILITY", 0) <= 0 or config.get("TARGET_PORTFOLIO_VOLATILITY", 0) > 1:
        errors.append("TARGET_PORTFOLIO_VOLATILITY must be between 0 and 1")
    
    if config.get("MAX_LEVERAGE", 0) <= 0:
        errors.append("MAX_LEVERAGE must be > 0")
    
    # Validate optimization method
    valid_methods = ["ensemble", "mean_variance", "black_litterman", "risk_parity", "hierarchical_risk_parity"]
    if config.get("OPTIMIZATION_METHOD") not in valid_methods:
        errors.append(f"OPTIMIZATION_METHOD must be one of: {valid_methods}")
    
    # Validate rebalancing method
    valid_rebalancing_methods = ["calendar", "threshold", "adaptive"]
    if config.get("REBALANCING_METHOD") not in valid_rebalancing_methods:
        errors.append(f"REBALANCING_METHOD must be one of: {valid_rebalancing_methods}")
    
    return errors

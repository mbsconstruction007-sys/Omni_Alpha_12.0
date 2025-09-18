"""
OMNI ALPHA 5.0 - CORE CONFIGURATION MANAGEMENT
==============================================
Production-ready environment configuration with encryption and validation
"""

import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json

try:
    from pydantic import BaseSettings, validator, Field
    from cryptography.fernet import Fernet
    PYDANTIC_AVAILABLE = True
    CRYPTO_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    CRYPTO_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
env_files = [
    '.env.local',
    'alpaca_live_trading.env', 
    'step1_environment_template.env',
    '.env'
]

for env_file in env_files:
    if Path(env_file).exists():
        load_dotenv(env_file)
        break

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class TradingMode(Enum):
    """Trading modes"""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"
    SIMULATION = "simulation"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    # PostgreSQL (Primary)
    postgres_url: str = os.getenv('POSTGRES_URL', 'postgresql://trader:password@localhost:5432/omni_alpha')
    postgres_pool_size: int = int(os.getenv('POSTGRES_POOL_SIZE', '20'))
    postgres_max_overflow: int = int(os.getenv('POSTGRES_MAX_OVERFLOW', '40'))
    postgres_pool_timeout: int = int(os.getenv('POSTGRES_POOL_TIMEOUT', '30'))
    
    # SQLite (Fallback)
    sqlite_url: str = os.getenv('SQLITE_URL', 'sqlite:///omni_alpha.db')
    
    # InfluxDB (Time-series)
    influxdb_url: str = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    influxdb_token: str = os.getenv('INFLUXDB_TOKEN', '')
    influxdb_org: str = os.getenv('INFLUXDB_ORG', 'omni_alpha')
    influxdb_bucket: str = os.getenv('INFLUXDB_BUCKET', 'market_data')
    
    # Redis (Cache)
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    redis_password: str = os.getenv('REDIS_PASSWORD', '')
    redis_db: int = int(os.getenv('REDIS_DB', '0'))

@dataclass
class APIConfig:
    """API configuration with encryption"""
    # Alpaca Trading
    alpaca_api_key: str = os.getenv('ALPACA_API_KEY', 'PK02D3BXIPSW11F0Q9OW')
    alpaca_secret_key: str = os.getenv('ALPACA_SECRET_KEY', '')
    alpaca_base_url: str = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    alpaca_stream_url: str = os.getenv('ALPACA_STREAM_URL', 'wss://stream.data.alpaca.markets/v2/iex')
    
    # Telegram Bot
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Google/Gemini AI
    google_api_key: str = os.getenv('GOOGLE_API_KEY', 'AIzaSyDpKZV5XTysC2T9lJax29v2kIAR2q6LXnU')
    google_project_id: str = os.getenv('GOOGLE_PROJECT_ID', 'hyper-gmhsi-trading-bot')
    
    # Alpha Vantage
    alpha_vantage_key: str = os.getenv('ALPHA_VANTAGE_KEY', '')
    
    # Yahoo Finance
    yahoo_finance_enabled: bool = os.getenv('YAHOO_FINANCE_ENABLED', 'true').lower() == 'true'
    
    # NSE/BSE
    nse_enabled: bool = os.getenv('NSE_ENABLED', 'true').lower() == 'true'
    bse_enabled: bool = os.getenv('BSE_ENABLED', 'false').lower() == 'true'
    
    # Encryption
    encryption_key: str = os.getenv('ENCRYPTION_KEY', '')
    
    def decrypt_if_encrypted(self, value: str) -> str:
        """Decrypt value if it's encrypted"""
        if not value or not self.encryption_key or not CRYPTO_AVAILABLE:
            return value
        
        try:
            if value.startswith('gAAAAA'):  # Fernet encrypted
                fernet = Fernet(self.encryption_key.encode())
                return fernet.decrypt(value.encode()).decode()
        except Exception:
            pass  # Return original if decryption fails
        
        return value
    
    def get_decrypted_alpaca_credentials(self) -> tuple[str, str]:
        """Get decrypted Alpaca credentials"""
        api_key = self.decrypt_if_encrypted(self.alpaca_api_key)
        secret_key = self.decrypt_if_encrypted(self.alpaca_secret_key)
        return api_key, secret_key

@dataclass
class TradingConfig:
    """Trading configuration"""
    # Position Limits
    max_position_size_dollars: float = float(os.getenv('MAX_POSITION_SIZE_DOLLARS', '10000'))
    max_position_size_percent: float = float(os.getenv('MAX_POSITION_SIZE_PERCENT', '0.10'))
    max_positions: int = int(os.getenv('MAX_POSITIONS', '20'))
    
    # Risk Limits
    max_daily_trades: int = int(os.getenv('MAX_DAILY_TRADES', '100'))
    max_daily_loss: float = float(os.getenv('MAX_DAILY_LOSS', '1000'))
    max_drawdown_percent: float = float(os.getenv('MAX_DRAWDOWN_PCT', '0.02'))
    stop_loss_percent: float = float(os.getenv('STOP_LOSS_PERCENT', '0.03'))
    take_profit_percent: float = float(os.getenv('TAKE_PROFIT_PERCENT', '0.06'))
    
    # Execution
    default_order_type: str = os.getenv('DEFAULT_ORDER_TYPE', 'market')
    default_time_in_force: str = os.getenv('DEFAULT_TIME_IN_FORCE', 'day')
    
    # Latency Thresholds (microseconds)
    max_order_latency_us: int = int(os.getenv('MAX_ORDER_LATENCY_US', '10000'))
    max_data_latency_us: int = int(os.getenv('MAX_DATA_LATENCY_US', '1000'))
    max_strategy_latency_us: int = int(os.getenv('MAX_STRATEGY_LATENCY_US', '5000'))

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    # Prometheus
    metrics_enabled: bool = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
    metrics_port: int = int(os.getenv('METRICS_PORT', '8001'))
    
    # Health Checks
    health_check_enabled: bool = os.getenv('HEALTH_CHECK_ENABLED', 'true').lower() == 'true'
    health_check_port: int = int(os.getenv('HEALTH_CHECK_PORT', '8000'))
    health_check_interval: int = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = os.getenv('CIRCUIT_BREAKER_ENABLED', 'true').lower() == 'true'
    max_consecutive_errors: int = int(os.getenv('MAX_CONSECUTIVE_ERRORS', '5'))
    error_cooldown_seconds: int = int(os.getenv('ERROR_COOLDOWN_SECONDS', '60'))
    
    # Alerts
    alert_webhook_url: str = os.getenv('ALERT_WEBHOOK_URL', '')
    email_alerts_enabled: bool = os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true'
    telegram_alerts_enabled: bool = os.getenv('TELEGRAM_ALERTS_ENABLED', 'true').lower() == 'true'

@dataclass
class DataCollectionConfig:
    """Data collection configuration"""
    # WebSocket Settings
    ws_reconnect_delay: int = int(os.getenv('WS_RECONNECT_DELAY', '5'))
    ws_max_reconnects: int = int(os.getenv('WS_MAX_RECONNECTS', '10'))
    ws_ping_interval: int = int(os.getenv('WS_PING_INTERVAL', '30'))
    
    # Tick Data
    tick_buffer_size: int = int(os.getenv('TICK_BUFFER_SIZE', '100000'))
    tick_flush_interval: int = int(os.getenv('TICK_FLUSH_INTERVAL', '60'))
    tick_compression: bool = os.getenv('TICK_COMPRESSION', 'true').lower() == 'true'
    tick_storage_days: int = int(os.getenv('TICK_STORAGE_DAYS', '30'))
    
    # Order Book
    order_book_levels: int = int(os.getenv('ORDER_BOOK_LEVELS', '20'))
    order_book_update_freq: int = int(os.getenv('ORDER_BOOK_UPDATE_FREQ', '1'))
    order_book_imbalance_threshold: float = float(os.getenv('ORDER_BOOK_IMBALANCE_THRESHOLD', '0.7'))
    
    # Corporate Actions
    corp_actions_update_interval: int = int(os.getenv('CORP_ACTIONS_UPDATE_INTERVAL', '3600'))
    adjust_for_splits: bool = os.getenv('ADJUST_FOR_SPLITS', 'true').lower() == 'true'
    track_dividends: bool = os.getenv('TRACK_DIVIDENDS', 'true').lower() == 'true'
    
    # News & Sentiment
    news_update_interval: int = int(os.getenv('NEWS_UPDATE_INTERVAL', '300'))
    sentiment_threshold: float = float(os.getenv('SENTIMENT_THRESHOLD', '0.6'))
    max_news_age_hours: int = int(os.getenv('MAX_NEWS_AGE_HOURS', '72'))
    
    # Data Quality
    max_spread_percent: float = float(os.getenv('MAX_SPREAD_PCT', '5.0'))
    max_price_movement_percent: float = float(os.getenv('MAX_PRICE_MOVEMENT_PCT', '10.0'))
    min_tick_interval_us: int = int(os.getenv('MIN_TICK_INTERVAL_US', '100'))

class OmniAlphaSettings:
    """Main settings class for Omni Alpha 5.0"""
    
    def __init__(self):
        self.load_settings()
    
    def load_settings(self):
        """Load all configuration sections"""
        # Core Settings
        self.app_name = os.getenv('APP_NAME', 'Omni Alpha 5.0')
        self.version = os.getenv('APP_VERSION', '5.0.0')
        self.environment = Environment(os.getenv('ENVIRONMENT', 'production'))
        self.trading_mode = TradingMode(os.getenv('TRADING_MODE', 'paper'))
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Configuration sections
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.monitoring = MonitoringConfig()
        self.data_collection = DataCollectionConfig()
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'omni_alpha.log')
        self.log_max_size = int(os.getenv('LOG_MAX_SIZE', '10485760'))  # 10MB
        self.log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))
        
        # Performance
        self.async_workers = int(os.getenv('ASYNC_WORKERS', '8'))
        self.batch_insert_size = int(os.getenv('BATCH_INSERT_SIZE', '1000'))
        self.cache_ttl_seconds = int(os.getenv('CACHE_TTL_SECONDS', '60'))
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return errors"""
        errors = []
        
        # Validate API keys
        if not self.api.alpaca_api_key:
            errors.append("ALPACA_API_KEY is required")
        
        if not self.api.alpaca_secret_key:
            errors.append("ALPACA_SECRET_KEY is required")
        
        if not self.api.telegram_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN is required")
        
        # Validate trading limits
        if self.trading.max_position_size_dollars <= 0:
            errors.append("MAX_POSITION_SIZE_DOLLARS must be positive")
        
        if self.trading.max_daily_trades <= 0:
            errors.append("MAX_DAILY_TRADES must be positive")
        
        # Validate latency thresholds
        if self.trading.max_order_latency_us <= 0:
            errors.append("MAX_ORDER_LATENCY_US must be positive")
        
        # Validate data collection settings
        if self.data_collection.tick_buffer_size <= 0:
            errors.append("TICK_BUFFER_SIZE must be positive")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'app_name': self.app_name,
            'version': self.version,
            'environment': self.environment.value,
            'trading_mode': self.trading_mode.value,
            'debug': self.debug,
            'database': {
                'postgres_url': self.database.postgres_url,
                'redis_url': self.database.redis_url,
                'influxdb_url': self.database.influxdb_url
            },
            'trading': {
                'max_position_size': self.trading.max_position_size_dollars,
                'max_daily_trades': self.trading.max_daily_trades,
                'max_daily_loss': self.trading.max_daily_loss
            },
            'monitoring': {
                'metrics_enabled': self.monitoring.metrics_enabled,
                'health_check_enabled': self.monitoring.health_check_enabled,
                'circuit_breaker_enabled': self.monitoring.circuit_breaker_enabled
            }
        }
    
    def get_sensitive_config(self) -> Dict[str, str]:
        """Get sensitive configuration (masked)"""
        api_key, secret_key = self.api.get_decrypted_alpaca_credentials()
        
        return {
            'alpaca_api_key': f"{api_key[:4]}...{api_key[-4:]}" if api_key else "Not configured",
            'alpaca_secret_key': f"{secret_key[:4]}...{secret_key[-4:]}" if secret_key else "Not configured",
            'telegram_token': f"{self.api.telegram_bot_token[:10]}..." if self.api.telegram_bot_token else "Not configured",
            'google_api_key': f"{self.api.google_api_key[:10]}..." if self.api.google_api_key else "Not configured"
        }
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled"""
        return self.trading_mode == TradingMode.LIVE
    
    def get_database_url(self, prefer_postgres: bool = True) -> str:
        """Get database URL with fallback"""
        if prefer_postgres and self.database.postgres_url:
            return self.database.postgres_url
        return self.database.sqlite_url

# Global settings instance
settings = OmniAlphaSettings()

# Validate on import
config_errors = settings.validate_configuration()
if config_errors:
    print(f"⚠️ Configuration warnings: {', '.join(config_errors)}")

def get_settings() -> OmniAlphaSettings:
    """Get global settings instance"""
    return settings

def reload_settings():
    """Reload settings from environment"""
    global settings
    settings = OmniAlphaSettings()
    return settings
